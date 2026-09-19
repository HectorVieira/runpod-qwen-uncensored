# Qwen3.8-27B OBLITERATED — RunPod Serverless (llama.cpp)

Worker serverless que serve o GGUF **Qwen3.8-27B-OBLITERATED** (abliterated, uncensored) com `llama-server` (llama.cpp).

```
RunPod Endpoint
      │  POST /run  ou  /runsync
      ▼
  handler.py  ──supervisiona──►  llama-server  ──mmap──►  /runpod-volume/models/*.gguf
      │                              │
      └──────── /v1/chat/completions ┘
```

## Por que llama.cpp e não Ollama/vLLM

- **Sem cópia do modelo.** O `llama-server` faz `mmap()` do GGUF direto do network volume. O Ollama copiava o blob para o disco efêmero do worker a cada cold start — inviável para 17–26 GB.
- **Tool calling nativo.** `--jinja` habilita o chat template do modelo, incluindo function calling.
- **Cold start previsível.** O tempo de start é o carregamento dos pesos na VRAM, não uma cópia de disco.

A imagem base é a oficial `ghcr.io/ggml-org/llama.cpp:server-cuda`, **pinada em `b11046`**. Não use `latest`: o endpoint faz autodeploy a cada push, e uma tag móvel mudaria o runtime sem aviso. Para subir de versão, edite o `ARG LLAMA_CPP_IMAGE` no `Dockerfile`.

## Arquivos

| Arquivo | Papel |
|---|---|
| `Dockerfile` | Imagem: base llama.cpp CUDA + SDK do RunPod |
| `handler.py` | Handler serverless, provisiona o modelo e supervisiona o `llama-server` |
| `.dockerignore` | Exclui lixo do contexto de build |

## Modelo

Origem: `OBLITERATUS/Qwen3.8-27B-OBLITERATED` (público, sem gating, Apache 2.0).

Base **Qwen/Qwen3.8-27B** com abliteração iterativa (V3). O card reporta MMLU 82,3% (−2,1pp do stock), 20/20 em geração de código e 7/8 em tarefas agênticas — paridade com o modelo stock. Recusa e desvio: 0%, auditados manualmente pelo autor.

| Arquivo (`MODEL_FILE`) | Tamanho | VRAM sugerida |
|---|---|---|
| `Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf` | 16,8 GB | 24 GB+ |
| `Qwen3.8-27B-OBLITERATED-Q5_K_M.gguf` | 19,5 GB | 32 GB+ |
| **`Qwen3.8-27B-OBLITERATED-Q6_K.gguf`** | **22,4 GB** | **32 GB+ (default)** |
| `Qwen3.8-27B-OBLITERATED-Q8_0.gguf` | 29,1 GB | 48 GB+ |

O default é **Q6_K (22,4 GB)** num volume de 40 GB: um segundo modelo desse porte não cabe ao lado.

### ⚠️ `REPEAT_PENALTY=1.15` é obrigatório

O card do OBLITERATED é enfático: sem `repetition_penalty` a decodificação greedy degenera repetindo imports e boilerplate, e em uso agêntico o modelo entra em loop de tool call. O `Dockerfile` já traz `REPEAT_PENALTY=1.15`. O card também recomenda `temperature 0` (0,1–0,3 em uso agêntico); como isso é por request, quem controla é o cliente.

### ⚠️ `REASONING=off` por padrão

O modelo é de raciocínio e, com thinking ligado, gasta o orçamento inteiro em `reasoning_content` **antes** de escrever qualquer `content`. Medido:

| Prompt | max_tokens | finish_reason | content | reasoning |
|---|---|---|---|---|
| Árvore binária completa | 300 | `length` | **0 chars** | 300 tokens |
| Árvore binária completa | 1200 | `length` | **0 chars** | 1200 tokens |
| Árvore binária completa | 3000 | `stop` | 4097 chars | — |
| Curto (palíndromo) | 300 | `stop` | 72 chars | 289 chars |

Ou seja: em prompt de código real, uma resposta com teto modesto volta **vazia**. Num harness de agente isso é pior que lento — parece que o modelo travou.

O default do `Dockerfile` é `REASONING=off`, que o próprio card do OBLITERATED recomenda ("enable_thinking OFF (recommended)"). Com ele o modelo responde direto, com `reasoning_content` vazio e `content` imediato. Se quiser raciocínio de volta, é só trocar para `REASONING=on` — mas então garanta `max_tokens` alto (3000+) no cliente.

### ⚠️ O template do GGUF não suporta tools

O template embutido no GGUF do OBLITERATED tem **506 caracteres e não implementa tool calling** — o `/props` do llama-server reporta `supports_tools: false` e `supports_tool_calls: false`. Na prática o modelo responde de memória em vez de chamar a ferramenta (cheguei a vê-lo inventar um clima para Recife). Esse mesmo template força `<think>\n\n</think>` vazio, desligando o raciocínio.

A correção é o **template oficial do Qwen3.8-27B** (8952 chars, com tools), versionado aqui como `chat_template_qwen38.jinja` e aplicado via `--chat-template-file`. Como a abliteração preservou a capacidade do modelo (MMLU −2,1pp do stock), o que faltava era só o template.

### Como trocar de modelo

O volume de 40 GB comporta **um** modelo desses, então trocar exige liberar espaço primeiro:

```bash
# 1. remove outros *.gguf do volume (nunca o configurado)
curl -X POST .../runsync -d '{"input":{"command":"cleanup"}}'
# 2. baixa o novo explicitamente, ou deixe o boot baixar sozinho
curl -X POST .../runsync -d '{"input":{"command":"download"}}'
```

O `cleanup` só remove `*.gguf` que sejam filhos diretos de `MODEL_DIR` e cujo nome difira de `MODEL_FILE`: nunca recursa, nunca toca em não-GGUF, nunca apaga o modelo configurado. O comando `status` informa `disk_free_gb` para você conferir antes.

### Por que não o Qwen3.6-35B-A3B anterior

Ele era bem mais rápido — MoE de 3B ativos, **173 tok/s medidos** — mas de uma geração anterior e sem fine-tune de código. Nos benchmarks oficiais do Qwen, o Qwen3.8-27B supera o Qwen3.6-27B em código agêntico por margem grande: **QwenSWEBench 79,0 vs 49,3** e **DeepSWE 42,2 vs 13,3**. O custo da troca é velocidade: um 27B denso fica na casa de 35–50 tok/s.

O `MTP` **é** suportado pelo llama.cpp (`--spec-type draft-mtp`), mas **não habilite**: há bug aberto de carregamento com `-ngl` em GPU única ([issue #29044](https://github.com/ggml-org/llama.cpp/issues/29044), aberto um dia antes do build pinado) e regressão de ~57× no prefill ([#28790](https://github.com/ggml-org/llama.cpp/issues/28790)). Use sempre o quant **sem** MTP.

## Flags do llama-server

O build está pinado em `b11046`, e nessa versão algumas flags mudaram de forma que quebra a inicialização:

| Armadilha | Detalhe |
|---|---|
| `--no-mmap` | **Removido.** Não existe mais; aborta o start com "invalid argument". Use `--load-mode none` |
| `--flash-attn` / `--reasoning` | **Exigem valor.** O parser sempre consome o próximo token, então `-fa` sozinho engole o argumento seguinte e falha. Use `--flash-attn on` |
| `--mtp` | Existe no parser mas só para o exemplo de download; o `llama-server` o rejeita |
| `--spec-type mtp` | Valor errado. No b11046 o nome é `draft-mtp` |
| `--chat-template-kwargs '{"enable_thinking":...}'` | Deprecado. Use `--reasoning on\|off` |
| `--jinja` | Já é o default no b11046; mantido explícito para não depender do default |

O `handler.py` valida isso: um teste local garante que nenhuma flag removida é emitida e que toda flag que exige valor o recebe.


## Variáveis de ambiente

Todas configuráveis no endpoint, sem rebuild.

| Variável | Default | Descrição |
|---|---|---|
| `MODEL_DIR` | `/runpod-volume/models` | Onde o GGUF fica. **Serverless monta o volume em `/runpod-volume`**, não em `/workspace`. |
| `MODEL_REPO` | `OBLITERATUS/Qwen3.8-27B-OBLITERATED` | Repo do HuggingFace |
| `MODEL_FILE` | `...Q6_K.gguf` | Arquivo a servir |
| `MODEL_ALIAS` | `qwen3.8-27b-obliterated` | Nome do modelo nas respostas |
| `MODEL_AUTO_DOWNLOAD` | `1` | Baixa o GGUF no primeiro boot se estiver faltando |
| `MMPROJ_FILE` | *(vazio)* | Preencha com `mmproj-...gguf` para habilitar visão |
| `CTX_SIZE` | `32768` | Contexto. Valores altos consomem VRAM no KV cache |
| `GPU_LAYERS` | `99` | Camadas na GPU |
| `LOAD_MODE` | `mmap` | `auto\|none\|mmap\|mlock\|mmap+mlock\|dio`. Não existe `--no-mmap` |
| `N_CPU_MOE` | *(vazio)* | Offload de N camadas de experts MoE para CPU. Use se o GGUF quase encher a VRAM |
| `FLASH_ATTN` | *(vazio)* | `on\|off\|auto`. Vazio deixa o default do build (`auto`) |
| `REASONING` | *(vazio)* | `on\|off\|auto`. Use `off` para desligar o modo de raciocínio e ganhar latência |
| `PARALLEL` | *(vazio)* | Slots concorrentes do llama-server. Útil porque `workersMax=1` |
| `REPEAT_PENALTY` | `1.15` | Obrigatório para o OBLITERATED: sem ele a decodificação entra em loop |
| `CHAT_TEMPLATE_FILE` | `/app/chat_template_qwen38.jinja` | Substitui o template do GGUF, que não tem tools |
| `SERVER_START_TIMEOUT` | `900` | Segundos aguardando o `/health` |
| `LLAMA_EXTRA_ARGS` | *(vazio)* | Flags extras não modeladas acima |
| `LLAMA_SERVER_BIN` | `/app/llama-server` | Binário na imagem |

## Deploy

### 0. Pré-requisito: o volume precisa estar anexado ao endpoint

O código lê o modelo de `/runpod-volume/models`. Isso só funciona se o network volume estiver **anexado ao endpoint** no console do RunPod. Se não estiver, `/runpod-volume` não existe no worker e o download cairia no disco efêmero — sendo perdido no próximo cold start.

Confirme com:

```bash
curl -s https://rest.runpod.io/v1/endpoints/968l6bh7yree8t \
  -H "Authorization: Bearer $RUNPOD_API_KEY" | python3 -m json.tool | grep networkVolumeId
```

Um valor vazio (`""`) significa **sem volume anexado**.

### 1. Network volume

O volume precisa estar montado no endpoint em **`/runpod-volume`** (o RunPod monta aí automaticamente para serverless).

Se o GGUF ainda não estiver no volume, `MODEL_AUTO_DOWNLOAD=1` baixa em background no primeiro boot. O download é retomável: se o worker for morto pelo idle timeout no meio, o próximo start continua de onde parou. Enquanto isso, `/run` devolve `downloading: true` em vez de travar.

Para forçar e acompanhar o download, envie um job:

```bash
curl -X POST https://api.runpod.ai/v2/968l6bh7yree8t/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"command": "download"}}'
```

O mesmo endpoint aceita `{"input": {"command": "status"}}` para ver prontidão, caminho do modelo e erros.

### 2. Endpoint

- **GPU**: o endpoint está configurado como `ADA_48_PRO` — RTX 6000 Ada, L40 ou L40S. Todas têm 48 GB e são **sm_89**, cobertas nativamente (`89-real`) pela imagem do llama.cpp.
- **Container disk**: pequeno serve (o modelo não é copiado para o disco local). 10 GB bastam.
- **Idle Timeout**: hoje está em **5 s**. Com `workersMin=1` o worker não chega a escalar para zero, mas o valor é agressivo: se o worker for reciclado, o modelo recarrega do zero. Considere 60–300 s.
- **Expose HTTP Port**: **não é necessário**. Este é um worker serverless com handler Python, não um serviço HTTP atrás do proxy.

### 3. Invocação

O endpoint é serverless (`api.runpod.ai/v2/<id>`), então usa-se `/run` ou `/runsync` — **não** `proxy.runpod.net/v1/chat/completions`.

```bash
curl -X POST https://api.runpod.ai/v2/968l6bh7yree8t/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "messages": [{"role": "user", "content": "Hello!"}],
      "max_tokens": 256,
      "temperature": 0.6
    }
  }'
```

A resposta vem em `output`. Como o handler é um **gerador** (necessário para o gateway OpenAI transmitir SSE), o RunPod agrega os chunks: `output` é uma **lista** com um elemento, não o objeto direto.

```json
{"output": [{"choices": [{"message": {"content": "..."}}], "usage": {...}}]}
```

Formas de entrada aceitas:

| Entrada | Quando usar |
|---|---|
| `{"input": {"messages": [...]}}` | Forma curta |
| `{"input": {"prompt": "..."}}` | Turno único |
| `{"input": {"openai_route": ..., "openai_input": {...}}}` | O que o gateway OpenAI do RunPod envia |
| `{"input": {"openai_route": "/v1/models"}}` | Rota sem corpo vira GET |

`temperature`, `top_p`, `max_tokens`, `tools`, `stop` são repassados ao `llama-server`. Na forma curta (`messages`/`prompt`) o `stream` é forçado para `false`, porque o RunPod devolve um único documento JSON e não há stream para retransmitir.

### 4. Endpoint compatível com OpenAI

O RunPod expõe um gateway que traduz chamadas OpenAI em jobs e retransmite o SSE do worker:

```
https://api.runpod.ai/v2/968l6bh7yree8t/openai/v1
```

Use a **RunPod API key** como chave (não é uma chave OpenAI):

```bash
curl https://api.runpod.ai/v2/968l6bh7yree8t/openai/v1/chat/completions \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.8-27b-obliterated",
       "messages": [{"role": "user", "content": "Ola"}],
       "max_tokens": 512, "stream": true}'
```

Duas restrições do gateway, verificadas experimentalmente:

- **`stream: true` é obrigatório.** Sem ele o gateway responde `500` — ele só retransmite Server-Sent Events. É por isso que o handler é um gerador assíncrono.
- `GET /v1/models` funciona, mas clientes que dependem dele para descobrir modelos devem usar o alias exato `qwen3.8-27b-obliterated`.

Para configurar no **DeepSeek Harness** (Settings → Models → Add a custom provider):

| Campo | Valor |
|---|---|
| Endpoint | `https://api.runpod.ai/v2/968l6bh7yree8t/openai/v1` |
| API protocol | `openai-completions` |
| API key | a RunPod API key |
| Model ID | `qwen3.8-27b-obliterated` |


## Troubleshooting

| Sintoma | Causa provável |
|---|---|
| `network volume is not mounted at /runpod-volume` | O volume não está anexado ao endpoint. O handler recusa baixar 25 GB para o disco efêmero |
| `"downloading": true` por muito tempo | Download inicial em andamento. Acompanhe os logs do worker |
| `model not found ... and MODEL_AUTO_DOWNLOAD is off` | Volume sem o GGUF e download desativado |
| `llama-server exited early` | Veja os logs `[llama-server]` acima da mensagem — normalmente flag inválida, ou GGUF maior que a VRAM |
| `llama-server not ready after Ns` | Modelo grande demais para a GPU, ou volume lento. Aumente `SERVER_START_TIMEOUT` |
| Erro de CUDA / `no kernel image is available` | Build do llama.cpp sem suporte à arquitetura da GPU. Em RTX 5090 (sm_120) exige CUDA 12.8+ |
| Resposta sem `tool_calls` | Confirme que a flag `--jinja` está ativa (é default no `handler.py`) |

## Notas

- A imagem base é CUDA 12.8 / Ubuntu 24.04, compatível com Blackwell (RTX 5090, sm_120).
- `LD_LIBRARY_PATH=/app` é necessário porque o llama.cpp é buildado com `GGML_BACKEND_DL=ON` e carrega os backends `.so` relativos ao binário.
- Logs do `llama-server` são replicados no stdout do worker com o prefixo `[llama-server]`, e o handler os supervisiona: se o processo morrer, ele é reiniciado.
