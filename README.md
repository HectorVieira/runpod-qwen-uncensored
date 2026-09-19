# Qwen3.6-35B-A3B Uncensored — RunPod Serverless (llama.cpp)

Worker serverless que serve o GGUF **Qwen3.6-35B-A3B-Uncensored-Genesis-Hermes** com `llama-server` (llama.cpp).

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

Origem: `LuffyTheFox/Qwen3.6-35B-A3B-Uncensored-Genesis-Hermes-V6-GGUF` (público, sem gating).

| Arquivo (`MODEL_FILE`) | Tamanho | VRAM sugerida |
|---|---|---|
| `Hermes3.6-35B-A3B-Uncensored-Genesis-Final-APEX-Compact.gguf` | 17,4 GB | 24 GB+ |
| `Hermes3.6-35B-A3B-Uncensored-Genesis-Final-MTP-APEX-Compact.gguf` | 18,3 GB | 24 GB+ |
| `Hermes3.6-35B-A3B-Uncensored-Genesis-Final-APEX.gguf` | 25,8 GB | 32 GB+ |
| `Hermes3.6-35B-A3B-Uncensored-Genesis-Final-MTP-APEX.gguf` | 26,7 GB | 32 GB+ |
| `Hermes3.6-35B-A3B-Uncensored-Genesis-Final-Q8_K_P.gguf` | 43,6 GB | 48 GB+ |
| `mmproj-Hermes3.6-35B-A3B-Uncensored-Genesis.gguf` | 0,9 GB | opcional (visão) |

O **default é `APEX` (25,8 GB)**, dimensionado para este endpoint: 48 GB de VRAM e volume de 40 GB. Sobram ~22 GB para KV cache, o que acomoda `CTX_SIZE=32768` com folga.

**`Q8_K_P` (43,6 GB) não cabe** no volume de 40 GB — exigiria aumentar o volume primeiro.

O `MTP` **é** suportado pelo llama.cpp (`--spec-type draft-mtp`), mas **não habilite**: há bug aberto de carregamento com `-ngl` em GPU única ([issue #29044](https://github.com/ggml-org/llama.cpp/issues/29044), aberto um dia antes do build pinado) e regressão de ~57× no prefill ([#28790](https://github.com/ggml-org/llama.cpp/issues/28790)). Por isso o default é o arquivo **sem** MTP.

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
| `MODEL_REPO` | `LuffyTheFox/...Hermes-V6-GGUF` | Repo do HuggingFace |
| `MODEL_FILE` | `...APEX-Compact.gguf` | Arquivo a servir |
| `MODEL_ALIAS` | `qwen3.6-35b-uncensored` | Nome do modelo nas respostas |
| `MODEL_AUTO_DOWNLOAD` | `1` | Baixa o GGUF no primeiro boot se estiver faltando |
| `MMPROJ_FILE` | *(vazio)* | Preencha com `mmproj-...gguf` para habilitar visão |
| `CTX_SIZE` | `32768` | Contexto. Valores altos consomem VRAM no KV cache |
| `GPU_LAYERS` | `99` | Camadas na GPU |
| `LOAD_MODE` | `mmap` | `auto\|none\|mmap\|mlock\|mmap+mlock\|dio`. Não existe `--no-mmap` |
| `N_CPU_MOE` | *(vazio)* | Offload de N camadas de experts MoE para CPU. Use se o GGUF quase encher a VRAM |
| `FLASH_ATTN` | *(vazio)* | `on\|off\|auto`. Vazio deixa o default do build (`auto`) |
| `REASONING` | *(vazio)* | `on\|off\|auto`. Use `off` para desligar o modo de raciocínio e ganhar latência |
| `PARALLEL` | *(vazio)* | Slots concorrentes do llama-server. Útil porque `workersMax=1` |
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

A resposta vem em `output`, no formato OpenAI (`choices[0].message.content`).

Formas de entrada aceitas:

| Entrada | Quando usar |
|---|---|
| `{"input": {"messages": [...]}}` | Forma curta |
| `{"input": {"prompt": "..."}}` | Turno único |
| `{"input": {"openai_input": {...}}}` | Cliente que já fala o formato do proxy OpenAI do RunPod |

`temperature`, `top_p`, `max_tokens`, `tools`, `stop` são repassados ao `llama-server`. `stream` é sempre forçado para `false` internamente: o RunPod devolve um único documento JSON, e pedir SSE ao llama-server quebraria a resposta.

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
