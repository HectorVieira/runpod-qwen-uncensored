# Qwen3.6-35B-A3B Uncensored - RunPod Serverless

Docker image para RunPod Serverless com o modelo **Qwen3.6-35B-A3B-Uncensored-Genesis-Hermes-V6**.

## Modelo
- **Base**: Qwen3.6-35B-A3B (MoE, 35B total, 3B ativos)
- **Quantização**: Q8_0 (alta qualidade)
- **特性**: Uncensored, function calling suportado
- **VRAM necessária**: ~18GB (Q8_0 com 35B params MoE)

## Deploy no RunPod

### 1. Build e push da imagem
```bash
# Build
docker build -t runpod/qwen36-uncensored .

# Tag para RunPod
docker tag runpod/qwen36-uncensored YOUR_REGISTRY/qwen36-uncensored:latest

# Push (substitua pelo seu registry)
docker push YOUR_REGISTRY/qwen36-uncensored:latest
```

### 2. Criar endpoint no RunPod
1. Acesse https://www.runpod.io/console/serverless
2. Clique "New Endpoint"
3. Selecione a imagem pushada
4. GPU: **A100 40GB** ou **A100 80GB** (recomendado)
5. GPU Count: 1
6. Expose HTTP Port: **8080**
7. Idle Timeout: 5s (para economizar)

### 3. Testar
```bash
curl https://YOUR_ENDPOINT_ID.proxy.runpod.net/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-35b-uncensored",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### 4. Configurar no Hermes
```yaml
model:
  base_url: https://YOUR_ENDPOINT_ID.proxy.runpod.net/v1
  api_key: rpa_YOUR_API_KEY
  api_mode: chat_completions
  default: qwen3.6-35b-uncensored
  provider: custom:runpod
  timeout: 300
```

## Specs
- **Modelo**: Qwen3.6-35B-A3B-Uncensored-Genesis-Hermes-V6-Q8_0.gguf
- **Tamanho**: ~18GB (Q8_0)
- **VRAM**: ~20GB recomendado (A100 40GB mínimo)
- **Contexto**: 8192 tokens
- **Throughput**: ~30-50 tokens/s (A100 40GB)
