# Qwen3.6-35B-A3B Uncensored - RunPod Serverless

Docker image para RunPod Serverless com o modelo **Qwen3.6-35B-A3B-Uncensored-Genesis-Hermes-V6**.

## Arquitetura

```
RunPod Endpoint → handler.py → Ollama → /runpod-volume/models/*.gguf
```

- **Modelo**: Armazenado no Network Volume (`/runpod-volume/models/`)
- **Persistente**: Modelo baixado uma vez, reutilizado em todos os cold starts
- **GPU**: RTX 5090 (32GB) ou A6000/L40S (48GB)

## Deploy no RunPod

### 1. Criar Network Volume
1. Acesse https://console.runpod.io/volumes
2. Clique "Create Network Volume"
3. Selecione o datacenter com **RTX 5090** ou **48GB GPU**
4. Tamanho: **50GB** mínimo

### 2. Build e push da imagem
```bash
cd runpod-qwen-uncensored

# Build
docker build -t runpod/qwen36-uncensored .

# Tag
docker tag runpod/qwen36-uncensored YOUR_REGISTRY/qwen36-uncensored:latest

# Push
docker push YOUR_REGISTRY/qwen36-uncensored:latest
```

### 3. Criar endpoint no RunPod
1. Acesse https://www.runpod.io/console/serverless
2. Clique "New Endpoint"
3. Selecione a imagem pushada
4. GPU: **RTX 5090** (32GB) ou **A6000/L40S** (48GB)
5. **Attach Network Volume**: Selecione o volume criado
6. Mount Path: `/runpod-volume`
7. Expose HTTP Port: **8080**
8. Idle Timeout: 5s

### 4. Testar
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

## Modelo

| Quantização | Tamanho | GPU 32GB | GPU 48GB |
|---|---|---|---|
| APEX | 23.9 GB | ✅ | ✅ |
| APEX-Compact | 16.2 GB | ✅ | ✅ |
| Q8_0 | 34.4 GB | ❌ | ✅ |

**Default**: APEX (23.9GB) — melhor equilíbrio qualidade/performance

## Performance Estimada

| GPU | Tokens/s | Latência |
|---|---|---|
| RTX 5090 (32GB) | ~30-50 | ~100ms/token |
| RTX A6000 (48GB) | ~20-40 | ~150ms/token |
| L40S (48GB) | ~25-45 | ~120ms/token |
