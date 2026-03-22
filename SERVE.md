# llm-serve: LLaMA Inference Server

A lightweight, zero-dependency LLaMA inference server written in pure C. Loads GGUF model files and exposes an OpenAI-compatible HTTP API for text completion.

## Table of Contents

- [Overview](#overview)
- [Building](#building)
- [Running the Server](#running-the-server)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Quantization Support](#quantization-support)
- [Deployment Patterns](#deployment-patterns)
- [Performance Tuning](#performance-tuning)
- [Limitations](#limitations)
- [Testing](#testing)

---

## Overview

`llm-serve` is designed to be the simplest possible path from a GGUF model file to a running inference endpoint. It's a single C file (`serve.c`) with header-only dependencies, no external libraries beyond libc and optionally OpenMP.

**What it's good for:**

- **Local development & prototyping** — spin up a model on your machine and hit it with curl or any OpenAI-compatible client
- **Edge / embedded inference** — minimal footprint, no Python runtime, no framework overhead
- **Learning & experimentation** — the entire inference pipeline (tokenizer, forward pass, sampling, HTTP) is readable in ~900 lines of C
- **Model testing** — quickly validate that a GGUF model loads and produces reasonable output before integrating into a larger system
- **Architecture research** — the code is simple enough to modify for custom attention patterns, activation functions, or architectural experiments

**What it's NOT designed for:**

- **Production at scale** — single-threaded request handling (no concurrent requests), no batching, no request queuing
- **Maximum throughput** — CPU-only inference without SIMD-optimized kernels; good enough for interactive use, not for bulk processing
- **Full OpenAI API compatibility** — only `/v1/completions` is implemented (no `/v1/chat/completions`, embeddings, etc.)

---

## Building

### Prerequisites

- A C compiler (clang or gcc)
- make
- OpenMP (optional but strongly recommended for performance)

**Install OpenMP:**

```bash
# Ubuntu/Debian
sudo apt-get install libomp-dev

# macOS (Homebrew)
brew install libomp

# Fedora/RHEL
sudo dnf install libomp-devel
```

### Compile

```bash
make llm-serve
```

This produces the `./llm-serve` binary. The Makefile auto-detects OpenMP and compiles with `-Ofast -march=native` for best performance on your hardware.

### Debug build

For stepping through with gdb/lldb, build with debug symbols:

```bash
CC=gcc CFLAGS="-g -O0 -fopenmp -DOMP" make llm-serve
```

---

## Running the Server

### Basic usage

```bash
./llm-serve -m /path/to/model.gguf
```

The server loads the model, prints config info to stderr, and begins listening on port 8080. You'll see output like:

```
OpenMP threads: 4
Loading GGUF model: /path/to/model.gguf
  arch=llama layers=22 heads=32 kv_heads=4 embed=2048 ffn=5632 ctx=2048
  vocab_size=32000 bos=1 eos=2
Model loaded.
llm-serve listening on http://0.0.0.0:8080
  POST /v1/completions
  GET  /health
```

### Quick test

```bash
# Health check
curl http://localhost:8080/health

# Generate text
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!", "max_tokens": 32}'
```

---

## Configuration

All configuration is via command-line flags. There is no config file.

| Flag | Long form | Default | Description |
|------|-----------|---------|-------------|
| `-m` | `--model` | *(required)* | Path to a GGUF model file |
| `-p` | `--port` | `8080` | TCP port to listen on |
| `-c` | `--context` | model default | Context window size (tokens). Overrides the model's default if smaller. Larger context = more RAM for KV cache |
| `-t` | `--threads` | `4` | Number of OpenMP threads for parallel matrix-vector operations |

### Context size

The context window determines the maximum combined length of prompt + generated tokens. Memory usage scales linearly with context size:

```
KV cache memory ≈ 2 × n_layers × context_len × n_kv_heads × head_dim × 4 bytes
```

For TinyLlama 1.1B with 256 context:
```
2 × 22 × 256 × 4 × 64 × 4 = ~3.5 MB
```

For TinyLlama 1.1B with 2048 context (default):
```
2 × 22 × 2048 × 4 × 64 × 4 = ~28 MB
```

Use `-c` to cap context when you want to save memory or don't need long sequences:

```bash
# Short context for quick tests
./llm-serve -m model.gguf -c 256

# Full context for longer generation
./llm-serve -m model.gguf -c 2048
```

### Thread count

More threads help with the matrix-vector multiplications in the forward pass. General guidelines:

| Scenario | Threads |
|----------|---------|
| Laptop / casual use | 2–4 |
| Desktop / workstation | 4–8 |
| Server with many cores | 8–16 |
| Diminishing returns | >16 for small models |

Set via flag or environment variable:

```bash
# Via flag
./llm-serve -m model.gguf -t 8

# Or via OMP environment (flag takes precedence)
OMP_NUM_THREADS=8 ./llm-serve -m model.gguf
```

### Port

Change the listening port with `-p`:

```bash
# Custom port
./llm-serve -m model.gguf -p 9090

# Multiple instances with different models
./llm-serve -m small.gguf -p 8080 &
./llm-serve -m large.gguf -p 8081 &
```

---

## API Reference

### `GET /health`

Returns server health status.

**Response:**

```json
{"status": "ok"}
```

Use this for readiness probes, load balancer health checks, or waiting for the server to finish loading.

### `POST /v1/completions`

OpenAI-compatible text completion endpoint.

**Request body (JSON):**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | *(required)* | Input text to complete |
| `max_tokens` | integer | `128` | Maximum number of tokens to generate |
| `temperature` | float | `0.7` | Sampling temperature. `0` = greedy/argmax, higher = more random |
| `top_p` | float | `0.9` | Nucleus sampling threshold. `1.0` = disabled |
| `stream` | boolean | `false` | Enable Server-Sent Events (SSE) streaming |

**Non-streaming response:**

```json
{
  "choices": [
    {
      "text": " Paris, the largest city in France...",
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 7,
    "completion_tokens": 12
  }
}
```

`finish_reason` is `"stop"` for both EOS and max_tokens (the server doesn't distinguish).

**Streaming response:**

When `stream: true`, the server sends SSE events:

```
data: {"choices":[{"text":" Paris","finish_reason":null}]}

data: {"choices":[{"text":",","finish_reason":null}]}

data: {"choices":[{"text":" the","finish_reason":null}]}

data: [DONE]
```

**Error responses:**

| Status | Condition |
|--------|-----------|
| 400 | Empty body or missing `prompt` field |
| 404 | Unknown endpoint |

```json
{"error": {"message": "missing prompt", "code": 400}}
```

### Examples

**Greedy decoding (deterministic):**

```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The meaning of life is",
    "max_tokens": 64,
    "temperature": 0
  }'
```

**Creative generation:**

```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a haiku about programming:\n",
    "max_tokens": 32,
    "temperature": 0.9,
    "top_p": 0.95
  }'
```

**Streaming with curl:**

```bash
curl -N -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "max_tokens": 100,
    "temperature": 0.8,
    "stream": true
  }'
```

**Python client (requests):**

```python
import requests

resp = requests.post("http://localhost:8080/v1/completions", json={
    "prompt": "Hello,",
    "max_tokens": 32,
    "temperature": 0.7
})
print(resp.json()["choices"][0]["text"])
```

**Python client (OpenAI SDK):**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
resp = client.completions.create(
    model="local",  # ignored by llm-serve, but required by the SDK
    prompt="The answer is",
    max_tokens=32
)
print(resp.choices[0].text)
```

---

## Quantization Support

llm-serve loads GGUF files and dequantizes tensor data on the fly during inference.

### Supported tensor types

| Type | Bits/weight | Description | Status |
|------|-------------|-------------|--------|
| `F32` | 32 | Full precision float | ✅ Supported |
| `F16` | 16 | Half precision float | ✅ Supported |
| `Q8_0` | 8.5 | 8-bit quantization | ✅ Supported |
| `Q4_0` | 4.5 | 4-bit quantization (basic) | ✅ Supported |
| `Q4_1` | 5.0 | 4-bit with min offset | ✅ Supported |
| `Q4_K` | ~4.5 | 4-bit k-quant (super blocks) | ✅ Supported (via `gguf_dequant.h`) |
| `Q5_K` | ~5.5 | 5-bit k-quant | ✅ Supported (via `gguf_dequant.h`) |
| `Q6_K` | ~6.6 | 6-bit k-quant | ✅ Supported (via `gguf_dequant.h`) |
| `Q2_K` | ~3.35 | 2-bit k-quant | ✅ Supported (via `gguf_dequant.h`) |
| `Q3_K` | ~3.4 | 3-bit k-quant | ✅ Supported (via `gguf_dequant.h`) |
| `Q5_0` | ~5.5 | 5-bit quantization (basic) | ✅ Supported (via `gguf_dequant.h`) |
| `Q5_1` | ~5.5 | 5-bit with min offset | ✅ Supported (via `gguf_dequant.h`) |
| `IQ4_NL` | ~4.5 | Importance-weighted 4-bit | ❌ Not yet |

### Choosing a quantization

For local inference with llm-serve:

- **Q4_K_M** — best balance of quality and size for most models. This is the sweet spot.
- **Q8_0** — near-lossless quality, ~2x the size of Q4_K_M.
- **Q4_0** — slightly worse quality than Q4_K but faster dequantization.
- **Q6_K** — good quality when you can spare the extra memory.
- **Q2_K / Q3_K** — for squeezing large models into limited RAM; noticeable quality loss.

### Model sources

Pre-quantized GGUF models are available from:

- [Hugging Face](https://huggingface.co/models?library=gguf) — search for "GGUF" in any model repo
- [TheBloke](https://huggingface.co/TheBloke) — extensive collection of quantized models

### Extending dequantization

The k-quant types are implemented in `llmc/gguf_dequant.h` using a registry pattern. To add a new quantization format:

1. Add the type enum to the `GGMLType` enum in `gguf.h`
2. Register block size and type size in `gguf.h`
3. Implement a `gguf_dequant_<type>_row` function in `gguf_dequant.h`
4. Register it with `gguf_dequant_register()` in the init function

See the [gguf_dequant.h](llmc/gguf_dequant.h) header for the full pattern.

---

## Deployment Patterns

### Local development

The simplest setup — run the server and point your app at it:

```bash
./llm-serve -m model.gguf -p 8080 -t 4
```

### systemd service

For persistent deployment on a Linux server:

```ini
# /etc/systemd/system/llm-serve.service
[Unit]
Description=llm-serve inference server
After=network.target

[Service]
Type=simple
ExecStart=/opt/llm-serve/llm-serve -m /opt/models/model.gguf -p 8080 -t 8
Restart=on-failure
RestartSec=5
User=llmserve
WorkingDirectory=/opt/llm-serve

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now llm-serve
```

### Behind a reverse proxy (nginx)

```nginx
upstream llm_backend {
    server 127.0.0.1:8080;
}

server {
    listen 443 ssl;
    server_name llm.example.com;

    location /v1/ {
        proxy_pass http://llm_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_read_timeout 300s;  # long timeout for generation

        # SSE streaming support
        proxy_set_header Accept-Encoding "";
        proxy_buffering off;
        chunked_transfer_encoding off;
    }

    location /health {
        proxy_pass http://llm_backend;
    }
}
```

### Docker

```dockerfile
FROM ubuntu:22.04 AS builder
RUN apt-get update && apt-get install -y build-essential libomp-dev
COPY . /src
WORKDIR /src
RUN make llm-serve

FROM ubuntu:22.04
RUN apt-get update && apt-get install -y libomp5 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /src/llm-serve /usr/local/bin/
EXPOSE 8080
ENTRYPOINT ["llm-serve"]
CMD ["-m", "/models/model.gguf", "-p", "8080", "-t", "4"]
```

```bash
docker build -t llm-serve .
docker run -v /path/to/models:/models -p 8080:8080 llm-serve
```

---

## Performance Tuning

### Thread count

Profile with different thread counts to find the sweet spot for your hardware:

```bash
# Quick benchmark: time a single request
time curl -s -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello","max_tokens":64,"temperature":0}' > /dev/null
```

### Context size

Smaller context = faster generation (attention is O(n²) over context length):

```bash
# Fast testing with short context
./llm-serve -m model.gguf -c 256

# Production with full context
./llm-serve -m model.gguf -c 2048
```

### Model selection

Inference speed is dominated by model size. Rough CPU generation speeds on a modern x86 machine:

| Model size | Q4_K_M | Q8_0 | Speed (tokens/sec, 8 threads) |
|-----------|--------|------|-------------------------------|
| ~1B params | ~700 MB | ~1.1 GB | 5–15 tok/s |
| ~3B params | ~1.8 GB | ~3.2 GB | 2–6 tok/s |
| ~7B params | ~4.1 GB | ~7.2 GB | 1–3 tok/s |

These are ballpark numbers — actual speed depends on your CPU, memory bandwidth, and quantization.

### Memory usage

Total memory ≈ model file size + KV cache + run state (~1 MB) + overhead

The model file is loaded entirely into RAM. There is no mmap or lazy loading (yet).

---

## Limitations

Things to be aware of:

- **Single request at a time** — the server handles one HTTP request to completion before accepting the next. No concurrency.
- **No chat template** — only raw text completion. You'll need to format chat prompts yourself (apply the model's chat template in your client).
- **No KV cache reuse** — each request resets the KV cache. No session/conversation continuity between requests.
- **No GPU acceleration** — CPU inference only. For GPU inference, use the CUDA training code or llama.cpp.
- **Basic JSON parsing** — the request parser is naive (no nested objects, no arrays). Stick to the documented fields.
- **No auth** — no API key validation. Don't expose directly to the internet without a reverse proxy.
- **No graceful shutdown** — Ctrl+C kills it. No drain period for in-flight requests.

---

## Testing

### Integration test

A bash integration test validates the full happy path:

```bash
# Run with the default TinyLlama model
./test_serve_integration.sh

# Specify a model
./test_serve_integration.sh /path/to/model.gguf
```

**Requirements:** `curl`, `jq`

The test builds the server, starts it on port 18199, runs 6 test groups (health check, greedy completion, sampled completion, error handling, 404, streaming), and tears down. Takes 1–2 minutes depending on model load time.

See the [README](README.md#integration-test) for details on what each test checks.

### Manual smoke test

```bash
# 1. Build & start
make llm-serve
./llm-serve -m model.gguf -p 8080 -c 256 -t 4

# 2. In another terminal — health check
curl http://localhost:8080/health
# → {"status":"ok"}

# 3. Generate something
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"2+2=","max_tokens":4,"temperature":0}'
# → {"choices":[{"text":"4","finish_reason":"stop"}],...}

# 4. Test streaming
curl -N -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Count to 5: 1,","max_tokens":16,"temperature":0,"stream":true}'
# → data: {"choices":[{"text":" 2","finish_reason":null}]}
# → data: {"choices":[{"text":",","finish_reason":null}]}
# → ...
# → data: [DONE]
```
