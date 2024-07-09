# Demo for LMCache

![image](https://github.com/LMCache/demo/assets/25103655/64fcf08d-d094-46e5-a280-2439fd0cb445)


## Installation

```bash
pip install openai streamlit
```

## Run the demo

The demo needs to talk with 2 vLLM serving engine at localhost:8000 and localhost:8001.

It assumes vLLM at port 8000 has the LMCache optimizations and the other one is non-optimized.

In the example.yaml, several config parameters need to be set:
```
cache_path: path where precomputed KV cahce are stored (default: "cache.pt")
local_device: "cpu" or "gpu" (default: "cpu")
separator_id: chunk separator which does not occur in the text chunks (default: "[422,422]")
```

Precompute KV cache and store them on disk
```
python precompute.py --lmcache-config-file example.yaml --data-path data
```

Start vLLM w/ LMCache (precomputed KV cache are automattically loaded to local device)
```
python3 -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.2 --port 8000 --gpu-memory-utilization 0.6 --lmcache-config-file example.yaml
```

Start vLLM w/o LMCache
```
CUDA_VISIBLE_DEVICES=1 python3 -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.2 --port 8001 --gpu-memory-utilization 0.6
```

Run the frontend
```
streamlit run frontend_blend.py
```
