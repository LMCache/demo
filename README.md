# Demo for LMCache

## Installation

```bash
pip install openai streamlit
```

## Run the demo

The demo needs to talk with 2 vLLM serving engine at localhost:8000 and localhost:8001.

It assumes vLLM at port 8000 has the LMCache optimizations and the other one is non-optimized.

Start vLLM w/ LMCache
```
python3 -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.2 --port 8000 --gpu-memory-utilization 0.6 --lmcache-config-file example.yaml
```

Start vLLM w/o LMCache
```
CUDA_VISIBLE_DEVICES=1 python3 -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.2 --port 8001 --gpu-memory-utilization 0.6
```

Run the frontend
```
streamlit run frontend.py
```
