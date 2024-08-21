# LMCache demo 2: share KV across different vLLM instances

![image](https://github.com/user-attachments/assets/123aba98-0bb9-4067-a061-f2b311a6cafd)

This demo shows how to share KV across different vLLM instances using LMCache


## Prerequisites
To run the quickstart demo, your server should have 2 GPUs and the [docker environment](https://docs.docker.com/engine/install/) with the [nvidia-runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed.

Depending on the server configuration, you may need sudo access to run the docker.

This demo will use the port 8000, 8001 (for vLLM), 65432 (for LMCache backend server), and 8501 (for the frontend).

## Clone the repo to local
```bash
git clone https://github.com/LMCache/demo
cd demo/demo2-multi-node-sharing
```

## Start the LMCache + vLLM with docker 

First, we need to do some configuration before starting the docker.
```bash
cp run-server.sh.template run-server.sh
vim run-server.sh
```

Edit the folloing lines based on your local environment:
```bash
MODEL=mistralai/Mistral-7B-Instruct-v0.2    # LLM model name
LOCAL_HF_HOME=                              # the HF_HOME on local machine. vLLM will try finding/dowloading the models here
HF_TOKEN=                                   # (optional) the huggingface token to access some special models
```

Then, start the docker images. 
```bash
bash ./run-server.sh  # This might need sudo
```

Now, you should have 3 docker images running.
- `lmcache-server` is the LMCache backend server that shares KV across multiple vLLM instaces
- `lmcache-vllm1` and `lmcache-vllm2` are the LMCache-integrated vLLM instaces 

## Start the frontend

The frontend is uses `openai` and `streamlit` python packages. Install them using:
```bash
pip install openai streamlit
```

Then, start the web server using:
```bash
streamlit run frontend.py
```

You should be able to access the frontend from your browser at `http://<your server's IP>:8501`.

In the demo, you can select different texts to make a long context, and ask questions to different vLLM instaces.

### What to expect:

- If the new context shares the same prefix as previously-used context, LMCache should be able to reduce the response delay by reusing the prefix KV cache.
- After vLLM instace 1 processes the context, vLLM instance 2 should be able to response much faster when loading the same context. This is because it can load KV cache from the LMCache server backend.
