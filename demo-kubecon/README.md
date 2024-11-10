# LMCache demo 3: Speed-up RAG by LMCache's KV cache blending feature

![image](https://github.com/user-attachments/assets/2a61160a-162c-4d9a-833a-8f5e02547484)

Usually, we cannot do prefix sharing in RAG use cases, because the retrieved documents can be very different across different requests.

To speed up such use cases, LMCache support quickly blending the KV caches from standalone documents/text chunks.

This demo demonstrates the capability of using LMCache in RAG use cases.

## Prerequisites
To run the quickstart demo, your server should have 2 GPUs and the [docker environment](https://docs.docker.com/engine/install/) with the [nvidia-runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed.

Depending on the server configuration, you may need sudo access to run the docker.

This demo will use the port 8000, 8001 (for vLLM), and 8501 (for the frontend).

## Clone the repo to local
```bash
git clone https://github.com/LMCache/demo
cd demo/demo3-KV-blending
```

## Start the LMCache + vLLM with docker 

First, we need to do some configuration before starting the docker.
```bash
cp run-server.sh.template run-server.sh
vim run-server.sh
```

Edit the following lines based on your local environment:
```bash
MODEL=mistralai/Mistral-7B-Instruct-v0.2    # LLM model name
LOCAL_HF_HOME=                              # the HF_HOME on local machine. vLLM will try finding/dowloading the models here
HF_TOKEN=                                   # (optional) the huggingface token to access some special models
```

Then, start the docker images. 
```bash
bash ./run-server.sh  # This might need sudo
```

The script will first load all the text chunks in `ffmpeg/` folder and calculate the KV cache for each chunk separately.

Then, it will start two docker images, `lmcache-vllm1` (vLLM with LMCache) and `lmcache-vllm2` (vLLM w/o LMCache).

You can monitor the logs by:
```bash
sudo docker logs --follow lmcache-vllm1 
```

The vLLM serving engine is ready after you see the following lines in the log:
```text
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## Start the frontend

The frontend uses `openai` and `streamlit` python packages. Install them using:
```bash
pip install openai streamlit sentencepiece langchain-text-splitters
```

After the serving engine is ready, start the frontend web server using:
```bash
streamlit run frontend.py
```

You should be able to access the frontend from your browser at `http://<your server's IP>:8501`.

In the demo, you can select different text chunks and re-order them to make a long context, and ask questions to different vLLM instances (with or without LMCache).

## Stop the docker images
```bash
sudo bash stop-dockers.sh
```

### What to expect:

- With the help of LMCache, the vLLM should be able to answer the questions with a lower response delay (time to first token).
