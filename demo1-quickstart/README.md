# LMCache demo 1: quickstart

![image](https://github.com/user-attachments/assets/f0577071-c3ba-4094-a7d2-5b288fbd411b)


This demo will help you set up the vLLM + LMCache and a QA frontend.


## Prerequisites
To run the quickstart demo, your server should have 1 GPU and the [docker environment](https://docs.docker.com/engine/install/) with the [nvidia-runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed.

Depending on the server configuration, you may need sudo access to run the docker.

This demo will use the port 8000 (for vLLM) and 8501 (for the frontend).

## Clone the repo to local
```bash
git clone https://github.com/LMCache/demo
cd demo/demo1-quickstart
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
LOCAL_HF_HOME=                              # the HF_HOME on local machine. vLLM will try finding/downloading the models here
HF_TOKEN=                                   # (optional) the huggingface token to access some special models
```

Then, start the docker
```bash
bash ./run-server.sh  # This might need sudo
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
pip install openai streamlit
```

Then, start the web server using:
```bash
streamlit run frontend.py
```

You should be able to access the frontend from your browser at `http://<your server's IP>:8501`.

### What to expect:

LMCache should be able to reduce the response delay since the second question.
