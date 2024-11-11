import time
import os, sys
import numpy as np
import pandas as pd
import streamlit as st
import chat_session
from typing import List, Dict
from transformers import AutoTokenizer

from huggingface_hub import login

# Set up the Hugging Face Hub credentials
hf_token = os.getenv("HF_TOKEN")
if hf_token is not None:
    login(token=hf_token)
    
MODEL_NAME = "lmsys/longchat-7b-16k"
PORT_LMCACHE = 8000
PORT_DEFAULT = 8001

@st.cache_resource
def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer


tokenizer = get_tokenizer()


@st.cache_data
def read_chunks(file_folder) -> Dict[str, str]:
    """
    Read all the txt files in the folder and return the filenames
    """
    filenames = os.listdir(file_folder)
    ret = {}
    for filename in filenames:
        if not filename.endswith("txt"):
            continue
        if filename == "sys_prompt.txt":
            continue
        key = filename.removesuffix(".txt")
        with open(os.path.join(file_folder, filename), "r") as fin:
            value = fin.read()
        ret[key] = value

    return ret

chunks = read_chunks("data/")
selected_chunks = st.multiselect(
    "Select the chunks into the context",
    list(chunks.keys()),
    default = [],
    placeholder = "Select in the drop-down menu")

# TODO(Jiayi): add system prompt support?
f = open("data/sys_prompt.txt")
sys_prompt = f.read()

container = st.container(border=True)
#container.header("The context given to LLM:", divider = "grey")


with st.sidebar:
    sys_container = st.container(border=True)
    sys_container.header("System prompt")
    sys_container.markdown(
        sys_prompt
    )
    temperature = st.slider("Temperature: ", 0.0, 1.0, 0.0)
    optimization = st.checkbox("Enable LMCacheBlend optimization")
    port = PORT_LMCACHE if optimization else PORT_DEFAULT

    print("The port is:", port)
    print("Current temparature is:", temperature)

    session = chat_session.ChatSession(port)



    session.temperature = temperature
    session.separator = " # # " if optimization else ""

    session.set_context([sys_prompt] + [chunks[key] for key in selected_chunks])
    num_tokens = tokenizer.encode(session.get_context())
    container.header(f"The context given to LLM: ({len(num_tokens)} tokens)", divider = "grey")
    container.text(session.get_context())

    messages = st.container(height=400)
    if prompt := st.chat_input("Type the question here"):
        messages.chat_message("user").write(prompt)
        messages.chat_message("assistant").write_stream(session.chat(prompt))

