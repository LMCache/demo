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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
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

chunks = read_chunks("ffmpeg/")

# Will not have a dropdown but need to show we use the same 
# chunks in some way.
selected_chunks = st.multiselect(
    "Select the chunks into the context",
    list(chunks.keys()),
    default = [],
    placeholder = "Select in the drop-down menu")

f = open("ffmpeg/sys_prompt.txt", "r")
sys_prompt = f.read()

gap_width = 0.013
col_width = (1 - gap_width) / 2
col1, col3, col2 = st.columns([col_width, gap_width, col_width], gap="small")

temperature = 0.0

with st.sidebar:
    st.markdown("### Settings")
    temperature = st.slider("Temperature: ", 0.0, 1.0, 0.0)
    st.markdown("### Optimization")
    kv_blend = st.checkbox("Use LMCache KV Blending")
    no_kv_blend = st.checkbox("Don't use LMCache")

with col1:
    port = PORT_LMCACHE
    st.markdown("#### LMCache + KV Blending")
    print("The port is:", port)
    print("Current temparature is:", temperature)

    session = chat_session.ChatSession(port)

    session.temperature = temperature
    session.separator = " # # "

    session.set_context([sys_prompt] + [chunks[key] for key in selected_chunks])
    num_tokens = tokenizer.encode(session.get_context())

    messages = st.container(height=400)
    messages.header(f"The context given to LLM: ({len(num_tokens)} tokens)", divider = "grey")
    # messages.text(session.get_context())

    if kv_blend:
        if prompt := st.chat_input("Type the question here", key = "blend"):
            messages.chat_message("user").write(prompt)
            messages.chat_message("assistant").write_stream(session.chat(prompt))

with col2:
    port = PORT_DEFAULT
    st.markdown("#### No LMCache")
    print("The port is:", port)
    print("Current temparature is:", temperature)

    session = chat_session.ChatSession(port)

    session.temperature = temperature
    session.separator = ""

    session.set_context([sys_prompt] + [chunks[key] for key in selected_chunks])
    num_tokens = tokenizer.encode(session.get_context())

    messages = st.container(height=400)
    messages.header(f"The context given to LLM: ({len(num_tokens)} tokens)", divider = "grey")
    # messages.text(session.get_context())

    if no_kv_blend:
        if prompt := st.chat_input("Type the question here", key = "no blend"):
            messages.chat_message("user").write(prompt)
            messages.chat_message("assistant").write_stream(session.chat(prompt))


    
