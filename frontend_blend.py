import time
import os, sys
import numpy as np
import pandas as pd
import streamlit as st
import chat_session_blend
from typing import List, Dict
from transformers import AutoTokenizer

@st.cache_resource
def get_tokenizer():
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
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
    sys_container.text(
        sys_prompt
    )
    temperature = st.slider("Temperature: ", 0.0, 1.0, 0.0)
    optimization = st.checkbox("Enable LMCacheBlend optimization")
    port = 8000 if optimization else 8001
    
    print("The port is:", port)
    print("Current temparature is:", temperature)

    session = chat_session_blend.ChatSession(port)

    
    
    session.temperature = temperature
    session.separator = " # # " if optimization else ""
    
    session.set_context([sys_prompt] + [chunks[key] for key in selected_chunks])
    container.text(session.get_context())

    messages = st.container(height=400)
    if prompt := st.chat_input("Type the question here"):
        messages.chat_message("user").write(prompt)
        messages.chat_message("assistant").write_stream(session.chat(prompt))
