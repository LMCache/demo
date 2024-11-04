import time
import os, sys
import numpy as np
import pandas as pd
import streamlit as st
import chat_session
from typing import List, Dict
from transformers import AutoTokenizer

# Change the following variables as needed
MODEL_NAME = "lmsys/longchat-7b-16k"
PORT_MAPPING = {
        "vLLM w/ LMCache: A": 8000,
        "vLLM w/ LMCache: B": 8001,
        "Original vLLM: A": 8002,
        "Original vLLM: B": 8003,
    }

@st.cache_resource
def preheat():
    global PORT_MAPPING 
    preheat_context = "This is dummy text. " * 500
    for key in PORT_MAPPING.keys():
        port = PORT_MAPPING[key]
        session = chat_session.ChatSession(port)
        session.set_context([preheat_context])
        stream = session.chat("Please just say 'hello': ")
        for s in stream:
            print(s, end = "", flush = True)
        print("")
        session.set_context([preheat_context])
        stream = session.chat("Please just say 'hey': ")
        for s in stream:
            print(s, end = "", flush = True)
        print("")

preheat() 

@st.cache_resource
def get_tokenizer():
    global MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer

tokenizer = get_tokenizer()

@st.cache_data
def read_context() -> str:
    with open("context.txt", "r") as fin:
        context = fin.read()
    return context

context = read_context()

preheat()

container = st.container(border=True)

with st.sidebar:
    option = st.selectbox(
        "Select the serving engine",
        list(PORT_MAPPING.keys()),
    )

    port = PORT_MAPPING[option]

    session = chat_session.ChatSession(port)

    system_prompt = st.text_area(
            "System prompt:",
            "You are a helpful assistant. I will now give you a document and "
            "please answer my question afterwards based on the content in document"
        )

    session.set_context([system_prompt] + [context])
    num_tokens = tokenizer.encode(session.get_context())
    container.header(f"The context given to LLM: ({len(num_tokens)} tokens)", divider = "grey")
    container.text(session.get_context())

    messages = st.container(height=400)
    if prompt := st.chat_input("Type the question here"):
        messages.chat_message("user").write(prompt)
        messages.chat_message("assistant").write_stream(session.chat(prompt))
