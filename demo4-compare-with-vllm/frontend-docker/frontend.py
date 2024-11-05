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

if len(sys.argv) != 5:
    print(f"Usage: streamlit run {sys.argv[0]} <vLLM IP> <vLLM port> <engine name> <1 or 0>")
    exit(-1)


IP = sys.argv[1]
PORT = int(sys.argv[2])
ENGINE_NAME = sys.argv[3]


@st.cache_resource
def preheat():
    global IP, PORT
    preheat_context = "This is dummy text. " * 750
    session = chat_session.ChatSession(IP, PORT)
    stream = session.chat("Please just say 'hello': ")
    for s in stream:
        print(s, end = "", flush = True)
    print("")
    session.set_context([preheat_context])
    stream = session.chat("Please just say 'hey': ")
    for s in stream:
        print(s, end = "", flush = True)
    print("")


if int(sys.argv[4]) > 0:
    preheat() 

@st.cache_resource
def get_tokenizer():
    global MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    return tokenizer

tokenizer = get_tokenizer()

@st.cache_data
def read_context() -> str:
    with open("context.txt", "r") as fin:
        context = fin.read()
    return context

context = read_context()

container = st.container(border=True)

with st.sidebar:
    port = PORT

    st.header(f"Current engine: _'{ENGINE_NAME}'_")

    session = chat_session.ChatSession(IP, port)

    system_prompt = st.text_area(
            "System prompt:",
            "You are a helpful assistant. I will now give you a document and "
            "please answer my question afterwards based on the content in document"
        )

    session.set_context([system_prompt] + [context])
    num_tokens = tokenizer.encode(session.get_context())
    container.header(f"The context given to LLM: ({len(num_tokens)} tokens)", divider = "grey")
    container.text(session.get_context())

    messages = st.container(height=300)
    if prompt := st.chat_input("Type the question here"):
        messages.chat_message("user").write(prompt)
        messages.chat_message("assistant").write_stream(session.chat(prompt))
