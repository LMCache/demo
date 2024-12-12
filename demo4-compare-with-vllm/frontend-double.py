import time
import os, sys
import numpy as np
import pandas as pd
import streamlit as st
import chat_session
from typing import List, Dict
from transformers import AutoTokenizer

#st.set_page_config(layout="wide")

# Change the following variables as needed
MODEL_NAME = "lmsys/longchat-7b-16k"

#if len(sys.argv) != 5:
#    print(f"Usage: streamlit run {sys.argv[0]} <preheat type>")
#    exit(-1)


#IP = sys.argv[1]
#PORT = int(sys.argv[2])
#ENGINE_NAME = sys.argv[3]

IP_VLLM = "localhost"
IP_LMCACHE = "localhost"
PORT_VLLM = "8001"
PORT_LMCACHE = "8003"


@st.cache_resource
def PrepareEngines():
    def consume_stream(stream):
        for s in stream:
            print(s, end = "", flush = True)
        print("")

    preheat_context = "This is dummy text. " * 750
    session1 = chat_session.ChatSession(IP_VLLM, PORT_VLLM)
    session2 = chat_session.ChatSession(IP_LMCACHE, PORT_LMCACHE)
    session1.set_context([preheat_context])
    session2.set_context([preheat_context])
    stream1 = session1.chat("Please just say 'hello': ")
    stream2 = session2.chat("Please just say 'hello': ")
    consume_stream(stream1)
    consume_stream(stream2)
    stream1 = session1.chat("Please just say 'hey': ")
    stream2 = session1.chat("Please just say 'hey': ")
    consume_stream(stream1)
    consume_stream(stream2)

    global MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    return tokenizer

tokenizer = PrepareEngines()

@st.cache_data
def read_context() -> str:
    with open("context.txt", "r") as fin:
        context = fin.read()
    return context

context = read_context()


gap_width = 0.013
col_width = (1 - gap_width) / 2
col1, col3, col2 = st.columns([col_width, gap_width, col_width], gap="small")

with col3:
    st.text("")
    st.image("bar.png", use_container_width=True)

with col1:
    port = PORT_VLLM
    ip = IP_VLLM

    st.markdown("#### Standard vLLM engine")

    session = chat_session.ChatSession(ip, port)

    system_prompt = "You are a helpful assistant. I will now give you a document and please answer my question afterwards based on the content in document"

    session.set_context([system_prompt] + [context])
    num_tokens = tokenizer.encode(session.get_context())

    messages = st.container(height=300)
    if prompt := st.chat_input("Type your question about this document", key = "vllm"):
        messages.chat_message("user").write(prompt)
        messages.chat_message("assistant").write_stream(session.chat(prompt))

with col2:
    port = PORT_LMCACHE
    ip = IP_LMCACHE

    st.markdown("#### LMCache + vLLM engine (new)")

    session = chat_session.ChatSession(ip, port)

    system_prompt = "You are a helpful assistant. I will now give you a document and please answer my question afterwards based on the content in document"

    session.set_context([system_prompt] + [context])
    num_tokens = tokenizer.encode(session.get_context())

    messages = st.container(height=300)
    if prompt := st.chat_input("Type your question about this document", key = "lmcache"):
        messages.chat_message("user").write(prompt)
        messages.chat_message("assistant").write_stream(session.chat(prompt))

container = st.container(border=True, height = 300)
container.markdown("#### Background Document:")
container.markdown("**13K words, 25K tokens**") #, divider = "grey")
container.text("\n" + context)

