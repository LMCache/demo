import time
import os, sys
import numpy as np
import pandas as pd
import streamlit as st
import chat_session
from typing import List, Dict
from transformers import AutoTokenizer

# Change the following variables as needed
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2" 
PORT1 = 8000
PORT2 = 8001

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

container = st.container(border=True)

with st.sidebar:
    system_prompt = st.text_area(
            "System prompt:",
            "You are a helpful assistant. I will now give you a document and "
            "please answer my question afterwards based on the content in document"
        )

    session = chat_session.ChatSession(PORT1)
    session2 = chat_session.ChatSession(PORT2)

    session.set_context([system_prompt] + [context])
    session2.set_context([system_prompt] + [context])

    num_tokens = tokenizer.encode(session.get_context())
    container.header(f"The context given to LLM: ({len(num_tokens)} tokens)", divider = "grey")
    container.text(session.get_context())

    messages = st.container(height=400)
    if prompt := st.chat_input("Type the question here"):
        messages.chat_message("user").write(prompt)
        messages.chat_message("assistant").write_stream(session.chat(prompt))

    messages2 = st.container(height=400)
    if prompt2 := st.chat_input("Type the question here"):
        messages2.chat_message("user").write(prompt2)
        messages2.chat_message("assistant").write_stream(session2.chat(prompt2))
