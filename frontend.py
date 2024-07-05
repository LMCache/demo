import time
import os, sys
import numpy as np
import pandas as pd
import streamlit as st
import chat_session
from typing import List, Dict

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

container = st.container(border=True)
container.header("The context given to LLM:", divider = "grey")

with st.sidebar:
    optimization = st.checkbox("Enable LMCache optimization")
    port = 8000 if optimization else 8001
    session = chat_session.ChatSession(port)

    system_prompt = st.text_area(
            "System prompt:",
            "You are a helpful assistant. I will now give you a few paragraphs and "
            "please answer my question afterwards based on the content in the paragraphs"
        )

    session.set_context([system_prompt] + [chunks[key] for key in selected_chunks])
    container.text(session.get_context())

    messages = st.container(height=400)
    if prompt := st.chat_input("Type the question here"):
        messages.chat_message("user").write(prompt)
        messages.chat_message("assistant").write_stream(session.chat(prompt))
