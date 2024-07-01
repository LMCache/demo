import time
import numpy as np
import pandas as pd
import streamlit as st
import chat_session

_LOREM_IPSUM = """
This document mainly talks about FFmpeg, a fast video and audio converter that can also grab live sources. It explains how to use FFmpeg to convert files or streams, and covers topics such as input and output options, stream selection, filtering, and various formats and codecs. The document also mentions the use of filters and provides examples of different command line options.
"""

file_dict = {
    "Stardew Valley Background 1": "data/intro.txt",
    "Abigail": "data/abi.txt",
    "Alex": "data/alex.txt",
    ("Abigail", "Player 1"): "data/abi-chat-1.txt",
    ("Abigail", "Player 2"): "data/abi-chat-2.txt",
    ("Alex", "Player 1"): "data/alex-chat-1.txt",
    ("Alex", "Player 2"): "data/alex-chat-2.txt",
}


container = st.container(border=True)
container.header("The context given to LLM:", divider = "grey")

with st.sidebar:
    background = st.selectbox(
        "Game background description",
        ["Stardew Valley Background 1"])

    npc = st.selectbox(
        "NPC background",
        ["Abigail", "Alex"])

    chat_history = st.selectbox(
        "Recent interactions between NPC and player",
        ["Player 1", "Player 2"])

    optimization = st.checkbox("Enable LMCache optimization")
    port = 8000 if optimization else 8001

    print("The port is:", port)

    session = chat_session.ChatSession(port)
    session.set_context([
        file_dict[background],
        file_dict[npc],
        file_dict[(npc, chat_history)],
    ])
    context = session.get_context()
    container.text(context)

    messages = st.container(height=300)
    if prompt := st.chat_input("Say something"):
        messages.chat_message("user").write(prompt)
        messages.chat_message("assistant").write_stream(session.chat(prompt))
