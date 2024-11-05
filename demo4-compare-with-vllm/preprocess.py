import time
import os, sys
import numpy as np
import pandas as pd
import chat_session
from typing import List, Dict

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
PORT_VLLM = "8000"
PORT_LMCACHE = "8002"


def PrepareEngines(contexts):
    def consume_stream(stream):
        for s in stream:
            print(s, end = "", flush = True)
        print("")

    session1 = chat_session.ChatSession(IP_LMCACHE, PORT_LMCACHE)
    session2 = chat_session.ChatSession(IP_VLLM, PORT_VLLM)
    session1.set_context(contexts)
    session2.set_context(contexts)
    stream1 = session1.chat("Please just say 'hello': ")
    stream2 = session2.chat("Please just say 'hello': ")
    consume_stream(stream1)
    consume_stream(stream2)

def read_context() -> str:
    with open("context.txt", "r") as fin:
        context = fin.read()
    return context

context = read_context()
system_prompt = "You are a helpful assistant. I will now give you a document and please answer my question afterwards based on the content in document"
PrepareEngines([system_prompt, context])
time.sleep(5)
