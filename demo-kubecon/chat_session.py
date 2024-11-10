from openai import OpenAI
import threading
import sys
from io import StringIO
import time
from transformers import AutoTokenizer
import json
import pdb

class ChatSession:
    def __init__(self, port):
        openai_api_key = "EMPTY"
        openai_api_base = f"http://localhost:{port}/v1"

        self.client = client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        models = client.models.list()
        self.model = models.data[0].id

        self.messages = [

        ]

        self.final_context = ""
        self.separator = " # # "
        self.temperature = 0.0

    def set_context(self, context_list):
        input_prompt = ""
        for context in context_list:
            input_prompt +=(self.separator + context)
        self.final_context = input_prompt
        self.messages.append({"role":"user", "content":input_prompt})

    def get_context(self):
        return self.final_context

    def on_user_message(self, message, display=True):
        if display:
            print("User message:", message)
        self.messages.append({"role": "user", "content": message})

    def on_server_message(self, message, display=True):
        if display:
            print("Server message:", message)
        self.messages.append({"role": "assistant", "content": message})

    def chat(self,question):
        #self.on_user_message(question)
        self.messages[0]["content"] = self.messages[0]["content"] + self.separator + question
        start = time.perf_counter()
        end = None
        chat_completion = self.client.chat.completions.create(
            messages=self.messages,
            model=self.model,
            temperature=self.temperature,
            stream=True,
            #stop=['\n']
        )

        output_buffer = StringIO()
        server_message = []
        for chunk in chat_completion:
            chunk_message = chunk.choices[0].delta.content
            if chunk_message is not None:
                if end is None:
                    end = time.perf_counter()
                yield chunk_message
                server_message.append(chunk_message)

        #self.on_server_message("".join(server_message))
        yield f"\n\n(Response delay: {end - start:.2f} seconds)"

