from openai import OpenAI
import threading
import sys
from io import StringIO
import time



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
            {
                "role": "user",
                "content": f"I want you to act as a NPC in stardew Valley and talk with me. I will now provide some basic knowledge about the Stardew Valley and the basic information of the NPC you are going to act as. "
                "The general guideline are following: " 
                "1. do NOT talk too much! Just 2-3 sentences will be good enough. "
                "2. make sure you are acting! Just generate the texts said by the NPC, don't generate anything else! "
                "Please reponse 'Got it' if you understand and are ready to act as the NPC",
            }, 
            {
                "role": "assistant",
                "content": "Got it"
            }, 
        ]

        self.final_context = ""


    def set_context(self, context_files):
        contexts = []
        for file in context_files:
            with open(file, "r") as fin:
                context = fin.read()
            contexts.append(context)

            # Evil hacking: write the first context twice to increase the number of tokens in total
            if len(contexts) == 1:
                contexts.append(context)
                contexts.append(context)

        contexts.append("Please start acting now!")
        self.final_context =  "\n".join(contexts) 
        self.on_user_message(self.final_context, display=False)
        self.on_server_message("Got it! Now I'm the NPC.", display=False)

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

    def chat(self, question):
        self.on_user_message(question)

        start = time.perf_counter()
        end = None
        chat_completion = self.client.chat.completions.create(
            messages=self.messages,
            model=self.model,
            temperature=0.5,
            stream=True,
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

        self.on_server_message("".join(server_message))
        yield f"\n\n(Response delay: {end - start:.2f} seconds)"

if __name__ == "__main__":
    session = ChatSession(8000)
    session.set_context(["data/intro.txt", "data/abi.txt"])
    
    ret = session.chat("Hey, here's a present for you, it's moss.")
    for text in ret:
        print(text)
