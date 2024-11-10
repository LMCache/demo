import os, sys
from langchain_text_splitters import TokenTextSplitter

context_file = "context.txt"

def create_chunks(chunk_size = 2000, output_dir = "ffmpeg", 
                    chunk_overlap = 0) -> None:
    """
    Split the context into chunks and save them into the data folder
    """
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, 
                        chunk_overlap=chunk_overlap)
    
    with open(context_file, "r") as fin:
        context = fin.read()

    splitter = text_splitter.split_text(context)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, chunk in enumerate(splitter):
        with open(os.path.join(output_dir, f"chunk_{i}.txt"), "w") as fout:
            fout.write(chunk)

def prepare_system_prompt(output_dir = "ffmpeg", file_name = "sys_prompt.txt") -> None:
    """
    Prepare the system prompt
    """
    
    text = "You are a helpful assistant. I will now give you a few paragraphs and \
            please answer my question afterwards based on the content in the paragraphs."

    with open(os.path.join(output_dir, file_name), "w") as fout:
        fout.write(text)

create_chunks()
prepare_system_prompt()

