from vllm import LLM, SamplingParams
from tqdm import tqdm
import torch
import json
import pdb
import argparse
import yaml
import os

#python precompute.py --lmcache-config-file example.yaml --data-path data

parser = argparse.ArgumentParser()
parser.add_argument(
        "--lmcache-config-file",
        type=str,
        help="The path to the lmcache configuration yaml file. Empty means disabling the lmcache")

parser.add_argument(
        "--data-path",
        type=str,
        help="The path where text chunks are stored.")

parser.add_argument(
        "--model",
        type=str,
        help="The LLM model name")

args = parser.parse_args()
config_path = args.lmcache_config_file
with open(config_path, 'r') as fin:
    config = yaml.safe_load(fin)
cache_path = config.get("cache_path","cache.pt")
if os.path.exists(cache_path):
    print("Found existing cache at", cache_path)
    print("Now skipping the pre-compute.")
    print("Please remove the existing cache file if you want to redo the pre-compute")
    exit(1)

data_path = args.data_path
data_list = os.listdir(data_path)
model_name = args.model
file_list = [data_path+"/"+file for file in data_list if file.endswith(".txt")]

llm = LLM(model=model_name, gpu_memory_utilization=0.5,
          #tensor_parallel_size=2
          )
tokenizer = llm.llm_engine.tokenizer.tokenizer



doc_prompts = []
for file in file_list:
    f = open(f"{file}")
    doc_prompts.append(f.read())

doc_chunk_ids = [tokenizer.encode(doc)[1:] for doc in doc_prompts]


# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, max_tokens=1)


cache_driver = llm.llm_engine.model_executor.driver_worker.model_runner.lmcache_driver

cache_fuse_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata
cache_fuse_metadata['collect'] = False
cache_fuse_metadata['check'] = False

s_start_full = [1,733, 16289, 28793, 28705]
s_start_len = len(s_start_full) + 1

s_start = []
s_start_1_len = len(s_start) + 1

s_end = [733, 28748, 16289, 28793]
s_end_len = len(s_end)
old_kvs = []

doc_chunk_ids = [s_start+chunk_ids for chunk_ids in doc_chunk_ids]
doc_chunk_ids = [s_start_full] + doc_chunk_ids


cache_fuse_metadata['collect'] = True
cache_fuse_metadata["check"] = False

input_ids = []
doc_chunk_ids_store = []
for i in range(len(doc_chunk_ids)):
    if i == 0:
        temp_ids = [1] + doc_chunk_ids[i]
    else:
        temp_ids = doc_chunk_ids[i][s_start_1_len-1:]
    input_ids += temp_ids
    doc_chunk_ids_store.append(temp_ids)
    
# Concatenate old KVs
for i in tqdm(range(len(doc_chunk_ids))):
    if i==0:
        prompts = ["<s>[INST] "]
    else:
        prompts = [tokenizer.decode(doc_chunk_ids[i])]
    llm.generate(prompts, sampling_params)
    
    k_tensor = None
    v_tensor = None
    
    llm_layers = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
    for j in range(len(llm_layers)):
        past_key_values = llm_layers[j].self_attn.hack_kv
        if i == 0:
            temp_k = past_key_values[0][:s_start_len].clone() # do not chage with s_start_1
            temp_v = past_key_values[1][:s_start_len].clone()
        else:
            temp_k = past_key_values[0][s_start_1_len:len(doc_chunk_ids[i])+1].clone()
            temp_v = past_key_values[1][s_start_1_len:len(doc_chunk_ids[i])+1].clone()    

        if j == 0:
            k_tensor = torch.unsqueeze(temp_k, dim=0)
            v_tensor = torch.unsqueeze(temp_v, dim=0)
        else:
            k_tensor = torch.cat((k_tensor,torch.unsqueeze(temp_k, dim=0)), dim=0)
            v_tensor = torch.cat((v_tensor,torch.unsqueeze(temp_v, dim=0)), dim=0)
    k_tensor = torch.unsqueeze(k_tensor, dim=0)
    v_tensor = torch.unsqueeze(v_tensor, dim=0)
    kv_tensor = torch.cat([k_tensor, v_tensor], dim=0)
    cache_driver.collect_kv_and_store(torch.tensor(doc_chunk_ids_store[i]),
                                      kv_tensor.cpu())
#cache_driver.dump(cache_path)


   

