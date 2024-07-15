from vllm import LLM, SamplingParams
import torch
import json
import pdb
import argparse
import yaml
import os
import ray
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

args = parser.parse_args()
config_path = args.lmcache_config_file
with open(config_path, 'r') as fin:
    config = yaml.safe_load(fin)
cache_path = config.get("cache_path","cache.pt")
data_path = args.data_path
data_list = os.listdir(data_path)
file_list = [data_path+"/"+file for file in data_list if file.endswith(".txt")]

llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2", gpu_memory_utilization=0.5,
          tensor_parallel_size=2
          )
tokenizer = llm.llm_engine.tokenizer.tokenizer



doc_prompts = []
for file in file_list:
    f = open(f"{file}")
    doc_prompts.append(f.read())

doc_chunk_ids = [tokenizer.encode(doc)[1:] for doc in doc_prompts]


# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, max_tokens=1)
'''
torch.distributed.barrier()
torch.cuda.synchronize()
if torch.distributed.get_rank()==0:
    import pdb
    pdb.set_trace()
torch.distributed.barrier()
torch.cuda.synchronize()
'''

print((llm.llm_engine).__dir__())
print((llm.llm_engine.model_executor.workers[0]).__dir__())
#print((llm.llm_engine.model_executor.workers[0]).worker.__dir__())
#cache_driver_d = llm.llm_engine.model_executor.driver_worker.worker.model_runner.lmcache_driver
#cache_driver_w = llm.llm_engine.model_executor._run_workers()
executor = llm.llm_engine.model_executor

executor._run_workers("set_collect_check", False, False)


s_start_full = [1,733, 16289, 28793, 28705]
s_start_len = len(s_start_full) + 1

s_start = []
s_start_1_len = len(s_start) + 1

s_end = [733, 28748, 16289, 28793]
s_end_len = len(s_end)
old_kvs = []

doc_chunk_ids = [s_start+chunk_ids for chunk_ids in doc_chunk_ids]
doc_chunk_ids = [s_start_full] + doc_chunk_ids

executor._run_workers("set_collect_check", True, False)


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
for i in range(len(doc_chunk_ids)):
    if i==0:
        prompts = ["<s>[INST] "]
    else:
        prompts = [tokenizer.decode(doc_chunk_ids[i])]
    llm.generate(prompts, sampling_params)
    
    executor._run_workers("concat_hack_kvs",
                          i,
                          doc_chunk_ids[i], doc_chunk_ids_store[i],
                          s_start_1_len, s_start_len)    
    
executor._run_workers("dump", cache_path) 
#cache_driver.dump(cache_path)


   

