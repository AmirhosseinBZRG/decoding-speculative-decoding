import torch
import torch.nn as nn
import re
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import deepspeed
import torch.distributed as dist
import json
import os
def json_loader(file_path):
    """
    Input: file_path: Path to the json file containing all the queries.

    File format looks like the following:
    {"prompt": "A seated man cleans a shoe in a classroom setting with other individuals. the man"}
    {"prompt": "Two girls are sailing on a lake. they"}

    Output: This function returns a list of prompts to be used by the draft LLM.
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data
import huggingface_hub
if __name__ == "__main__":
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if dist.is_initialized():
       dist.destroy_process_group()
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)
# The LLaMA tokenizer does not have a pad token.
# Modify the tokenizer to add a pad token and change the model configs accordingly.
    huggingface_hub.login("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")
    access_token = "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"
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/Sheared-LLaMA-2.7B", padding_side="left", torch_dtype=torch.float16)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    
    model_name = "princeton-nlp/Sheared-LLaMA-2.7B"
    config = AutoConfig.from_pretrained(model_name, cache_dir="/content/decoding-speculative-decoding")
    
    target_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
    
    
    target_model.eval()
  
max_new_tokens = 200
batch_count = 0
output_file = "Llama-2-2.7B-Hellaswag.txt"
hellaswag_test = json_loader("/content/decoding-speculative-decoding/hellaswag.json")
prompts = [item["prompt"] for item in hellaswag_test]

# Ensure the model is on the correct device
device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
target_model = target_model.to(device)

for batch in hellaswag_test:
    prompt = batch['prompt']
    batch_count += 1

    if local_rank == 0:
        print(batch_count)

    inputs = tokenizer.encode(prompt, padding="longest", return_tensors="pt")
    input_tensors = inputs.to(device)
    outputs = target_model.generate(input_tensors, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id, use_cache=True)
    outputs = outputs.to("cpu")

    if local_rank == 0 and rank == 0:
        print(tokenizer.decode(outputs[0]))
        print(tokenizer.decode(outputs[0, inputs.size(0):]))
        with open(output_file, "a") as f:
            f.write(str(outputs[:, input_tensors.size(0):]) + str("\n"))

    if batch_count == 200:
        break
