import sys
sys.path.append("../")
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from util import nethook
from util.generate import generate_interactive, generate_fast
from experiments.py.demo import demo_model_editing, stop_execution

# import pdb
# pdb.set_trace()

MODEL_NAME = "../models/gpt-j-6b"
ALG_NAME = "MEMIT"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# loading model and tokenizer
print(f"Loading model from {MODEL_NAME} ...")
model, tok = (
    AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        low_cpu_mem_usage=False,
        torch_dtype=(torch.float16 if "20b" in MODEL_NAME else None),
    ).to("cuda"),
    AutoTokenizer.from_pretrained(MODEL_NAME),
)
tok.pad_token = tok.eos_token
print("Successfully loaded!")

# request
request = [
    {
        "prompt": "The mother of {} is",
        "subject": "Alice",
        "target_new": {"str": "Bob"},
    }
]
generation_prompts = [
    "The mother of Alice is",
    "The parent of Alice is",
    "Alice is the daughter of"
]

# Restore fresh copy of model
try:
    with torch.no_grad():
        for k, v in orig_weights.items():
            nethook.get_parameter(model, k)[...] = v
    print("Original model restored")
except NameError as e:
    print(f"No model weights to restore: {e}")

# Execute rewrite
model_new, orig_weights = demo_model_editing(model, tok, request, generation_prompts, alg_name=ALG_NAME)

# prompting to see the result
prompt = "The mother of Alice is"
generate_fast(model_new, tok, [prompt], max_out_len=30)