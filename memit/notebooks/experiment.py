'''
In this scripe,
we evaluate unmodified LM on NBA dataset
'''
import sys
sys.path.append("../")

import json
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional
import unicodedata
from util.generate import generate_fast

def id2entity(idx, entity_table):
    return entity_table[idx]

if __name__ == "__main__":
    # load model

    MODEL_NAME = "../models/gpt-j-6b"
    print(f"loading model from {MODEL_NAME}...")
    model, tok = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=(torch.float16 if "20b" in MODEL_NAME else None),
        ).to("cuda"),
        AutoTokenizer.from_pretrained(MODEL_NAME),
    )
    tok.pad_token = tok.eos_token
    print("successfully loaded!")

    # prepare map from id to entities
    with open("../dataset/id2entity.json",'r') as f:
        entity_table = json.load(f)
    with open("../dataset/id2relation.json",'r') as f:
        relation_table = json.load(f)

    # load data
    print("preparing data...")
    data_dir = "../dataset/NBAtransactions_test_50.json"
    output_dir = "../output/gpt2_wo_memit.json"

    with open(data_dir, 'r') as f:
        samples = json.load(f)

    results = []
    top_k = 1
    max_out_len = 45

    # generate results
    print("begin evaluation...")
    for sample in tqdm(samples):
        for triple_idx, triple in enumerate(sample["modified_triples"]):
            fact = f"The {id2entity(str(triple[1]), relation_table)} of {id2entity(str(triple[0]), entity_table)} is {id2entity(str(triple[2]), entity_table)}."
            prompt = f"{fact} Is this correct?\nThe answer(yes or no) is"
            answer = generate_fast(model, tok, [prompt], n_gen_per_prompt=1, top_k=top_k, max_out_len=max_out_len)
            # save answer and other necessary features to `results`

            output_sample = {
                "sample_id": sample["id"],
                "triple_id": triple_idx,
                "fact": fact,
                "answer": answer,
                "news": sample["text"]
            }
            results.append(output_sample)

    with open(output_dir, 'w') as f_out:
        json.dump(results, f_out)