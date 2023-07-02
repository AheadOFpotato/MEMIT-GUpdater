"""
In this script, we test MEMIT on the test set of family tree. (3 trees, 78 updates)
"""
import sys
sys.path.append("../")
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook
from util.generate import generate_interactive, generate_fast

from experiments.py.demo import demo_model_editing, stop_execution

from tqdm import tqdm

import re

# import pdb
# pdb.set_trace()

def load_dataset(path_to_test_set: str = "../family_tree_data"):
    new_samples = []
    with open(f"{path_to_test_set}/family_tree_test.json", "r") as f:
        samples = json.load(f)
    for sample in samples:
        sample["triple_added"] = list(set([tuple(triple) for triple in sample["subgraph_after"]]) - set([tuple(triple) for triple in sample["subgraph_before"]]))
        new_samples.append(sample)
    return new_samples

def triple_to_request(triple: list, path_to_test_set: str = "../family_tree_data"):
    '''
    convert triple needed to be inserted into GPT to request.\n
    return `request: dict` with keys:
    * "prompt"
    * "subject"
    * "target_new"
    '''
    with open(f"{path_to_test_set}/id2relation.json", "r") as f:
        id2relation = json.load(f)
    with open(f"{path_to_test_set}/id2entity.json", "r") as f:
        id2entity = json.load(f)
    head = id2entity[str(triple[0])]
    relation = id2relation[str(triple[1])].split("O")[0]
    tail = id2entity[str(triple[2])]
    return {
        "prompt": f"The {relation} of " + "{} is",
        "subject": tail,
        "target_new": {"str": head}
    }


def main():
    OUTPUT_PATH = "../output"

    # LOADING MODEL
    MODEL_NAME = "../models/gpt-j-6b"

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
    print(f"Successfully loaded!")

    # LOADING DATASET
    test_samples = load_dataset()
    output = []
    for sample in tqdm(test_samples):
        sample_answer = []
        request = [triple_to_request(triple) for triple in sample["triple_added"]]

        # INSERT REQUEST TO GPT THROUGH MEMIT
        model_new, orig_weights = demo_model_editing(model, tok, request, [request[0]["prompt"].format(request[0]["subject"])], alg_name="MEMIT")
        correct = 0
        all = 0
        for idx, update_request in enumerate(request):
            prompt = update_request["prompt"].format(update_request["subject"])
            answer = generate_fast(model_new, tok, [prompt], top_k=1, max_out_len=15)
            answer_name = answer[0].split(prompt)[1].split()[0].lower()
            answer_name = re.sub(r"\W", "", answer_name)
            target_name = update_request["target_new"]["str"]
            print(f"==============={idx}/{len(request)}=================")
            print(f"REQUEST: {prompt}" + target_name)
            print(f"MODEL_ANSWER: {answer}")
            print(f"ANSWER_NAME: {answer_name}")
            if answer_name == target_name:
                correct += 1
            all += 1
        sample["correct"] = correct
        sample["all"] = all
        sample["accuracy"] = correct / all
        output.append(sample)

        with open(f"{OUTPUT_PATH}/{MODEL_NAME.split('/')[-1]}_family_tree_result_T=0.json", 'w') as f:
            try:
                f.seek(0)
                f.truncate()
                json.dump(output, f)
            except:
                print("Error occurs when rewriting the output file")



if __name__ == "__main__":
    main()