import json

def load_str_dict(path, seperator="\t", reverse=False):
    ''' load string dict '''
    dictionary, reverse_dictionay = {}, {}
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            try:
                key, value = line.strip().split(seperator)
                dictionary[key] = int(value)
                reverse_dictionay[int(value)] = key
            except:
                pass
    if reverse:
        return dictionary, reverse_dictionay, len(dictionary)
    return dictionary, len(dictionary)

files = ["entity","relation","token"]
for file in files:
    with open(f"id2{file}.json","r") as f:
        d = json.load(f)
    s = f"{len(d)}\n"
    for key in d:
        s += f"{d[key]}\t{key}\n"
    with open(f"{file}2id.txt", "w") as f_out:
        f_out.write(s)
    print(load_str_dict(f"{file}2id.txt"))