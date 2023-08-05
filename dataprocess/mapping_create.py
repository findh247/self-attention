import json

speaker2id = {}
for i in range(10001, 11252):
    speaker_id = f"id{i:05d}"
    speaker2id[speaker_id] = i - 10001
id2speaker = {value:key for key,value in speaker2id.items()}
new_dict = {**{'speaker2id': speaker2id}, **{'id2speaker': id2speaker}}

path=r"D:\article\data\dataset\mapping.json"

with open(path, 'w') as f:
    json.dump(new_dict, f, indent=4)