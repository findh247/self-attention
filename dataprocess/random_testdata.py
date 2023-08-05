import json
import random

# 读取 JSON 文件
with open(r"D:\article\data\dataset\testdata.json", 'r') as f:
    dict = json.load(f)

# 打乱 "utterances" 键对应的值
random.shuffle(dict["utterances"])

# 写回 JSON 文件
with open(r"D:\article\data\dataset\testdata.json", 'w') as f:
    json.dump(dict, f, indent=4)
