import json
import pandas as pd

path_json = r"D:\article\data\dataset\alldata.json"
path_csv = r"D:\article\SA_2\output.csv"

counter = 0
for i in range(0, 8558):
    with open(path_csv) as file:
        data_csv = pd.read_csv(file)
        speakers_uttr = data_csv.iloc[i, 0]
        speakers_csv = data_csv.iloc[i, 1]
        with open(path_json) as f:
            str = f.read()
            data_json = json.loads(str)
            speakers_id = data_json["speakers"]  # speakers_id is a dict
            for key, value in speakers_id.items():
                if speakers_csv in key:
                    list_b = []
                    for item in speakers_id[speakers_csv]:
                        list_b.append(item['feature_path'])
                    if speakers_uttr in list_b:
                        counter = counter + 1

print(counter)

g = counter / 8558
print(g)

# with open(path_json) as f:
#   data_json = json.load(f)
#   speakers_json=data_json["speakers"]
#   id=speakers_json["id03074"]
# print(id[0])

# speakers_json = data_json[k]["speakers"]
# if speakers_json== speakers_csv:
#   for n in speakers_json["feature_path"]:
#     id=speakers_json[n]["feature_path"]
#     if speakers_uttr == id:
#       h=h+1
