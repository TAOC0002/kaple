import pandas as pd
import json

valid_file = open("valid.jsonl", "w")
train_file = open("train.jsonl", "w")

train = pd.read_csv('train.tsv', sep='\t').sort_values(by='date')
valid_size = train.shape[0] // 10
valid = train[-valid_size:].to_json(orient="records")
train = train[:-valid_size].to_json(orient="records")
valid_file.write(json.dumps(json.loads(valid), indent=4))
train_file.write(json.dumps(json.loads(train), indent=4))

valid_file.close()
train_file.close()