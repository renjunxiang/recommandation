import json
from NLP.supervised_classify import

with open('D:/github/machine-learning/NLP/data/keyword.json', encoding='utf-8') as f:
    keyword = json.load(f)
with open('D:/github/machine-learning/NLP/data/train_data.json', encoding='utf-8') as f:
    train_data = json.load(f)

print(keyword,train_data)
