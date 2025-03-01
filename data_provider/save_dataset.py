import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import pandas as pd
from datasets import Dataset, DatasetDict

train = pd.read_csv('./dataset_csv/train.csv',encoding='utf-8-sig')
valid = pd.read_csv('./dataset_csv/valid.csv',encoding='utf-8-sig')


train_data = {"id":[] , "text":[], "label": []}
valid_data = {"id":[] , "text":[], "label": []}

max_seq_len = 100
for i in range(len(train)):
    sentences = train.iloc[i]['essay'].split('#@문장구분#')
    sentences = [sent.strip() for sent in sentences if sent.strip() != '']
    seq_len = max_seq_len - len(sentences)
    padded_sentences = [""]*seq_len
    sentences.extend(padded_sentences)
    train_data['text'].append(sentences)
    score = train.iloc[i]['essay_score_avg'].split('#')
    score = [float(s) for s in score]
    train_data['label'].append(score)
    essay_id = train.iloc[i]['essay_id']
    train_data['id'].append(essay_id)

for i in range(len(valid)):
    sentences = valid.iloc[i]['essay'].split('#@문장구분#')
    sentences = [sent.strip() for sent in sentences if sent.strip() != '']
    seq_len = max_seq_len - len(sentences)
    padded_sentences = [""]*seq_len
    sentences.extend(padded_sentences)
    valid_data['text'].append(sentences)
    score = valid.iloc[i]['essay_score_avg'].split('#')
    score = [float(s) for s in score]
    valid_data['label'].append(score)
    essay_id = valid.iloc[i]['essay_id']
    valid_data['id'].append(essay_id)


train_df = pd.DataFrame(train_data)
valid_df = pd.DataFrame(valid_data)

train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)

dataset = DatasetDict({
    "train": train_dataset,
    "test": valid_dataset
})
dataset.save_to_disk("./aes_dataset")