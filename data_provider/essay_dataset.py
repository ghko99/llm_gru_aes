import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class EssayDataset(Dataset):
    def __init__(self, texts, scores, tokenizer, max_length=200):
        self.texts = texts
        self.scores = torch.tensor(scores, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inputs =  [self.tokenizer.batch_encode_plus(text, max_length = self.max_length, padding="max_length", truncation=True) for text in tqdm(texts)]
        self.input_ids = [torch.tensor(input['input_ids']) for input in tqdm(self.inputs)]
        self.attention_mask = [torch.tensor(input['attention_mask']) for input in tqdm(self.inputs)]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': self.scores[idx]
        }