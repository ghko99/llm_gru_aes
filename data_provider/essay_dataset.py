import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class EssayDataset(Dataset):
    def __init__(self, texts, scores, tokenizer, max_length=200):
        self.texts = texts
        self.scores = torch.tensor(scores, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        inputs = self.tokenizer.batch_encode_plus(self.texts[idx], 
                                                  max_length=self.max_length,
                                                  padding="max_length",
                                                  truncation=True
                                                  )
        input_ids = torch.tensor(inputs['input_ids']).cuda()
        attention_mask = torch.tensor(inputs['attention_mask']).cuda()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': self.scores[idx]
        }