from datasets import load_from_disk
from data_provider.essay_dataset import EssayDataset
from kobert_transformers import get_tokenizer
from torch.utils.data import DataLoader
from models.essay_scorer import EssayScorer

dataset = load_from_disk('./aes_dataset')

train, valid = dataset['train'] , dataset['test']

tokenizer = get_tokenizer()

train_dataset = EssayDataset(train['text'], train['label'], tokenizer)
valid_dataset = EssayDataset(valid['text'], valid['label'], tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=2, shuffle=False)

model = EssayScorer().cuda()
IDX = 0
for batch in train_dataloader:
    if IDX == 5:
        break
    x = model(batch)
    print(x)
    IDX += 1

