from datasets import load_from_disk
from data_provider.essay_dataset import EssayDataset
from transformers import TrainingArguments, Trainer
from kobert_transformers import get_tokenizer
from models.essay_scorer import EssayScorer

def load_aes_dataset():
    dataset = load_from_disk('./aes_dataset')

    train, valid = dataset['train'] , dataset['test']

    tokenizer = get_tokenizer()

    train_dataset = EssayDataset(train['text'], train['label'], tokenizer)
    valid_dataset = EssayDataset(valid['text'], valid['label'], tokenizer)

    return train_dataset, valid_dataset

def aes_train():

    train_dataset, valid_dataset = load_aes_dataset()
    model = EssayScorer()
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=1e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        fp16=True,
        num_train_epochs=20,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=1,
        save_total_limit=2,
        load_best_model_at_end=True
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset
    )
    trainer.train()
    trainer.save_model("./essay_scorer_model")

if __name__ == "__main__":
    aes_train()
