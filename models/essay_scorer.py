from kobert_transformers import get_tokenizer, get_kobert_model
from layers.gru_layer import GRUScoreModule
import torch.nn as nn
import torch

class EssayScorer(nn.Module):
    def __init__(self, output_dim=11, hidden_dim=128):
        super(EssayScorer, self).__init__()
        self.bert = get_kobert_model()
        self.gru = GRUScoreModule(output_dim=output_dim, hidden_dim=hidden_dim)

    def forward(self, batch, labels=None):
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        B, S, L = input_ids.size()
        flattened_input_ids = input_ids.view(B * S, L)
        flattened_attention_mask = attention_mask.view(B * S, L)

        outputs = self.bert(input_ids=flattened_input_ids, attention_mask=flattened_attention_mask)
        embedded_outputs = outputs[0][:,0,:]

        embedded_outputs = embedded_outputs.view(B,S,-1)
        gru_output = self.gru(embedded_outputs)
        
        loss = None

        if labels is not None:
            criterion = nn.MSELoss()
            loss = criterion(gru_output, labels)  # MSE 손실 계산

        if loss is not None:
            return {"loss": loss, "logits": gru_output}
        else:
            return {"logits": gru_output}  # loss 없이 logits만 반환