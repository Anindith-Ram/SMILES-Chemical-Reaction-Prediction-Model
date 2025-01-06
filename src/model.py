import torch
import torch.nn as nn
from transformers import RobertaModel

class TransformerDecoderWithAttention(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, num_layers=6, num_heads=8, dropout=0.1):
        super(TransformerDecoderWithAttention, self).__init__()
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout),
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, decoder_input_ids, memory):
        outputs = self.decoder(tgt=decoder_input_ids, memory=memory)
        return self.fc_out(outputs)

class RoBERTaWithMaxPoolingAndAttention(nn.Module):
    def __init__(self, roberta_model, decoder, tokenizer, hidden_size=768, max_length=256, dropout_rate=0.1):
        super(RoBERTaWithMaxPoolingAndAttention, self).__init__()
        self.roberta = roberta_model
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size
        self.max_length = max_length

        self.product_embedding = nn.Embedding(self.tokenizer.vocab_size, self.hidden_size)
        self.pooling = nn.AdaptiveMaxPool1d(256)
        self.product_embedding_dropout = nn.Dropout(p=dropout_rate)
        self.roberta_output_dropout = nn.Dropout(p=dropout_rate)

    def forward(self, input_ids, attention_mask, products):
        reactant_encoding = self.roberta(input_ids, attention_mask=attention_mask)
        memory = reactant_encoding.last_hidden_state
        memory = self.roberta_output_dropout(memory)
        memory = memory.permute(0, 2, 1)
        memory = self.pooling(memory)
        memory = memory.permute(0, 2, 1)
        decoder_input_ids = self.product_embedding(products)
        decoder_input_ids = self.product_embedding_dropout(decoder_input_ids)
        outputs = self.decoder(decoder_input_ids, memory)
        return outputs
