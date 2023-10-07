import torch
import torch.nn as nn
from TorchCRF import CRF
import numpy as np

# 1. input is a sentence
# 2. output is start and end position of all entities in the sentence
# 3. use CRF to decode the output

class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_maxtrix = None, embedding_dim=384, hidden_dim=384, units='lstm'):
        super(BiLSTMCRF, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        if embedding_maxtrix is not None:
            embedding_dim = embedding_maxtrix.shape[1]
        
        # embedding layer
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)

        if embedding_maxtrix is not None:
            self.word_embeds.weight.data.copy_(torch.from_numpy(embedding_maxtrix))
            self.word_embeds.weight.requires_grad = False
        
        if units == 'lstm':
            # bidirectional LSTM layer
            self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        elif units == 'gru':
            # bidirectional GRU layer
            self.lstm = nn.GRU(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        elif units == 'rnn':
            # bidirectional RNN layer
            self.lstm = nn.RNN(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        else:
            raise ValueError('units must be one of lstm, gru or rnn')
        
        # linear layer to find hidden state representation of tags
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        
        # initialize CRF
        self.crf = CRF(self.tagset_size)
        
    def _get_lstm_features(self, sentence):
        # sentence: (seq_len, batch_size)
        # embeds: (seq_len, batch_size, embedding_dim)
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        
        # lstm_out: (seq_len, batch_size, hidden_dim)
        lstm_out, _ = self.lstm(embeds)
        
        # lstm_feats: (seq_len, batch_size, tagset_size)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats
    
    def neg_log_likelihood(self, sentence, tags):
        # sentence: (seq_len, batch_size)
        # tags: (seq_len, batch_size)
        # feats: (seq_len, batch_size, tagset_size)
        feats = self._get_lstm_features(sentence)
        
        # forward_score: (batch_size)
        forward_score = self.crf.forward(feats, tags)
        
        # gold_score: (batch_size)
        gold_score = self.crf.score_sentence(feats, tags)
        
        # loss: (batch_size)
        loss = forward_score - gold_score
        return loss
    
    def forward(self, sentence):
        # sentence: (seq_len, batch_size)
        # feats: (seq_len, batch_size, tagset_size)
        feats = self._get_lstm_features(sentence)
        
        # tag_seq: (seq_len, batch_size)
        tag_seq = self.crf.viterbi_decode(feats)
        return tag_seq
    
    def save(self, file_path):

        torch.save(self.state_dict(), file_path)

    def load(self, file_path):
        
        self.load_state_dict(torch.load(file_path))