import torch.nn as nn


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, vocab_size, embedding_size, hidden_size, num_layers, dropout=0.5, tie=False):
        super(RNNModel, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.Embedding(vocab_size, embedding_size)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(embedding_size, hidden_size, num_layers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(hidden_size, vocab_size)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie:
            if hidden_size != embedding_size:
                raise ValueError('When using the tied flag, hidden_size must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def init_weights(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, input, hidden=None):
        emb = self.dropout(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.dropout(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.num_layers, bsz,
                                     self.hidden_size), weight.new_zeros(self.num_layers, bsz, self.hidden_size))
        else:
            return weight.new_zeros(self.num_layers, bsz, self.hidden_size)