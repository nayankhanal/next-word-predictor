import torch.nn as nn


class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=50, hidden_size=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, question):
        embedded_question = self.embedding(question)
        hidden, final = self.rnn(embedded_question)
        output = self.fc(final.squeeze(0))
        return output