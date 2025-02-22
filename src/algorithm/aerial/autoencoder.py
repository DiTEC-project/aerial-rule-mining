import torch
import os
from torch import nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    """
    This autoencoder is used to create a numerical representation for the categorical values.
    """

    def __init__(self, data_size):
        """
        :param data_size: size of the categorical features in the knowledge graph, after one-hot encoding
        """
        super().__init__()
        self.data_size = data_size
        self.encoder = nn.Sequential(
            nn.Linear(self.data_size, int(1 * self.data_size / 8)),
        )
        self.decoder = nn.Sequential(
            nn.Linear(int(1 * self.data_size / 8), self.data_size),
        )

        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        """
        all weights are initialized with values sampled from uniform distributions with the Xavier initialization
        and the biases are set to 0, as described in the paper by Delong et al. (2023)
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.zero_()

    def save(self, p):
        torch.save(self.encoder.state_dict(), p + 'cat_encoder.pt')
        torch.save(self.decoder.state_dict(), p + 'cat_decoder.pt')

    def load(self, p):
        if os.path.isfile(p + 'cat_encoder.pt') and os.path.isfile(p + 'cat_decoder.pt'):
            self.encoder.load_state_dict(torch.load(p + 'cat_encoder.pt'))
            self.decoder.load_state_dict(torch.load(p + 'cat_decoder.pt'))
            self.encoder.eval()
            self.decoder.eval()
            return True
        else:
            return False

    def forward(self, x, input_vector_category_indices):
        y = self.encoder(x)
        y = self.decoder(y)

        # Split the tensor into chunks based on the ranges
        chunks = [y[:, start:end] for start, end in input_vector_category_indices]

        # Apply softmax to each chunk
        softmax_chunks = [F.softmax(chunk, dim=1) for chunk in chunks]

        # Concatenate the chunks back together
        y = torch.cat(softmax_chunks, dim=1)

        return y
