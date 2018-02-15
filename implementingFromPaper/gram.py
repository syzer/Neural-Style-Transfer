import torch
import torch.nn as nn

# gram matrix holds encoded shape, spacial information and so on
class GramMatrix(nn.Module):

    def forward(self, input):
        x, y, x2, y2 = input.size()
        features = input.view(x * y, x2 * y2)
        G = torch.mm(features, features.t())

        return G.div(x * y * x2 * y2)
