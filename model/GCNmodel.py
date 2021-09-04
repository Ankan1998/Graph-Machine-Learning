import torch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import GCNConv

class GCNModel(nn.Module):
    def __init__(self, num_features,num_classes):
        super(GCNModel, self).__init__()
        self.conv_1 = GCNConv(num_features,4)
        self.conv_2 = GCNConv(4, 4)
        self.conv_3 = GCNConv(4, 2)
        self.classifier = Linear(2, num_classes)

    def forward(self, x, edge_index):

        h = self.conv_1(x, edge_index)
        h = h.tanh()
        h = self.conv_2(h, edge_index)
        h = h.tanh()
        h = self.conv_3(h, edge_index)
        h = h.tanh()

        out = self.classifier(h)

        return out, h

if __name__=="__main__":
    # num_features = node feature vector dimension
    # Num_classes = 4 classes

    gcnmodel = GCNModel(34,4)
    print(gcnmodel)