import torch
from torch.nn import Linear
import torch.nn.Functionnal as F
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.nn import global_max_pool as gmp, global_mean_pool as gap


embedding = 64

class GNN(torch.nn.Module):
    def __init__(self, data):
        super(GNN, self).__init__()
        
        self.data = data

        # Care about the design of the NN here
        self.initial_conv = GCNConv(in_channels = data.num_features, out_channels = embedding)
        self.conv_layer1 = GCNConv(in_channels = embedding, out_channels = embedding)
        self.conv_layer2 = GCNConv(in_channels = embedding, out_channels = embedding)
        self.conv_layer3 = GCNConv(in_channels = embedding, out_channels = embedding)

        self.out = Linear(in_features = embedding, out_features = 1)
        
    def forward(self, x, edge_index, batch_index):
        y = self.initial_conv(x, edge_index)
        # activation 
        y = self.conv_layer1(y, edge_index)
        # activation 
        y = self.conv_layer2(y, edge_index)
        # activation 
        y = self.conv_layer3(y, edge_index)


        # Pooling Layer here 
        out = self.out(y)
    
        return out, y