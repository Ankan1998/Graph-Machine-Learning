from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from data_preparation import GNNdataset
from model.GCNmodel import GCNModel
from train import Train
from visualizer import visualize

def main_training(n_epoch):
    data, num_features, num_labels = GNNdataset()
    model = GCNModel(num_features,num_labels)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model_init = Train(model, criterion, optimizer)
    for epoch in tqdm(range(n_epoch)):
        loss, h = model_init.training(data)
        # Visualize the node embeddings every 10 epochs
        if epoch % 50 == 0:
            visualize(h, color=data.y, epoch=epoch, loss=loss)


if __name__=="__main__":
    main_training(500)