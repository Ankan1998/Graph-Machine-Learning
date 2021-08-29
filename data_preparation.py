from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx

from visualizer import visualize


def GNNdataset(stat=True, viz=True):

    dataset = KarateClub()
    data = dataset[0]
    if stat:
        print(f'Number of graphs: {len(dataset)}')
        print(f'Number of features: {dataset.num_features}')
        print(f'Number of classes: {dataset.num_classes}')
        print(f'Number of nodes: {data.num_nodes}')
        print(f'Number of edges: {data.num_edges}')
        print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
        print(f'Number of training nodes: {data.train_mask.sum()}')
        print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
        print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
        print(f'Contains self-loops: {data.contains_self_loops()}')
        print(f'Is undirected: {data.is_undirected()}')
    if viz:
        G = to_networkx(data, to_undirected=True)
        visualize(G, color=data.y)

    return data


