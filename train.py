class Train:
    def __init__(self,model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def training(self,data):
        self.optimizer.zero_grad()
        out, h = self.model(data.x, data.edge_index)
        loss = self.criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss, h