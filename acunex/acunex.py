#!/usr/bin/env python
import torch
from torch import tensor



class Trainer():
    def __init__(self, model, learning_rate=0.001, reduction='mean'):
        self.loss_fn = torch.nn.MSELoss(reduction=reduction)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.model = model

    def train_epoch(self, inputData, target):
        result = self.model(inputData)

        loss = self.loss_fn(result, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def trainUntil(self, inputData, target, iterations=None, target_loss=None):
        if target_loss:
            target_loss = float(target_loss)
        if target_loss or iterations:
            if iterations:
                if isinstance(iterations, int):
                    for i in range(iterations):
                        loss = self.train_epoch(inputData, target)
                        if target_loss:
                            if loss <= target_loss:
                                return loss
            elif target_loss:
                loss = target_loss+1.0
                while loss > target_loss:
                    loss = self.train_epoch(inputData, target)
                return loss

        return self.train_epoch(inputData, target)



class Network(torch.nn.Sequential):

    def add_layer(self, name, input_shape, output_shape):
        self.add_module( name, torch.nn.Linear(input_shape, output_shape) )


if __name__ == "__main__":
    inputData = tensor( [[0,0], [0,1], [1,0], [1,1]] ).float()
    targetData = tensor( [[0,1,1,1]] ).float().resize_(4,1)

    network = Network(
        torch.nn.Linear(2, 3),
        torch.nn.Sigmoid(),
        torch.nn.Linear(3, 1),
    )

    trainer = Trainer(network)

    for i in range(10000):
        loss = trainer.train_epoch(inputData, targetData)
        if not i % 10:
            print(i, loss)
        if loss < 0.001:
            break

    print(network(inputData))
