#!/usr/bin/env python
import torch
from torch import tensor
import matplotlib.pyplot as plt



class Trainer():
    def __init__(self, model, optimizer="adam", learning_rate="default", reduction='mean'):
        self.optimizers = {
            "adam": torch.optim.Adam,
            "adadelta": torch.optim.Adadelta,
            "adagrad": torch.optim.Adagrad,
            "adamax": torch.optim.Adamax,
            "adamw": torch.optim.AdamW,
            "asgd": torch.optim.ASGD,
            "rmsprop": torch.optim.RMSprop,
            "rprop": torch.optim.Rprop,
            "sgd": torch.optim.SGD,
        }

        if optimizer == "sgd" and learning_rate == "default":
            learning_rate = 0.1

        self.learning_rate = learning_rate
        self.loss_fn = torch.nn.MSELoss(reduction=reduction)
        self.model = model
        self.iterations = 0
        self.history = {}   # Structure: {<epoch>: <loss>}
        self.optimizer_name = optimizer

        if self.learning_rate == "default":
            self.optimizer = self.optimizers[optimizer.lower()](model.parameters())
            self.learning_rate = self.optimizer.defaults["lr"]
        else:
            self.optimizer = self.optimizers[optimizer.lower()](model.parameters(), lr=self.learning_rate)

    def set_optimizer(self, optimizer):
        self.optimizer = self.optimizers[optimizer.lower()](self.model.parameters(), lr=self.learning_rate)

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        self.optimizer.defaults["lr"] = learning_rate

    def train_epoch(self, inputData, target):
        result = self.model(inputData)

        loss = self.loss_fn(result, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.iterations += 1
        self.history[self.iterations] = loss.item()

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

    lossList = []
    epochList = []

    for i in range(10000):
        loss = trainer.trainUntil(inputData, targetData, iterations=20)
        print(trainer.iterations, loss)
        lossList.append(loss)
        epochList.append(trainer.iterations)
        if loss < 0.001:
            break
    for epoch in trainer.history:
        plt.plot(epochList, lossList)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Optimizer: {}".format(trainer.optimizer_name))
    plt.show()

    print(network(inputData))
