#!/usr/bin/env python
import torch
from torch import tensor
import matplotlib.pyplot as plt



def expand_list(lst):
    result = []
    for i in range(len(lst)):
        result.append(lst[i])
        if i and i != len(lst)-1:
            result.append(lst[i])
    return result


def graph_loss(net_trainer, interval=50):
    if not net_trainer.history:
        return

    epochList = [i for i in net_trainer.history]
    lossList = [net_trainer.history[i] for i in net_trainer.history]
    netLen = len( [i for i in net_trainer.model.children()] )

    plt.plot(epochList, lossList)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Optimizer: {} | Layers: {}".format(net_trainer.optimizer_name, netLen))
    plt.show()


class Trainer():
    def __init__(self, model, optimizer="adam", learning_rate="default", reduction="mean"):
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

    trainer_setup = False
    trainer = None

    def setup_trainer(self, optimizer="adam", learning_rate="default", reduction="mean"):
        self.trainer = Trainer(self, optimizer=optimizer, learning_rate=learning_rate, reduction=reduction)
        self.trainer_setup = True



class Network_Builder():
    def __init__(self):
        self.layers = []
        self.network = None

    def output_size(self):
        return self.layers[len(self.layers)-1][1]

    def adjust_layers(self):
        prevLayer = None
        for layer in self.layers:
            if prevLayer:
                if "input" in layer and "output" in layer:
                    if "output" in prevLayer:
                        if prevLayer["output"] != layer["input"]:
                            prevLayer["output"] = layer["input"]
                    prevLayer = layer
            else:
                prevLayer = layer

    def build_layers(self):
        builtList = []
        for layer in self.layers:
            newLayer = None
            if layer["type"] == "linear":
                newLayer = torch.nn.Linear(layer["input"], layer["output"])
            elif layer["type"] == "sigmoid":
                newLayer = torch.nn.Sigmoid()

            if newLayer:
                builtList.append(newLayer)
        return builtList

    def build(self):
        if not self.layers:
            return

        self.adjust_layers()   # Should adjust the weights so each layer matches correctly

        builtList = self.build_layers()
        self.network = Network(*builtList)
        return self.network

    def add_linear_layers(self, layers):
        if isinstance(layers, list):
            mList = expand_list(layers)
            mList = iter(mList)
            for i in mList:
                try:
                    newLayer = {"input": i, "output": next(mList), "type": "linear"}
                    self.layers.append(newLayer)
                except StopIteration:
                    newLayer = {"input": i, "output": i, "type": "linear"}
                    self.layers.append(newLayer)

    def add_sigmoid(self):
        newLayer = {"type": "sigmoid"}
        self.layers.append(newLayer)



if __name__ == "__main__":
    inputData = tensor( [[0,0], [0,1], [1,0], [1,1]] ).float()
    targetData = tensor( [[0,1,1,1]] ).float().resize_(4,1)

    network = Network(
        torch.nn.Linear(2, 3),
        torch.nn.Linear(3, 1),
        torch.nn.Sigmoid(),
    )

    netBuilder = Network_Builder()

    netBuilder.add_linear_layers([2])
    netBuilder.add_sigmoid()
    netBuilder.add_linear_layers([3, 1])

    network = netBuilder.build()

    trainer = Trainer(network)

    lossList = []
    epochList = []

    while True:
        loss = trainer.trainUntil(inputData, targetData, iterations=50, target_loss=0.001)
        print(trainer.iterations, loss)
        lossList.append(loss)
        epochList.append(trainer.iterations)
        if trainer.iterations >= 30000:
            break
        if loss < 0.001:
            break

    graph_loss(trainer)

    print(network(inputData))
