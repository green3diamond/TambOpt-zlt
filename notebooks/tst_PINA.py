import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pina import Trainer, Condition, LabelTensor
from pina.problem import AbstractProblem
from pina.solver import GAROM
from pina.callback import MetricTracker
from pina.model import FeedForward
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class Generator(nn.Module):

    def __init__(
        self,
        input_dimension=1,
        parameters_dimension=1,
        noise_dimension=1,
        activation=torch.nn.SiLU,
    ):
        super().__init__()

        self._noise_dimension = noise_dimension
        self._activation = activation
        self.model = FeedForward(6 * noise_dimension, input_dimension)
        self.condition = FeedForward(parameters_dimension, 5 * noise_dimension)

    def forward(self, param):
        # uniform sampling in [-1, 1]
        z = (
            2
            * torch.rand(
                size=(param.shape[0], self._noise_dimension),
                device=param.device,
                dtype=param.dtype,
                requires_grad=True,
            )
            - 1
        )
        return self.model(torch.cat((z, self.condition(param)), dim=-1))


# Simple Discriminator Network


class Discriminator(nn.Module):

    def __init__(
        self,
        input_dimension=1,
        parameter_dimension=1,
        hidden_dimension=2,
        activation=torch.nn.ReLU,
    ):
        super().__init__()

        self._activation = activation
        self.encoding = FeedForward(input_dimension, hidden_dimension)
        self.decoding = FeedForward(2 * hidden_dimension, input_dimension)
        self.condition = FeedForward(parameter_dimension, hidden_dimension)

    def forward(self, data):
        x, condition = data
        encoding = self.encoding(x)
        conditioning = torch.cat((encoding, self.condition(condition)), dim=-1)
        decoding = self.decoding(conditioning)
        return decoding




# Generate noisy sine wave data
x = np.linspace(0, 2*np.pi, 1000)
noise = 0.1 * np.random.normal(size=x.shape)
y = np.sin(x) + noise
data = pd.DataFrame({'x': x, 'noisy_sin': y})

input_columns = ['x']
output_columns = ['noisy_sin']

# Normalize outputs
y_scaler = StandardScaler()
y_normalized = y_scaler.fit_transform(data[output_columns])

# Wrap inputs and outputs
x_pina = LabelTensor(data[input_columns].values, input_columns)
y_pina = LabelTensor(y_normalized, output_columns)

class SineWaveProblem(AbstractProblem):
    input_variables = input_columns
    output_variables = output_columns
    conditions = {"data": Condition(input=x_pina, target=y_pina)}

problem = SineWaveProblem()

# Create generator and discriminator with input/output dimension 1
generator = Generator()
discriminator = Discriminator()

solver = GAROM(problem, generator, discriminator)

trainer = Trainer(
    solver=solver,
    max_epochs=500,
    logger=True,
    callbacks=[MetricTracker()],
    accelerator="cpu",
    train_size=0.7,
    test_size=0.2,
    val_size=0.1,
)

trainer.train()

# plot loss
trainer_metrics = trainer.callbacks[0].metrics
loss = trainer_metrics["train_loss"]
epochs = range(len(loss))
plt.plot(epochs, loss.cpu())
# plotting
plt.xlabel("epoch")
plt.ylabel("loss")
plt.yscale("log")
plt.show()

all_outputs = None
all_targets = None

trainer.data_module.setup("test")
with torch.no_grad():
    for data in trainer.data_module.test_dataloader():
        # for data in trainer.data_module.train_dataloader():
        inputs, target = data[0][1]["input"], data[0][1]["target"]
        outputs = solver(inputs)

        if all_outputs is None:
            all_outputs = LabelTensor(outputs, labels=output_columns)
            all_targets = target
        else:
            all_outputs.append(LabelTensor(outputs, labels=output_columns))
            all_targets.append(target)
        break

# plot targets vs predictions for validation set
y_mean = all_outputs.detach()
true_output = all_targets.detach()

plt.figure(figsize=(18, 10))
# use 3 columns per row
for i, col in enumerate(output_columns):
    plt.subplot(len(output_columns)//4+1, 4, i+1)
    plt.scatter(true_output[:, i], y_mean[:, i], alpha=0.5)
    plt.plot([true_output[:, i].min(), true_output[:, i].max()],
             [true_output[:, i].min(), true_output[:, i].max()], 'r--')
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(col)
plt.tight_layout(pad=1.0)
plt.show()
