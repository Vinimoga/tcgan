import torch
import torch.nn as nn
import torch.nn.functional as F


class TCGAN_Discriminator(nn.Module):
    def __init__(self, input_shape, units, kernel_size, strides, layer_steps, device):
        super(TCGAN_Discriminator, self).__init__()
        self.l1 = nn.Conv1d(
            input_shape[-1],
            units[0],
            kernel_size,
            stride=strides,
            padding=6,
            device=device,
        )
        self.l2 = nn.Conv1d(
            units[0], units[1], kernel_size, stride=strides, padding=4, device=device
        )
        self.l3 = nn.Conv1d(
            units[1], units[2], kernel_size, stride=strides, padding=4, device=device
        )
        self.l4 = nn.BatchNorm1d(units[2], device=device)
        self.l5 = nn.Conv1d(
            units[2], units[3], kernel_size, stride=strides, padding=4, device=device
        )
        self.l6 = nn.BatchNorm1d(units[3], device=device)
        self.l7 = nn.Linear(layer_steps[0] * units[-1], 1, device=device)

    def forward(self, x):
        x = self.l1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.l2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.l3(x)
        x = self.l4(x)
        x = F.leaky_relu(x, 0.2)
        x = self.l5(x)
        x = self.l6(x)
        x = F.leaky_relu(x, 0.2)
        x = torch.transpose(x, 1, 2)
        x = torch.flatten(x, start_dim=1)
        x = self.l7(x)
        return x


class TCGAN_Generator(nn.Module):
    def __init__(
        self, input_layer, conv_units, kernel_size, strides, layer_steps, device
    ):
        super(TCGAN_Generator, self).__init__()
        self.layer_steps = layer_steps
        self.conv_units = conv_units

        self.l1 = nn.Linear(
            input_layer, layer_steps[0] * conv_units[0] * 2, device=device
        )
        self.l2 = nn.BatchNorm1d(layer_steps[0] * conv_units[0] * 2, device=device)
        self.l3 = nn.ConvTranspose1d(
            conv_units[0] * 2,
            conv_units[0],
            kernel_size,
            stride=strides,
            padding=4,
            device=device,
        )
        self.l4 = nn.BatchNorm1d(conv_units[0], device=device)
        self.l5 = nn.ConvTranspose1d(
            conv_units[0],
            conv_units[1],
            kernel_size,
            stride=strides,
            padding=4,
            device=device,
        )
        self.l6 = nn.BatchNorm1d(conv_units[1], device=device)
        self.l7 = nn.ConvTranspose1d(
            conv_units[1],
            conv_units[2],
            kernel_size,
            stride=strides,
            padding=4,
            device=device,
        )
        self.l8 = nn.BatchNorm1d(conv_units[2], device=device)
        self.l9 = nn.ConvTranspose1d(
            conv_units[2],
            conv_units[3],
            kernel_size,
            stride=strides,
            padding=6,
            output_padding=0,
            device=device,
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = F.relu(x)
        x = x.reshape(x.shape[0], self.layer_steps[0], self.conv_units[0] * 2)
        x = torch.transpose(x, 1, 2)
        x = self.l3(x)
        x = self.l4(x)
        x = F.relu(x)
        x = self.l5(x)
        x = self.l6(x)
        x = F.relu(x)
        x = self.l7(x)
        x = self.l8(x)
        x = F.relu(x)
        x = self.l9(x)
        return x

class TCGAN_Encoder(nn.Module):
    def __init__(self, input_shape, units, kernel_size, strides, device):
        super(TCGAN_Discriminator, self).__init__()
        self.l1 = nn.Conv1d(
            input_shape[-1],
            units[0],
            kernel_size,
            stride=strides,
            padding=6,
            device=device,
        )
        self.l2 = nn.Conv1d(
            units[0], units[1], kernel_size, stride=strides, padding=4, device=device
        )
        self.l3 = nn.Conv1d(
            units[1], units[2], kernel_size, stride=strides, padding=4, device=device
        )
        self.l4 = nn.BatchNorm1d(units[2], device=device)
        self.l5 = nn.Conv1d(
            units[2], units[3], kernel_size, stride=strides, padding=4, device=device
        )
        self.l6 = nn.BatchNorm1d(units[3], device=device)

    def forward(self, x):
        x = self.l1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.l2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.l3(x)
        x = self.l4(x)
        x = F.leaky_relu(x, 0.2)
        x = self.l5(x)
        x = self.l6(x)
        x = F.leaky_relu(x, 0.2)
        x = torch.flatten(x, start_dim=1)
        return x
