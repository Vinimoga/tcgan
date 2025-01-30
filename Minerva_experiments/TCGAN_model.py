import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from minerva.models.nets.tfc import IgnoreWhenBatch1

class TCGAN_Discriminator(nn.Module):
    def __init__(
        self, 
        input_shape = (60,6), 
        noise_shape = 100, 
        n_layers = 4, 
        kernel_size = 10, 
        strides = 2, 
        g_units_base = 32,
        device = 'cpu',
        batch_1_correction: bool = False
    ):
        """The TCGAN discriminator is designed to classify time series data using a 
        convolutional neural network (CNN) architecture. It takes a time series input 
        and outputs a classification score.
        Parameters
        ----------
        input_shape : tuple, optional
            Shape of the input time series data, by default (60, 6)
        noise_shape : int, optional
            Shape of the noise vector, by default 100
        n_layer : int, optional
            Number of convolutional layers, by default 4
        kernel_size : int, optional
            Size of the convolutional kernel, by default 10
        strides : int, optional
            Stride size for the convolutional layers, by default 2
        g_units_base : int, optional
            Base number of units for the generator, by default 32
        device : str, optional
            Device to run the model on, by default 'cpu'
        """        
        super(TCGAN_Discriminator, self).__init__()
        self.input_shape = input_shape
        self.noise_shape = noise_shape
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.strides = strides
        self.g_units_base = g_units_base
        self.device = device

        self.calculate_constants()

        self.l1 = nn.Conv1d(
            self.input_shape[-1], self.units[0], self.kernel_size, stride=self.strides, padding=6, device=self.device
        )
        self.l2 = nn.Conv1d(
            self.units[0], self.units[1], self.kernel_size, stride=self.strides, padding=4, device=self.device
        )
        self.l3 = nn.Conv1d(
            self.units[1], self.units[2], self.kernel_size, stride=self.strides, padding=4, device=self.device
        )
        self.l4 = IgnoreWhenBatch1(nn.BatchNorm1d(self.units[2], device=self.device), active=batch_1_correction)
        self.l5 = nn.Conv1d(
            self.units[2], self.units[3], self.kernel_size, stride=self.strides, padding=4, device=self.device
        )
        self.l6 = IgnoreWhenBatch1(nn.BatchNorm1d(self.units[3], device=self.device), active=batch_1_correction)

        self.l7 = nn.Linear(
            self.layer_steps[0] * self.units[-1], 1, device=self.device
        )

        self.initweights()

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
    
    def calculate_constants(self):
        layer_steps = [self.input_shape[0]]
        for i in range(self.n_layers):
            layer_steps.append(int(np.ceil(float(layer_steps[-1]) / float(self.strides))))
        layer_steps.reverse()
        self.layer_steps = layer_steps

        conv_units = []
        if self.n_layers > 1:
            conv_units.append(self.g_units_base)
            for _ in range(self.n_layers - 2):  # minus the first and the last layers
                conv_units.append(conv_units[-1] * 2)
        conv_units.reverse()
        # the last layer must be aligned to the number of dimensions of input.
        conv_units.append(self.input_shape[-1])
        self.conv_units = conv_units

        units = [32]
        for _ in range(self.n_layers - 1):  # exclude the first layer.
            units.append(units[-1] * 2)
        self.units = units


    def initweights(self):
            self.l1.weight = nn.init.trunc_normal_(self.l1.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
            self.l1.bias.data.zero_()
            self.l2.weight = nn.init.trunc_normal_(self.l2.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
            self.l2.bias.data.zero_()
            self.l3.weight = nn.init.trunc_normal_(self.l3.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
            self.l3.bias.data.zero_()
            self.l4.module.weight = nn.init.trunc_normal_(self.l4.module.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
            self.l4.module.bias.data.zero_()
            self.l5.weight = nn.init.trunc_normal_(self.l5.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
            self.l5.bias.data.zero_()
            self.l6.module.weight = nn.init.trunc_normal_(self.l6.module.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
            self.l6.module.bias.data.zero_()
            self.l7.weight = nn.init.trunc_normal_(self.l7.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
            self.l7.bias.data.zero_()
            '''for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    m.weight= nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight= nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
                    m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight= nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
                    m.bias.data.zero_()
            '''


class TCGAN_Generator(nn.Module):
    def __init__(
        self, 
        input_shape = (60,6),
        noise_shape = 100, 
        n_layers = 4, 
        kernel_size = 10, 
        strides = 2, 
        g_units_base = 32, 
        device = 'cpu',
        batch_1_correction: bool = False
    ):
        super(TCGAN_Generator, self).__init__()
        self.input_shape = input_shape
        self.noise_shape = noise_shape
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.strides = strides
        self.g_units_base = g_units_base
        self.device = device

        self.calculate_constants()

        self.l1 = nn.Linear(
            noise_shape, self.layer_steps[0] * self.conv_units[0] * 2, device=self.device
        )
        self.l2 = IgnoreWhenBatch1(nn.BatchNorm1d(self.layer_steps[0] * self.conv_units[0] * 2, device=self.device), active=batch_1_correction)

        self.l3 = nn.ConvTranspose1d(
            self.conv_units[0] * 2, self.conv_units[0], self.kernel_size, stride=self.strides, padding=4, device=self.device
        )
        self.l4 = IgnoreWhenBatch1(nn.BatchNorm1d(self.conv_units[0], device=device), active=batch_1_correction)

        self.l5 = nn.ConvTranspose1d(
            self.conv_units[0], self.conv_units[1], self.kernel_size, stride=self.strides, padding=4, device=self.device
        )
        self.l6 = IgnoreWhenBatch1(nn.BatchNorm1d(self.conv_units[1], device=device), active=batch_1_correction)

        self.l7 = nn.ConvTranspose1d(
            self.conv_units[1], self.conv_units[2], self.kernel_size, stride=self.strides, padding=4, device=self.device
        )
        self.l8 = IgnoreWhenBatch1(nn.BatchNorm1d(self.conv_units[2], device=self.device), active=batch_1_correction)
        
        self.l9 = nn.ConvTranspose1d(
            self.conv_units[2], self.conv_units[3], self.kernel_size, stride=self.strides, padding=6, output_padding=0, device=self.device 
        )
        
        self.initweights()

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

    def calculate_constants(self):
        steps = self.input_shape[0]
        layer_steps = [steps]
        for i in range(self.n_layers):
            layer_steps.append(int(np.ceil(float(layer_steps[-1]) / float(self.strides))))
        layer_steps.reverse()
        self.layer_steps = layer_steps

        conv_units = []
        if self.n_layers > 1:
            conv_units.append(self.g_units_base)
            for _ in range(self.n_layers - 2):  # minus the first and the last layers
                conv_units.append(conv_units[-1] * 2)
        conv_units.reverse()
        # the last layer must be aligned to the number of dimensions of input.
        conv_units.append(self.input_shape[-1])
        self.conv_units = conv_units

        units = [32]
        for _ in range(self.n_layers - 1):  # exclude the first layer.
            units.append(units[-1] * 2)
        self.units = units


    def initweights(self):
            self.l1.weight = nn.init.trunc_normal_(self.l1.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
            self.l1.bias.data.zero_()
            self.l2.module.weight = nn.init.trunc_normal_(self.l2.module.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
            self.l2.module.bias.data.zero_()
            self.l3.weight = nn.init.trunc_normal_(self.l3.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
            self.l3.bias.data.zero_()
            self.l4.module.weight = nn.init.trunc_normal_(self.l4.module.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
            self.l4.module.bias.data.zero_()
            self.l5.weight = nn.init.trunc_normal_(self.l5.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
            self.l5.bias.data.zero_()
            self.l6.module.weight = nn.init.trunc_normal_(self.l6.module.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
            self.l6.module.bias.data.zero_()
            self.l7.weight = nn.init.trunc_normal_(self.l7.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
            self.l7.bias.data.zero_()
            self.l8.module.weight = nn.init.trunc_normal_(self.l8.module.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
            self.l8.module.bias.data.zero_()
            self.l9.weight = nn.init.trunc_normal_(self.l9.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
            self.l9.bias.data.zero_()


class TCGAN_Encoder(nn.Module):
    def __init__(
        self, 
        input_shape = (60,6), 
        noise_shape = 100, 
        n_layer = 4, 
        kernel_size = 10, 
        strides = 2, 
        g_units_base = 32,
        device = 'cpu',
        batch_1_correction: bool = False
    ):
        """The TCGAN discriminator is designed to classify time series data using a 
        convolutional neural network (CNN) architecture. It takes a time series input 
        and outputs a classification score.
        Parameters
        ----------
        input_shape : tuple, optional
            Shape of the input time series data, by default (60, 6)
        noise_shape : int, optional
            Shape of the noise vector, by default 100
        n_layer : int, optional
            Number of convolutional layers, by default 4
        kernel_size : int, optional
            Size of the convolutional kernel, by default 10
        strides : int, optional
            Stride size for the convolutional layers, by default 2
        g_units_base : int, optional
            Base number of units for the generator, by default 32
        device : str, optional
            Device to run the model on, by default 'cpu'
        """        
        super(TCGAN_Encoder, self).__init__()
        self.input_shape = input_shape
        self.noise_shape = noise_shape
        self.n_layer = n_layer
        self.kernel_size = kernel_size
        self.strides = strides
        self.g_units_base = g_units_base
        self.device = device
        

        self.calculate_constants()

        self.l1 = nn.Conv1d(
            input_shape[-1], self.units[0], kernel_size, stride=strides, padding=6, device=device
        )
        self.l2 = nn.Conv1d(
            self.units[0], self.units[1], kernel_size, stride=strides, padding=4, device=device
        )
        self.l3 = nn.Conv1d(
            self.units[1], self.units[2], kernel_size, stride=strides, padding=4, device=device
        )
        self.l4 = IgnoreWhenBatch1(nn.BatchNorm1d(self.units[2], device=device), active = batch_1_correction)

        self.l5 = nn.Conv1d(
            self.units[2], self.units[3], kernel_size, stride=strides, padding=4, device=device
        )
        self.l6 = IgnoreWhenBatch1(nn.BatchNorm1d(self.units[3], device=device), active = batch_1_correction)

        self.initweights()

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
    
    def calculate_constants(self):
        steps = self.input_shape[0]
        layer_steps = [steps]
        for i in range(self.n_layers):
            layer_steps.append(int(np.ceil(float(layer_steps[-1]) / float(self.strides))))
        layer_steps.reverse()
        self.layer_steps = layer_steps

        conv_units = []
        if self.n_layers > 1:
            conv_units.append(self.g_units_base)
            for _ in range(self.n_layers - 2):  # minus the first and the last layers
                conv_units.append(conv_units[-1] * 2)
        conv_units.reverse()
        # the last layer must be aligned to the number of dimensions of input.
        conv_units.append(self.input_shape[-1])
        self.conv_units = conv_units

        units = [32]
        for _ in range(self.n_layers - 1):  # exclude the first layer.
            units.append(units[-1] * 2)
        self.units = units


    def initweights(self):
            self.l1.weight = nn.init.trunc_normal_(self.l1.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
            self.l1.bias.data.zero_()
            self.l2.weight = nn.init.trunc_normal_(self.l2.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
            self.l2.bias.data.zero_()
            self.l3.weight = nn.init.trunc_normal_(self.l3.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
            self.l3.bias.data.zero_()
            self.l4.module.weight = nn.init.trunc_normal_(self.l4.module.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
            self.l4.module.bias.data.zero_()
            self.l5.weight = nn.init.trunc_normal_(self.l5.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
            self.l5.bias.data.zero_()
            self.l6.module.weight = nn.init.trunc_normal_(self.l6.module.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
            self.l6.module.bias.data.zero_()