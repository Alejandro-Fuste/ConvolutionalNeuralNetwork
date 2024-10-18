import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()

        # Define various layers here, such as in the tutorial example

        # step 1: fully connected layer with 100 neurons
        self.fc1 = nn.Linear(28 * 28, 100)

        # step 2: insert two convolutional layers
        self.conv1 = nn.Conv2d(1, 40, 5)
        self.conv2 = nn.Conv2d(40, 40, 5)
        # self.fc1 = nn.Linear(40 * 28 * 28, 100)

        # This will select the forward pass function based on mode for the ConvNet.
        # Based on the question, you have 5 modes available for step 1 to 5.
        # During creation of each ConvNet model, you will assign one of the valid mode.
        # This will fix the forward function (and the network graph) for the entire training/testing
        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        elif mode == 3:
            self.forward = self.model_3
        elif mode == 4:
            self.forward = self.model_4
        elif mode == 5:
            self.forward = self.model_5
        else:
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)

    # Baseline model. step 1
    def model_1(self, x):
        # ======================================================================
        # One fully connected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        x = x.view(-1, self.flatten_features(x))

        fcl = F.sigmoid(self.fc1(x))

        return fcl

    # Use two convolutional layers.
    def model_2(self, x):
        # ======================================================================
        # Two convolutional layers + one fully connected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        x = F.max_pool2d(F.sigmoid(self.conv1(x)), (2, 1))
        x = F.max_pool2d(F.sigmoid(self.conv2(x)), (2, 1))

        x = x.view(-1, self.flatten_features(x))

        x = F.sigmoid(self.fc1(x))
        # Uncomment the following return stmt once method implementation is done.
        return x

    # Replace sigmoid with ReLU.
    def model_3(self, x):
        # ======================================================================
        # Two convolutional layers + one fully connected layer, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 1))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 1))

        x = x.view(-1, self.flatten_features(x))

        x = F.relu(self.fc1(x))

        # Uncomment the following return stmt once method implementation is done.
        return x

    # Add one extra fully connected layer.
    def model_4(self, X):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        return NotImplementedError()

    # Use Dropout now.
    def model_5(self, X):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        # and  + Dropout.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #

        # Uncomment the following return stmt once method implementation is done.
        # return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        return NotImplementedError()

    def flatten_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s

        return num_features
