import torch
import torch.nn as nn

class DDQN(nn.Module):
    def __init__(self, state_size, action_size, layer_size):
        super(DDQN, self).__init__()

        self.input_shape = state_size
        self.action_size = action_size
        self.flatten = nn.Flatten()
        self.head_1 = nn.Linear(self.input_shape, layer_size)
        
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.ff_2 = nn.Linear(layer_size, action_size)

    def forward(self, input):
        """
        
        """
        input = self.flatten(input)
        x = torch.relu(self.head_1(input))
        x = torch.relu(self.ff_1(x))
        out = self.ff_2(x)
        
        return out

class DDQN_Online(nn.Module):
    def __init__(self, state_size, action_size):
        super(DDQN_Online, self).__init__()

        self.input_shape = state_size
        self.action_size = action_size
        # self.conv = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0),
        #                             nn.ReLU(),
        #                             nn.MaxPool2d(kernel_size=4, stride=2),
        #                             nn.Conv2d(in_channels=6, out_channels=10, kernel_size=5, stride=1, padding=0),
        #                             nn.ReLU(),
        #                             nn.MaxPool2d(kernel_size=4, stride=4),
        #                             nn.Conv2d(in_channels=10, out_channels=14, kernel_size=5, stride=1, padding=0),
        #                             nn.ReLU(),
        #                             nn.Flatten())

        # self.ff_1 = nn.Linear(3920, 1240)
        # self.ff_2 = nn.Linear(1240, 512) 
        # self.ff_3 = nn.Linear(512, 64)
        # self.ff_4 = nn.Linear(64, action_size)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 4, stride = 4),
            nn.ReLU(),
            nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 4, stride = 2),
            nn.ReLU(),
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 4, stride = 1),
            nn.ReLU(),
            nn.Flatten())

        self.ff_1 = nn.Linear(1152, 520)
        self.ff_2 = nn.Linear(520, 128)
        self.ff_3 = nn.Linear(128, 64)
        self.ff_4 = nn.Linear(64, action_size)

    def forward(self, input):
        """
        
        """
        x = self.conv(input)
        # print(x.shape)
        x = torch.relu(self.ff_1(x))
        x = torch.relu(self.ff_2(x))
        x = torch.relu(self.ff_3(x))
        out = self.ff_4(x)
        
        return out