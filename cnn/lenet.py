import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F

# H = (H-F +2P)/S  + 1


class LeNet(nn.Module):
    def __init__(self, n_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5) # 6 x28x28x
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)       # 6x14x14

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.out = nn.Linear(84, n_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        print(x.shape)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # except batch size dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


if __name__ == "__main__":
    model = LeNet(n_classes=10)
    summary(model, (1, 32, 32), batch_size=2)
    # x  = torch.rand((28, 28))
    # print(model(x))
