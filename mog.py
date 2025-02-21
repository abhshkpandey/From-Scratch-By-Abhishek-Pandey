import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# Define a small CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define the MoG gradient mixing function
def mixture_of_gradients(model, loss_fn, optimizer, data, target):
    optimizer.zero_grad()
    output = model(data)
    
    # Define different loss functions (gradient experts)
    loss_1 = loss_fn(output, target)  # Standard Cross-Entropy Loss
    loss_2 = torch.mean((output - output.mean())**2)  # Contrastive-like Loss
    loss_3 = torch.mean(torch.log(1 + torch.exp(-output)))  # Smoothed Loss
    
    # Compute gradients from different experts
    loss_1.backward(retain_graph=True)
    grad_1 = [param.grad.clone() for param in model.parameters()]
    
    optimizer.zero_grad()
    loss_2.backward(retain_graph=True)
    grad_2 = [param.grad.clone() for param in model.parameters()]
    
    optimizer.zero_grad()
    loss_3.backward()
    grad_3 = [param.grad.clone() for param in model.parameters()]
    
    # Compute weights dynamically using softmax
    weights = torch.softmax(torch.tensor([loss_1.item(), loss_2.item(), loss_3.item()]), dim=0)
    
    # Apply mixture of gradients
    optimizer.zero_grad()
    for param, g1, g2, g3 in zip(model.parameters(), grad_1, grad_2, grad_3):
        param.grad = weights[0] * g1 + weights[1] * g2 + weights[2] * g3
    
    optimizer.step()
    return loss_1.item()

# Training loop
model = CNN().cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5):  # Train for 5 epochs
    total_loss = 0
    for data, target in dataloader:
        data, target = data.cuda(), target.cuda()
        loss = mixture_of_gradients(model, loss_fn, optimizer, data, target)
        total_loss += loss
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")
