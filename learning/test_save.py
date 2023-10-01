import torch
import torch.nn as nn

# Define a simple PyTorch module
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)
dev = "cuda"
# Create an instance of the module
model = MyModule().to(dev)
print("Save Model = ", model)
# Save the module to a file called 'model.pth'
torch.save(model, 'model.pth')

load_model = torch.load("model.pth",map_location=dev)

print("Load Model = ", load_model)