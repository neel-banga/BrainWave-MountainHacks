import hand_data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle 
from PIL import Image
import torchvision.transforms as transforms


def load_data():
    set_x, set_y = hand_data.return_data()

    with open('shuffled_arrays.pickle', 'wb') as f:
        pickle.dump((set_x, set_y), f)

#load_data()

with open('shuffled_arrays.pickle', 'rb') as f:
    set_x, set_y = pickle.load(f)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10000, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 64)
        self.fc6 = nn.Linear(64, 64)
        self.fc7 = nn.Linear(64, 64)
        self.fc8 = nn.Linear(64, 64)
        self.fc9 = nn.Linear(64, 64)
        self.fc10 = nn.Linear(64, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = self.fc10(x)
        return F.log_softmax(x,dim=-1)

def train_model():
    model = Model()

    EPOCHS = 100
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        for i in range(len(set_x)):
            x = set_x[i]
            y = set_y[i]
            model.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        print(loss)

    torch.save(model.state_dict(), 'model.pth')

train_model()

correct = 0
loss = 0
total = 0

net = Model()
net.load_state_dict(torch.load('model.pth'))
net.eval()

y = set_y[60]
x = set_x[60]


output = net(x)
output_list = output.tolist()
mv = max(output_list)
index = output_list.index(mv)+1


def evaluate(image_path):
    img = Image.open(image_path)
    transform = transforms.ToTensor()
    tensor = transform(img)
    output_size = (1, 10000)
    adaptive_pool = torch.nn.AdaptiveMaxPool2d(output_size)
    tensor = adaptive_pool(tensor)
    print(tensor.size())
    
    with torch.no_grad():
        output = net(tensor)
    output_list = output.tolist()

    print(output_list)

    mv = max(output_list)

    output_list = output.tolist()
    index = output_list.index(mv)+1

    return index
