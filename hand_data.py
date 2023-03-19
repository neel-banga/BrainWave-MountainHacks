import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import random

PADDING_SIZE = 9917000
RANDOM_SEED = 66

def purge_folder(folder_path):
    files = os.listdir(folder_path)

    random.shuffle(files)
    half_index = len(files) // 2

    for file_name in files[:half_index]:

        file_path = os.path.join(folder_path, file_name)

        if os.path.isfile(file_path):

            os.remove(file_path)

        elif os.path.isdir(file_path):

            os.rmdir(file_path)

def join_label(label):

    if label == 0:
        return torch.tensor([1, 0], dtype=torch.float32)
    if label == 2:
        return torch.tensor([0, 1], dtype=torch.float32)



def image_to_tensor(file_path):
    img = Image.open(file_path)

    transform = transforms.Compose([
        transforms.PILToTensor()
    ])
    
    img_tensor = transform(img)
    
    return img_tensor

def process_data():
    train_set = []
    set_y = []

    for folder in os.listdir(os.path.join(os.getcwd(), 'SignNums')):

        if not folder.startswith('.'):

            folder_path = os.path.join(os.getcwd(), 'SignNums', folder)
            #purge_folder(folder_path)

            for filename in os.listdir(folder_path):
                f = os.path.join(os.getcwd(), 'SignNums', folder, filename)
                if os.path.isfile(f):
                    tensor = image_to_tensor(f)
                    tensor = tensor.type(torch.float32)
                    train_set.append(tensor.view(-1))
                    #set_y.append(join_label(int(folder)))
                    set_y.append(torch.tensor(int(folder)))
     
    set_x = torch.nn.utils.rnn.pad_sequence(train_set, batch_first=True)
    
    #set_x = set_x.view(PADDING_SIZE)
    
    # Randomize both datasets with seed so that they are both randomized in the same order

    return set_x, set_y

def return_data():
    random.seed(RANDOM_SEED)
    set_x, set_y = process_data()
    random.shuffle(set_x), random.shuffle(set_y)
    return set_x, set_y
