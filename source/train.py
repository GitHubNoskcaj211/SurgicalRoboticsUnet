from model import UNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

image_rows = 1024
image_cols = 1280
image_channels = 3

def load_dataset():
    data_path = '../Dataset/instrument_1_4_training/'
    
    input_paths = glob.glob(f'{data_path}instrument_dataset*/left_frames/*.png', recursive=True)
    
    
    # input_paths = input_paths[:3]
    
    num_frames = len(input_paths)
    
    inputs = np.empty(shape = (num_frames, image_rows, image_cols, image_channels), dtype=np.uint8)
    labels = np.zeros(shape = (num_frames, image_rows, image_cols), dtype=np.bool_)
    
    print('Getting Frames:', end='', flush=True)
    c = 0
    for frame in range(num_frames):
        input_path = input_paths[frame]
        
        inputs[frame] = cv2.imread(input_path)[28:-28,320:-320]
        
        for label_file in glob.glob(f'{data_path}{input_path.split(chr(47))[3]}/ground_truth/**/{input_path.split(chr(47))[5]}', recursive=True):
            labels[frame] = np.logical_or(labels[frame], cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)[28:-28,320:-320] != 0)
        if c % 50 == 0:
            print('.', end='', flush = True)
        c += 1
    return torch.as_tensor(inputs).permute(0, 3, 1, 2), torch.as_tensor(labels).unsqueeze(3).permute(0, 3, 1, 2)

def show_image(image):
    plt.imshow(image.permute(1, 2, 0))
    plt.show()

def prepare_inputs(inputs):
    return torch.div(inputs.to(torch.float32), 255.0)

def prepare_labels(labels):
    return labels.to(torch.float32)

def minibatch(model, optimizer, criterion, inputs, labels, batch_size, losses):
    permutation = torch.randperm(inputs.size()[0])
    
    c = 0
    for i in range(0, 70, batch_size): #inputs.size()[0], batch_size):
        print(f'Batch: {c}')
        c += 1
        optimizer.zero_grad()

        indices = permutation[i : i + batch_size]
        batch_inputs, batch_labels = inputs[indices], labels[indices]
        
        outputs = model.forward(prepare_inputs(batch_inputs))
        loss = criterion(outputs, prepare_labels(batch_labels))
        
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        
def plot_losses(losses):
    plt.plot(losses)
    plt.show()

if __name__ == "__main__":
    epoch_num = 7
    max_epochs = 50
    batch_size = 10
    
    inputs, labels = load_dataset()
    print(np.shape(inputs))
    print(np.shape(labels))
    
    # show_image(inputs[0])
    # show_image(labels[0])
    
    # Define the model parameters
    input_channels = 3
    output_channels = 1
    
    learning_rate = 0.00001
    
    unet = UNet(3, 1)
    unet.load_state_dict(torch.load("../Models/model_6.pth"))
    optimizer = torch.optim.Adam(unet.parameters(), lr = learning_rate)
    criterion = nn.BCELoss()
    
    total_params = sum(p.numel() for p in unet.parameters())
    print(f"Total number of parameters in network unet: {total_params}")
    
    # Define the learning parameters
    
    while(epoch_num < max_epochs):
        print(f'Epoch: {epoch_num}')
        
        losses = []

        minibatch(unet, optimizer, criterion, inputs, labels, batch_size, losses)

        torch.save(unet.state_dict(), f"../Models/model_{epoch_num}.pth")
        
        with open(f'../Losses/losses_epoch_{epoch_num}.txt', 'w') as f:
            for item in losses:
                f.write("%s\n" % item)
        
        epoch_num += 1
