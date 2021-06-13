import numpy as np
from numpy.lib.npyio import load
import torch
import os
import model
import tensorflow as tf
import PIL.Image
from torchvision import transforms

root_dir = "G:\\projects\\pythonProjects\\data\\CUB_200_2011\\CUB_200_2011\\images\\001.Black_footed_Albatross\\"
filenames = [os.path.join(root_dir,d) for d in tf.io.gfile.listdir(root_dir)]

image_size = 224
transform = transforms.Compose(
    [
        transforms.Resize(int(image_size / 0.875)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ),
    ]
)

def load_image(filename):
    image = PIL.Image.open(tf.io.gfile.GFile(filename,"rb")).convert("RGB")
    input = transform(image)
    return input

def test(filenames):
    with torch.no_grad():
        filenames = filenames[:20].copy()
        # print(filenames)
        np.random.shuffle(filenames)
        inputs = torch.empty((0,3,224,224))
        for filename in filenames:
            input = load_image(filename)
            input = input.view(1,3,224,224)
            inputs = torch.cat([inputs,input],dim=0)
        print(inputs.shape)
        
        mymodel = model.CUBResNet50Wrapper("./cub_200_2011_labels.txt", "./82.12_best_model.tar")
        mymodel.model.eval()
        res = mymodel.forward(inputs)
        print(res.shape)
        print("results: ",torch.argmax(res,dim=1))

# mymodel = model.CUBResNet50Wrapper("./cub_200_2011_labels.txt", "./82.12_best_model.tar")
# mymodel.model.eval()
# print(torch.argmax(mymodel.forward(load_image(filenames[0])),dim=1))
test(filenames)