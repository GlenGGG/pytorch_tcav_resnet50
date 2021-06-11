import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from abc import ABCMeta
from abc import abstractmethod
import numpy as np
import tensorflow as tf
import gc

import os
from torchvision.transforms.functional import _is_numpy_image

from torchvision.transforms.transforms import CenterCrop

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ModelWrapper(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        self.model = None
        self.model_name = None

    @abstractmethod
    def get_cutted_model(self, bottleneck):
        pass

    def get_gradient(self, acts, y, bottleneck_name):
        inputs = torch.autograd.Variable(
            torch.tensor(acts).to(device), requires_grad=True
        )
        targets = (y[0] * torch.ones(inputs.size(0))).long().to(device)

        cutted_model = self.get_cutted_model(bottleneck_name).to(device)
        cutted_model.eval()
        outputs = cutted_model(inputs)

        # y=[i]
        grads = -torch.autograd.grad(outputs[:, y[0]], inputs)[0]

        grads = grads.detach().cpu().numpy()

        cutted_model = None
        gc.collect()

        return grads

    def reshape_activations(self, layer_acts):
        return np.asarray(layer_acts).squeeze()

    @abstractmethod
    def label_to_id(self, label):
        pass

    def run_examples(self, examples, bottleneck_name):

        global bn_activation
        bn_activation = None

        def save_activation_hook(mod, inp, out):
            global bn_activation
            bn_activation = out

        handle = self.model._modules[bottleneck_name].register_forward_hook(
            save_activation_hook
        )

        self.model.to(device)
        inputs = torch.FloatTensor(examples).permute(0, 3, 1, 2).to(device)
        self.model.eval()
        self.model(inputs)
        acts = bn_activation.detach().cpu().numpy()
        handle.remove()

        return acts


class ImageModelWrapper(ModelWrapper):
    """Wrapper base class for image models."""

    def __init__(self, image_shape):
        super(ModelWrapper, self).__init__()
        # shape of the input image in this model
        self.image_shape = image_shape

    def get_image_shape(self):
        """returns the shape of an input image."""
        return self.image_shape


class PublicImageModelWrapper(ImageModelWrapper):
    """Simple wrapper of the public image models with session object."""

    def __init__(self, labels_path, image_shape):
        super(PublicImageModelWrapper, self).__init__(image_shape=image_shape)
        self.labels = tf.io.gfile.GFile(labels_path).read().splitlines()

    def label_to_id(self, label):
        return self.labels.index(label)


class ResNet50_cutted(torch.nn.Module):
    def __init__(self, resnet50, bottleneck):
        super(ResNet50_cutted, self).__init__()
        names = list(resnet50._modules.keys())
        # print("ResNet50_cutted.names: ", names)
        layers = list(resnet50.children())
        # print("ResNet50_cutted.layers: ", layers)
        self.layers = torch.nn.ModuleList()
        self.layers_names = []

        bottleneck_met = False
        for name, layer in zip(names, layers):
            if name == bottleneck:
                bottleneck_met = True
                continue  # because we already have the output of the bottleneck layer
            if not bottleneck_met:
                continue

            self.layers.append(layer)
            self.layers_names.append(name)
        # print("ResNet50_cutted self.layers", self.layers)
        # print("ResNet50_cutted self.layers_names", self.layers_names)

    def forward(self, x):
        y = x
        for i in range(len(self.layers)):
            # pre-forward process
            if self.layers_names[i] == "fc":
                y = torch.flatten(y, 1)
            y = self.layers[i](y)
        return y


class InceptionV3_cutted(torch.nn.Module):
    def __init__(self, inception_v3, bottleneck):
        super(InceptionV3_cutted, self).__init__()
        names = list(inception_v3._modules.keys())
        layers = list(inception_v3.children())
        # print("Inception_v3_cutted.names: ", names)
        # print("Inception_v3_cutted.layers: ", layers)

        self.layers = torch.nn.ModuleList()
        self.layers_names = []

        bottleneck_met = False
        for name, layer in zip(names, layers):
            if name == bottleneck:
                bottleneck_met = True
                continue  # because we already have the output of the bottleneck layer
            if not bottleneck_met:
                continue
            if name == "AuxLogits":
                continue

            self.layers.append(layer)
            self.layers_names.append(name)
        # print("Inception_v3_cutted self.layers", self.layers)
        # print("Inception_v3_cutted self.layers_names", self.layers_names)

    def forward(self, x):
        y = x
        for i in range(len(self.layers)):
            # pre-forward process
            # if self.layers_names[i] == 'Conv2d_3b_1x1':
            #     y = F.max_pool2d(y, kernel_size=3, stride=2)
            # elif self.layers_names[i] == 'Mixed_5b':
            #     y = F.max_pool2d(y, kernel_size=3, stride=2)
            # elif self.layers_names[i] == 'fc':
            #     y = F.adaptive_avg_pool2d(y, (1, 1))
            #     y = F.dropout(y, training=self.training)
            #     y = y.view(y.size(0), -1)
            if self.layers_names[i] == "fc":
                y = y.view(y.size(0), -1)

            # print(self.layers_names[i])

            # print(y.shape)
            y = self.layers[i](y)
        return y


class InceptionV3Wrapper(PublicImageModelWrapper):
    def __init__(self, labels_path):
        image_shape = [299, 299, 3]
        super(InceptionV3Wrapper, self).__init__(
            image_shape=image_shape, labels_path=labels_path
        )
        self.model = torchvision.models.inception_v3(
            pretrained=True, transform_input=True
        )
        # self.model = torchvision.models.resnet50(pretrained=True)
        self.model_name = "InceptionV3_public"

    def forward(self, x):
        return self.model.forward(x)

    def get_cutted_model(self, bottleneck):
        return InceptionV3_cutted(self.model, bottleneck)


class ResNet50Wrapper(PublicImageModelWrapper):
    def __init__(self, labels_path):
        image_shape = [256, 256, 3]
        super(ResNet50Wrapper, self).__init__(
            image_shape=image_shape, labels_path=labels_path
        )
        self.model = torchvision.models.resnet50(pretrained=True)
        self.model_name = "ResNet50_public"
        self.transform = transforms.Compose(
            [
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def forward(self, x):
        x = self.transform(x)
        return self.model.forward(x)

    def get_cutted_model(self, bottleneck):
        return ResNet50_cutted(self.model, bottleneck)


class CUBResNet50Wrapper(PublicImageModelWrapper):
    def __init__(self, labels_path, model_dict=None):
        image_size = 448
        image_shape = [int(image_size/0.875), int(image_size/0.875), 3]
        super(CUBResNet50Wrapper, self).__init__(
            image_shape=image_shape, labels_path=labels_path
        )
        if model_dict:
            self.load_model(model_dict)
        else:
            self.model = torchvision.models.resnet50(pretrained=True)
        self.model_name = "CUBResNet50_public"
        self.transform = transforms.Compose(
            [
                # transforms.Resize(int(image_size / 0.875)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def forward(self, x):
        x = self.transform(x)
        return self.model.forward(x)

    def get_cutted_model(self, bottleneck):
        return ResNet50_cutted(self.model, bottleneck)

    def load_model(self, model_dict):
        self.model = torchvision.models.resnet50()
        self.model.fc = nn.Linear(self.model.fc.in_features, 200)
        dict = torch.load(model_dict)["state"]
        new_dict = {}
        for key, val in dict.items():
            new_dict[key.replace("res50.", "")] = val
        self.model.load_state_dict(new_dict)
