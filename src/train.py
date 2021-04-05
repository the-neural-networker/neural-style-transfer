from argparse import ArgumentParser
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import random_split

from torchvision import transforms
from torchvision.models import vgg19

from PIL import Image
import matplotlib.pyplot as plt

from models.vgg19 import VGG19
from losses import ContentLoss, StyleLoss

from tqdm import tqdm

def main(input_image="content"):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"training on device {device}")

    # content and style images
    content_dir = "../images/content/neckarfront.jpg"
    style_dir = "../images/style/kandinsky.jpg"
    content = image_loader(content_dir, device)
    style = image_loader(style_dir, device)

    # input image
    if input_image == "content":
        x = content.clone()
    elif input_image == "style":
        x = style.clone()
    else:
        x = torch.randn(content.data.size(), device=device)

    # mean and std for vgg19
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # vgg19 model
    model = VGG19(mean=mean, std=std).to(device=device)
    model = load_vgg19_weights(model)
    optimizer = optim.LBFGS([x.requires_grad_()])

    # computing content and style representations
    content_outputs = model(content)
    style_outputs = model(style)

    # defining content and style losses
    content_loss = ContentLoss(content_outputs["conv4"][1], device)
    style_losses = []
    for i in range(1, 6):
        style_losses.append(StyleLoss(style_outputs[f"conv{i}"][0], device))

    # run style transfer
    output = train(model, optimizer, content_loss, style_losses, content, style, x)
    output = output.detach().to("cpu")

    # save result
    plt.imsave("../result/result.jpg", output[0].permute(1, 2, 0))
    # torch.save(output.detach().to("cpu"), "../result/result_new.pt")

def image_loader(path, device=torch.device("cuda")):
    """Loads and resizes the image."""
    transform = transforms.Compose([
                    transforms.Resize((512, 512)),
                    transforms.ToTensor(),
                ])
    img = Image.open(path)
    img = transform(img)
    img = img.unsqueeze(0).to(device=device)
    return img

def load_vgg19_weights(model):
    """Loads VGG19 pretrained weights from ImageNet for style transfer"""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pretrained_model = vgg19(pretrained=True).features.to(device).eval()

    matching_keys = {
        "conv1.conv1.weight": "0.weight",
        "conv1.conv1.bias": "0.bias",
        "conv1.conv2.weight": "2.weight",
        "conv1.conv2.bias": "2.bias",

        "conv2.conv1.weight": "5.weight",
        "conv2.conv1.bias": "5.bias",
        "conv2.conv2.weight": "7.weight",
        "conv2.conv2.bias": "7.bias",

        "conv3.conv1.weight": "10.weight",
        "conv3.conv1.bias": "10.bias",
        "conv3.conv2.weight": "12.weight",
        "conv3.conv2.bias": "12.bias",
        "conv3.conv3.weight": "14.weight",
        "conv3.conv3.bias": "14.bias",
        "conv3.conv4.weight": "16.weight",
        "conv3.conv4.bias": "16.bias",

        "conv4.conv1.weight": "19.weight",
        "conv4.conv1.bias": "19.bias",
        "conv4.conv2.weight": "21.weight",
        "conv4.conv2.bias": "21.bias",
        "conv4.conv3.weight": "23.weight",
        "conv4.conv3.bias": "23.bias",
        "conv4.conv4.weight": "25.weight",
        "conv4.conv4.bias": "25.bias",

        "conv5.conv1.weight": "28.weight",
        "conv5.conv1.bias": "28.bias",
        "conv5.conv2.weight": "30.weight",
        "conv5.conv2.bias": "30.bias",
        "conv5.conv3.weight": "32.weight",
        "conv5.conv3.bias": "32.bias",
        "conv5.conv4.weight": "34.weight",
        "conv5.conv4.bias": "34.bias",
    }

    pretrained_dict = pretrained_model.state_dict()
    model_dict = model.state_dict()

    for key, value in matching_keys.items():
        model_dict[key] = pretrained_dict[value]
    
    model.load_state_dict(model_dict)

    return model


def train(model, optimizer, content_loss, style_losses, 
          content, style, x, iterations=100, alpha=1000, beta=1000000, style_weight=1):
    """Train the neural style transfer algorithm."""

    with tqdm(range(iterations)) as iterations:
        for iteration in iterations:
            iterations.set_description(f"Iteration: {iteration}")

            def closure():
                optimizer.zero_grad()

                # correcting to 0-1 range
                x.data.clamp_(0, 1)
                outputs = model(x)

                # input content and style representations
                content_feature_maps = outputs["conv4"][1]
                style_feature_maps = []
                for i in range(1, 6):
                    style_feature_maps.append(outputs[f"conv{i}"][0])

                # input content and style losses
                total_content_loss = content_loss(content_feature_maps)

                total_style_loss = 0
                for feature_map, style_loss in zip(style_feature_maps, style_losses):
                    total_style_loss += (style_weight * style_loss(feature_map))

                # total loss
                loss = (alpha * total_content_loss) + (beta * total_style_loss)
                loss.backward()

                iterations.set_postfix({
                    "content loss": total_content_loss.item(),
                    "style loss": total_style_loss.item(),
                    "total loss": loss.item()
                })

                return loss

            optimizer.step(closure)

    # final correction
    x.data.clamp_(0, 1)
    return x


if __name__ == "__main__":
    # choose input image
    input_image = "content"
    main(input_image)



        

        

        
        


    

    
