import torch 
from torch import nn 
import torch.nn.functional as F 

class ContentLoss(nn.Module):
    """
    Content Loss for the neural style transfer algorithm.
    """
    def __init__(self, target: torch.Tensor, device: torch.device) -> None:
        super(ContentLoss, self).__init__()
        batch_size, channels, height, width = target.size()
        target = target.view(batch_size * channels, height * width)
        self.target = target.detach().to(device)

    def __str__(self) -> str:
        return "Content loss"

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = input.size()
        input = input.view(batch_size * channels, height * width)
        return F.mse_loss(input, self.target)


class StyleLoss(nn.Module):
    """
    Style loss for the neural style transfer algorithm.
    """
    def __init__(self, target: torch.Tensor, device: torch.device) -> None:
        super(StyleLoss, self).__init__()
        self.target = self.compute_gram_matrix(target).detach().to(device)

    def __str__(self) -> str:
        return "Style loss"

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = self.compute_gram_matrix(input)
        return F.mse_loss(input, self.target)

    def compute_gram_matrix(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = input.size()
        input = input.view(batch_size * channels, height * width)
        return torch.matmul(input, input.T).div(batch_size * channels * height * width)