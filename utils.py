import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import save_image
import config
# from skimage.color import lab2rbg


def connect_x_y(x, y):
    connected = np.zeros(256 * 256 * 3)
    x, y = x.cpu(), y.cpu()
    connected.resize([256, 256, 3])
    connected[:, :, 0] = x
    connected[:, :, 1] = y[0, :, :]
    connected[:, :, 2] = y[1, :, :]
    connected = lab2rbg(connected)
    return transforms.ToTensor()(connected)


def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = connect_x_y(x, y_fake[0, :, :, :])
        save_image(y_fake, folder + f"/generated_20122023_{epoch}.png")
        if epoch == 0:
            real = connect_x_y(x, y_fake[0, :, :, :])
            save_image(real, folder + f"/input_20122023_0.png")

    gen.train()  # return the model to a train mode


def load_checkpoint(checkpoint_file, model, optimizer, learning_rate):
    print("=>loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for p_group in optimizer.param_groups:
        p_group['lr'] = learning_rate



