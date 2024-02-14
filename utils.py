import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import save_image
# import config
import PIL.Image as Image
import cv2
# import config


# from skimage.color import lab2rbg


# def connect_x_y(x, y):
#     connected = np.zeros(256 * 256 * 3)
#     x, y = x.config.DEVICE(), y.config.DEVICE()
#     connected.resize([256, 256, 3])
#     connected[:, :, 0] = x
#     connected[:, :, 1] = y[0, :, :]
#     connected[:, :, 2] = y[1, :, :]
#     connected = lab2rbg(connected)
#     return transforms.ToTensor()(connected)

def connect_x_y(x, y):
    print("=> Saving examples")
    # print(x.shape)
    # print(y.shape)
    # x = x[]
  # Convert x and y to numpy arrays
    x_array = x.cpu().numpy()
    y_array = y.cpu().numpy()
    
    # Resize the arrays to have the same dimensions (256x256 in this case)
    x_resized = np.transpose(x_array, (1, 2, 0))
    y_resized = np.transpose(y_array, (1, 2, 0))
    x_resized = cv2.resize(x_resized, (256, 256))
    y_resized = cv2.resize(y_resized, (256, 256))
    
    # Create a new tensor of zeros with the desired shape
    connected = torch.zeros((3, 256, 512))
    
    # Copy x into the left half of the connected tensor
    connected[:, :, :256] = torch.from_numpy(np.transpose(x_resized, (2, 0, 1)))
    
    # Copy y into the right half of the connected tensor
    connected[:, :, 256:] = torch.from_numpy(np.transpose(y_resized, (2, 0, 1)))
    
    return connected

def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = connect_x_y(x, y_fake[0])
        save_image(y_fake, folder + f"/generated_{config.CURRENT_DATE}_epoch_{epoch}_{config.IMG_SAVE_ID}.png")
        #if epoch == 0:
        #    real = connect_x_y(x, y_fake[0])
        #   save_image(real, folder + f"/input_{config.CURRENT_DATE}_0.png")

    gen.train()  # return the model to a train mode


def load_checkpoint(checkpoint_file, model, optimizer, learning_rate):
    print("=>loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for p_group in optimizer.param_groups:
        p_group['lr'] = learning_rate


def save_checkpoint(model, optimizer, filename):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, filename)


def main():
  t1 = torch.randn(size = (3,256,256))
  t2 = torch.randn(size = (3,256,256))
  save_image(connect_x_y(t1,t2), "./prod/123.png")

if __name__ == "__main__":
  main()