import torch
import torchvision
from timeit import default_timer as timer
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from generator import Generator
from discriminator import Discriminator
import config


def print_train_time(start: float, end: float, device: torch.device):
    """
    Print difference between start and end time
    :param start:
    :param end:
    :param device:
    :return: float: time between start and end in seconds (higher is longer)
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


def train(gen: torch.nn.Module, disc: torch.nn.Module, data_loader: DataLoader,
          loss_fn_gen: torch.nn.Module, loss_fn_disc: torch.nn.Module,
          gen_optimizer: torch.optim.Optimizer, disc_optimizer: torch.optim.Optimizer,
          gen_scalar: torch.cuda.amp.grad_scaler, disc_scalar: torch.cuda.amp.grad_scaler):
    loop = tqdm(data_loader, leave=True)
    for index, (x, y) in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        with torch.autocast(device_type=config.DEVICE):
            y_fake = gen(x)
            disc_real = disc(x, y)
            disc_fake = disc(x, y_fake)
            disc_real_loss = loss_fn_disc(disc_real, torch.ones_like(disc_real))
            disc_fake_loss = loss_fn_disc(disc_fake, torch.zeros_like(disc_fake))
            disc_loss = (disc_real_loss + disc_fake_loss) / 2

        disc.zero_grad()
        disc_scalar.scale(disc_loss).backward()
        disc_scalar.step(disc_optimizer)
        disc_scalar.update()

        with torch.autocast(device_type=config.DEVICE):
            disc_fake = disc(x, y_fake)
            gen_fake_loss = loss_fn_disc(disc_fake, torch.ones_like(disc_fake))
            L1 = loss_fn_gen(y, y_fake) * config.L1_LAMBDA
            gen_loss = gen_fake_loss + L1

        gen.zero_grad()
        gen.scale(gen_loss).backward()
        gen_scalar.step(gen_optimizer)
        gen_scalar.update()


def main():
    gen = Generator(in_channels=3).to(config.DEVICE)
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen_optimizer = torch.optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    disc_optimizer = torch.optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    gen_loss_fn = torch.nn.L1Loss().to(config.DEVICE)
    disc_loss_fn = torch.nn.BCEWithLogitsLoss().to(config.DEVICE)
    gen_scalar = torch.cuda.amp.GradScaler()
    disc_scalar = torch.cuda.amp.GradScaler()
    dataSet=[]
    data = DataLoader(dataSet, batch_size=config.BATCH_SIZE, shuffle=True)
    train(gen, disc, data, gen_loss_fn, disc_loss_fn, gen_optimizer, disc_optimizer, gen_scalar,
         disc_scalar)


if __name__ == '__main__':
    main()
