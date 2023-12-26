from discriminator import Discriminator
from generator import Generator, print_train_time
import torch
import config
from timeit import default_timer as timer
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

x = timer()
disc = Discriminator().cpu()
gen = Generator().cpu()
# t = torch.randn(size=(1, 3, 256, 256))
sketch_path = "./5881.jpg"
image = Image.open(sketch_path)
resize_transform = transforms.Resize((256, 256))
image = resize_transform(image)
to_tensor_transform = transforms.ToTensor()
t = to_tensor_transform(image)
t = t.unsqueeze(0)
gen.eval()
with torch.no_grad():
    result = gen(t.cpu())
save_image(result[0], "./prod/house.jpg")
disc.eval()
tens = torch.randn(size=(1, 3, 510, 510)).cpu()
with torch.no_grad():
    z = disc(result.detach(), tens.detach())
print(z)
print(z.shape)
disc_loss_fn = torch.nn.BCEWithLogitsLoss()
d_real_loss = disc_loss_fn(z, torch.ones_like(z))
d_fake_loss = disc_loss_fn(z, torch.zeros_like(z))
y = timer()

d_loss = (d_real_loss + d_fake_loss) * 0.5
print(f"real: {d_real_loss}")
print(f"fake: {d_fake_loss}")
print(f"loss: {d_loss}")

z = z[0]
save_image(z, "./prod/z.png")
print_train_time(x, y, config.DEVICE)
