from discriminator import Discriminator
from generator import Generator, print_train_time
import torch
import config
from timeit import default_timer as timer

x = timer()
disc = Discriminator().cpu()
gen = Generator().cpu()
t = torch.randn(size=(1, 3, 256, 256))
gen.eval()
with torch.no_grad():
  result = gen(t.cpu())
disc.eval()
with torch.no_grad():
  z= disc(result.detach(),result.detach())
print(z)
print(z.shape)
disc_loss_fn = torch.nn.BCEWithLogitsLoss()
d_real_loss = disc_loss_fn(z,torch.ones_like(z))
d_fake_loss = disc_loss_fn(z, torch.zores(z))
y = timer()
if not d_real_loss:
  d_loss = (d_real_loss + d_fake_loss) /2
else:
  d_loss = 0
print(d_loss)
print_train_time(x,y)