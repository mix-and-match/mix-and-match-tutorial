from model import MixMatchVAE
import torch
from scipy.interpolate import interp1d
import numpy as np
import torchvision


def sample():
    n_sample = 20
    x, y = next(iter(testloader))
    result = torch.zeros(x.shape[0] * n_sample, 1, 28, 28).to(device)
    C = torch.zeros(x.shape[0] * n_sample, 1, 28, 28).to(device)
    z_val = [z for z in np.arange(-2, 2, 0.1)]
    with torch.no_grad():
        cnt = 0
        for i in range(x.shape[0]):
            reshaped_image = x[i:i+1].view(1, -1)
            data, condition = reshaped_image[:, :500], \
                              reshaped_image[:, 500:]
            data, condition = data.to(device), condition.to(device)
            result[cnt] = x[i]
            cnt += 1
            C_image = torch.zeros(reshaped_image.shape).to(device)
            C_image[:, 500:] = condition
            C_image = C_image.view(x[i:i + 1].shape)
            result[cnt] = C_image
            cnt += 1
            for j in range(2, n_sample):
                z = torch.ones(1, 32).to(device)
                z *= z_val[j]
                reconstructed = model.sample(condition, 64)
                rec_image = torch.zeros(reshaped_image.shape).to(device)

                rec_image[:, :500] = reconstructed
                rec_image[:, 500:] = condition

                rec_image = rec_image.view(x[i:i+1].shape)
                result[cnt] = rec_image
                cnt += 1
    torchvision.utils.save_image(result, nrow=20, fp="results/sampled.png")


if __name__ == "__main__":
    device = "cuda"

    class MixMatchArgs(object):
        def __init__(self):
            self.input_dim = 500  # (28 * 28) // 2
            self.output_dim = 500  # (28 * 28) // 2
            self.condition_dim = (28 * 28) - 500
            self.hidden_dim = 128
            self.latent_dim = 32
            self.device = device

    model = MixMatchVAE(MixMatchArgs()).to(device)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=20,
                                             shuffle=False, num_workers=0)

    model.load_state_dict(torch.load("checkpoint/mix_match_48.pth"))
    model.eval()
    sample()