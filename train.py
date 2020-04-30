from model import MixMatchVAE
import torch
import numpy as np
from torch import optim
import torchvision


def kl_anneal_function(anneal_function, step, k, x0):
    if anneal_function == 'logistic':
        return float(1/(1+np.exp(-k*(step-x0))))
    elif anneal_function == 'linear':
        return min(1, step/x0)


def train(model, trainloader, alpha):
    reconstruction_loss = torch.nn.BCELoss(reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    MAX_EPOCH = 50
    step = 0

    for epoch in range(1, MAX_EPOCH + 1):
        for batch_idx, (image, _) in enumerate(trainloader):
            reshaped_image = image.view(image.shape[0], -1)

            data, condition = reshaped_image[:, :500], reshaped_image[:, 500:]

            data, condition = data.to(device), condition.to(device)

            optimizer.zero_grad()
            reconstructed, mu, sigma = model(data, condition, alpha)
            # reconstruction loss
            rec_loss = reconstruction_loss(reconstructed, data)
            # KL divergence loss
            KL_loss = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
            # KL annealing weight
            KL_weight = kl_anneal_function('logistic', step, 0.002, 2500)
            # computing the total loss for the current mini batch
            loss = rec_loss + KL_weight * KL_loss

            loss.backward()
            optimizer.step()

            step += 1

        print(
            epoch,
            '\tRec Loss:', round(rec_loss.item(), 3),
            '\tKL Loss:', round(KL_loss.item(), 3),
            '\tAnneal:', round(KL_weight, 5)
        )

        # visualize results
        rec_image = torch.zeros(reshaped_image.shape).to(device)
        rec_image[:, :500] = reconstructed
        rec_image[:, 500:] = condition

        rec_image = rec_image.view(image.shape)
        torchvision.utils.save_image(rec_image[0:100], nrow=10, fp="results/rec_" + str(epoch).zfill(2) + ".png")
        torch.save(model.state_dict(), "checkpoint/mix_match_" + str(epoch).zfill(2) + ".pth")


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

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                             shuffle=False, num_workers=0)

    train(model, trainloader, 64)
