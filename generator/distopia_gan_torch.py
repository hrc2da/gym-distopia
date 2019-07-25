import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import save_image
from torch.autograd import Variable
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
from distopia.app.agent import VoronoiAgent

NUM_DISTRICTS = 8
BLOCKS_PER_DISTRICT = 2
STATE_SHAPE = (NUM_DISTRICTS, BLOCKS_PER_DISTRICT, 2)
LATENT_DIM = NUM_DISTRICTS * BLOCKS_PER_DISTRICT * 2
adversarial_loss = nn.BCELoss()


def load_data(path, batch_size=32):
    data = []
    with open(path) as infile:
        data = np.array(json.load(infile))
    num_samples, _ = data.shape
    data = data.reshape(num_samples, 16, 2)
    data[:, :, 0] /= 1920
    data[:, :, 1] /= 1080
    data = data.reshape(num_samples, 32)
    data = torch.from_numpy(data)
    data = data.type(torch.FloatTensor)
    dataset = TensorDataset(data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


def construct_layout(block_locs):
    obs_dict = {}
    # added = {} taken out to allow for double blocks in gan--note that voronoi should fail so it should be fine
    for d in range(0, NUM_DISTRICTS):
        obs_dict[d] = []
        for b in range(0, BLOCKS_PER_DISTRICT):
            index = 2 * (d * BLOCKS_PER_DISTRICT + b)
            coords = [
                block_locs[index],
                block_locs[index + 1],
            ]  # already in pixel space
            if block_locs[index] > 100:  # if the x is far enough to the right
                obs_dict[d].append(coords)
            # assert self.hash_loc(coords) not in added # just double check to ensure we aren't passing two blocks in same loc
            # added[self.hash_loc(coords)] = (d,b)
    return obs_dict


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(LATENT_DIM, 64)
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.fc2 = nn.Linear(64, LATENT_DIM)

    def forward(self, z):
        x = F.relu(self.bn1(self.fc1(z)))
        x = F.relu(self.fc2(x))
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(np.prod(np.array(STATE_SHAPE)), 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x


class FlatGAN:
    def __init__(self):
        self.G = Generator()
        self.D = Discriminator()
        self.g_optim = optim.Adam(self.G.parameters())
        self.d_optim = optim.Adam(self.D.parameters())
        self.data_loader = load_data("data/trimmed.json", batch_size=16)
        self.voronoi = VoronoiAgent()
        self.voronoi.load_data()

    def check_validity(self, config):
        config = config.reshape(16, 2)
        config[:, 0] *= 1920
        config[:, 1] *= 1080
        config = config.reshape(32)
        layout = construct_layout(config)
        districts = self.voronoi.get_voronoi_districts(layout)

        if len(districts) < NUM_DISTRICTS:
            return 0
        try:
            state_metrics, district_metrics = self.voronoi.compute_voronoi_metrics(
                districts
            )
        except Exception as e:
            print("Couldn't compute Voronoi for {}:{}".format(districts, e))
            return 0

        return 1

    def train(self, n_epochs):
        valid_states = []
        for epoch_n in range(n_epochs):
            self.G.train()
            self.D.train()
            for batch_i, data in enumerate(self.data_loader):
                data = data[0]  # some weird dereference bug

                # ground_truth variables
                real = Variable(Tensor(data.size(0), 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(data.size(0), 1).fill_(0.0), requires_grad=False)

                # train generator
                self.g_optim.zero_grad()
                z = Variable(
                    Tensor(np.random.normal(0, 1, (data.shape[0], LATENT_DIM)))
                )

                generated_states = self.G(z)
                generator_loss = adversarial_loss(self.D(generated_states), real)
                generator_loss.backward()
                self.g_optim.step()

                # train discriminator
                self.d_optim.zero_grad()

                real_loss = adversarial_loss(self.D(data), real)
                fake_loss = adversarial_loss(self.D(generated_states.detach()), fake)

                discriminator_loss = (real_loss + fake_loss) / 2
                discriminator_loss.backward()

                self.d_optim.step()

                if batch_i % 100 == 0:
                    print(
                        "Train Epoch: {} [{}/{} ({:.0f}%)]\tG_Loss: {:.6f} \tD_Loss: {:.6f}".format(
                            epoch_n,
                            batch_i * len(data),
                            len(self.data_loader.dataset),
                            100.0 * batch_i / len(self.data_loader),
                            generator_loss.item() / len(data),
                            discriminator_loss.item() / len(data),
                        )
                    )

            # test the generator on the oracle
            with torch.no_grad():
                sample_z = torch.randn(100, LATENT_DIM)
                generated_states = self.G(sample_z)
                results = np.zeros(100)
                for i, state in enumerate(generated_states):
                    results[i] = self.check_validity(state.data.numpy())
                    if results[i]:
                        valid_states.append(state.data.numpy())
                print(
                    "{} of 100 random samples generated valid states in epoch {}".format(
                        np.sum(results), epoch_n + 1
                    )
                )
        torch.save(self.G, "models/torch_generator.pickle")
        torch.save(self.D, "models/torch_discriminator.pickle")
        with open("generated_valid_states.pickle", "wb") as outfile:
            pickle.dump(valid_states, outfile)


gan = FlatGAN()
gan.train(25)

