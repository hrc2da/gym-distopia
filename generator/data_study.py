from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle as pk
from distopia.app.agent import VoronoiAgent


def load_data(path):
    data = []
    with open(path) as infile:
        data = np.array(json.load(infile))
    return data


def tsne(data):
    print("fitting tsne")
    X_embedded = TSNE(n_components=2).fit_transform(data)
    print("plotting tsne")
    x_coord = X_embedded[:, 0]
    y_coord = X_embedded[:, 1]
    plt.scatter(x_coord, y_coord)
    plt.show()


def construct_layout(block_locs):
    obs_dict = {}
    # added = {} taken out to allow for double blocks in gan--note that voronoi should fail so it should be fine
    for d in range(0, 8):
        obs_dict[d] = []
        for b in range(0, 2):
            index = 2 * (d * 2 + b)
            coords = [
                block_locs[index],
                block_locs[index + 1],
            ]  # already in pixel space
            if block_locs[index] > 100:  # if the x is far enough to the right
                obs_dict[d].append(coords)
            # assert self.hash_loc(coords) not in added # just double check to ensure we aren't passing two blocks in same loc
            # added[self.hash_loc(coords)] = (d,b)
    return obs_dict


def plot_coords(data, sample_num):
    n_samples, _ = data.shape
    data = data.reshape(n_samples, 16, 2)
    # data = data[600:1600, :, :]
    x_coord = data[sample_num, :, 0]
    y_coord = data[sample_num, :, 1]
    plt.scatter(x_coord, y_coord, s=3, alpha=0.5)
    plt.show()


sample_num = 700
data = load_data("trimmed.json")
layout = data[sample_num]
voronoi = VoronoiAgent()
voronoi.load_data()
layout_dict = construct_layout(layout)
districts = voronoi.get_voronoi_districts(layout_dict)
boundaries = []
for district in districts:
    boundaries.append(district.boundary)

with open('test_boundary.pickle', 'wb') as handle:
    pk.dump(boundaries, handle, protocol=pk.HIGHEST_PROTOCOL)
    

# plot_coords(data, sample_num)
