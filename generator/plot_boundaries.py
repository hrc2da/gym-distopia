from matplotlib import pyplot as plt

from matplotlib.path import Path
import matplotlib.patches as patches

import pickle as pkl
import numpy as np
import json
import time

import sys
from distopia.app.agent import VoronoiAgent


def plot_boundary(boundary_paths, raw_data, index):

    raw_data = raw_data.reshape(16, 2)
    for i, boundary_path in enumerate(boundary_paths):
        assert len(boundary_path) % 2 == 0
        n_verts = len(boundary_path)//2
        verts = np.reshape(np.array(boundary_path), (n_verts, 2))
        # verts[:, 0] += 100
        verts = np.vstack([verts, raw_data[i*2:i*2 + 2, :]])
        print(i, raw_data[i*2:i*2 + 2, :])
        plt.scatter(verts[:, 0], verts[:, 1])

    plt.text(1500, 900, "index: {}".format(index))
    plt.ylim(0, 1080)
    plt.xlim(-100, 1920)
    plt.show(block=False)
    plt.pause(0.05)
    plt.clf()
    plt.savefig("results/img/"+str(time.time())+".png")


def construct_layout(block_locs):
    obs_dict = {}
    # added = {} taken out to allow for double blocks in gan--note that voronoi should fail so it should be fine
    for d in range(0, 8):
        obs_dict[d] = []
        for b in range(0, 2):
            index = 2*(d*2 + b)
            coords = [block_locs[index], block_locs[index+1]]  # already in pixel space
            if block_locs[index] > 100:  # if the x is far enough to the right
                obs_dict[d].append(coords)
            # assert self.hash_loc(coords) not in added # just double check to ensure we aren't passing two blocks in same loc
            #added[self.hash_loc(coords)] = (d,b)
    return obs_dict


voronoi = VoronoiAgent()
voronoi.load_data()


def test_pt(points, index):
    point = points
    layout = construct_layout(points)
    districts = voronoi.get_voronoi_districts(layout)
    boundaries = []
    for district in districts:
        if district.precincts == []:
            return False
        boundaries.append(district.boundary)
    plot_boundary(boundaries, points, index)
    return True


# with open('test_boundary.pickle', 'rb') as infile:
#     boundary = pkl.load(infile)


def load_data(path):
    data = []
    with open(path, "r") as infile:
        data = np.array(json.load(infile))
    return data


def load_pkl(path):
    data = []
    with open(path, "rb") as infile:
        data = np.array(pkl.load(infile))
    return data


if __name__ == "__main__":
    # pt = int(sys.argv[1])
    raw_data = load_data("trimmed3.json")[500:]
    # raw_data = load_pkl("results/generated_valid_states.pickle")
    count = 0
    for pt in range(len(raw_data)):
        if pt < 0:
            raw_data[pt, 0] += 100
            if test_pt(raw_data[pt], pt):
                count += 1
        else:
            if test_pt(raw_data[pt], pt):
                count += 1
    print("{} 8 district assignments".format(count))
