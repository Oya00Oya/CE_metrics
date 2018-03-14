import torch
import numpy as np
import fid_score as fid
from inception import InceptionV3

########
# PATHS
########
data_path = '/home/orashi/datasets/Colorize_C/color'  # set path to training set images
output_path = 'fid_stats_color_192.npz'  # path for where to store the statistics
batch_size = 64
# loads all images into memory (this might require a lot of RAM!)
print("load images..", end=" ", flush=True)
data = fid.CreateDataLoader(data_path, batch_size, 2333)
print("%d images found and loaded" % len(data))

print("create inception graph..", end=" ", flush=True)
model = torch.nn.DataParallel(InceptionV3([1]))
model.cuda()
print("ok")

print("calculte FID stats..", end=" ", flush=True)

mu, sigma = fid.calculate_activation_statistics(data, model, batch_size,
                                                dims=192, cuda=True, verbose=True)
np.savez_compressed(output_path, mu=mu, sigma=sigma)
print("finished")
