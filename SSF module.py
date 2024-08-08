from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
!pip install dotmap
import math
import tensorflow as tf
import time
import matplotlib
import math
import time
import random
import numpy as np
import pandas as pd
from dotmap import DotMap
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
!pip install dotmap
import math
import tensorflow as tf
import time
import matplotlib
import math
import time
import random
import numpy as np
import pandas as pd
from dotmap import DotMap
from PIL import Image
from keras.models import Model, Sequential, load_model
from keras.layers import Conv2D,  Dropout, Flatten, Dense, Input, Reshape
from keras.layers import Activation, Conv2DTranspose, UpSampling2D, BatchNormalization, Embedding, multiply
from keras.layers import LeakyReLU
from keras.datasets import mnist, fashion_mnist
from keras.optimizers.legacy import Adam, RMSprop
from keras.utils import to_categorical
from PIL import Image
from tqdm import tqdm
from scipy.io import loadmat
from __future__ import print_function
import pandas as pd
from scipy.io import loadmat
import numpy as np
import math
np.set_printoptions(suppress=True)
import os
import time
import random
import matplotlib.pyplot as plt
from importlib import reload
import numpy as np
import os
import matplotlib.pyplot as plt
from importlib import reload
from matplotlib import pyplot as plt
from sklearn.preprocessing import scale
from scipy.interpolate import griddata
from dotmap import DotMap
import matplotlib.pyplot as plt
from importlib import reload
from multiprocessing import Process
#from model import main
%matplotlib inline
def cart2sph(x, y, z):
    """
    Transform Cartesian coordinates to spherical
    :param x: X coordinate
    :param y: Y coordinate
    :param z: Z coordinate
    :return: radius, elevation, azimuth
    """
    x2_y2 = x**2 + y**2
    r = math.sqrt(x2_y2 + z**2)                    # r
    elev = math.atan2(z, math.sqrt(x2_y2))            # Elevation
    az = math.atan2(y, x)                          # Azimuth
    return r, elev, az

def pol2cart(theta, rho):
    """
    Transform polar coordinates to Cartesian
    :param theta: angle value
    :param rho: radius value
    :return: X, Y
    """
    return rho * math.cos(theta), rho * math.sin(theta)

def makePath(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

def monitor(process, multiple, second):
    while True:
        sum = 0
        for ps in process:
            if ps.is_alive():
                sum += 1
        if sum < multiple:
            break
        else:
            time.sleep(second)
def get_logger(name, log_path):
    import logging
    reload(logging)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logfile = makePath(log_path) + "/Train_" + name + ".log"
    print ("logggggger data", name)
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    if log_path == "./result/test":
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger
def azim_proj(pos):
    """
    Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
    Imagine a plane being placed against (tangent to) a globe. If
    a light source inside the globe projects the graticule onto
    the plane the result would be a planar, or azimuthal, map
    projection.

    :param pos: position in 3D Cartesian coordinates
    :return: projected coordinates using Azimuthal Equidistant Projection
    """
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
    return pol2cart(az, math.pi / 2 - elev)

def gen_images(data, args):
    locs = loadmat('/content/drive/MyDrive/Colab Notebooks/locs_orig.mat')
    locs_3d = locs['data']
    locs_2d = []
    for e in locs_3d:
        locs_2d.append(azim_proj(e))

    locs_2d_final = np.array(locs_2d)
    grid_x, grid_y = np.mgrid[
                     min(np.array(locs_2d)[:, 0]):max(np.array(locs_2d)[:, 0]):args.image_size * 1j,
                     min(np.array(locs_2d)[:, 1]):max(np.array(locs_2d)[:, 1]):args.image_size * 1j]

    images = []
    for i in range(1):
        images.append(griddata(locs_2d_final, data[i, :], (grid_x, grid_y), method='cubic', fill_value=np.nan))
    images = np.stack(images, axis=0)
    print (images.shape)
    time.sleep(2.4)
    images[~np.isnan(images)] = scale(images[~np.isnan(images)])
    images = np.nan_to_num(images)
    imgplot = plt.imshow(images.T)
    plt.show()
    return images
def read_prepared_data(args):
    data = []

    for j in range(len(args.ConType)):
        for k in range(args.trail_number):
            filename = args.data_document_path + "/" + args.ConType[j] + "/" + args.name + "Tra" + str(k + 1) + ".csv"
            print ("name datammmmmmm", filename)
            print ("jjjjjjjjjjjjjjjjjjjjjjjj", j)
            print ("len arg", len(args.ConType))
            data_pf = pd.read_csv(filename, header=None)
            eeg_data = data_pf.iloc[:, 2 * args.audio_channel:]

            data.append(eeg_data)

    data = pd.concat(data, axis=0, ignore_index=True)
    print ("preprossss data", data.shape)
    return data
# output shape: [(time, feature) (window, feature) (window, feature)]
def window_split(data, args):
    random.seed(args.random_seed)
    # init
    test_percent = args.test_percent
    window_lap = args.window_length * (1 - args.overlap)
    overlap_distance = max(0, math.floor(1 / (1 - args.overlap)) - 1)
    print("overlap distanceeeeeeeeeeeeeeeeeeeeeeeeeeeeee",overlap_distance)
    train_set = []
    test_set = []

    for l in range(len(args.ConType)):
        label = pd.read_csv(args.data_document_path + "/csv/" + args.name + args.ConType[l] + ".csv")
        # split trial
        for k in range(args.trail_number):
            # the number of windows in a trial
            window_number = math.floor(
                (args.cell_number - args.window_length) / window_lap) + 1
            print("window_numberrrrrrrrrrrr",window_number)

            test_window_length = math.floor(
                (args.cell_number * test_percent - args.window_length) / window_lap)
            test_window_length = test_window_length if test_percent == 0 else max(
                0, test_window_length)
            test_window_length = test_window_length + 1

            test_window_left = random.randint(0, window_number - test_window_length)
            test_window_right = test_window_left + test_window_length - 1
            target = label.iloc[k, args.label_col]

            # split window
            for i in range(window_number):
                left = math.floor(k * args.cell_number + i * window_lap)
                right = math.floor(left + args.window_length)
                # train set or test set
                if test_window_left > test_window_right or test_window_left - i > overlap_distance or i - test_window_right > overlap_distance:
                    train_set.append(np.array([left, right, target, len(train_set), k, args.subject_number]))
                elif test_window_left <= i <= test_window_right:
                    test_set.append(np.array([left, right, target, len(test_set), k, args.subject_number]))

    # concat
    train_set = np.stack(train_set, axis=0)
    test_set = np.stack(test_set, axis=0) if len(test_set) > 1 else None

    return np.array(data), train_set, test_set

def to_alpha(data, window, args):
    alpha_data = []
    for window_index in range(window.shape[0]):
        start = window[window_index][args.window_metadata.start]
        end = window[window_index][args.window_metadata.end]
        window_data = np.fft.fft(data[start:end, :], n=args.window_length, axis=0)
        window_data = np.abs(window_data) / args.window_length
        window_data = np.sum(np.power(window_data[args.point_low:args.point_high, :], 2), axis=0)
        alpha_data.append(window_data)
    alpha_data = np.stack(alpha_data, axis=0)
    return alpha_data