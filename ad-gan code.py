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

class ACGAN:

  def __init__(self, rows=28, cols=28, channels=1):
    self.rows = rows
    self.cols = cols
    self.channels = channels
    self.shape = (self.rows, self.cols, self.channels)
    self.latent_size = 100
    self.sample_rows = 1
    self.sample_cols = 2
    self.sample_path = 'images'
    self.num_classes = 2

    #optimizer = RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8)
    optimizer = Adam(0.0002, 0.5)

    image_shape = self.shape
    seed_size = self.latent_size

    #Get the discriminator and generator Models
    print("Build Discriminator")
    self.discriminator = self.build_discriminator()

    self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    print("Build Generator")

    self.generator = self.build_generator()

    random_input = Input(shape=(seed_size,))
    label = Input(shape=(1,))

    #Pass noise/random_input and label as input to the generator
    generated_image = self.generator([random_input,label])

    #Put discriminator.trainable to False. We do not want to train the discriminator at this point in time
    self.discriminator.trainable =False

    #Pass generated image and label as input to the discriminator
    validity, label_out = self.discriminator(generated_image)
    print('validity',validity)
    print('label_discr',label_out)
    #Pass radom input and label as input to the combined model
    self.combined_model = Model([random_input,label], [validity,label_out])
    self.combined_model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])

  def build_discriminator(self):

    input_shape = self.shape
    model = Sequential()
    model.add(Conv2D(64, (3,3), strides=2, padding='same', input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(128,(3,3), strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3,3), strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(1,activation='sigmoid'))

    output = model
    inp = Input(shape=input_shape)

    model.summary()

    input_image = Input(shape=input_shape)

    #Extrating features from the model
    features = model(input_image)

    # AC GAN has 2 outputs, 1 for real or Fake using sigmoid activation. Another for class prediction using softmax.

    validity = Dense(1, activation='sigmoid', name='Dense_validity')(features)
    print('validity',validity)
    aux = Dense(self.num_classes, activation='softmax', name ='Dense_Aux')(features)
    print('aux',aux)
    return Model(input_image,[validity,aux])

  def build_generator(self):

    seed_size = self.latent_size
    model = Sequential()
    model.add(Dense(7*7*256, input_dim=seed_size))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))

    model.add(Reshape((7,7,256)))
    model.add(Dropout(0.4))

    model.add(Conv2DTranspose(128,(5,5),padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))
    model.add(UpSampling2D())

    model.add(Conv2DTranspose(64,(3,3),padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))
    model.add(UpSampling2D())

    model.add(Conv2DTranspose(32,(3,3),padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(1,(3,3),padding='same'))
    model.add(Activation('sigmoid'))

    noise = Input(shape=(seed_size,))
    label = Input(shape = (1,), dtype='int32')

    label_embeddings = Flatten()(Embedding(self.num_classes,self.latent_size)(label))

    model_input = multiply([noise,label_embeddings])

    generated_image = model(model_input)

    model.summary()

    return(Model([noise,label],generated_image))

  def plot_sample_images(self, epoch, noise):
    r, c = self.sample_rows, self.sample_cols
    #noise = np.random.normal(0, 1, (r * c, self.latent_size))

    sampled_labels = np.arange(0, self.num_classes).reshape(-1, 1)

    gen_imgs = self.generator.predict([noise,sampled_labels])
    #print ('genrated image',gen_imgs)
    filename = os.path.join(self.sample_path,'%d.png'% epoch)
    #fig, axs = plt.subplots(1, 1)
    cnt = 0
    #for i in range(r):
      #for j in range(c):
        #axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
        #axs[i,j].axis('off')
        #cnt += 1

    #imgplot = plt.imshow(gen_imgs[cnt, :,:,0])
    #plt.show()
    #matplotlib.pyplot.imshow(gen_imgs[cnt, :,:,0])
    #plt.savefig('geerated.png')
    #fig.savefig(filename)
    #plt.close()

  def plot_loss(self,losses):
    """
    @losses.keys():
      0: loss
      1: accuracy
    """
    d_loss = [v[0] for v in losses["D"]]
    g_loss = [v[0] for v in losses["G"]]

    plt.figure(figsize=(10,8))
    plt.plot(d_loss, label="Discriminator loss")
    plt.plot(g_loss, label="Generator loss")

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

  def train(self, save_freq=2, batch_size=32,epochs=1, name="S3", data_document_path="/content/drive/MyDrive/Colab Notebooks"):
    args = DotMap()
    args.name = name
    args.subject_number = 16
    #print ("gggggggggggggggggg", int(args.name[1:]))
    args.data_document_path = data_document_path
    args.ConType = ["No"]
    args.fs = 128
    args.window_length = math.ceil(args.fs * 1)
    print ("window lengthtttttttttt", math.ceil(args.fs * 1) )
    args.overlap = 0.8
    args.batch_size = 32
    args.max_epoch = 4
    args.random_seed = time.time()
    args.image_size = 28
    args.people_number = 16
    args.eeg_channel = 64
    args.audio_channel = 1
    args.channel_number = args.eeg_channel + args.audio_channel * 2
    args.trail_number = 8
    args.cell_number = 46080
    args.test_percent = 0.1
    args.vali_percent = 0.1
    args.label_col = 0
    args.alpha_low = 8
    args.alpha_high = 13
    args.log_path = "./result"
    args.frequency_resolution = args.fs / args.window_length
    args.point_low = math.ceil(args.alpha_low / args.frequency_resolution)
    args.point_high = math.ceil(args.alpha_high / args.frequency_resolution) + 1
    args.window_metadata = DotMap(start=0, end=1, target=2, index=3, trail_number=4, subject_number=5)
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
    logger = get_logger(args.name, args.log_path)
    # load data 和 label
    data = read_prepared_data(args)
    seed_size = self.latent_size
    data, train_window, test_window = window_split(data, args)
    train_label = train_window[:, args.window_metadata.target]
    test_label = test_window[:, args.window_metadata.target]
    print("data.shape",data.shape)
    print("train_label.shape",train_label.shape)
    print("test_label.shape",test_label.shape)
    # fft
    train_data = to_alpha(data, train_window, args)
    test_data = to_alpha(data, test_window, args)
    print("train_data.shape",train_data.shape)
    print("test_data.shape",test_data.shape)
    del data

    train_data = gen_images(train_data, args)
    test_data = gen_images(test_data, args)

    train_data = np.expand_dims(train_data, axis=-1)
    test_data = np.expand_dims(test_data, axis=-1)
    train_label = train_label.reshape(-1,1)
    test_label = test_label.reshape(-1,1)
    #Load Dataset
    #(x_train,y_train),(_,_) = mnist.load_data()
    #(x_train,y_train),(_,_) = fashion_mnist.load_data()
    print("train_data.shape: {}".format(train_data.shape))
    #normalize and reset train set in range (0,1) # normalizing to (-1,1) seems to be not working.

    #x_train = (x_train.astype('float32') / 127.5 ) - 1. # Normalizing this way doesn't work during training.

    #x_train = x_train.astype('float32')/255.0 #Normalizing  this way does work during training.

    print("train_data.shape",train_data.shape)

    #Ground Truth. Setting real images labels to True
    y_real = np.ones((batch_size,1))

    #Setting fake images labels to False
    y_fake = np.zeros((batch_size,1))
    #fixed_seed = np.random.normal(0,1,size=[25,seed_size])

    cnt = 1

    #Generating Fixed noise to be passed for sampling with same inputs after set of epochs and seeing the results
    noise_input = np.random.normal(0,1,size=[self.sample_rows*self.sample_cols,seed_size])

    #Setup loss vector to store losses for Generator and Discriminator

    losses = {"D":[], "G":[]}

    path = self.sample_path
    if not os.path.isdir(path):
      os.mkdir(path)

    for epoch in range(epochs):

      #Training of Discriminator. Taking random samples of batch_size #
      noise = np.random.normal(0,1,size=[batch_size,seed_size])

      #take random batched of indexes for x_train
      idx = np.random.randint(0,train_data.shape[0],size=batch_size)

      #print('hhhhhhhhhhh',idx[0:10])
      x_real, y_real_label = train_data[idx], train_label[idx]
      #print ('ftftf',y_real_label)
      #######matplotlib.pyplot.imshow(x_real[0, :,:,0])
      ######plt.savefig('acutal.png')
      #Generate random labels for fake image generation
      y_fake_label = np.random.randint(0, self.num_classes, (batch_size, 1))
      #Generate some fake images
      x_fake = self.generator.predict([noise,y_fake_label])
      #####matplotlib.pyplot.imshow(x_fake[0, :,:,0])
      plt.savefig('fake.png')
      x = np.concatenate((x_real,x_fake))
      #print ('xxxxxxx',x.shape)
      y_label = np.concatenate((y_real_label, y_fake_label))
      #print ('ffff',y_fake_label)
      #print ('rrrrrr',y_real_label)
      #print ('ggggggg',y_label)
      y_real_or_fake = np.ones([2*batch_size,1]) #putting all images as real
      y_real_or_fake[batch_size:,:] = 0 #putting 0 for fake images
      #print(y_real_or_fake)
      #Train discriminator on real and fake
      encoded_y_label = to_categorical(y_label-1,num_classes=2)
      #print(encoded_y_label)
      d_loss = self.discriminator.train_on_batch(x,[y_real_or_fake,encoded_y_label])
      #Train Generator on Calculated loss
      y_real_or_fake = np.ones([batch_size, 1])
      noise = np.random.normal(0,1,size=[batch_size,seed_size])
      sampled_labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)
      encoded_sampled_labels = to_categorical(sampled_labels,num_classes=self.num_classes)
      g_loss = self.combined_model.train_on_batch([noise,sampled_labels],[y_real_or_fake,encoded_sampled_labels])
      print("genertaor losssss", g_loss)
      losses["D"].append(d_loss)
      losses["G"].append(g_loss)

      #Time for an update

      if save_freq > 0:
        if epoch % save_freq == 0:

          print ("epoch %d: [D loss: %f, acc.: %.2f%%] [G loss: %f, acc.: %.2f%%]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], 100*g_loss[1]))
          self.plot_sample_images(epoch, noise_input)

          cnt+=1

    self.plot_loss(losses)
if __name__ == '__main__':

  gan = ACGAN()
  gan.train(epochs=14000, batch_size=32, save_freq=200)
  gan.train()



