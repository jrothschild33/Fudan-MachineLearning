'''
Homework4:Auto-Encoder
@Author:周嘉楠 19210980081
@Date:2020-5-31
@Prerequisites:
    scipy == 1.2.1
    Pillow == 7.1.2
    tensorflow == 1.15.3
    numpy == 1.18.3
    torch == 1.2.0
'''
# =================================================================================================================
'''
第0步：导入所需的库
'''
# =================================================================================================================

import gzip
import os
import numpy
from scipy import ndimage
from six.moves import urllib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave
from scipy.misc import imresize
import torch
from torch import nn
from torch.nn import functional as F
import warnings
warnings.filterwarnings('ignore')


# =================================================================================================================
'''
第1步：下载数据
* 从网站 http://yann.lecun.com/exdb/mnist/ 上下载数据并自动解压到data文件夹
'''
# =================================================================================================================

# Download Website
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
DATA_DIRECTORY = "Data"

# Params for MNIST
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000


# Download MNIST data
def maybe_download(filename):
    """Download the data from Yann's website, unless it's already here."""
    if not tf.io.gfile.exists(DATA_DIRECTORY):
        tf.gfile.MakeDirs(DATA_DIRECTORY)
    filepath = os.path.join(DATA_DIRECTORY, filename)
    if not tf.io.gfile.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath


# Extract the images
def extract_data(filename, num_images, norm_shift=False, norm_scale=True):
    """
    Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
        if norm_shift:
            data = data - (PIXEL_DEPTH / 2.0)
        if norm_scale:
            data = data / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        data = numpy.reshape(data, [num_images, -1])
    return data


# Extract the labels
def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
        num_labels_data = len(labels)
        one_hot_encoding = numpy.zeros((num_labels_data, NUM_LABELS))
        one_hot_encoding[numpy.arange(num_labels_data), labels] = 1
        one_hot_encoding = numpy.reshape(one_hot_encoding, [-1, NUM_LABELS])
    return one_hot_encoding


# Augment training data
def expend_training_data(images, labels):
    expanded_images = []
    expanded_labels = []
    j = 0  # counter
    for x, y in zip(images, labels):
        j = j + 1
        if j % 100 == 0:
            print('expanding data : %03d / %03d' % (j, numpy.size(images, 0)))

        # register original data
        expanded_images.append(x)
        expanded_labels.append(y)

        # get a value for the background
        # zero is the expected value, but median() is used to estimate background's value
        bg_value = numpy.median(x)  # this is regarded as background's value
        image = numpy.reshape(x, (-1, 28))

        for i in range(4):
            # rotate the image with random degree
            angle = numpy.random.randint(-15, 15, 1)
            new_img = ndimage.rotate(image, angle, reshape=False, cval=bg_value)

            # shift the image with random distance
            shift = numpy.random.randint(-2, 2, 2)
            new_img_ = ndimage.shift(new_img, shift, cval=bg_value)

            # register new training data
            expanded_images.append(numpy.reshape(new_img_, 784))
            expanded_labels.append(y)

    # images and labels are concatenated for random-shuffle at each epoch
    # notice that pair of image and label should not be broken
    expanded_train_total_data = numpy.concatenate((expanded_images, expanded_labels), axis=1)
    numpy.random.shuffle(expanded_train_total_data)

    return expanded_train_total_data


# Prepare MNISt data
def prepare_MNIST_data(use_norm_shift=False, use_norm_scale=True, use_data_augmentation=False):
    # Get the data.
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, 60000, use_norm_shift, use_norm_scale)
    train_labels = extract_labels(train_labels_filename, 60000)
    test_data = extract_data(test_data_filename, 10000, use_norm_shift, use_norm_scale)
    test_labels = extract_labels(test_labels_filename, 10000)

    # Generate a validation set.
    validation_data = train_data[:VALIDATION_SIZE, :]
    validation_labels = train_labels[:VALIDATION_SIZE, :]
    train_data = train_data[VALIDATION_SIZE:, :]
    train_labels = train_labels[VALIDATION_SIZE:, :]

    # Concatenate train_data & train_labels for random shuffle
    if use_data_augmentation:
        train_total_data = expend_training_data(train_data, train_labels)
    else:
        train_total_data = numpy.concatenate((train_data, train_labels), axis=1)
    train_size = train_total_data.shape[0]

    return train_total_data, train_size, validation_data, validation_labels, test_data, test_labels

# =================================================================================================================
'''
第2步：定义常用函数
* 定义Auto-Decoder解析代码生成图片的函数：Plot_Reproduce_Performance()
'''
# =================================================================================================================

# 定义使用Decoder解析代码生成图片的函数
class Plot_Reproduce_Performance():
    def __init__(self, DIR, n_img_x=8, n_img_y=8, img_w=28, img_h=28, resize_factor=1.0):
        self.DIR = DIR
        assert n_img_x > 0 and n_img_y > 0
        self.n_img_x = n_img_x
        self.n_img_y = n_img_y
        self.n_tot_imgs = n_img_x * n_img_y

        assert img_w > 0 and img_h > 0
        self.img_w = img_w
        self.img_h = img_h

        assert resize_factor > 0
        self.resize_factor = resize_factor

    def save_images(self, images, name='result.jpg'):
        images = images.reshape(self.n_img_x * self.n_img_y, self.img_h, self.img_w)
        imsave(self.DIR + "/" + name, self._merge(images, [self.n_img_y, self.n_img_x]))

    def _merge(self, images, size):
        h, w = images.shape[1], images.shape[2]
        h_ = int(h * self.resize_factor)
        w_ = int(w * self.resize_factor)
        img = np.zeros((h_ * size[0], w_ * size[1]))
        for idx, image in enumerate(images):
            i = int(idx % size[1])
            j = int(idx / size[1])
            image_ = imresize(image, size=(w_, h_), interp='bicubic')
            img[j * h_:j * h_ + h_, i * w_:i * w_ + w_] = image_
        return img

# =================================================================================================================
'''
第3步：建立Auto-Encoder和Decoder模型
'''
# =================================================================================================================

class Encoder(nn.Module):
    def __init__(self, imgsz, n_hidden, n_output, keep_prob):
        super(Encoder, self).__init__()
        self.imgsz = imgsz
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.keep_prob = keep_prob
        self.net = nn.Sequential(
            nn.Linear(imgsz, n_hidden),
            nn.ELU(inplace=True),
            nn.Dropout(1 - keep_prob),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Dropout(1 - keep_prob),
            nn.Linear(n_hidden, n_output * 2)
        )

    def forward(self, x):
        mu_sigma = self.net(x)
        mean = mu_sigma[:, :self.n_output]
        stddev = 1e-6 + F.softplus(mu_sigma[:, self.n_output:])
        return mean, stddev

class Decoder(nn.Module):
    def __init__(self, dim_z, n_hidden, n_output, keep_prob):
        super(Decoder, self).__init__()
        self.dim_z = dim_z
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.keep_prob = keep_prob
        self.net = nn.Sequential(
            nn.Linear(dim_z, n_hidden),
            nn.Tanh(),
            nn.Dropout(1 - keep_prob),

            nn.Linear(n_hidden, n_hidden),
            nn.ELU(),
            nn.Dropout(1 - keep_prob),

            nn.Linear(n_hidden, n_output),
            nn.Sigmoid()
        )

    def forward(self, h):
        return self.net(h)

def init_weights(encoder, decoder):
    def init_(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    for m in encoder.modules():
        m.apply(init_)
    for m in decoder.modules():
        m.apply(init_)
    print('weights inited!')

def get_ae(encoder, decoder, x):
    # encoding
    mu, sigma = encoder(x)
    # sampling by re-parameterization technique
    z = mu + sigma * torch.randn_like(mu)
    # decoding
    y = decoder(z)
    y = torch.clamp(y, 1e-8, 1 - 1e-8)
    return y

def get_z(encoder, x):
    # encoding
    mu, sigma = encoder(x)
    # sampling by re-parameterization technique
    z = mu + sigma * torch.randn_like(mu)
    return z

def get_loss(encoder, decoder, x, x_target):
    batchsz = x.size(0)
    mu, sigma = encoder(x)
    z = mu + sigma * torch.randn_like(mu)
    # decoding
    y = decoder(z)
    y = torch.clamp(y, 1e-8, 1 - 1e-8)
    # loss
    marginal_likelihood = -F.binary_cross_entropy(y, x_target, reduction='sum') / batchsz
    KL_divergence = 0.5 * torch.sum(
        torch.pow(mu, 2) +
        torch.pow(sigma, 2) -
        torch.log(1e-8 + torch.pow(sigma, 2)) - 1
    ).sum() / batchsz
    ELBO = marginal_likelihood - KL_divergence
    loss = -ELBO
    return y, z, loss, marginal_likelihood, KL_divergence

# =================================================================================================================
'''
第4步：主程序2:运行Auto-Decoder
'''
# =================================================================================================================

IMAGE_SIZE_MNIST = 28

def main():
    device = torch.device('cuda')

    # Basic Parameters
    RESULTS_DIR = 'Results'                     # File path of output images
    n_hidden = 500                              # Number of hidden units in MLP
    dim_img = IMAGE_SIZE_MNIST ** 2             # Image size of MNIST
    dim_z = 20                                  # Dimension of latent vector

    # Train Parameters
    n_epochs = 20                               # The number of epochs to run
    batch_size = 128                            # Batch size
    learn_rate = 1e-3                           # Learning rate for Adam optimizer


    # Plot Parameters:Auto-Decoder
    PRR = True                                  # Plot Reproduce Result
    PRR_n_img_x = 10                            # Number of images along x-axis
    PRR_n_img_y = 10                            # Number of images along y-axis
    PRR_resize_factor = 1.0                     # Resize factor for each displayed image


    """ prepare MNIST data """
    train_total_data, train_size, _, _, test_data, test_labels = prepare_MNIST_data()
    n_samples = train_size

    """ create network """
    keep_prob = 0.99
    encoder = Encoder(dim_img, n_hidden, dim_z, keep_prob).to(device)
    decoder = Decoder(dim_z, n_hidden, dim_img, keep_prob).to(device)
    # + operator will return but .extend is inplace no return.
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learn_rate)
    # vae.init_weights(encoder, decoder)

    """ training """
    # Plot for reproduce performance
    if PRR:
        PRR = Plot_Reproduce_Performance(RESULTS_DIR, PRR_n_img_x, PRR_n_img_y, IMAGE_SIZE_MNIST,
                                                    IMAGE_SIZE_MNIST, PRR_resize_factor)
        x_PRR = test_data[0:PRR.n_tot_imgs, :]
        x_PRR_img = x_PRR.reshape(PRR.n_tot_imgs, IMAGE_SIZE_MNIST, IMAGE_SIZE_MNIST)
        PRR.save_images(x_PRR_img, name='input.jpg')
        print('saved:', 'input.jpg')
        x_PRR = torch.from_numpy(x_PRR).float().to(device)

    # train
    total_batch = int(n_samples / batch_size)
    min_tot_loss = np.inf
    for epoch in range(n_epochs):
        # Random shuffling
        np.random.shuffle(train_total_data)
        train_data_ = train_total_data[:, :-NUM_LABELS]
        # Loop over all batches
        encoder.train()
        decoder.train()
        for i in range(total_batch):
            # Compute the offset of the current minibatch in the data.
            offset = (i * batch_size) % (n_samples)
            batch_xs_input = train_data_[offset:(offset + batch_size), :]
            batch_xs_target = batch_xs_input
            batch_xs_input, batch_xs_target = torch.from_numpy(batch_xs_input).float().to(device), \
                                              torch.from_numpy(batch_xs_target).float().to(device)
            assert not torch.isnan(batch_xs_input).any()
            assert not torch.isnan(batch_xs_target).any()
            y, z, tot_loss, loss_likelihood, loss_divergence = \
                                        get_loss(encoder, decoder, batch_xs_input, batch_xs_target)

            optimizer.zero_grad()
            tot_loss.backward()
            optimizer.step()

        # print cost every epoch
        print("epoch %d: L_tot %03.2f L_likelihood %03.2f L_divergence %03.2f" % (
                                                epoch, tot_loss.item(), loss_likelihood.item(), loss_divergence.item()))

        encoder.eval()
        decoder.eval()
        # if minimum loss is updated or final epoch, plot results
        if min_tot_loss > tot_loss.item() or epoch + 1 == n_epochs:
            min_tot_loss = tot_loss.item()
            # Plot for reproduce performance
            if PRR:
                y_PRR = get_ae(encoder, decoder, x_PRR)

                y_PRR_img = y_PRR.reshape(PRR.n_tot_imgs, IMAGE_SIZE_MNIST, IMAGE_SIZE_MNIST)
                PRR.save_images(y_PRR_img.detach().cpu().numpy(), name="/PRR_epoch_%02d" % (epoch) + ".jpg")
                print('saved:', "/PRR_epoch_%02d" % (epoch) + ".jpg")


if __name__ == '__main__':
    main()

