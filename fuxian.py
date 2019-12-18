import numpy as np
import matplotlib.pyplot as plt

import glob
import keras
import pandas as pd
import nibabel as nib
from scipy.ndimage.interpolation import zoom
import math
import tensorflow as tf
import datetime
from PIL import Image
from keras import layers
from keras.models import Model, Sequential
from keras.layers import *
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.layers.core import Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras import objectives
from keras.backend import *
from keras.layers import Concatenate, Reshape

import readpet

#T.config.nvcc.fastmath = False
# Firstly, run coregistration using matlab batch..

img_rows = 192
img_cols = 256
dout_size = (6, 8)


# Functions
def get_unet():
    inputs = Input((1, img_rows, img_cols))
    # 192 256
    # e1 = Convolution2D(64, 3, 3, subsample=(2, 2), init='uniform', border_mode='same')(inputs)
    e1 = Conv2D(64, (3, 3), strides=(2, 2), padding="same", kernel_initializer="glorot_uniform")(inputs)
    e1 = BatchNormalization()(e1)
    x = LeakyReLU(0.2)(e1)
    # 96 128
    # e2 = Conv2D(128, 3, 3, subsample=(2, 2), init='uniform', border_mode='same')(x)
    e2 = Conv2D(128, (3, 3), strides=(2, 2), padding="same", kernel_initializer="glorot_uniform")(x)
    e2 = BatchNormalization()(e2)
    x = LeakyReLU(0.2)(e2)
    # 48 64
    e3 = Conv2D(256, (3, 3), strides=(2, 2), padding="same", kernel_initializer="glorot_uniform")(x)
    e3 = BatchNormalization()(e3)
    x = LeakyReLU(0.2)(e3)
    # 24 32
    # e4 = Convolution2D(512, 3, 3, subsample=(2, 2), init='uniform', border_mode='same')(x)
    e4 = Conv2D(256, (3, 3), strides=(2, 2), padding="same", kernel_initializer="glorot_uniform")(x)
    e4 = BatchNormalization()(e4)
    x = LeakyReLU(0.2)(e4)
    # 12 16
    # e5 = Conv2D(512, 3, 3, subsample=(2, 2), init='uniform', border_mode='same')(x)
    e5 = Conv2D(512, (3, 3), strides=(2, 2), padding="same", kernel_initializer="glorot_uniform")(x)
    e5 = BatchNormalization()(e5)
    x = LeakyReLU(0.2)(e5)
    # 6 8
    e6 = Conv2D(512, (2, 2), strides=(2, 2), padding="same", kernel_initializer="glorot_uniform")(x)
    e6 = BatchNormalization()(e6)
    x = LeakyReLU(0.2)(e6)
    # 3 4

    # d1 = Deconvolution2D(512, 3, 3,output_shape=(None,512, int(img_rows / 32),int(img_cols / 32)), init='uniform',
    #                       border_mode='same',subsample=(2, 2),input_shape=(512, 3, 4))(x)
    #d1 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same',init='glorot_uniform',input_shape=(512, 3, 4,1))(x)
    d1 = Conv2D(512, 3, padding='same', kernel_initializer='glorot_uniform',input_shape=(512, 3, 4,1))(UpSampling2D(size=(2, 2))(x))
    d1 = BatchNormalization()(d1)
    # 6 8
    d1 = Dropout(0.5)(d1)
    x = Concatenate(axis=1)([d1, e5])
    x = LeakyReLU(0.2)(x)

    #d2 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same',init='glorot_uniform')(x)
    d2 = Conv2D(512, (3,3), padding='same', kernel_initializer='glorot_uniform',input_shape=(512, 6, 8,1))(UpSampling2D(size=(2, 2))(x))
    d2 = BatchNormalization()(d2)
    d2 = Dropout(0.5)(d2)#输出大小等于输入
    x = Concatenate(axis=1)([d2, e4])#e4:12,16,
    x = LeakyReLU(0.2)(x)

    #d3 = Conv2DTranspose(256, (3, 3), subsample=(2, 2), padding='same',init='glorot_uniform')(x)
    d3 = Conv2D(256, (3, 3), padding='same', kernel_initializer='glorot_uniform', input_shape=(512, 12, 16, 1))(UpSampling2D(size=(2, 2))(x))
    d3 = BatchNormalization()(d3)
    d3 = Dropout(0.5)(d3)
    x = Concatenate( axis=1)([d3, e3])
    x = LeakyReLU(0.2)(x)

    #d4 = Conv2DTranspose(128, (3, 3), subsample=(2, 2), padding='same',init='glorot_uniform')(x)
    d4 = Conv2D(128, (3, 3), padding='same', kernel_initializer='glorot_uniform', input_shape=(256, 24, 32, 1))(UpSampling2D(size=(2, 2))(x))
    d4 = BatchNormalization()(d4)
    d4 = Dropout(0.5)(d4)
    x = Concatenate( axis=1)([d4, e2])
    x = LeakyReLU(0.2)(x)

    #d5 = Conv2DTranspose(64, (3, 3), subsample=(2, 2), padding='same',init='glorot_uniform')(x)
    d5 = Conv2D(64, (3, 3), padding='same', kernel_initializer='glorot_uniform', input_shape=(128, 48, 64, 1))(
        UpSampling2D(size=(2, 2))(x))
    d5 = BatchNormalization()(d5)
    d5 = Dropout(0.5)(d5)
    x = Concatenate( axis=1)([d5, e1])
    x = LeakyReLU(0.2)(x)

    #d6 = Conv2DTranspose(1, (3, 3), subsample=(2, 2), padding='same',init='glorot_uniform')(x)
    d6 = Conv2D(1, (3, 3), padding='same', kernel_initializer='glorot_uniform', input_shape=(512, 96, 128, 1))(
        UpSampling2D(size=(2, 2))(x))
    xout = Activation('tanh')(d6)
    model = Model(inputs=inputs, outputs=xout)

    return model


def unet(pretrained_weights=None, input_size=(192, 256, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    e1 = BatchNormalization()(conv1)
    x = LeakyReLU(0.2)(e1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    e2 = BatchNormalization()(conv2)
    x = LeakyReLU(0.2)(e2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    e3 = BatchNormalization()(conv3)
    x = LeakyReLU(0.2)(e3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    # pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    e4 = BatchNormalization()(drop4)
    x = LeakyReLU(0.2)(e4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    # merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    # merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    # merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    # merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def get_unet_basic():
    inputs = Input((1, img_rows, img_cols))
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, 3, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, 3, activation='relu', padding='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Conv2D(256, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, 3, activation='relu', padding='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Conv2D(128, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, 3, activation='relu', padding='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64,  3, activation='relu', padding='same')(up8)
    conv8 = Convolution2D(64,  3, activation='relu', padding='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32,  3, activation='relu', padding='same')(up9)
    conv9 = Convolution2D(32,  3, activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    return model


def DiscriminatorModel(d_optim):
    model = Sequential()
    model.add(Conv2D(64, (5, 5), subsample=(2, 2), input_shape=(2, img_rows, img_cols), padding='same'))
    model.add(LeakyReLU(0.2))
    # model.add(BatchNormalization())
    model.add(Conv2D(64, (5, 5), subsample=(2, 2), padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(128, (5, 5), subsample=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(256, (5, 5), subsample=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(512, (5, 5), subsample=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('sigmoid'))

    def d_loss(y_true, y_pred):
        return objectives.binary_crossentropy(K.batch_flatten(y_true), K.batch_flatten(y_pred))

    model.compile(optimizer=d_optim, loss=d_loss)

    return model


def generator_containing_discriminator(generator, discriminator, g_optim, alpha=100):
    inputs = Input((1, img_rows, img_cols))
    inputs_target = Input((1, img_rows, img_cols))
    x_generator = generator(inputs)
    # merged = merge([inputs, x_generator], mode='concat', concat_axis=1)
    merged = Concatenate(axis=1)([inputs, x_generator])
    discriminator.trainable = False
    x_discriminator = discriminator(merged)
    model = Model(inputs=[inputs, inputs_target], outputs=x_discriminator)

    def g_loss(y_true, y_pred):
        y_true_flat = K.batch_flatten(y_true)
        y_pred_flat = K.batch_flatten(y_pred)

        L_adv = objectives.binary_crossentropy(y_true_flat, y_pred_flat)  # Adversarial Loss

        inputs_target_flat = K.batch_flatten(inputs_target)
        x_generator_flat = K.batch_flatten(x_generator)
        L_1 = K.mean(K.abs(inputs_target_flat - x_generator_flat))

        return L_adv + alpha * L_1

    model.compile(optimizer=g_optim, loss=g_loss)
    return model


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[2:]
    image = np.zeros((height * shape[0], width * shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = img[:, :]
    return image


def train(X_train, y_train, BATCH_SIZE):
    d_optim = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08)#辨别器优化，Adam优化方法，学习率为0.0002，beta1：一阶矩估计的指数衰减率
    g_optim = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08)#beta2：二阶矩估计的指数衰减率
    d_lossList=[]
    g_lossList = []
    # generator = get_unet()
    generator = get_unet()#获取网络
    discriminator = DiscriminatorModel(d_optim)
    discriminator_on_generator = generator_containing_discriminator(generator,
                                                                    discriminator,
                                                                    g_optim)

    generator.compile(loss='binary_crossentropy', optimizer="SGD")#采用梯度下降


    discriminator.trainable = True#开始先训练辨别器

    for epoch in range(30):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0] / BATCH_SIZE))
        for index in range(int(X_train.shape[0] / BATCH_SIZE)):#index:将总共的切片以batchSize为一组，index为每一组的索引
            y_batch = y_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]#对应的一组图像标签pixel to pixel

            generated_images = generator.predict(X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE])#只是预测了一次，不分验证集

            # Visualizing...
            if index % 20 == 0:
                image = combine_images(generated_images)
                image = (image + 1) * 255 * 0.5#原始图像不是通过255归一化，这里是否用255？--是
                Image.fromarray(image.astype(np.uint8)).save(
                    "./image_example/" + str(epoch) + "_" + str(index) + ".png")

            real_pairs = np.concatenate((X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE, :, :, :],
                                         y_batch), axis=1)#(8,2,91,109)
            fake_pairs = np.concatenate((X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE, :, :, :],
                                         generated_images), axis=1)#横向拼接
            X = np.concatenate((real_pairs, fake_pairs))#纵向拼接#(16,2,91,109)
            y = np.zeros((X.shape[0], 1) + dout_size)#（16,1,6,8）
            y[:real_pairs.shape[0]] = 1


            discriminator.train_on_batch(X, y)
            discriminator.train_on_batch(X, y)
            d_loss = discriminator.train_on_batch(X, y)
            # ###################################################
            # #再训练一遍D
            # y_batch = y_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]  # 对应的一组图像标签pixel to pixel
            #
            # generated_images = generator.predict(X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE])  # 只是预测了一次，不分验证集
            #
            # xx = X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE, :, :, :]
            # real_pairs = np.concatenate((xx,
            #                              y_batch), axis=1)  # (8,2,91,109)
            # fake_pairs = np.concatenate((X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE, :, :, :],
            #                              generated_images), axis=1)  # 横向拼接
            # X = np.concatenate((real_pairs, fake_pairs))  # 纵向拼接#(8,2,91,109)
            # y = np.zeros((X.shape[0], 1) + dout_size)
            # y[:real_pairs.shape[0]] = 1
            #
            # d_loss = discriminator.train_on_batch(X, y)
            #          ##########################################################################
            print("batch %d / %d d_loss : %f" % (index, int(X_train.shape[0] / BATCH_SIZE), d_loss))


            discriminator.trainable = False  #然后训练生成器
            g_loss = discriminator_on_generator.train_on_batch(
                [X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE, :, :, :], y_batch],
                np.ones((BATCH_SIZE, 1) + dout_size)
            )
            d_lossList.append(d_loss);
            g_lossList.append(g_loss);

            discriminator.trainable = True

            print("batch %d g_loss : %f" % (index, g_loss))
#            generated_images2 = generated_images
#            generated_images2 = np.transpose(generated_images2,(0,2,3,1))
            temp_path = "tensorboard"
            if len(os.listdir("tensorboard"))!=0:
                name = os.listdir("tensorboard")[0]
                temp_path = os.path.join(temp_path, name)
                if len(os.listdir(temp_path))!=0:
                    #os.system('rm ' + logdir + '/' + 'events*')
                    name = os.listdir(temp_path)[0]
                    temp_path = os.path.join(temp_path, name)
                    os.remove(temp_path)
            summary = sess.run(merged, feed_dict={D_LOSS: d_loss, G_LOSS: g_loss})
            writer.add_summary(summary, index+epoch*(int(X_train.shape[0] / BATCH_SIZE)))


        generator.save_weights('./model/generator_0510' + str(epoch) + '.h5', True)
        discriminator.save_weights('./model/discriminator_0510' + str(epoch) + '.h5', True)


x_train = []
y_train = []
path='D:/work/ADNI_AV45_TI_HC_MCI_AD/test/AV45_1'
path2='D:/work/ADNI_AV45_TI_HC_MCI_AD/test/t1_1'

x_train, _ = readpet.read_img(path,100)
y_train, _ = readpet.read_img(path2,100)
x_train =np.array(x_train)
y_train =np.array(y_train)
maxindex = np.unravel_index(np.argmax(x_train),x_train.shape)
x_train/=x_train[maxindex]#归一化
maxindex = np.unravel_index(np.argmax(y_train),y_train.shape)
y_train/=y_train[maxindex]#归一化

#start tensorboard
sess=tf.Session()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)
D_LOSS = tf.placeholder(tf.float32, [])
G_LOSS = tf.placeholder(tf.float32, [])
IMAGES = tf.placeholder(tf.float32,shape=[None,192,256,1])
tf.summary.scalar("D_LOSS", D_LOSS)#一个d_loss
tf.summary.scalar("G_LOSS", G_LOSS)#一个g_loss
#tf.summary.image("IMAGES", IMAGES,6)
merged=tf.summary.merge_all()#合并标签为D_LOSS的集合和G_LOSS的集合
#end tensorboard


train(x_train, y_train, 1)


def remove_logdir(self):
    print(os.system('ls checkpoints/' + self.model_name + '/' + 'events*'))
    os.system('rm ' + logdir + '/' + 'events*')
