from __future__ import print_function, division

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution

from tqdm import tqdm



disable_eager_execution()



def batches(x, batch_size):
    """
    :x: shape (N, d)
    :return: Yields batches
    """
    x = x[np.random.permutation(len(x))][:int(len(x)/batch_size)*batch_size]
    for i in range(0, len(x), batch_size):
        yield x[i:i+batch_size]
        
class DCGAN():
    def __init__(self, prior, img_shape=(28, 28, 1), load_from=None):
        self.img_shape = img_shape
        self.channels = img_shape[-1]
        self.latent_dim = prior.d
        self.prior = prior

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        
        self.G = self.generator
        self.D = self.discriminator
        self.optimizer = optimizer
        
        
        if load_from:
            self.generator.load_weights(load_from+"_g.h5")
            self.discriminator.load_weights(load_from+"_d.h5")

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.prior.d))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        noise = Input(shape=(self.prior.d,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train_step(self, imgs, d_steps, g_steps, pruning):
        batch_size = len(imgs)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        # Sample noise and generate a batch of new images
        for _ in range(d_steps):
            noise = self.prior(batch_size)
            gen_imgs = self.generator.predict(noise)
            # fakes_val = gen_imgs[:10]
            
            # ------------------------------- PRUNING --------------------------------
            if pruning != 1:
                gen_scores = self.discriminator.predict(gen_imgs)
                gen_imgs_sorted_ids = tf.argsort(gen_scores,
                                                 axis=0,
                                                 direction='DESCENDING')
                
                leftover = int(gen_imgs_sorted_ids.shape[0]//(1/pruning))
                fake_output_best_ids = gen_imgs_sorted_ids[:leftover, :]
                
                gen_imgs = tf.gather_nd(gen_imgs, fake_output_best_ids)
                fake = np.zeros((gen_imgs.shape[0], 1))
            # ------------------------------------------------------------------------

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        #assert np.allclose(self.generator.predict(noise)[:10], fakes_val), "D update effects G"

        #  Train Generator
        g_loss = -1
        for _ in range(g_steps):
            # Train the generator (wants discriminator to mistake images as real)
            y_val = self.discriminator.predict(imgs[:10])
            g_loss = self.combined.train_on_batch(noise, valid)
            #assert np.allclose(y_val, self.discriminator.predict(imgs[:10])), "G update effects D"
    
    def train(self, X_train, epochs, batch_size=128, callbacks=[], d_steps=1, g_steps=1, pruning=1):
        print('Training DCGAN')
        X_train = np.reshape(X_train, [-1]+list(self.img_shape))

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            for batch_id, imgs in enumerate(batches(X_train, batch_size)):
                self.train_step(imgs, d_steps, g_steps, pruning)
                
            for callback, period in callbacks:
                if (epoch+1) % period == 0:
                    callback()
    
    def save(self, file_prefix='dcgan'):
        self.generator.save_weights(f"{file_prefix}_g.h5")
        self.discriminator.save_weights(f"{file_prefix}_d.h5")


class MiniGan(DCGAN):
    def __init__(self, prior, img_shape=[2], load_from=None, neurons=512):
        self.neurons = neurons
        super().__init__(prior, img_shape, load_from)
        
    def build_generator(self):

        model = Sequential()

        model.add(Dense(self.neurons, activation="relu", input_dim=self.prior.d))
        model.add(Dense(self.neurons, activation="relu", input_dim=self.prior.d))
        model.add(Dense(self.neurons, activation="relu", input_dim=self.prior.d))
        model.add(Dense(2, activation="tanh"))

        noise = Input(shape=(self.prior.d,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(self.neurons, activation="tanh", input_dim=self.prior.d))
        model.add(Dense(self.neurons, activation="tanh", input_dim=self.prior.d))
        model.add(Dense(self.neurons, activation="tanh", input_dim=self.prior.d))
        
        model.add(Dense(1, activation='sigmoid'))

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)
   

class Uniform():
    def __init__(self, d=100):
        self.d = d
        self.low = -1
        self.high = 1
    
    def __call__(self, n):
        return np.random.uniform(low=-1, high=1, size=(n, self.d))


class Unconnected(Uniform):
    def __init__(self, d=100):
        self.d = d
        self.low = -1.1
        self.high = 1.1
        
    def __call__(self, n):
        z = np.random.uniform(low=-1, high=1, size=(n, self.d))
        z += np.sign(z)*0.1
        return z


class GradientInverser():
    def __init__(self, gan):
        self.gan = gan
        g = gan.generator
        z = Input([gan.prior.d])
        reals = Input(gan.img_shape)
        fakes = g(z)
        loss = K.square(g.outputs[0]-reals)
        gradient = K.gradients(loss, g.inputs[0])[0]
        self.iterate = K.function([g.inputs[0], reals], [loss, gradient])

    def invert(self, x):
        z = self.gan.prior(1000)
        fakes = self.gan.generator.predict(z).reshape([1000, -1])
        x_ = x.reshape([-1, fakes.shape[1]])
        z=z[cdist(fakes, x.reshape([-1, fakes.shape[1]])).argmin(0)]
        lr = z*0+0.1
        momentum = 0.5
        old_gradient = 0
        for i in range(40):
            loss, gradient = self.iterate([z, x])
            gradient = (1-momentum)*gradient + momentum*old_gradient
            old_gradient = gradient
            loss = np.sum(loss, axis=(1, 2, 3))[...,None]
            new_z = z - lr*gradient
            new_z = np.clip(new_z, -1, 1)
            new_loss, _ = self.iterate([new_z, x])
            new_loss = np.sum(new_loss, axis=(1, 2, 3))[...,None]
            z = np.where(new_loss<loss, new_z, z+np.random.normal(0, 0.01, z.shape))
            lr = np.where(new_loss>loss, lr/2, lr*1.1)*0.9
        return z
    
    def __call__(self, x):
        return self.invert(x)


class AutoencodingGAN(DCGAN):
    def __init__(self, prior, img_shape=(28, 28, 1), load_from=None, train_decoder=False):
        super().__init__(prior, img_shape, load_from)

        self.encoder = self.build_encoder()
        x = Input(shape=(self.img_shape))
        encoded = self.encoder(x)
        if not train_decoder:
            self.generator.trainable = False
        decoded = self.generator(encoded)
        self.autoencoder = Model(x, decoded)
        self.autoencoder.compile(loss='mean_squared_error', optimizer=self.optimizer)


        if load_from:
            self.generator.load_weights(load_from+"_g.h5")
            self.discriminator.load_weights(load_from+"_d.h5")
            try:
                self.encoder.load_weights(load_from+"_enc.h5")
            except Exception as e:
                print('WARNING: loaded the gan but the autoencoder is not loaded')
                
    def build_encoder(self):
        # let's use the discriminator architecture
        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(self.prior.d, activation='tanh')) 
        # we assume that the prior is scaled from -1 to 1, otherwise 
        # we would need to scale explicitly here

        img = Input(shape=self.img_shape)
        encoded = model(img)

        return Model(img, encoded) 
        
        
    def train(self, X_train, epochs, batch_size=128, callbacks=[], d_steps=1, g_steps=1, ae_steps=1):
        print('Training AutoencodingGAN')
        X_train = np.reshape(X_train, [-1]+list(self.img_shape))
    
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
    
        for epoch in tqdm(range(epochs)):
    
            # ---------------------
            #  Train Discriminator
            # ---------------------
    
            for batch_id, imgs in enumerate(batches(X_train, batch_size)):
                self.train_step(imgs, d_steps, g_steps, ae_steps=ae_steps)
    
            for callback, period in callbacks:
                if (epoch+1) % period == 0:
                    callback()
            
        
    def train_step(self, imgs, d_steps, g_steps, ae_steps):
        batch_size = len(imgs)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        ## Train the autoencoder
        #ae_steps_done = 0
        #while (ae_steps - ae_steps_done) > np.random.uniform(0, 1):
        #    self.autoencoder.train_on_batch(imgs, imgs)
        #    ae_steps_done += 1
            
        # Train the autoencoder 'ae_steps' times
        for _ in range(ae_steps):
            self.autoencoder.train_on_batch(imgs, imgs)
        
        # Sample noise and generate a batch of new images
        for _ in range(d_steps):
            noise = self.prior(batch_size)
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        #  Train Generator
        g_loss = -1
        for _ in range(g_steps):
            y_val = self.discriminator.predict(imgs[:10])
            g_loss = self.combined.train_on_batch(noise, valid)
            
    def encode(self, x):
        return self.encoder.predict(x)
    
    def generate(self, z):
        return self.generator.predict(z)
    
    def discriminate(self, x):
        return self.discriminator.predict(x)