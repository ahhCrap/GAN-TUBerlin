import numpy as np
from matplotlib import pyplot as plt
import keras
from keras import layers
from keras.models import Model
import time
from keras.models import load_model
from keras import backend as K


# Data generation
na = None

def show_2d(x_real, x_fake=None, title=None, space='x'):
    plt.plot(x_real[:, 0], x_real[:, 1], '.', label='Real')
    if x_fake is not None:
        plt.plot(x_fake[:, 0], x_fake[:, 1], '.', label='Fake')
        plt.legend()
    if title is not None:
        plt.title(title)
    plt.xlabel(space+'_1')
    plt.ylabel(space+'_2')
    plt.show()


def get_mode(center, n_per_mode, r):
    samples = np.random.normal(center, r, size=(3*n_per_mode, 2))
    dist = np.linalg.norm(samples-center[na,...], axis=1)
    return samples[dist<r][:n_per_mode]

def generate_2d(n_per_mode, radius=0.2):
    centers = [
        np.array([-1, -1]),
        np.array([-1, 1]),
        np.array([1, -1]),
        np.array([1, 1])
    ]
    return np.concatenate([get_mode(c, n_per_mode, radius) for c in centers], axis=0)


x = np.load('data/2d.npy')


class Uniform():
    def __init__(self, d=2):
        self.d = d
    
    def __call__(self, n):
        return np.random.uniform(low=-1, high=1, size=(n, self.d))

uniform = Uniform()

def show_learned_distribution(prior, G):
    z = prior(500)
    show_2d(z, title='Prior', space='z')
    show_2d(G(z), title='G(z)')

# ## D and G architecture
# 
# We have the data, now let's create a generator and a discriminator. We will experiment with different architectures and we often want to use a symmetric architecture for D and G, so let's build it modular such that the hidden layers will be passed as argument.



class Subnet():
    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x
    
    def __call__(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

mini_arch = lambda: [
    layers.Dense(4, use_bias=False, input_shape=(2,)),
    layers.BatchNormalization(),
    layers.LeakyReLU(),
    layers.Dense(4, use_bias=False),
    layers.BatchNormalization(),
    layers.LeakyReLU()
]

big_arch = lambda: [
    layers.Dense(100, use_bias=False, input_shape=(2,)),
    layers.BatchNormalization(),
    layers.LeakyReLU(),
    layers.Dense(100, use_bias=False),
    layers.BatchNormalization(),
    layers.LeakyReLU()
]

def flexibel_arch(neurons):
    return lambda: [
        layers.Dense(neurons, use_bias=False, input_shape=(2,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Dense(neurons, use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU()
    ]
    

class Discriminator(Subnet):
    def __init__(self, hidden_layers):
        # 2 hidden layers a 4 neurons, random init, batchnorm
        inputs=keras.Input(shape=[2])
        self.layers = [layers.BatchNormalization()] + hidden_layers() + [layers.Dense(1, activation='sigmoid')]
        self.model = Model(inputs=inputs, outputs=self.forward(inputs))
        self.optimizer = keras.optimizers.Adam(0.0002, 0.5)
        
        
class Generator(Subnet):
    def __init__(self, hidden_layers, outputlayer=lambda: layers.Dense(2)):
        # 2 hidden layers a 4 neurons, random init, batchnorm
        inputs=keras.Input(shape=[2])
        self.layers = hidden_layers() + [outputlayer()]
        self.model = Model(inputs=inputs, outputs=self.forward(inputs))
        self.optimizer = keras.optimizers.Adam(0.0002, 0.5)
        
    def loss(self, D):
        # D: keras model
        def l(y_target, x_fake):
            # y_target: always 1
            y_fake = D(x_fake)
            return keras.losses.binary_crossentropy(y_target, y_fake)
        return l
    

# ## Training Analysis and Callbacks
# 
# Now let's train a GAN to sample from this distribution. We can analyze the training by repeatedly checking the following questions:
# 1. how does the learned distribution change over time?
# 2. what is $ D(G(z)) $ for a grid of $z$s?
# 3. what is the mapping $G$? We can associate each point $z$ with a color $C(G(z))$, where the color mapping $C$ is e.g. an rgb color: $(x_1, x_2) -> rgb(n(x_1), n(x_2), 0) $ where $n$ is an affine function.
# 
# Let's create the plot functions we can use to analyze our GAN one by one. During training, we can pass it as callbacks that are executed after each epoch, to see how training evolves.
# 

def make_grid():
    ticks = np.arange(-1.5, 1.5, 0.05)
    return np.dstack(np.meshgrid(ticks, ticks)).reshape(-1, 2)

def grid_labels():
    return range(0, 61, 6), [f"{i/50:.2g}" for i in range(-75, 76, 15)] 

grid = make_grid()

def colors_of(x):
    c = np.zeros([x.shape[0], 3])
    c[:, :2] = (x+1.5)/3
    return np.clip(c, 0, 1)

def color_plot(mapping, title=None):
    z = make_grid()
    x = mapping(z)
    
    fig, ax = plt.subplots(ncols=2)
    
    ax[0].imshow(colors_of(z).reshape(60, 60, 3))
    ax[0].set_title('C(Z)')
    ticks, labels = grid_labels()
    ax[0].set_xticks(ticks)
    ax[0].set_xticklabels(labels)
    ax[0].set_yticks(ticks)
    ax[0].set_yticklabels(labels)
    ax[0].set_xlabel('z0')
    ax[0].set_ylabel('z1')
    
    ax[1].imshow(colors_of(x).reshape(60, 60, 3))
    ax[1].set_title('C(G(Z))')
    ax[1].set_xticks(ticks)
    ax[1].set_xticklabels(labels)
    ax[1].set_yticks(ticks)
    ax[1].set_yticklabels(labels)
    ax[1].set_xlabel('z0')
    ax[1].set_ylabel('z1')
    
    if title is not None:
        fig.suptitle(title)
    
    plt.show()
  

def score_over_z(G, D, title=None):
    z = make_grid()
    dgz = D(G(z))
    plt.imshow(dgz.reshape(60, 60))
    ticks, labels = grid_labels()
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    plt.xlabel('z0')
    plt.ylabel('z1')
    plt.colorbar()
    if title is not None:
        plt.title(title)
    plt.show()
    


# ## GAN Training
# 
# We have everything prepared, now we can train our GAN. We want to keep it modular with respect to the `Generator` and `Discriminator` class as well as the `prior` function, so that we can use it on another dataset like MNIST as well, or test different priors.

# ### The GAN

def batches(x, batch_size):
    """
    :x: shape (N, d)
    :return: Yields batches
    """
    x = x[np.random.permutation(len(x))][:int(len(x)/batch_size)*batch_size]
    for i in range(0, len(x), batch_size):
        yield x[i:i+batch_size]
        
        
class GAN():
    def __init__(self, prior, G, D):
        self.prior = prior
        
        D_loss = keras.losses.binary_crossentropy
        D.model.compile(D.optimizer, D_loss)
        self.D = D
        
        #D.model.trainable = False # D is already compiled, this will only effect the Generator loss
        g_loss = G.loss(D.model)
        G.model.compile(G.optimizer, g_loss)
        self.G = G
    
    def get_mixed_batch(self, x_real):
        n = x_real.shape[0]
        z = self.prior(n)
        x_fake = self.G(z)
        x_mix = np.concatenate([x_fake, x_real], axis=0)
        y_target = np.concatenate([np.zeros(n), np.ones(n)], axis=0)
        return x_mix, y_target
        
    def train_step(self, x_real, g_updates, d_updates):
        n = x_real.shape[0]
        
        test_y = self.D(x_real)
        for _ in range(g_updates):
            z = self.prior(n)
            self.G.model.train_on_batch(z, np.ones(n))
        assert (self.D(x_real) == test_y).all(), "Generator update effects D"
        
        z = self.prior(n)
        test_fakes = self.G(z)
        for _ in range(d_updates):
            x_mix, y_target = self.get_mixed_batch(x_real)
            self.D.model.train_on_batch(x_mix, y_target)
        assert (self.G(z) == test_fakes).all(), "Discriminator update effects G"

    def fit(self, x, epochs, file_prefix, callbacks=[], g_updates=1, d_updates=1, batch_size=128):
        """
        x: samples from the real distribution
        epochs: how many epochs to train
        file_prefix: saves the model to ./models/file_prefix<_g.npy and _d.npy>
        callbacks: [(f(z_at_epoch), num_epochs), ...] to execute every num_epochs
        g_updates: int, so many generator updates happen per train step
        d_updates: int, so many discriminator updates happen per train step
        returns: z_at_epoch: [G_at_epoch_i(fixed_z)) for i in range(epochs)]
        """
        for epoch in range(epochs):
            start = time.time()

            for x_real in batches(x, batch_size):
                self.train_step(x_real, g_updates, d_updates)
            print('.', end='')
            
            for callback, rhythm in callbacks:
                if (epoch+1) % rhythm == 0:
                    callback()
            #print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    
    def save(self, file_prefix):
        self.G.model.save(f"{file_prefix}_g.h5")
        self.D.model.save(f"{file_prefix}_d.h5")
    
    @staticmethod
    def load(file_prefix, prior):
        G = load_model(f"{file_prefix}_g.h5")
        D = load_model(f"{file_prefix}_d.h5")
        return GAN(prior, G, D)

# ### More callbacks
# 
# We will define a few more callbacks that track and plot loss or accuracy, and one that shows how the projection of a batch of fixed $z$s evolves during training

class EvolvingCallback():
    def __init__(self, gan, x):
        self.gan = gan
        self.z = gan.prior(100) # for plotting how the projection evolves
        self.z_at_epoch = [gan.G(self.z)]
        self.x = x
    
    def track(self):
        self.z_at_epoch += [self.gan.G(self.z)]
        
    def plot(self):
        t = np.stack(self.z_at_epoch, axis=0)
        epochs = len(self.z_at_epoch)
        plt.clf()
        plt.plot(self.x[:,0], self.x[:,1], 'o')
        for p in range(t.shape[1]):
            plt.plot(t[:,p,0], t[:,p,1], alpha=0.5, color='orange')
        plt.plot(t[-1,:,0], t[-1,:,1], 'o', color='red')
        plt.title(f"Epoch {epochs}")
        plt.show()

        
class LossMetrics():
    def __init__(self, gan, x_test):
        self.gan = gan
        self.x_test = x_test
        self.z_test = gan.prior(len(x_test))
        self.g_losses = []
        self.d_losses = []
        self.d_loss_real = []
        self.d_loss_fake = []
        
    def track(self, *ignored_args, **ignored_kwargs):
        x_mix, y_target = gan.get_mixed_batch(self.x_test)
        self.g_losses.append(self.gan.G.model.evaluate(self.z_test, np.ones(len(self.z_test)), verbose=0))
        self.d_losses.append(self.gan.D.model.evaluate(x_mix, y_target, verbose=0))
        self.d_loss_real.append(self.gan.D.model.evaluate(self.x_test, np.ones(len(x)), verbose=0))
        self.d_loss_fake.append(self.gan.D.model.evaluate(self.gan.G(self.z_test), np.zeros(len(self.z_test)), verbose=0))
    
    def plot(self, *ignored_args, **ignored_kwargs):    
        plt.plot(self.g_losses, label='G')
        plt.plot(self.d_losses, label='D')
        plt.plot(self.d_loss_real, label='D(real)')
        plt.plot(self.d_loss_fake, label='D(fake)')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()
        
        
class PMetrics():
    def __init__(self, gan, x_test):
        self.gan = gan
        self.x_test = x_test
        self.z_test = gan.prior(len(x_test))
        self.p_fake, self.fake_std = [], []
        self.p_real, self.real_std = [], []
        
    def track(self, *ignored_args, **ignored_kwargs):
        y_fake = self.gan.D(self.gan.G(self.z_test))
        y_real = self.gan.D(self.x_test)
        self.p_fake += [y_fake.mean()]
        self.p_real += [y_real.mean()]
        self.fake_std += [y_fake.std()]
        self.real_std += [y_real.std()]
        
    def plot(self, *ignored_args, **ignored_kwargs):
        epochs = range(len(self.p_fake))
        plt.errorbar(epochs, self.p_fake, self.fake_std, label='D(G(z))', color='blue')
        #plt.plot(epochs, np.array(self.p_fake) + np.array(self.fake_std), '--', color='blue')
        #plt.plot(epochs, np.array(self.p_fake) - np.array(self.fake_std), '--', color='blue')
        plt.errorbar(epochs, self.p_real, self.real_std, label='D(x)', color='orange', alpha=0.75)
        #plt.plot(epochs, np.array(self.p_real) + np.array(self.real_std), '--', color='orange')
        #plt.plot(epochs, np.array(self.p_real) - np.array(self.real_std), '--', color='orange')
        plt.title('P(real)')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()


def d_landscape(d, x, title=None):
    # make grid
    x0_min = x[:,0].min() - 0.1
    x0_max = x[:,0].max() + 0.1

    x1_min = x[:,1].min() - 0.1
    x1_max = x[:,1].max() + 0.1

    x0_ticks = np.arange(x0_min, x0_max+0.0025, 0.05)

    x1_ticks = np.arange(x1_min, x1_max+0.0025, 0.05)[::-1]
    grid = np.dstack(np.meshgrid(x0_ticks, x1_ticks))

    # plot the landscape
    plt.imshow(d(grid.reshape(-1, 2)).reshape(grid.shape[:-1]))

    # axis annotation
    xticks = [0, int(len(x0_ticks)/2), len(x0_ticks)-1]
    xlabels = [f"{x0_ticks[i]:.1f}" for i in xticks]

    yticks = [0, int(len(x1_ticks)/2), len(x1_ticks)-1]
    ylabels = [f"{x1_ticks[i]:.1f}" for i in yticks]

    plt.xticks(xticks, xlabels)
    plt.yticks(yticks, ylabels)

    plt.xlabel('x0')
    plt.ylabel('x1')
    
    if title is not None:
        plt.title(title)

    plt.colorbar()
    plt.show()


class DLandscapeCallback():
    def __init__(self, gan, data=x):
        self.gan = gan
        self.data = x
    
    def plot(self, *args, **kwargs):
        d_landscape(self.gan.D, np.concatenate([self.data,self.gan.G(self.gan.prior(400))], axis=0))
 


# ## Experiments
# Now we can do experiments.
# 
# 

# ### Initial Experiments: can G fit a fixed D, vice verca and can both fit our toy data
# Let's start with the following three:
# - train only G and see that it gets better at fooling D
# - train only D and see that it gets better at recognizing fakes and trues
# - train G and D and see if it learns our real distribution

def d_g_ratio_experiment(g_steps, d_steps, experiment_title, arch=flexibel_arch(512)):
    print("================================================")
    print(experiment_title)
    print("================================================")
    gan = GAN(uniform, Generator(arch), Discriminator(arch))
    d_landscape(gan.D, x, title=experiment_title+" D landscape before training")
    
    e = EvolvingCallback(gan)
    m = LossMetrics(gan, x)
    p = PMetrics(gan, x)
    
    callbacks = [
        (e.track, 1),
        (e.plot, 10),
        (DLandscapeCallback(gan).plot, 10),
        (p.track, 1),
        (p.plot, 10)
    ]
    gan.fit(x,
        epochs=100,
        file_prefix='models/2d_uniform',
        callbacks=callbacks,
        d_updates=d_steps,
        g_updates=g_steps
    )
    d_landscape(gan.D, x, title=experiment_title+" D landscape after training")


# This doesn't look good at all. We note a few things:
# - G is not able to fit D very well if we train only D
# - D is not able to fit a randomly initialized fix G and our distribution very well, if we train only D
# - If we train both together.
# 
# It seems that our mini architecture is not complex enough, but let's also see if D can fit our distribution if we use a generator that just maps the identity.
# 

# ### Train D with identity G


class IdentityG(Generator):
    def __init__(self, scale=1.5):
        inputs=keras.Input(shape=[2])
        self.scale=scale
        self.model = Model(inputs=inputs, outputs=self.forward(inputs))
        self.optimizer = keras.optimizers.Adam(0.0002, 0.5)
    
    def forward(self, z):
        return layers.Lambda(lambda x: x *self.scale)(z)
    
    
def train_d_against_identity_g(): 
    gan = GAN(uniform, IdentityG(), Discriminator(flexibel_arch(512)))
    
    d_landscape(gan.D, x, title="Big D landscape before training")

    callbacks = [
        (DLandscapeCallback(gan).plot, 10)
    ]
    gan.fit(x,
        epochs=100,
        file_prefix='models/2d_uniform',
        callbacks=callbacks,
        d_updates=1,
        g_updates=0
    )


# ### Compare to optimal D
# 
# We also know the optimal classifier and compare against it:


def optimal_d_for_uniform_fakes(x):
    y = np.zeros(len(x))
    for c in [[-1, -1], [-1, 1], [1, -1], [1, 1]]:
        y[np.linalg.norm(x-np.array([c]), axis=1)<0.2] = 1
    y *= 0.9208103130755064 # because there are some fakes in the true blops as well, we need to scale down
    return y

#d_landscape(optimal_d_for_uniform_fakes, x)

def evaluate_d_on_uniform(d, real):
    fake = np.random.uniform(low=real.min(), high=real.max(), size=real.shape)
    x = np.concatenate([real, fake], axis=0)
    y_pred = d(x)
    y_opti = optimal_d_for_uniform_fakes(x)
    print(f"D: avg real={y_pred[:len(fake)].mean():.2f}\n       fake=={y_pred[len(fake):].mean():.2f}")
    print(f"Opti   real={y_opti[:len(fake)].mean():.2f}\n       fake=={y_opti[len(fake):].mean():.2f}")


# D is not optimal but it can fit the problem reasonably. Maybe the result gets better when we pretrain D.

# ### Initial experiments with pretrained D
# 
# This still doesn't look like the true distribution, but it's much closer. So let's repeat the first experiment and pretrain D against a uniform distribution.


def pretrain_d(D, x):
    pretrain_gan = GAN(uniform, IdentityG(scale=3), D)
    pretrain_gan.fit(x,epochs=5,file_prefix='models/2d_uniform', g_updates=0, d_updates=1)
    



class GAN2(GAN):
    def get_mixed_batch(self, x_real):
        n = x_real.shape[0]
        z = self.prior(n)
        x_fake = self.G(z)
        x_mix = np.concatenate([x_fake, x_real], axis=0)
        x_uni = np.stack([
                    np.random.uniform(low=x_mix[:,0].min(), high=x_mix[:,0].max(), size=n),
                    np.random.uniform(low=x_mix[:,0].min(), high=x_mix[:,0].max(), size=n)],
                    axis=1)
        x_mix = np.concatenate([x_mix, x_uni], axis=0)
        y_target = np.concatenate([np.zeros(n), np.ones(n), np.zeros(n)], axis=0)
        return x_mix, y_target
    