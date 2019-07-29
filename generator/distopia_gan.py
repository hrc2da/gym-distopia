from keras.models import Sequential, Model
from keras.layers import Input, Dense, LeakyReLU, BatchNormalization
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.models import load_model
from random import shuffle

import tensorflow as tf
import os
import json
import time
import numpy as np

from threading import Thread, Lock

from distopia.app.agent import VoronoiAgent

#TODO:
'''
* try flipping 1 and 0 labels
* try normalizing inputs to -1,1 and using tanh instead of sigmoid
* dump the gradients so we get an idea of how it's learning
* try deeper hidden arch or wider input/latent space
* thread the generator loss/validity check and logfile(s) dumping
'''

# stealing from tensorflow cnngan tutorial and here: https://www.datacamp.com/community/tutorials/generative-adversarial-networks
class DistopiaGAN:
    num_districts = 8
    blocks_per_district = 2
    noise_dim = 32
    batch_size = 500
    state_shape = num_districts * blocks_per_district * 2
    padding = 100
    num_precincts = 72
    def __init__(self, data_path,verbose=True,g_model=None,d_model=None):
        self.logfile_lock = Lock()
        self.verbose = verbose
        self.voronoi = VoronoiAgent()
        self.voronoi.load_data()
        self.failed_voronoi = []
        self.missing_districts = []
        self.empty_district = []
        self.missing_precinct = []
        if data_path is None:

            return
        # self.trim_data(data_path)
        self.load_data(data_path, check=False)

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = Adam(1e-4)
        self.discriminator_optimizer = Adam(1e-4)
        self.build_generator(g_model)
        self.build_discriminator(d_model)
        self.discriminator.trainable = False

        # now setup the gan
        gan_input = Input(shape=(self.noise_dim,))
        x = self.generator(gan_input)
        gan_output = self.discriminator(x)
        self.gan = Model(inputs=gan_input, outputs=gan_output)
        self.gan.compile(loss="binary_crossentropy", optimizer=self.generator_optimizer)
        # checkpoint_dir = './training_checkpoints'
        # checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        # self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
        #                                 discriminator_optimizer=self.discriminator_optimizer,
        #                                 generator=self.generator,
        #                                 discriminator=self.discriminator)

    def setup_normalization(self,x_max,y_max,nrange=[0,1]):
        self.x_max = x_max
        self.y_max = y_max
        self.n_magnitude = nrange[1]-nrange[0]
        target_center = nrange[0] + magnitude/2
        actual_center = magnitude/2
        self.n_shift = target_center - actual_center

    def normalize(self,coord):
        normalized_x = (coord[0]/self.x_max) * self.magnitude + self.shift
        normalized_y = (coord[1]/self.y_max) * self.magnitude + self.shift
        return (normalized_x,normalized_y)

    def denormalize(self,coord):
        denormalized_x = (coord[0] + self.shift) / self.magnitude * self.x_max
        denormalized_y = (coord[1] + self.shift) / self.magnitude * self.y_max
        return (denormalized_x,denormalized_y)

    def trim_data(self, path):
        real_samples = []
        valid_samples = []
        sampled = {}
        self.failed_voronoi = []
        self.missing_districts = []
        self.empty_district = []
        self.missing_precinct = []
        with open(path) as infile:
            real_samples = json.load(infile)
            counter = 0
            for i, sample in enumerate(real_samples):
                if self.check_validity(sample) == True:
                    counter += 1
                    valid_samples.append(sample)
                if i % 500 == 0:
                    print("Valid Samples: {} out of {}".format(counter, i))
                    print("Failed Voronoi: {}, Missing Districts: {}, Empty District: {}, Missing Precinct: {}".format(len(self.failed_voronoi),len(self.missing_districts),len(self.empty_district), len(self.missing_precinct)))
            print("Valid Samples: {} out of {}".format(counter, len(real_samples)))
            print("Failed Voronoi: {}, Missing Districts: {}, Empty District: {}, Missing Precinct: {}".format(len(self.failed_voronoi),len(self.missing_districts),len(self.empty_district), len(self.missing_precinct)))
        with open("trimmed3.json", "w+") as outfile:
            json.dump(valid_samples, outfile)

        with open("failures.json", "w+") as outfile:
            json.dump({'failed_voronoi': self.failed_voronoi,
                        'missing_districts': self.missing_districts,
                        'empty_district': self.empty_district,
                        'missing_precinct': self.missing_precinct}, outfile)

    def load_data(self, path, check=False):
        #self.real_samples = []
        #self.sampled = {}
        # with open(path) as infile:
        #     self.real_samples = json.load(infile)
        #     if check == True:
        #         counter = 0
        #         for sample in self.real_samples:
        #             if self.check_validity(sample) != True:
        #                 counter += 1
        #         print(counter)
        #         assert counter == 0
        # self.dataset = (
        #     []
        # )  # tf.data.Dataset.from_tensor_slices(self.real_samples).shuffle(len(self.real_samples)).batch(self.batch_size)
        data = []
        with open(path) as infile:
            data = np.array(json.load(infile))
        num_samples, _ = data.shape
        data = data.reshape(num_samples, 16, 2)
        # normalize here
        data[:, :, 0] /= 1920
        data[:, :, 0] *= 2 
        data[:, :, 0] -= 1       # *2 -1 to get to -1,1
        data[:, :, 1] /= 1080 
        data[:, :, 1] *= 2
        data[:, :, 1] -= 1
        self.scalar = (1920,1080)
        data = data.reshape(num_samples, 32)
        num_batches = num_samples // self.batch_size + 1
        #elf.real_samples = data
        self.dataset = []
        for i in range(num_batches - 1):
            self.dataset.append(
                data[i * self.batch_size : (i + 1) * self.batch_size]
            )
        # add the last separately in case there is not enough
        self.dataset.append(data[(num_batches - 1) * self.batch_size :])

    def construct_layout(self, block_locs, scalar=1):
        '''
            input: [x0,y0,x1,y1,....,xn,yn]

                    scalar: if even scale, just a scalar, if x/y, then (x,y) tuple
                    for example: if in grid space, we are going to scale up to pixels, so scalar=grid_width
                        if in pixels, no scale
                        if in normalized 0-1: scale is (1920,1080)
            
            output is in pixels w/ staging area blocks removed:
            {0:[[x0*scale_x,y0*scale_y],[x1*scale_x,y1*scale_y]], 1:[[x3*scale_x,y3*scale_y]]...,7:[[xn*scale_x,yn*scale_y]]}
        '''
        if type(scalar) == list or type(scalar) == tuple or type(scalar) == np.ndarray:
            scale_x = scalar[0]
            scale_y = scalar[1]
        else:
            scale_x = scale_y = scalar
        obs_dict = {}
        # added = {} taken out to allow for double blocks in gan--note that voronoi should fail so it should be fine
        for d in range(0, self.num_districts):
            obs_dict[d] = []
            for b in range(0, self.blocks_per_district):
                index = 2 * (d * self.blocks_per_district + b)
                coords = [
                    (block_locs[index]+1)*0.5*scale_x, #+1 * 0.5 to rescale from -1,1 to 0,1 then to pixls
                    (block_locs[index + 1]+1)*0.5*scale_y,
                ]  # already in pixel space
                if (
                    (block_locs[index]+1)*0.5*scale_x > self.padding #padding is in pixel scale
                ):  # if the x is far enough to the right
                    obs_dict[d].append(coords)
                # assert self.hash_loc(coords) not in added # just double check to ensure we aren't passing two blocks in same loc
                # added[self.hash_loc(coords)] = (d,b)
        
        return obs_dict

    def hash_loc(self, loc):
        loc_type = type(loc)
        if loc_type is list:
            return str(tuple(loc))
        elif loc_type is tuple:
            return str(loc)
        elif loc_type is np.ndarray:
            return str(tuple(loc))
        else:
            raise TypeError("Location should be a tuple or a list or a numpy array")

    def vprint(self,string):
        if self.verbose == True:
            print(string)

    def check_validity(self, layout, scalar=None, bypass=[]):

        if scalar is None:
            if hasattr(self, "scalar"):
                scalar = self.scalar
            else:
                scalar = 1
        
        layout_dict = self.construct_layout(layout, scalar)
        districts = self.voronoi.get_voronoi_districts(layout_dict)
        #import pdb; pdb.set_trace()
        if len(districts) == 0 and 'empty' not in bypass:
            self.vprint("zero districts! probably voronoi falure!")
            self.failed_voronoi.append(layout)
            return False
        if len(districts) < self.num_districts and 'missing_districts' not in bypass:

            self.vprint("missing districts! less than 8 districts found!")
            self.missing_districts.append(layout)
            return False
        assigned_precincts = 0
        for i,d in enumerate(districts):
            if len(d.precincts) < 1:
                self.vprint("empty district {} (1-indexed)!".format(i+1))
                self.empty_district.append(layout)
                return False
            assigned_precincts += len(d.precincts)
        
        if assigned_precincts < self.num_precincts:
            self.vprint("missing precincts! only assigned {}.".format(assigned_precincts))
            self.missing_precinct.append(layout)
            return False
        try:
            state_metrics, district_metrics = self.voronoi.compute_voronoi_metrics(
                districts
            )

        except Exception as e:
            self.vprint("Couldn't compute Voronoi for {}:{}".format(districts, e))
            return False


        # try:
        #     objectives = self.extract_objectives(district_metrics)
        #     #print("{}:{}".format(self.n_calls,cost))
        # except ValueError as v:
        #     print("Problem calculating the metrics: {}".format(v))
        #     return False
        return True

    def build_generator(self,model):
        if model is None:
            self.generator = Sequential()
            self.generator.add(Dense(64, input_dim=self.noise_dim))
            self.generator.add(BatchNormalization(momentum=0.8))
            self.generator.add(LeakyReLU(alpha=0.2))
            self.generator.add(Dense(self.state_shape, activation="tanh"))

            self.generator.compile(
                loss="binary_crossentropy", optimizer=self.generator_optimizer
            )
        else:
            self.generator = load_model(model)
        self.generator.summary()

    def build_discriminator(self,model):
        if model is None:
            self.discriminator = Sequential()
            self.discriminator.add(Dense(128, input_dim=self.state_shape))
            self.discriminator.add(LeakyReLU(alpha=0.2))
            self.discriminator.add(Dense(64))
            self.discriminator.add(LeakyReLU(alpha=0.2))
            self.discriminator.add(Dense(1, activation="sigmoid"))
            self.discriminator.compile(
                loss="binary_crossentropy", optimizer=self.discriminator_optimizer
            )
        else:
            self.discriminator = load_model(model)
        self.discriminator.summary()

    # def generator_loss(fake_output):
    #     return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    # def discriminator_loss(real_output, fake_output):
    #     real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
    #     fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
    #     total_loss = real_loss + fake_loss
    #     return total_loss

    def train_step(self, real_layouts):
        self.discriminator.trainable = False
        noise = np.random.normal(0, 1, size=[self.batch_size, self.noise_dim]) # 0 mean, std 1 seed
        generated_layouts = self.generator.predict(noise)

       
        X = np.concatenate([real_layouts, generated_layouts])
        # Labels for generated and real data
        y_dis = np.zeros(len(real_layouts) + self.batch_size)
        assert len(y_dis) == len(real_layouts) + self.batch_size
        # One-sided label smoothing
        # y_dis[: len(real_layouts)] = np.random.uniform(low=0.7,high=1.2, size=(len(real_layouts),))
        # y_dis[len(real_layouts):] = np.random.uniform(low = 0, high=0.3, size=(self.batch_size,))
        # try flipping 0 and 1...
        y_dis[: len(real_layouts)] = np.random.uniform(low = 0, high=0.3, size=(len(real_layouts),))
        y_dis[len(real_layouts):] = np.random.uniform(low=0.7,high=1.0, size=(self.batch_size,))

        # Train discriminator
        self.discriminator.trainable = True

        dis_loss = self.discriminator.train_on_batch(X, y_dis)

        # Train generator
        noise = np.random.normal(0, 1, size=[self.batch_size, self.noise_dim])
        #y_gen = np.ones(self.batch_size)
        y_gen = np.zeros(self.batch_size)
        self.discriminator.trainable = False

        gen_loss = self.gan.train_on_batch(noise, y_gen)

        # with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        #     generated_layouts = self.generator(noise)

        #     real_output = discriminator(real_layouts)
        #     fake_output = discriminator(generated_layouts)

        #     gen_loss = generator_loss(fake_output)
        #     disc_loss = discriminator_loss(real_output, fake_output)

        #     gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        #     gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        #     self.generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        #     self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        return dis_loss, gen_loss

    def test_gen_validity(self,generated_layouts):
        valid_count = 0
        i=0
        for layout in generated_layouts:
            if self.check_validity(layout) == True:
                valid_count += 1
            i+= 1
            print(i)
        return valid_count / len(generated_layouts)
    def log_stats(self,epoch,dl,ganl,generated_layouts,logfile):
        print("Logging stats for epoch {}".format(epoch))
        gl = self.test_gen_validity(generated_layouts)
        print(
                "Final Discriminator Loss: {}, Final Generator Loss: {}, Final GAN Loss: {}".format(
                    dl, gl, ganl
                )
            )
        self.logfile_lock.acquire()
        logfile.write("{},{},{},{}\n".format(epoch,dl, gl, ganl))
        logfile.flush() # force a buffer write on each epoch
        self.logfile_lock.release()

    def log_basic_stats(self,epoch,dl,ganl,logfile):
        '''
        Only log the losses, don't test the generator's outputs
        '''
        print(
            "Final Discriminator Loss: {}, Final GAN Loss: {}".format(
                dl, ganl
            )
        )
        logfile.write("{},{},{}\n".format(epoch, dl, ganl))
        logfile.flush() # force a buffer write on each epoch

    def train(self, epochs):
        logfile = open("generator_log_threaded.csv", "w+")
        for epoch in range(epochs):
            print("starting epoch {}".format(epoch))
            start = time.time()
            shuffle(self.dataset)

            for image_batch in self.dataset:
                dl, ganl = self.train_step(image_batch)

            #noise = np.random.normal(0, 1, size=[self.batch_size, self.noise_dim])
            #generated_layouts = self.generator.predict(noise)
            self.log_basic_stats(epoch,dl,ganl,logfile)
            #Thread(target=self.log_stats,args=(epoch,dl,ganl,generated_layouts,logfile)).start() #hm, I probably need to lock the logfile
            # valid_count = 0
            # for layout in generated_layouts:
            #     if self.check_validity(layout) == True:
            #         valid_count += 1
            # gl = valid_count / len(generated_layouts)

            # # Produce images for the GIF as we go
            # display.clear_output(wait=True)
            # generate_and_save_images(generator,
            #                         epoch + 1,
            #                         seed)

            # Save the model every 15 epochs
            # if (epoch + 1) % 15 == 0:
            #     self.checkpoint.save(file_prefix = checkpoint_prefix)

            print("Time for epoch {} is {} sec".format(epoch + 1, time.time() - start))
            # print(
            #     "Final Discriminator Loss: {}, Final Generator Loss: {}, Final GAN Loss: {}".format(
            #         dl, gl, ganl
            #     )
            # )
            self.generator.save("generator_"+str(epoch)+"threaded.h5")
            self.discriminator.save("discriminator_"+str(epoch)+"threaded.h5")
            # logfile.write("{},{},{}\n".format(dl, gl, ganl))
            # logfile.flush() # force a buffer write on each epoch
        # # Generate after the final epoch
        # display.clear_output(wait=True)
        # self.generate_and_save_images(generator,
        #                         epochs,
        #                         seed)
        logfile.close()

    # def generate_and_save_images(self,generator,epochs,seed):
    #     ...


if __name__ == "__main__":
    gan = DistopiaGAN("/home/dev/research/distopia/gym-distopia/generator/trimmed3.json",verbose=False)#,
                    # g_model='/home/dev/research/distopia/gym-distopia/generator/run2/generator_9999threaded.h5',
                    # d_model='/home/dev/research/distopia/gym-distopia/generator/run2/discriminator_9999threaded.h5',)
    #gan.trim_data("/home/dev/research/distopia/gym-distopia/generator/data/trimmed.json")
    gan.train(10000)

