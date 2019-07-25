from keras.models import Sequential, Model
from keras.layers import Input, Dense, LeakyReLU, BatchNormalization
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from random import shuffle

import tensorflow as tf
import os
import json
import time
import numpy as np

from distopia.app.agent import VoronoiAgent


# stealing from tensorflow cnngan tutorial and here: https://www.datacamp.com/community/tutorials/generative-adversarial-networks
class DistopiaGAN:
    num_districts = 8
    blocks_per_district = 2
    noise_dim = 8
    batch_size = 500
    state_shape = num_districts * blocks_per_district * 2
    padding = 100

    def __init__(self, data_path):

        self.voronoi = VoronoiAgent()
        self.voronoi.load_data()
        # self.trim_data(data_path)
        self.load_data(data_path, check=False)

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = Adam(1e-4)
        self.discriminator_optimizer = Adam(1e-4)
        self.build_generator()
        self.build_discriminator()
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

    def trim_data(self, path):
        real_samples = []
        valid_samples = []
        sampled = {}
        with open(path) as infile:
            real_samples = json.load(infile)
            counter = 0
            for i, sample in enumerate(real_samples):
                if self.check_validity(sample) == True:
                    counter += 1
                    valid_samples.append(sample)
                if i % 500 == 0:
                    print("Valid Samples: {} out of {}".format(counter, i))
            print("Valid Samples: {} out of {}".format(counter, len(real_samples)))
        with open("trimmed.json", "w+") as outfile:
            json.dump(valid_samples, outfile)

    def load_data(self, path, check=False):
        self.real_samples = []
        self.sampled = {}
        # with open(path) as infile:
        #     raw_data = json.load(infile)
        #     for sample in raw_data:
        #         abort = False
        #         if str(sample) in self.sampled:
        #             continue
        #         flattened = []
        #         for district in sample:
        #             if len(district) != 2:
        #                 abort = True
        #             else:
        #                 for block in district:
        #                     flattened += block
        #         if abort == False:
        #             self.sampled[str(sample)] = 1
        #             self.real_samples.append(flattened)
        with open(path) as infile:
            self.real_samples = json.load(infile)
            if check == True:
                counter = 0
                for sample in self.real_samples:
                    if self.check_validity(sample) != True:
                        counter += 1
                print(counter)
                assert counter == 0
        self.dataset = (
            []
        )  # tf.data.Dataset.from_tensor_slices(self.real_samples).shuffle(len(self.real_samples)).batch(self.batch_size)
        num_batches = len(self.real_samples) // self.batch_size + 1
        for i in range(num_batches - 1):
            self.dataset.append(
                self.real_samples[i * self.batch_size : (i + 1) * self.batch_size]
            )
        # add the last separately in case there is not enough
        self.dataset.append(self.real_samples[(num_batches - 1) * self.batch_size :])

    def construct_layout(self, block_locs):
        obs_dict = {}
        # added = {} taken out to allow for double blocks in gan--note that voronoi should fail so it should be fine
        for d in range(0, self.num_districts):
            obs_dict[d] = []
            for b in range(0, self.blocks_per_district):
                index = 2 * (d * self.blocks_per_district + b)
                coords = [
                    block_locs[index],
                    block_locs[index + 1],
                ]  # already in pixel space
                if (
                    block_locs[index] > self.padding
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

    def check_validity(self, layout):
        layout_dict = self.construct_layout(layout)
        districts = self.voronoi.get_voronoi_districts(layout_dict)
        if len(districts) < self.num_districts:
            return False
        try:
            state_metrics, district_metrics = self.voronoi.compute_voronoi_metrics(
                districts
            )

        except Exception as e:
            print("Couldn't compute Voronoi for {}:{}".format(districts, e))
            return False
        # try:
        #     objectives = self.extract_objectives(district_metrics)
        #     #print("{}:{}".format(self.n_calls,cost))
        # except ValueError as v:
        #     print("Problem calculating the metrics: {}".format(v))
        #     return False
        return True

    def build_generator(self):
        self.generator = Sequential()
        self.generator.add(Dense(8, input_dim=self.noise_dim))
        self.generator.add(LeakyReLU(alpha=0.2))
        self.generator.add(BatchNormalization(momentum=0.8))
        self.generator.add(Dense(16))
        self.generator.add(LeakyReLU(alpha=0.2))
        self.generator.add(BatchNormalization(momentum=0.8))
        self.generator.add(Dense(32))
        self.generator.add(LeakyReLU(alpha=0.2))
        self.generator.add(BatchNormalization(momentum=0.8))
        self.generator.add(Dense(self.state_shape, activation="tanh"))
        self.generator.compile(
            loss="binary_crossentropy", optimizer=self.generator_optimizer
        )
        self.generator.summary()

    def build_discriminator(self):
        self.discriminator = Sequential()
        self.discriminator.add(Dense(32, input_dim=self.state_shape))
        self.discriminator.add(LeakyReLU(alpha=0.2))
        self.discriminator.add(Dense(16))
        self.discriminator.add(LeakyReLU(alpha=0.2))
        self.discriminator.add(Dense(8))
        self.discriminator.add(LeakyReLU(alpha=0.2))
        self.discriminator.add(Dense(1, activation="sigmoid"))
        self.discriminator.compile(
            loss="binary_crossentropy", optimizer=self.discriminator_optimizer
        )
        self.discriminator.summary()

    def generator_loss(fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def train_step(self, real_layouts):
        self.discriminator.trainable = False
        noise = np.random.normal(0, 1, size=[self.batch_size, self.noise_dim])
        generated_layouts = self.generator.predict(noise)
        X = np.concatenate([real_layouts, generated_layouts])
        # Labels for generated and real data
        y_dis = np.zeros(len(real_layouts) + self.batch_size)
        # One-sided label smoothing
        y_dis[: len(real_layouts)] = 0.9

        # Train discriminator
        self.discriminator.trainable = True

        dis_loss = self.discriminator.train_on_batch(X, y_dis)

        # Train generator
        noise = np.random.normal(0, 1, size=[self.batch_size, self.noise_dim])
        y_gen = np.ones(self.batch_size)
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

    def train(self, epochs):
        logfile = open("generator_log.csv", "w+")
        for epoch in range(epochs):
            start = time.time()
            shuffle(self.dataset)

            for image_batch in self.dataset:
                dl, ganl = self.train_step(image_batch)

            noise = np.random.normal(0, 1, size=[self.batch_size, self.noise_dim])
            generated_layouts = self.generator.predict(noise)
            valid_count = 0
            for layout in generated_layouts:
                if self.check_validity(layout) == True:
                    valid_count += 1
            gl = valid_count / len(generated_layouts)

            # # Produce images for the GIF as we go
            # display.clear_output(wait=True)
            # generate_and_save_images(generator,
            #                         epoch + 1,
            #                         seed)

            # Save the model every 15 epochs
            # if (epoch + 1) % 15 == 0:
            #     self.checkpoint.save(file_prefix = checkpoint_prefix)

            print("Time for epoch {} is {} sec".format(epoch + 1, time.time() - start))
            print(
                "Final Discriminator Loss: {}, Final Generator Loss: {}, Final GAN Loss: {}".format(
                    dl, gl, ganl
                )
            )
            logfile.write("{},{},{}\n".format(dl, gl, ganl))
        # # Generate after the final epoch
        # display.clear_output(wait=True)
        # self.generate_and_save_images(generator,
        #                         epochs,
        #                         seed)
        logfile.close()

    # def generate_and_save_images(self,generator,epochs,seed):
    #     ...


if __name__ == "__main__":
    gan = DistopiaGAN("/home/dev/scratch/gym-distopia/generator/trimmed.json")
    gan.train(10000)

