import numpy as np
import gym_distopia
import gym
from gym import wrappers

from keras.models import Sequential, model_from_yaml
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents import SARSAAgent
from rl.policy import BoltzmannQPolicy, GreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger

import os
import time

from rl.core import Processor

# class PatchedGreedyQPolicy(GreedyQPolicy):
#     '''
#         Monkey-patching to allow for multi-discrete action space (have to convert from scalar to array of scalars)
#     '''
#     def __init__(self, num_actions=0, num_blocks=0):
#         super(PatchedGreedyQPolicy, self).__init__()
#         if num_actions == 0 or num_blocks == 0:
#             raise ValueError("Need to pass info about actions and blocks")
#         self.num_actions = num_actions
#         self.num_blocks = num_blocks
#     def action2multidiscrete(self, action):
#         # takes a scalar, converts it to a multidiscrete
#         # where there are num_actions for each of num_blocks
#         # that is, the actions are arranged [b0N, b0S, b0E, b0W, b1N, b1S, b1E, b1W, etc.]
#         mdaction = []
#         for i in range(self.num_blocks):
#             mdaction.append(0)
#         # to figure out which block it is, divide by the number of actions
#         block_idx = action // self.num_actions
#         block_action = action % self.num_actions
#         mdaction[block_idx] = block_action
#         return mdaction
        
#     def select_action(self, q_values):
#         action = super(PatchedGreedyQPolicy, self).select_action(q_values)
#         # this gives me a scalar
#         return self.action2multidiscrete(action)  


# class PatchedBoltzmannQPolicy(BoltzmannQPolicy):
#     '''
#         Monkey-patching to allow for multi-discrete action space (have to convert from scalar to array of scalars)
#     '''
#     def __init__(self, tau=1., clip=(-500., 500.), num_actions=0, num_blocks=0):
#         super(PatchedBoltzmannQPolicy, self).__init__(tau,clip)
#         if num_actions == 0 or num_blocks == 0:
#             raise ValueError("Need to pass info about actions and blocks")
#         self.num_actions = num_actions
#         self.num_blocks = num_blocks
#     def action2multidiscrete(self, action):
#         # takes a scalar, converts it to a multidiscrete
#         # where there are num_actions for each of num_blocks
#         # that is, the actions are arranged [b0N, b0S, b0E, b0W, b1N, b1S, b1E, b1W, etc.]
#         mdaction = []
#         for i in range(self.num_blocks):
#             mdaction.append(0)
#         # to figure out which block it is, divide by the number of actions
#         block_idx = action // self.num_actions
#         block_action = action % self.num_actions
#         mdaction[block_idx] = block_action
#         return mdaction
        
#     def select_action(self, q_values):
#         action = super(PatchedBoltzmannQPolicy, self).select_action(q_values)
#         # this gives me a scalar
#         return self.action2multidiscrete(action)


class DistopiaProcessor(Processor):
    '''
        Converts actions from a scalar to a multidiscrete for use in the Distopia env
    '''
    def __init__(self,num_blocks,num_actions):
        self.num_blocks = num_blocks
        self.num_actions = num_actions
        super(DistopiaProcessor,self).__init__()

    def process_action(self,action):
        # takes a scalar, converts it to a multidiscrete
        # where there are num_actions for each of num_blocks
        # that is, the actions are arranged [b0N, b0S, b0E, b0W, b1N, b1S, b1E, b1W, etc.]
        mdaction = []
        for i in range(self.num_blocks):
            mdaction.append(0)
        # to figure out which block it is, divide by the number of actions
        block_idx = action // self.num_actions
        block_action = action % self.num_actions
        mdaction[block_idx] = block_action
        return mdaction

    '''
        do I need to process the observations? please check...
    '''


# from the keras_rl docs

class DistopiaSARSA:
   
    def __init__(self,env_name='distopia-initial4-v0',in_path=None,out_path=None,terminate_on_fail=False,reconstruct=False):
        self.ENV_NAME = env_name
        self.filename = self.ENV_NAME
        self.init_paths(in_path,out_path)
        self.init_env(terminate_on_fail)
        self.init_model(reconstruct)
        self.compile_agent()

    def init_paths(self, in_path, out_path):
        self.in_path = in_path #if self.in_path != None else './'
        self.out_path = out_path if out_path != None else './'
        self.log_path = "./logs/{}".format(time.time())
        os.mkdir(self.log_path)

    def init_env(self,terminate_on_fail):
        self.env = gym.make(self.ENV_NAME)
        self.env.terminate_on_fail = terminate_on_fail
        self.env.record_path = "{}/ep_".format(self.log_path)
        self.env = gym.wrappers.Monitor(self.env, "recording", force=True)
        np.random.seed(234)
        self.env.seed(234)
        self.nb_actions = np.sum(self.env.action_space.nvec)
        self.num_actions = self.env.NUM_DIRECTIONS
        self.num_blocks = self.env.NUM_DISTRICTS * self.env.BLOCKS_PER_DISTRICT


    def init_model(self, reconstruct=False):
        if self.in_path != None:
            if reconstruct == True:
                self.construct_model()
            else:
                yaml_file = open("{}/{}.yaml".format(self.in_path,self.filename), 'r')
                model_yaml = yaml_file.read()
                yaml_file.close()
                self.model = model_from_yaml(model_yaml)
            self.model.load_weights("{}/{}.h5".format(self.in_path,self.filename))
        else:
        # Next, we build a very simple model.
            self.construct_model()
        self.save_model()
        print(self.model.summary())

    def construct_model(self):
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(1,) + self.env.observation_space.shape))
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        # self.model.add(Dense(16))
        # self.model.add(Activation('relu'))
        self.model.add(Dense(self.nb_actions))
        self.model.add(Activation('linear'))

    def save_model(self):
        if self.out_path != None:
            with open(self.filename+".yaml", 'w+') as yaml_file:
                yaml_file.write(self.model.to_yaml())
            self.model.save_weights('{}/{}.h5'.format(self.out_path, self.ENV_NAME))

    def compile_agent(self):
        # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # even the metrics!
        processor = DistopiaProcessor(self.num_blocks,self.num_actions)
        #memory = SequentialMemory(limit=50000, window_length=1)
        #policy = PatchedBoltzmannQPolicy(num_actions = self.num_actions, num_blocks = self.num_blocks)
        #test_policy = PatchedGreedyQPolicy(num_actions = self.num_actions, num_blocks = self.num_blocks)
        policy = BoltzmannQPolicy()
        test_policy = GreedyQPolicy()
        self.sarsa = SARSAAgent(model=self.model, processor=processor, nb_actions=self.nb_actions, nb_steps_warmup=1000, 
                    policy=policy, test_policy=test_policy, gamma = 0.9)
        self.sarsa.compile(Adam(lr=1e-3), metrics=['mae'])

    def train(self, max_steps = 100, episodes = 100):
        # Okay, now it's time to learn something! We visualize the training here for show, but this
        # slows down training quite a lot. You can always safely abort the training prematurely using
        # Ctrl + C.
        self.env._max_steps = max_steps
        #for i in range(episodes):
        self.env.current_step = 0
        n_steps = max_steps*episodes
        logger = FileLogger(filepath='{}/{}.json'.format(self.out_path, self.ENV_NAME))
        self.sarsa.fit(self.env, nb_steps = n_steps, nb_max_episode_steps=max_steps, visualize=False, verbose=1, callbacks=[logger])
        #self.env.reset()
        
        # After episode is done, we save the final weights.
        self.sarsa.save_weights('{}/{}.h5'.format(self.out_path, self.ENV_NAME), overwrite=True)

    def test(self):
        # Finally, evaluate our algorithm for 5 episodes.
        self.sarsa.test(self.env, nb_episodes=5, nb_max_start_steps=0, visualize=True)



if __name__ == '__main__':
    d = DistopiaSARSA(reconstruct=True,terminate_on_fail=True)
    #lsd.dqn.load_weights('{}/{}.h5'.format(d.out_path,d.ENV_NAME))
    d.train(episodes=1000)
    #d.test()
