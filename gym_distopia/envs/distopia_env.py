import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_distopia.envs.distopia_rewards import PopulationStdEvaluator
from copy import deepcopy
import numpy as np

from matplotlib import pyplot as plt



class DistopiaEnv(gym.Env):
    """
    The main OpenAI Gym class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.
    The main API methods that users of this class need to know are:
        step
        reset
        render
        close
        seed
    And set the following attributes:
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards
    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.
    The methods are accessed publicly as "step", "reset", etc.. The
    non-underscored versions are wrapper methods to which we may add
    functionality over time.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, screen_size, reward_evaluator, num_districts = 4, blocks_per_district = 3,
                    grid_width = 50, raw_padding = 100, num_directions = 5, step_size = 1, 
                    init_state = None, skip_first_reset = False, always_reset_to_initial = False):
        '''
        OBSERVATION (STATE) SPACE:
        The state space for this environment is the x,y coordinates for each block
        There are up to BLOCKS_PER_DISTRICT blocks for each district
        The structure of the space is a TUPLE of districts, each composed of a TUPLE of blocks,
        each composed of an (x,y) tuple

        A note about the blocks: while all the blocks are represented in the space, not all must 
        be in the config. To accomplish this, there is a space of 100 px (by default) padding in x.
        When evaluating a config, we first subtract 100 px and strip out all blocks with a negative x.

        A note about evalutaion: the configs are passed to the evaluator as a dictionary of block coords
        keyed on the fiducial id

        REWARD FUNCTION:
        The reward function can either be a function or a list of supported aggregate stats ("sum,avg,std"). It cannot be None.

        ACTION SPACE
        The action space is up/down/left/right for each block. Note that currently (as a MultiDiscrete)
        the agent can move more than one block at a time.

        '''
        super().__init__()
        self.STEP_SIZE = step_size
        self.NUM_DISTRICTS = num_districts
        self.BLOCKS_PER_DISTRICT = blocks_per_district
        self.GRID_WIDTH = grid_width # width of a grid in pixels
        self.PADDING = raw_padding//self.GRID_WIDTH # px reserved for the "off-screen" blocks
        self.NUM_DIRECTIONS = num_directions # 0,1,2,3,4 --> Do nothing, N, S, E, W
        
        self.current_step = 0

        if reward_evaluator is None:
            raise ValueError("Must pass a reward evaluator instance! See distopia_rewards.py.")
        self.evaluator = reward_evaluator
        self.width, self.height = (dim//self.GRID_WIDTH for dim in screen_size)
        
        
        self.action_space = spaces.MultiDiscrete([self.NUM_DIRECTIONS]*self.NUM_DISTRICTS * self.BLOCKS_PER_DISTRICT)
        
        
        
        # the state space is the x,y coords of all blocks i.e. (x0,y0,x1,y1,x2,y2...)
        self.observation_space = spaces.Tuple(
            # list of districts
            [spaces.Tuple(
                # list of blocks
                [spaces.Tuple((spaces.Discrete(self.width + self.PADDING),spaces.Discrete(self.height))) 
                    for block in range(self.BLOCKS_PER_DISTRICT)]) 
            for district in range(self.NUM_DISTRICTS)])
        self.observation_space.shape = (self.NUM_DISTRICTS,self.BLOCKS_PER_DISTRICT,2) 
        self.reward_range = self.evaluator.calculate_reward_range({'NUM_DISTRICTS':self.NUM_DISTRICTS})
        if init_state is not None:
            skip_first_reset = True
        self.skip_reset = False
        self.always_reset_to_initial = always_reset_to_initial
        self.reset(initial=init_state, skip_next_reset=skip_first_reset) #passing skip flag to prevent auto-reset at the start of most agent episodes
        
    def get_staged_blocks_dict(self,observation):
        obs_dict = {}
        for i,district in enumerate(observation):
            obs_dict[i] = [block*self.GRID_WIDTH for block in district if block[0] > self.PADDING]
        return obs_dict

    def evaluate(self,observation):
        obs_dict = self.get_staged_blocks_dict(observation)
        return self.evaluator.evaluate(obs_dict)

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        self.last_action = action
        done = False
        info = {"error":""}
        self.current_step += 1
        self.prior_state = deepcopy(self.districts)
        observation, success = self._take_action(action)
        reward = self.evaluate(observation)
        if reward == False:
            #self.reset(initial=self.prior_state) # IMPORTANT: I think reverting the state will cause me to get stuck (since we're only taking 1-step; alternatively we could force starting from a legal state)
            reward = self.reward_range[0]
            info["error"] = "illegal voronoi"
        
        return (observation, reward, done, info)

    def _next_observation(self):
        return self.districts

    def _update_state(self,district,block,old_loc,new_loc):
        self.districts[district][block] = new_loc
        self.occupied[str(new_loc)] = (district,block)
        if old_loc is not None: # note that is ! the same as == here; == tries to do a value compare, is does identity compare, so == fails b/c np array has no truth value
            self.occupied.pop(str(old_loc))
    

    def _apply_action(self, district, block, action):
        '''
        Inputs:
            district: the district of the block to apply the action to
            block: the block in that district to apply the action to
            action: a number from {0,1,2,3,4} representing the action
        Updates:
            self.districts if possible to apply the action legally
        Returns:
            True if action succeeded, False if it failed

        '''
        success = False
        block_location = self.districts[district][block]
        if action == 0:
            # do nothing
            return True

        elif action == 1:
            # move north
            new_loc = block_location + (0, self.STEP_SIZE)
            if new_loc[1] > self.height or str(new_loc) in self.occupied:
                return False
            else:
                self._update_state(district,block,block_location,new_loc)
                return True

        elif action == 2:
            # move south
            new_loc = block_location + (0, -self.STEP_SIZE)
            if new_loc[1] < 0 or str(new_loc) in self.occupied:
                return False
            else:
                self._update_state(district,block,block_location,new_loc)
                return True

        elif action == 3:
            # move east
            new_loc = block_location + (self.STEP_SIZE, 0)
            if new_loc[0] > self.width or str(new_loc) in self.occupied:
                return False
            else:
                self._update_state(district,block,block_location,new_loc)
                return True

        elif action == 4:
            # move west
            new_loc = block_location + (-self.STEP_SIZE, 0)
            if new_loc[0] < 0 or str(new_loc) in self.occupied:
                return False
            else:
                self._update_state(district,block,block_location,new_loc)
                return True

        else:
            raise ValueError("Invalid action requested: {}. Please pass actions between 0 and {}.".format(action,self.NUM_DIRECTIONS-1))

    
    def _take_action(self, action):
        '''
        Inputs:
            action: an array of numbers between 0 and the number of moves
            the array is flattened (num_districts * num_blocks per district)
        Returns:
            the current block layout after taking the action
            a boolean list parallel to the input, indicating which succeeded and which failed
        
        Note that some actions may make other illegal. This function will perform the actions in
        random order in an attempt to reduce bias in which actions get nullified by prior actions.
        '''
        # generate a random order in which to execute the actions
        indices = np.arange(len(action))
        np.random.shuffle(indices)
        successes = []
        # go through the actions, unflattening and applying each one if legal
        for index in indices:
            district = index // self.BLOCKS_PER_DISTRICT
            block = index % self.BLOCKS_PER_DISTRICT
            successes.append(self._apply_action(district,block,action[index]))
        
        return self.districts, successes

    def place_block(self,occupied,active=None):
        '''
        Get unoccupied x,y coords to place a block on the active zone
        of the table, non-active zone, or neither
        Inputs:
            occupied: a dict containing occupied coords as keys
            active: True to force the placement in the active area
                    False to force the placement in the PADDING area
                    None to not force either
        Returns:
            a numpy array x,y for a block
        '''
        xlow = self.PADDING if (active == True) else 0
        xhigh = self.PADDING if (active == False) else self.PADDING + self.width
        ylow = 0
        yhigh = self.height
        x = np.random.randint(low=xlow, high=xhigh)
        y = np.random.randint(low=ylow, high=yhigh)
        while str(np.asarray([x,y])) in occupied:
            x = np.random.randint(low=xlow, high=xhigh)
            y = np.random.randint(low=ylow, high=yhigh)
        return np.asarray([x,y])
  
    def reset(self, initial=None, min_active=1, max_active=None, skip_next_reset = False):
        """
        Resets the state of the environment and returns an initial observation.
            initial: a complete inital block layout
            min_active: the min number of blocks that should be in the active area for each district
            max_active: the max number of blocks that can be in the active area for each district
        Returns: 
            observation (object): the initial observation.
        """
        if self.skip_reset == True:
            self.skip_reset = False
            return self.districts
        if skip_next_reset == True:
            self.skip_reset = True
        if initial is not None:
            self.districts = []
            for d in initial:
                new_district = [np.asarray(b) for b in d]
                self.districts.append(new_district)
            self.occupied = {}
            for district in self.districts:
                for block in district:
                    self.occupied[str(block)] = [district,block]
            self.initial_state = deepcopy(self.districts)
            self.initial_occupancy = deepcopy(self.occupied)

        elif self.always_reset_to_initial == True:
            print("resetting from start!")
            self.districts = deepcopy(self.initial_state)
            self.occupied = deepcopy(self.initial_occupancy)

        else:
            if max_active is None:
                max_active = self.BLOCKS_PER_DISTRICT
            # in order to prevent overlaps, hash each block placed on its coords
            self.occupied = {}

            self.districts = [
                [
                    np.zeros(2) for b in range(self.BLOCKS_PER_DISTRICT)
                ] for d in range(self.NUM_DISTRICTS)
            ]

            for i,district in enumerate(self.districts):
                for j,block in enumerate(district[:min_active]):
                    #self.districts[i][j] = block = self.place_block(self.occupied,active=True)
                    #self.occupied[str(block)] = (i,j)
                    self._update_state(i,j,None,self.place_block(self.occupied,active=True))
                for k,block in enumerate(district[min_active:max_active],start=min_active):
                    #self.districts[i][k] = block = self.place_block(self.occupied,active=None) # None-->don't force active, False-->force not active
                    #self.occupied[str(block)] = (i,k)
                    self._update_state(i,k,None,self.place_block(self.occupied,active=None))
        return self.districts

    def render(self, mode='human'):
        """
        Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        Args:
            mode (str): the mode to render with
        Example:
        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}
            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        if mode == 'human':
            for i,d in enumerate(self.districts):
                x,y = zip(*d)
                plt.scatter(x,y,label='D{}'.format(i))
                plt.axvline(self.PADDING)
                plt.legend(loc=0)
                plt.title(self.last_action)
            plt.show(block=False)
            plt.pause(0.05)
            plt.clf()
            #plt.close()
        else:
            # will raise an exception
            super(DistopiaEnv, self).render(mode=mode)
    def close(self):
        """
        Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        return



if __name__ == '__main__':
    ev = PopulationStdEvaluator()
    de = DistopiaEnv((1920,1080),ev)
    import pdb
    pdb.set_trace()