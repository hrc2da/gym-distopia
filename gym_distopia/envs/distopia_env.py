import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_distopia.envs.distopia_rewards import PopulationStdEvaluator
from copy import deepcopy
import numpy as np
import json
from matplotlib import pyplot as plt
import pickle
import time



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
                    init_state = None, skip_first_reset = False, always_reset_to_initial = False, max_steps = 100, 
                    record_path = None, terminate_on_fail = False):
        '''
        OBSERVATION (STATE) SPACE:
        The state space for this environment is the x,y coordinates for each block PLUS the current district assignments for each county.
        If a county is unassigned, then its district is set to -1.
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
        self._max_steps = max_steps
        self.STEP_SIZE = step_size
        self.NUM_DISTRICTS = num_districts
        self.BLOCKS_PER_DISTRICT = blocks_per_district
        self.NUM_BLOCKS = self.NUM_DISTRICTS * self.BLOCKS_PER_DISTRICT
        self.GRID_WIDTH = grid_width # width of a grid in pixels
        self.PADDING = raw_padding//self.GRID_WIDTH # px reserved for the "off-screen" blocks
        self.NUM_DIRECTIONS = num_directions # 0,1,2,3,4 --> Do nothing, N, S, E, W
        self.stats = []
        self.record_path = record_path
        self.current_step = 0
        self.terminate_on_fail = terminate_on_fail
        if reward_evaluator is None:
            raise ValueError("Must pass a reward evaluator instance! See distopia_rewards.py.")
        self.evaluator = reward_evaluator
        self.NUM_PRECINCTS = self.evaluator.num_precincts
        self.width, self.height = (dim//self.GRID_WIDTH for dim in screen_size)
        
        
        self.action_space = spaces.MultiDiscrete([self.NUM_DIRECTIONS for b in range(self.NUM_BLOCKS)])
        
        
        
        # the state space is the x,y coords of all blocks i.e. (x0,y0,x1,y1,x2,y2...) + the district assignments of all 72 precincts (p0=0,p1=1,p2=2,...)
        obs_dims = []
        for b in range(self.NUM_BLOCKS):
            obs_dims.append(self.width + self.PADDING) # x for each block
            obs_dims.append(self.height) # y for each block
        for p in range(self.NUM_PRECINCTS):
            obs_dims.append(self.NUM_DISTRICTS) # district assignment for each precinct
        self.observation_space = spaces.MultiDiscrete(obs_dims)
        #self.observation_space.shape = (self.NUM_DISTRICTS,self.BLOCKS_PER_DISTRICT,2) 
        self.reward_range = self.evaluator.calculate_reward_range({'NUM_DISTRICTS':self.NUM_DISTRICTS})
        if init_state is not None:
            skip_first_reset = True
        self.skip_reset = False
        self.always_reset_to_initial = always_reset_to_initial
        self.reset(initial=init_state, skip_next_reset=skip_first_reset) #passing skip flag to prevent auto-reset at the start of most agent episodes


    def set_max_steps(self,max_steps):
        # set max steps per episode:
        self._max_steps = max_steps

    def dump_stats(self):
        with open("{}_{}.pkl".format(self.record_path,time.time()),'wb') as outfile:
            pickle.dump(self.stats,outfile)

    def record_stat(self,state,reward,action,success):
        self.stats.append({"state":json.dumps(state),"reward":reward, "action": action, "success":success})

    def get_staged_blocks_dict(self,block_locs):
        # refactored
        obs_dict = {}
        added = {}
        for d in range(0,self.NUM_DISTRICTS):
            obs_dict[d] = []
            for b in range(0,self.BLOCKS_PER_DISTRICT):
                index = 2*(d*self.BLOCKS_PER_DISTRICT + b)
                coords = [block_locs[index]*self.GRID_WIDTH,block_locs[index+1]*self.GRID_WIDTH]
                if block_locs[index] > self.PADDING: # if the x is far enough to the right
                    obs_dict[d].append(coords)
                assert self.hash_loc(coords) not in added # just double check to ensure we aren't passing two blocks in same loc
                added[self.hash_loc(coords)] = (d,b)
                
        return obs_dict
        # for i,district in enumerate(observation):
        #     obs_dict[i] = [block*self.GRID_WIDTH for block in district if block[0] > self.PADDING]
        # return obs_dict

    def evaluate_dist(self,districts):
        return self.evaluator.evaluate(districts)

    def evaluate_current(self):
        return self.evaluator.evaluate(self.district_list)

    def evaluate_block_locs(self, block_locs):
        obs_dict = self.get_staged_blocks_dict(block_locs)
        assignments = self.get_updated_precicnts(block_locs)
        if assignments == False:
            return self.reward_range[0]
        else:
            precincts,districts = assisgnments
            return self.evaluator.evaluate(districts)

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
        
        info = {"error":""}
        self.current_step += 1
        if self.current_step < self._max_steps:
            done = False
        else:
            done = True
        block_locs, precincts, occupied, district_list, reward, success = self._take_action(action, evaluate = True)
        observation = self.block_locs + self.precincts
        #if the evaluation failed
        if reward == False:
            # don't carry out the action
            reward = self.reward_range[0]
            if self.terminate_on_fail == True:
                done = True # try this out--ending the episode if illegal action occurs
            info["error"] = "illegal voronoi"
        else:
            self.update_state(block_locs,precincts,occupied,district_list)
            self.last_reward = reward
        if self.record_path is not None:
            self.record_stat(observation,reward,action,success)
        return (observation, reward, done, info)

    def _next_observation(self):
        # refactored
        return self.block_locs + self.precincts

    def update_state(self,block_locs,precincts,occupied,district_list):
        '''
        updates the state
        block_locs: an array of block locations
        precincts: an array of precinct district assignments
        occupied: a dict keyed on grid coords holding the block_ids
        district_list: a list of districts with their precincts
        '''
        self.block_locs = block_locs
        self.precincts = precincts
        self.occupied = occupied
        self.district_list = district_list

    def get_updated_block_locs(self,block_idx,old_loc,new_loc,block_locs,occupied):
        # refactored
        #block_locs = self.block_locs[:]
        #occupied = deepcopy(self.occupied)
        block_locs[block_idx] = new_loc[0]
        block_locs[block_idx+1] = new_loc[1]
        occupied[self.hash_loc(new_loc)] = block_idx
        if old_loc is not None: # note that is ! the same as == here; == tries to do a value compare, is does identity compare, so == fails b/c np array has no truth value
            occupied.pop(self.hash_loc(old_loc))
        return block_locs, occupied    
    
    def parse_precincts(self, district_list):
        precincts = [-1 for p in range(self.NUM_PRECINCTS)]
        for district in district_list:
            for p in district.precincts:
                precincts[p.identity] = district.identity
        return precincts

    def get_updated_precincts(self, block_locs):
        # refactored
        district_list = self.evaluator.map_districts(self.get_staged_blocks_dict(block_locs)) # list of precinct assignments for each district
        if len(district_list) < 1:
            return False
        else:
            # if parsed is True:
            #     return self.parse_precincts(district_list)
            # else:
            #     return district_list
            return self.parse_precincts(district_list), district_list




    def _apply_action(self, block_idx, action, block_locs, precincts, occupied):
        # refactored
        '''
        Inputs:
            block_idx: the ACTUAL bloc_locs index of the x-coord (multiply your block_id by 2 before passing in)
            action: a number from {0,1,2,3,4} representing the action
        Updates:
            self.districts if possible to apply the action legally
        Returns:
            True if action succeeded, False if it failed

        '''
        success = False
        #block_location = self.districts[district][block]
        block_x_idx = block_idx
        block_y_idx = block_x_idx + 1
        block_x = block_locs[block_x_idx]
        block_y = block_locs[block_y_idx]
        if action == 0:
            # do nothing
            return block_locs, precincts, occupied

        elif action == 1:
            # move north
            target_x = block_x
            target_y = block_y + self.STEP_SIZE
            if target_y > self.height or self.hash_loc([target_x,target_y]) in occupied:
                return False

        elif action == 2:
            # move south
            target_x = block_x
            target_y = block_y - self.STEP_SIZE
            if target_y < 0 or self.hash_loc([target_x,target_y]) in occupied:
                return False
            
        elif action == 3:
            # move east
            target_x = block_x + self.STEP_SIZE
            target_y = block_y 
            if target_x > self.width or self.hash_loc([target_x,target_y]) in occupied:
                return False

        elif action == 4:
            # move west
            target_x = block_x - self.STEP_SIZE
            target_y = block_y
            if target_x < 0 or self.hash_loc([target_x,target_y]) in occupied:
                return False

        else:
            raise ValueError("Invalid action requested: {}. Please pass actions between 0 and {}.".format(action,self.NUM_DIRECTIONS-1))
        block_loc = [block_x,block_y]
        new_loc = [target_x,target_y]
        block_locs, occupied = self.get_updated_block_locs(block_x_idx, block_loc, new_loc, block_locs, occupied)
        assignments = self.get_updated_precincts(block_locs)
        if assignments == False:
            return False
        precincts, district_list = assignments
        return block_locs, precincts, occupied, district_list
        
    
    def _take_action(self, action, evaluate = False):
        # refactored
        '''
        Inputs:
            action: an array of numbers between 0 and the number of moves
            for each block, e.g. [0, 0, 0, 2] moves the 4th block south 1 step

            evaluate: if true, will return an evaluation for the config. otherwise,
            will return None for the evaluation
        Returns:
            bloc_locs: the current block layout after taking the action
            precincts: a (parsed) list of precinct assignments
            evaluation: evaluation of district state or None
            successes: a boolean list parallel to the input, indicating which succeeded and which failed
        
        Note that some actions may make other illegal. This function will perform the actions in
        random order in an attempt to reduce bias in which actions get nullified by prior actions.
        '''
        # generate a random order in which to execute the actions
        indices = np.arange(len(action))
        np.random.shuffle(indices)
        successes = [0]*len(action)
        changes = 0
        block_locs = self.block_locs[:]
        precincts = self.precincts[:]
        occupied = deepcopy(self.occupied)
        evaluation = None
        district_list =  self.district_list[:]
        # go through the actions, unflattening and applying each one if legal
        for index in indices:
            # district = index // self.BLOCKS_PER_DISTRICT
            # block = index % self.BLOCKS_PER_DISTRICT
            block_idx = index * 2
            res = self._apply_action(block_idx,action[index],block_locs,precincts,occupied) 
            if res is not False:
                successes[index] = 1
                if action[index] > 0:
                    changes += 1
                    block_locs, precincts, occupied, district_list = res
            num_failures = len(successes) - sum(successes)

        if evaluate is True:
            if num_failures > 0:
                # don't update the state
                return block_locs, precincts, occupied, district_list, False ,successes
                

            else:
                evaluation = self.evaluator.evaluate(district_list)
                if evaluation is False:
                    block_locs = self.block_locs
                    precincts = self.precincts
                    occupied = self.occupied
                    district_list = self.district_list
        else:
            evaluation = None
            
        return block_locs, precincts, occupied, district_list, evaluation, successes

    
        

    def place_block(self,occupied,active=None):
        # refactored!
        '''
        Get unoccupied x,y coords to place a block on the active zone
        of the table, non-active zone, or neither
        Inputs:
            occupied: a dict containing occupied coords as keys
            active: True to force the placement in the active area
                    False to force the placement in the PADDING area
                    None to not force either
        Returns:
            x,y for a block
        '''
        xlow = self.PADDING if (active == True) else 0
        xhigh = self.PADDING if (active == False) else self.PADDING + self.width
        ylow = 0
        yhigh = self.height
        x = np.random.randint(low=xlow, high=xhigh)
        y = np.random.randint(low=ylow, high=yhigh)
        while self.hash_loc([x,y]) in occupied:
            x = np.random.randint(low=xlow, high=xhigh)
            y = np.random.randint(low=ylow, high=yhigh)
        return x, y

    def hash_loc(self,loc):
        loc_type = type(loc)
        if loc_type is list:
            return str(tuple(loc))
        elif loc_type is tuple:
            return str(loc)
        elif loc_type is np.ndarray:
            return str(tuple(loc))
        else:
            raise TypeError("Location should be a tuple or a list or a numpy array")

  
    def reset(self, initial=None, min_active=1, max_active=None, skip_next_reset = False):
        # refactored!
        """
        Resets the state of the environment and returns an initial observation.
            initial: a complete inital block layout
            min_active: the min number of blocks that should be in the active area for each district
            max_active: the max number of blocks that can be in the active area for each district
        Returns: 
            observation (object): the initial observation, consisting of block_locs + precinct_assignments.
        """

        if len(self.stats) > 1 and self.record_path is not None:
            self.dump_stats()
            self.stats = []

        self.current_step = 0
        if self.skip_reset == True:
            self.skip_reset = False
            return self.block_locs + self.precincts
        if skip_next_reset == True:
            self.skip_reset = True
        if initial is not None:
            self.block_locs = []
            self.occupied = {}
            for d in initial:
                for b in d:
                    self.block_locs.append(b[0])
                    self.block_locs.append(b[1])
                    self.occupied[self.hash_loc(b)] = len(self.block_locs) - 1 # x-index 
            assignments = self.get_updated_precincts(self.block_locs)
            assert assignments is not False
            self.precincts, self.district_list = assignments

            self.initial_blocks = deepcopy(self.block_locs)
            self.initial_precincts = deepcopy(self.precincts)
            self.initial_occupancy = deepcopy(self.occupied)
            #TODO: I think this is ok: I NEVER access/update the district objects
            # so I shouldn't need to deepcopy
            # (the problem is that it won't deepcopy due to cython)
            # as long as I COPY the array, I think I should be fine.
            self.initial_district_list = self.district_list[:]

        elif self.always_reset_to_initial == True:
            print("resetting from start!")
            self.block_locs = deepcopy(self.initial_blocks)
            self.precincts = deepcopy(self.initial_precincts)
            self.occupied = deepcopy(self.initial_occupancy)
            self.district_list = self.initial_district_list[:]

        else:
            if max_active is None:
                max_active = self.BLOCKS_PER_DISTRICT
            # in order to prevent overlaps, hash each block placed on its coords
            assignments = False
            # this is VERY inefficient and it is NOT recommended for larger number of blocks
            # or AT ALL unless ABSOLUTELY necessary.
            while assignments == False:
                self.occupied = {}

                self.block_locs = [0 for i in range(self.NUM_BLOCKS*2)]

                for i in range(self.NUM_DISTRICTS):
                    
                    for j in range(min_active):
                        loc = self.place_block(self.occupied, active=True)
                        self.block_locs, self.occupied = self.get_updated_block_locs(2*(i*self.BLOCKS_PER_DISTRICT + j), None, loc, self.block_locs[:], deepcopy(self.occupied))
                        
                    for k in range(min_active,max_active):
                        loc = self.place_block(self.occupied, active=None)
                        #self.districts[i][k] = block = self.place_block(self.occupied,active=None) # None-->don't force active, False-->force not active
                        #self.occupied[str(block)] = (i,k)
                        self.block_locs, self.occupied = self.get_updated_block_locs(2*(i*self.BLOCKS_PER_DISTRICT + k), None, loc, self.block_locs[:], deepcopy(self.occupied))
                assignments = self.get_updated_precincts(self.block_locs)
            
            self.precincts, self.district_list = assignments
        return self.block_locs + self.precincts

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
            x = self.block_locs[:self.NUM_BLOCKS*2:2]
            y = self.block_locs[1:self.NUM_BLOCKS*2+1:2]
            plt.scatter(x,y)
            plt.axvline(self.PADDING)
            plt.show(block=False)
            plt.pause(0.05)
            plt.clf()

            # for i,d in enumerate(self.districts):
            #     x,y = zip(*d)
            #     plt.scatter(x,y,label='D{}'.format(i))
            #     plt.axvline(self.PADDING)
            #     plt.legend(loc=0)
            #     plt.title(self.last_action)
            # plt.show(block=False)
            # plt.pause(0.05)
            # plt.clf()
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
        if len(self.stats) > 1 and self.record_path is not None:
            self.dump_stats()
            self.stats = []

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