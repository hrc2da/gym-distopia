from gym_distopia.envs.distopia_env import DistopiaEnv
from gym_distopia.envs.distopia_rewards import PopulationStdEvaluator
from copy import deepcopy
import numpy as np



full_valid_state = [
    [[400,200],[491,200],[400,291]],
    [[600,200],[651,200],[600,251]],
    [[800,200],[851,200],[800,251]],
    [[400,500],[451,500],[400,551]],
    [[600,500],[651,500],[600,551]],
    [[800,500],[851,500],[800,551]],
    [[500,800],[551,800],[500,851]],
    [[700,800],[751,800],[700,851]]
]

valid_four_state = [
     [[12,8]],
    [[16,8]],
    [[15,15]],
    [[12,15]]
]

invalid_four_state = [
    [[11,11]],
    [[17,13]],
    [[18,18]],
    [[9,19]]
]

one_block_per_state = [
    [[400,200],[11,200],[40,201]],
    [[600,200],[21,200],[60,201]],
    [[800,200],[31,200],[80,201]],
    [[400,500],[41,500],[40,501]],
    [[600,500],[61,500],[60,501]],
    [[800,500],[81,500],[80,501]],
    [[500,800],[51,800],[50,801]],
    [[700,800],[71,800],[70,801]]
]
empty_state = [[np.zeros(2) for block in range(3)] for district in range(8)]


def scale_state(state, scale):
    scaled_state = []
    for district in state:
        #import pdb
        #pdb.set_trace()
        scaled_district = [[b[0]//scale,b[1]//scale] for b in district]
        scaled_state.append(scaled_district)
    return scaled_state

class TestDistopiaEnv:
    
    
    def setup_class(self):
        print('setting up')
        self.ev = PopulationStdEvaluator()
        for i in range(199):
            self.de = DistopiaEnv((1920,1080),self.ev)

    def teardown_class(self):
        print('tearing down')
        self.ev = None
        self.de = None

    def state_equal(self,state1,state2):
        for i,district in enumerate(state1):
            for j,block in enumerate(district):
                for k,coord in enumerate(block):
                    if coord != state2[i][j][k]:
                        return False
        return True
    
    def test_constructor(self):
        # make sure the constructor runs
        ev = PopulationStdEvaluator()
        de = DistopiaEnv((1920,1080),ev)
        assert de != None
        # check to see that things got created correctly
        # check to see that the action space is right
        # check the dimensions of the state space
        import pdb
        pdb.set_trace()
        
        assert np.asarray(de.observation_space).shape == (de.NUM_DISTRICTS,de.BLOCKS_PER_DISTRICT,2)
        assert np.asarray(de.districts).shape == (de.NUM_DISTRICTS,de.BLOCKS_PER_DISTRICT,2)
        # check something about the voronoi
    # def test_update_state(self):
    #     #de = DistopiaEnv((1920,1080),'population',minimize_std)
    #     old_loc = self.de.districts[1][2]
    #     self.de._update_state(1,2,old_loc,np.asarray((42,42)))
    #     #state = np.zeros()
    #     assert np.array_equal(self.de.districts[1][2],(42,42))
    #     assert str(np.asarray((42,42))) in self.de.occupied
    #     assert np.array_equal(self.de.occupied[str(np.asarray((42,42)))], (1,2))
    # def test_apply_action(self):
    #     #de = DistopiaEnv((1920,1080),'population',minimize_std)
    #     district = np.random.randint(self.de.NUM_DISTRICTS)
    #     block = np.random.randint(self.de.BLOCKS_PER_DISTRICT)
    #     # check that the "do nothing" action does nothing
    #     state = deepcopy(self.de.districts)
    #     res = self.de._apply_action(district,block,0)
    #     assert res == True
    #     assert self.state_equal(state,self.de.districts) == True
    #     # check that moving a block to an open spot updates the state
    #     for i in range(1):
    #         target_loc = self.de.districts[district][block] + (0,-1) # let's try going south
    #         while target_loc[1] < 0:
    #             district = np.random.randint(self.de.NUM_DISTRICTS)
    #             block = np.random.randint(self.de.BLOCKS_PER_DISTRICT)
    #             target_loc = self.de.districts[district][block] + (0,-1)
    #         if str(target_loc) in self.de.occupied:
    #             print("overlap in the test. let's move this block somewhere illegal")

    #             mvdist,mvblock = self.de.occupied[str(target_loc)]

    #             self.de._update_state(mvdist,mvblock,target_loc,np.asarray([-1,-1]))
    #         state = deepcopy(self.de.districts)
    #         res = self.de._apply_action(district,block,2)
    #         assert res == True
    #         assert self.state_equal(state,self.de.districts) != True

    #         assert np.array_equal(self.de.districts[district][block],target_loc)


    #     target_loc = self.de.districts[district][block] + (1,0) # let's try going east
    #     while target_loc[0] > self.de.width:
    #         district = np.random.randint(self.de.NUM_DISTRICTS)
    #         block = np.random.randint(self.de.BLOCKS_PER_DISTRICT)
    #         target_loc = self.de.districts[district][block] + (1,0)
    #     if str(target_loc) in self.de.occupied:
    #         print("overlap in the test. let's move this block somewhere illegal")
    #         mvdist,mvblock = self.de.occupied[str(target_loc)]
    #         self.de._update_state(mvdist,mvblock,target_loc,np.asarray([-1,-1]))
    #     state = deepcopy(self.de.districts)
    #     res = self.de._apply_action(district,block,3)
    #     assert res == True
    #     assert self.state_equal(state,self.de.districts) != True
    #     assert np.array_equal(self.de.districts[district][block],target_loc)

    #     # check that moving a block where there already is one fails
    #     occupied_loc = self.de.districts[district][block]
    #     if self.de.NUM_DISTRICTS > 1:
    #         target_district = district + 1 if district < (self.de.NUM_DISTRICTS-1) else district - 1
    #     if self.de.BLOCKS_PER_DISTRICT > 1:
    #         target_block = block + 1 if block < (self.de.BLOCKS_PER_DISTRICT-1) else block - 1
    #     target_old_loc = self.de.districts[target_district][target_block]
    #     target_loc = occupied_loc + (0, -1)
    #     self.de._update_state(target_district,target_block,target_old_loc,target_loc)
    #     state = deepcopy(self.de.districts)
    #     res = self.de._apply_action(target_district,target_block,1)
    #     assert res == False
    #     assert self.state_equal(state,self.de.districts)
    # def test_take_action(self):
    #     assert 1 == 1
    # def test_place_block(self):
    #     assert 1 == 1
    # def test_reset(self):
    #     #de = DistopiaEnv((1920,1080),'population',minimize_std)
    #     state = [[np.zeros(2) for block in range(self.de.BLOCKS_PER_DISTRICT)] for district in range(self.de.NUM_DISTRICTS)]
    #     self.de.reset(initial=state)
    #     assert  self.state_equal(self.de.districts, state)
    #     self.de.reset()
    #     assert not self.state_equal(self.de.districts,state)


    # def test_get_staged_blocks_dict(self):
    #     self.de = DistopiaEnv((1920,1080),self.ev,num_districts = 8)
    #     self.de.reset(initial=scale_state(one_block_per_state,self.de.GRID_WIDTH))
    #     obs_dict = self.de.get_staged_blocks_dict(self.de.districts)
    #     assert len(obs_dict) == 8
    #     assert np.all([len(obs_dict[district]) == 1 for district in obs_dict])
    #     assert np.all([np.array_equal(obs_dict[district], [self.de.districts[district][0]*self.de.GRID_WIDTH]) for district in obs_dict])
    #     self.de.reset(initial=scale_state(full_valid_state,self.de.GRID_WIDTH))
    #     obs_dict = self.de.get_staged_blocks_dict(self.de.districts)
    #     assert len(obs_dict) == 8
    #     assert np.all([len(obs_dict[district]) == 3 for district in obs_dict])
    #     assert np.all([np.array_equal(obs_dict[district], [block*self.de.GRID_WIDTH for block in self.de.districts[district]]) for district in obs_dict])

    # def test_evaluate(self):
    #     self.de = DistopiaEnv((1920,1080),self.ev,num_districts = 8)
    #     self.de.reset(initial=scale_state(full_valid_state,self.de.GRID_WIDTH))
    #     res = self.de.evaluate(self.de.districts)
    #     assert res != False
    #     self.de.reset(initial=empty_state)
    #     res = self.de.evaluate(self.de.districts)
    #     assert res == False


    # def test_four_evaluate(self):
    #     de = DistopiaEnv((1920,1080),self.ev,blocks_per_district=1,num_districts=4)
    #     de.reset(initial=valid_four_state)
    #     res = de.evaluate(de.districts)
    #     import pdb
    #     pdb.set_trace()
    #     assert res != False
    #     de.reset(initial=invalid_four_state)
    #     import pdb
    #     pdb.set_trace()
    #     res = de.evaluate(de.districts)
    #     assert res == False