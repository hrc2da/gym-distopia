import numpy as np
from distopia.app.agent import VoronoiAgent

def minimize_std(districts):
    return - np.std(districts)

def maximize_std(districts):
    return np.std(districts)

class RewardEvaluator:
    objective_ids = []
    def setup_voronoi(self):
        self.voronoi = VoronoiAgent()
        self.voronoi.load_data()
        self.num_precincts = len(self.voronoi.precincts)
    def setup_objectives(self,objectives):
        self.objective_ids = []
        try:
            for objective in objectives:
                self.objective_ids.append(self.voronoi.metrics.index(objective))
        except ValueError:
            raise ValueError("Trying to optimize on {} but it doesn't exist!".format(objective))

    def map_districts(self,fiducials):
        return self.voronoi.get_voronoi_districts(fiducials)

    def evaluate(self,observation):
        try:
            state_metrics, district_metrics = self.voronoi.compute_voronoi_metrics(observation)
            
        except Exception as e:
            print("Couldn't compute Voronoi for {}:{}".format(observation,e))
            return False
        
        try:
            return self.calculate_reward(state_metrics,district_metrics)
            #print("{}:{}".format(self.n_calls,cost))
        except ValueError as v:
            print("Problem calculating the metrics: {}".format(v))
            return False
        
    
    def calculate_reward_range(self,info):
        raise NotImplementedError
    def calculate_reward(self,state_metrics,district_metrics):
        raise NotImplementedError
    def extract_objectives(self,districts,objectives=None):
        if objectives is None:
            objectives = self.objective_ids
        all_objectives = {}        
        if len(districts) < 1:
            raise ValueError("No Districts")
        for obj in objectives:
            objective_vals = []
            for d in districts:
                data = districts[d][obj].get_data()
                if len(data['data']) < 1:
                    raise ValueError("Empty District {}".format(d))
                else:
                    objective_vals.append(data['scalar_value'])
            all_objectives[obj] = objective_vals
        return all_objectives

class PopulationStdEvaluator(RewardEvaluator):
    def __init__(self):
        self.setup_voronoi()
        self.setup_objectives(['population'])
        
    def calculate_reward_range(self,info):
        '''
        for each of the metrics, calculate (or hard-code) a reward range
        alternatively, you can choose to normalize
        
        this is a function that will be called by openai gym env's to set their own
        reward range
        '''
        NUM_DISTRICTS = info['NUM_DISTRICTS']
        WISCONSIN_POP = 5814000
        std_dummy = np.zeros(NUM_DISTRICTS)
        std_dummy[0] = WISCONSIN_POP
        self.max_punishment = np.std(std_dummy)
        #self.reward_range = (-self.max_punishment/self.max_punishment,0.0) #normalized
        self.reward_range = (-self.max_punishment/self.max_punishment+1,0.0+1) #normalized plus 1
        return self.reward_range
    
    

    def calculate_reward(self,state_metrics,district_metrics):
        objectives = self.extract_objectives(district_metrics)
        return minimize_std(objectives[self.objective_ids[0]])/self.max_punishment + 1
        