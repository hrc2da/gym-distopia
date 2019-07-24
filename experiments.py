"""
Credit: dylan-slack
https://gist.github.com/dylan-slack/b8c95b38d519b63cbba186d889b09330

A simple experiment manager.  You can add experiments as functions with the @register decorator.  Then, call the experiments you want to run
using the -r flag followed by the function names of the experiments or run all the experiments with the -a flag. 

Experiment runs are stored in an .experiments/ folder created on the first time you run an experiment.  Each run creates a folder inside
.experiments/ with the name of the function and the datetime it was run.  Every experiment should accept a logging_path argument.  
This is the path to the folder created for the specific experiment run.  You can use this to store experiment specific files.   
"""
import argparse
import datetime
import os
import re
import sys

from distopia_agents.d_dqn import DistopiaDQN
from distopia_agents.d_random import DistopiaRDQN

experiments = {}
parser = argparse.ArgumentParser(description='Run experiments.')
experiment_folder = './experiments'

STRIP = -7

def register(experiment):
	experiments[experiment.__name__] = experiment 

def make_note_file(path, info):
	start_time = datetime.datetime.strptime(info['start_time'],'%Y-%m-%d.%H:%M:%S')
	end_time = datetime.datetime.now()
	total_time = (end_time - start_time)
	total_seconds = total_time.total_seconds()

	with open(os.path.join(path,'notes.md'), 'w') as outfile:
		outfile.write('{}\n\n'.format(path))
		outfile.write('Job completed at {}\n\n'.format(str(end_time).replace(" ", ".")[:STRIP]))
		outfile.write('Time elapsed (H:M:S): {}\n'.format(total_time))
		outfile.write('Time elapsed in seconds: {}\n\n'.format(total_seconds))
		for key in info:
			outfile.write('* {}: {}\n'.format(key,info[key]))
		


##
# Begin Experiments:
# Each experiment function call should accept a logging path argument.
# This is the path of the directory created for each new run of this experiment 
# and can be used to store files specific to the experiment.
###
@register
def test_dqn_1(logging_path):
	name = sys._getframe().f_code.co_name
	n_eps = 10000
	steps_per_ep = 1000
	info = {
		'experiment': name,
		'start_time': re.findall(r'[^_]*$',logging_path)[0],
		'n_eps': n_eps,
		'steps_per_ep': steps_per_ep,
		'policy': "Linear Annealed Epsilon-Greedy (1 to 0 over 1,000,000 steps)"
	}
	print ("Experiment 1 ({}) here:{}".format(name,logging_path))
	make_note_file(logging_path,info)

	d = DistopiaDQN(reconstruct=True,terminate_on_fail=False, out_path=logging_path, revert_failures=False)
	d.train(max_steps = steps_per_ep, episodes = n_eps, visualize= False)




  
@register
def test_random_1(logging_path):
	name = sys._getframe().f_code.co_name
	n_eps = 10000
	steps_per_ep = 1000
	info = {
		'experiment': name,
		'start_time': re.findall(r'[^_]*$',logging_path)[0],
		'n_eps': n_eps,
		'steps_per_ep': steps_per_ep,
		'policy': "Random Policy"
	}
	print ("Experiment 2 ({}) here:{}".format(name,logging_path))
	make_note_file(logging_path,info)

	d = DistopiaRDQN(reconstruct=True,terminate_on_fail=False, out_path=logging_path, revert_failures=False)
	d.train(max_steps = steps_per_ep, episodes = n_eps, visualize= False)
###

def setup(exp):
	time_of_creation = str(datetime.datetime.now()).replace(" ", ".")[:STRIP]
	exp_path = os.path.join(experiment_folder,f"{exp}_{time_of_creation}")

	if not os.path.exists(exp_path):
		os.makedirs(exp_path)
	return exp_path

def main():
	parser.add_argument('-s', action='store_true', default=False,
                   help='show the current experiments avaliable to run')
	parser.add_argument('-r', metavar='--run', nargs='+',
				   help='run these experiments')
	parser.add_argument('-a', action='store_true', default=False,
				   help='run all the experiments avaliable')

	args = vars(parser.parse_args())

	if args['a']:
		print("running all")
		for exp in experiments:
			print (f"Beginning experiment {exp}")
			print ("-----------------------------")
			logging_path = setup(exp)
			experiments[exp](logging_path)
			print (f"Saving in {logging_path}")
			print ("-----------------------------")
		exit()


	if args['s'] or not args['r']:
		print ("Avaliable Experiments:")
		print ("----------------------")
		for key in experiments:
			print (key)
		print ("----------------------")	
		print ("Run experiments with -r flag.")
		exit()


	for exp in args['r']:
		if exp in experiments:
			print (f"Beginning experiment {exp}")
			print ("-----------------------------")
			logging_path = setup(exp)
			experiments[exp](logging_path)
			print (f"Saving in {logging_path}")
			print ("-----------------------------")
		else:
			print(f"Experiment {exp} not registered -- skipping!.")


if __name__ == "__main__":
	main()
