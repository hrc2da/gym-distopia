import pickle as pkl
import json
import glob

log_path = "/home/dev/scratch/gym-distopia/generator/pre_empt_logs2/logs/"

legal_ctr = 0
illegal_ctr = 0

legal_states = set()
for filename in glob.glob(log_path + '*.pkl'):
    with open(filename, 'rb') as infile:
        episode = pkl.load(infile)
        if episode[0]['reward'] > 0:
            legal_ctr += 1
        else:
            illegal_ctr += 1
        for step in episode:
            if float(step['reward']) > 0:
                state = json.loads(step['state'])[:32]
                legal_states.add(json.dumps(state))

out = []
while len(legal_states) > 0:
    out.append(json.loads(legal_states.pop()))

with open("sucked_files2.json", 'w+') as outfile:
    json.dump(out,outfile)

print("Legal: {}, Illegal: {}".format(legal_ctr,illegal_ctr))