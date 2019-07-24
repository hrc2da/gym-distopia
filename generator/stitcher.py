import json

sample_1 = "/home/dev/scratch/gym-distopia/generator/random_sample_1.json" # in 1920x1080 format, unflattened

run_1 = "/home/dev/scratch/gym-distopia/generator/sucked_files.json" # in grid format, flattened
run_2 = "/home/dev/scratch/gym-distopia/generator/sucked_files2.json" # in grid format, flattened

flattened_1 = []
with open(sample_1) as infile:
    unflattened = json.load(infile)
    for sample in unflattened:
        abort = False
        new_sample = []
        for district in sample:
            if len(district) != 2:
                abort = True
                break
            else:
                for block in district:
                    for coord in block:
                        new_sample.append(coord)
        if abort == False:
            flattened_1.append(new_sample)
assert len(flattened_1[0]) == 32
assert len(flattened_1[-1]) == 32

GRID_WIDTH = 50.0 # symmetric grid in x,y
parsed_1 = []
with open(run_1) as infile:
    unparsed = json.load(infile)
    for sample in unparsed:
        new_sample = []
        for coord in sample:
            new_sample.append(coord*GRID_WIDTH)
        parsed_1.append(new_sample)
assert len(parsed_1[0]) == 32
assert len(parsed_1[-1]) == 32

parsed_2 = []
with open(run_2) as infile:
    unparsed = json.load(infile)
    for sample in unparsed:
        new_sample = []
        for coord in sample:
            new_sample.append(coord*GRID_WIDTH)
        parsed_2.append(new_sample)
assert len(parsed_2[0]) == 32
assert len(parsed_2[-1]) == 32

final = flattened_1 + parsed_1 + parsed_2

with open("merged.json", 'w+') as outfile:
    json.dump(final,outfile)
