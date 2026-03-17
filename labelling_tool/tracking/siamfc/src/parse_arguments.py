import json
import os
from collections import namedtuple


def parse_arguments(in_hp={}, in_evaluation={}, in_run={}):

    param_dir = 'siamfc/parameters'
    # param_dir = 'parameters'
    with open(os.path.join(param_dir, 'hyperparams.json')) as json_file:
        hp = json.load(json_file)
    with open(os.path.join(param_dir, 'evaluation.json')) as json_file:
        evaluation = json.load(json_file)
    with open(os.path.join(param_dir, 'run.json')) as json_file:
        run = json.load(json_file)
    with open(os.path.join(param_dir, 'environment.json')) as json_file:
        env = json.load(json_file)
    with open(os.path.join(param_dir, 'design.json')) as json_file:
        design = json.load(json_file)                

    for name,value in in_hp.items():
        hp[name] = value
    for name,value in in_evaluation.items():
        evaluation[name] = value
    for name,value in in_run.items():
        run[name] = value
    
    hp = namedtuple('hp', hp.keys())(**hp)
    evaluation = namedtuple('evaluation', evaluation.keys())(**evaluation)
    run = namedtuple('run', run.keys())(**run)
    env = namedtuple('env', env.keys())(**env)
    design = namedtuple('design', design.keys())(**design)

    return hp, evaluation, run, env, design
