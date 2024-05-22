import random

import numpy as np
import torch


def set_global_seeds(i):
    """set seed for fair comparison"""
    torch.manual_seed(i)
    torch.cuda.manual_seed(i)
    torch.cuda.manual_seed_all(i)
    np.random.seed(i)
    random.seed(i)
    torch.backends.cudnn.deterministic = True


def global_list():
    record_list={ 'success':0,'global_coll':0,'iter':0,'makespan':0,'run_time':0,'sum_of_cost':0}
    return record_list

def global_dict():
    record_list={'success':[],'global_coll':[],'iter':[],'makespan':[],'run_time':[],'sum_of_cost':[]}
    return record_list

def global_std():
    record_list={'global_coll':0,'iter':0,'makespan':0,'run_time':0,'sum_of_cost':0}
    return record_list
