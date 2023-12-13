import json
import numpy as np


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def load_json(file):
    with open(file) as f:
        obj = json.load(f)
    return obj


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, cls=NpEncoder, indent=4)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)