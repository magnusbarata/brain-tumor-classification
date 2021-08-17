"""Utility functions"""
import os
import json
import shutil

def set_seed(seed):
    """Function setting the global seed for reproducible results"""
    import tensorflow as tf
    import numpy as np
    import random
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def continue_training(dir_path):
    """Function asking the user whether to continue training or not.
    
    Args:
        dir_path: Experiment directory to continue training.
    
    Returns:
        Bool: True if user chose to continue training.
    """
    if os.path.exists(dir_path):
        usr_in = input(f'Continue training the model on {dir_path}? (y/[n]): ')
        if usr_in.lower() == 'y':
            return True
        else:
            usr_in = input(f'Overwrite experiment on {dir_path}? (y/[n]): ')
            if usr_in.lower() == 'y':            
                shutil.rmtree(dir_path)
            else:
                print('Exiting...')
                raise SystemExit
    
    os.makedirs(dir_path)
    return False


class Hyperparams(dict):
    """Class to load/save hyperparameters from/to a json file.
    
    Example:
    ```
    params = Hyperparams(hyperparams.json)
    print(params.epoch)
    params.epochs = 100 # change the number of epochs in params
    ```
    """
    def __init__(self, arg={}, **kwargs):
        if isinstance(arg, str):
            with open(arg) as f:
                arg = json.load(f)
        
        super(Hyperparams, self).__init__(arg, **kwargs)
        if isinstance(arg, dict):
            for k, v in arg.items():
                self[k] = v
        
        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __addlist(self, ls):
        for elem in range(0, len(ls)):
            if isinstance(ls[elem], dict):
                ls[elem] = Hyperparams(ls[elem])
            elif isinstance(ls[elem], list):
                self.__addlist(ls[elem])

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = Hyperparams(value)
        elif isinstance(value, list):
            self.__addlist(value)
        super(Hyperparams, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Hyperparams, self).__delitem__(key)
        del self.__dict__[key]

    def _save(self, fpath):
        """Save hyperparameters to json file"""
        with open(fpath, 'w') as f:
            json.dump(self, f, indent=2)

    @staticmethod
    def _load(fpath):
        """Load hyperparameters from json file"""
        return Hyperparams(fpath)