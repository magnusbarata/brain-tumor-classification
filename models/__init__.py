"""Model definitions."""
import sys

def find_model(model_name):
    """Import a model using model name"""
    this_module = sys.modules[__name__]
    return getattr(this_module, f'{model_name}')


# Import exposed model functions
from models.baseline import baseline, baseline_v2
from models.ensemble import ensemble_model
