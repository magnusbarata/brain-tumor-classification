"""Model definitions."""
import sys

# Import exposed model functions
from models.baseline import baseline
from models.baseline_2d import baseline_2d

def find_model(model_name):
    """Import a model using model name"""
    this_module = sys.modules[__name__]
    return getattr(this_module, f'{model_name}')