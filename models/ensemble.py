import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Lambda
from models import find_model

def ensemble_model(
        input_shape,
        n_class=2,
        base_model_name='baseline',
        **kwargs):
    """Ensemble learning model.
    
    This model wraps a single classifier to do ensemble learning.
    The single classifier model acts as a feature extractor, and the final model
    can learn to put more weights on trusted classifiers.

    Args:
      input_shape: Shape of the input tensor, not including the `batch_size` dimension.
        Last dimension of the input must be channel.
      n_class: The number of class to be predicted.
      base_model_name: The name of the model to be used as the single classifier model.
        Default to `baseline`.

    Returns:
      A `keras.Model` instance.
    """
    base_input_shape = (*input_shape[:-1], 1)
    inputs = tf.keras.Input(shape=input_shape)
    
    base_model_outs = []
    for c in range(input_shape[-1]):
        x = Lambda(lambda x, c=c: tf.expand_dims(tf.gather(x, c, axis=-1), -1))(inputs)
        base_model = find_model(base_model_name)(base_input_shape, n_class)
        base_model = tf.keras.Model(base_model.input, base_model.layers[-2].output)
        base_model_outs.append(base_model(x))
    
    x = Concatenate()(base_model_outs)
    x = Dense(units=128, activation='relu')(x)
    outputs = Dense(n_class, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs, name='ensemble_model')