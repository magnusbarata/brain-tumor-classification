from tensorflow import keras
import efficientnet_3D.tfkeras as efn

def efficientnet(input_shape, n_class=2, variant='B0', **kwargs):
    """EfficientNet model from the paper https://arxiv.org/abs/1905.11946
    
    Args:
      input_shape: Shape of the input tensor, not including the `batch_size` dimension.
      n_class: The number of class to be predicted.
      variant: The variant of efficient net. Can be `B0`~`B7` for both 2D and 3D.
        `L2` is available for 3D version only.
    
    Returns:
      A `keras.Model` instance.
    """
    default_effnet_params = {
        'include_top': True,
        'weights': None,
        'input_shape': input_shape,
        'pooling': None,
        'classes': n_class,
    }
    try:
        if len(input_shape) == 4:
            effnet_layer = getattr(efn, f'EfficientNet{variant}')(**default_effnet_params)
            modal = '3D'
        elif len(input_shape) == 3:
            effnet_layer = getattr(keras.applications, f'EfficientNet{variant}')(
                classifier_activation='softmax', **default_effnet_params
            )
            modal = '2D'
        else:
            raise ValueError('input_shape is expected as an array ranked 3 or 4')
    except AttributeError:
        print(f'No EfficientNet variation found for {variant}')
        raise SystemExit

    inputs = keras.Input(input_shape)
    outputs = effnet_layer(inputs)

    return keras.Model(inputs, outputs, name=f'efficientnet_{modal}_{variant}')
