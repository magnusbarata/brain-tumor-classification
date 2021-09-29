from tensorflow import keras
import efficientnet_3D.tfkeras as efn

def efficientnet(input_shape, n_class=2, variant='B0', **kwargs):
    """EfficientNet model from the paper https://arxiv.org/abs/1905.11946"""
    default_effnet_params = {
        'include_top': True,
        'weights': None,
        'input_shape': input_shape,
        'pooling': None,
        'classes': n_class,
    }
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

    inputs = keras.Input(input_shape)
    outputs = effnet_layer(inputs)

    return keras.Model(inputs, outputs, name=f'efficientnet_{modal}_{variant}')
