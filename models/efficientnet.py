from tensorflow import keras
import efficientnet_3D.tfkeras as efn

def efficientnet(input_shape, n_class=2, n_filters=64, **kwargs):
    """Baseline model from the paper https://arxiv.org/abs/2007.13224"""
    if len(input_shape) == 4:
        effnet_layer = efn.EfficientNetB0(
            include_top=True,
            weights=None,
            input_shape=input_shape, 
            pooling=None,
            classes=2,
        )
        modal = '3D'
    elif len(input_shape) == 3:
        effnet_layer = keras.applications.EfficientNetB0(
            include_top=True,
            weights=None,
            input_shape=input_shape, 
            pooling=None,
            classes=2,
            classifier_activation='softmax'
        )
        modal = '2D'
    else:
        raise ValueError('input_shape is expected as an array ranked 3 or 4')

    inputs = keras.Input(input_shape)

    outputs = effnet_layer(inputs)

    if modal == '3D':
        outputs = keras.layers.Softmax()(outputs)

    return keras.Model(inputs, outputs, name=f'efficientnet_{modal}')
