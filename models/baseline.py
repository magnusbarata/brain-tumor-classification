from tensorflow import keras

def baseline(input_shape, n_class=2, n_filters=64, **kwargs):
    """Baseline model from the paper https://arxiv.org/abs/2007.13224"""
    if len(input_shape) == 4:
        gap_layer = keras.layers.GlobalAveragePooling3D
        modal = '3D'
    elif len(input_shape) == 3:
        gap_layer = keras.layers.GlobalAveragePooling2D
        modal = '2D'
    else:
        raise ValueError('input_shape is expected as an array ranked 3 or 4')

    inputs = keras.Input(input_shape)

    x = base_block(inputs, n_filters)
    x = base_block(x, n_filters)
    x = base_block(x, 2 * n_filters)
    x = base_block(x, 4 * n_filters)

    x = gap_layer()(x)
    x = keras.layers.Dense(units=512, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)

    outputs = keras.layers.Dense(n_class, activation='softmax')(x)

    return keras.Model(inputs, outputs, name=f'baseline_{modal}')


def baseline_v2(input_shape, n_class=2, n_filters=64, **kwargs):
    """Modified baseline model with the inspiration from https://www.kaggle.com/ammarnassanalhajali/brain-tumor-3d-training"""
    if len(input_shape) == 4:
        gap_layer = keras.layers.GlobalAveragePooling3D
        modal = '3D'
    elif len(input_shape) == 3:
        gap_layer = keras.layers.GlobalAveragePooling2D
        modal = '2D'
    else:
        raise ValueError('input_shape is expected as an array ranked 3 or 4')

    inputs = keras.Input(input_shape)
    
    x = base_block(inputs, n_filters)
    x = keras.layers.Dropout(0.01)(x)
    x = base_block(x, 2 * n_filters)
    x = keras.layers.Dropout(0.02)(x)
    x = base_block(x, 4 * n_filters)
    x = keras.layers.Dropout(0.03)(x)
    x = base_block(x, 8 * n_filters)
    x = keras.layers.Dropout(0.04)(x)

    x = gap_layer()(x)
    x = keras.layers.Dense(units=1024, activation='relu')(x)
    x = keras.layers.Dropout(0.08)(x)

    outputs = keras.layers.Dense(n_class, activation='softmax')(x)

    return keras.Model(inputs, outputs, name=f'baseline_v2_{modal}')


def base_block(x, filters):
    """Utility function for the base block (conv + pool + BN) of baseline model.
    
    Args:
      x: Input tensor.
      filters: Number of filters in conv layer.
    
    Returns:
      Output tensor after applying base block to x.
    """
    if len(x.shape) == 5:
        conv_layer = keras.layers.Conv3D
        pool_layer = keras.layers.MaxPool3D
    else:
        conv_layer = keras.layers.Conv2D
        pool_layer = keras.layers.MaxPool2D

    x = conv_layer(filters, kernel_size=3, activation='relu')(x)
    x = pool_layer(pool_size=2)(x)
    x = keras.layers.BatchNormalization()(x)
    return x
