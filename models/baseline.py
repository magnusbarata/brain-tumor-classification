from tensorflow import keras

def baseline(input_shape, n_class=2, n_filters=64, **kwargs):
    """Baseline model from the paper https://arxiv.org/abs/2007.13224"""
    if len(input_shape) == 4:
        conv_layer = keras.layers.Conv3D
        pool_layer = keras.layers.MaxPool3D
        gap_layer = keras.layers.GlobalAveragePooling3D
        modal = '3D'
    elif len(input_shape) == 3:
        conv_layer = keras.layers.Conv2D
        pool_layer = keras.layers.MaxPool2D
        gap_layer = keras.layers.GlobalAveragePooling2D
        modal = '2D'
    else:
        raise ValueError('input_shape is expected as an array ranked 3 or 4')

    inputs = keras.Input(input_shape)

    x = conv_layer(n_filters, kernel_size=3, activation='relu')(inputs)
    x = pool_layer(pool_size=2)(x)
    x = keras.layers.BatchNormalization()(x)

    x = conv_layer(n_filters, kernel_size=3, activation='relu')(x)
    x = pool_layer(pool_size=2)(x)
    x = keras.layers.BatchNormalization()(x)

    x = conv_layer(2 * n_filters, kernel_size=3, activation='relu')(x)
    x = pool_layer(pool_size=2)(x)
    x = keras.layers.BatchNormalization()(x)

    x = conv_layer(4 * n_filters, kernel_size=3, activation='relu')(x)
    x = pool_layer(pool_size=2)(x)
    x = keras.layers.BatchNormalization()(x)

    x = gap_layer()(x)
    x = keras.layers.Dense(units=512, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)

    outputs = keras.layers.Dense(n_class, activation='softmax')(x)

    return keras.Model(inputs, outputs, name=f'baseline_{modal}')
