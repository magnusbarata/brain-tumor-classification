from tensorflow import keras

def multi_channel(input_shape, n_class=2, **kwargs):
    """2D multi_channel model"""

    if len(input_shape) != 4:
        raise ValueError('input_shape is expected as an array ranked 4')

    inputs = keras.Input(input_shape)

    new_dim = tuple([x for x in inputs.shape.as_list() if x != 1 and x is not None])
    reshape_layer = keras.layers.Reshape(new_dim)(inputs)
    
    x = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu')(reshape_layer)
    x = keras.layers.MaxPool2D(pool_size=2)(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
    x = keras.layers.MaxPool2D(pool_size=2)(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu')(x)
    x = keras.layers.MaxPool2D(pool_size=2)(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu')(x)
    x = keras.layers.MaxPool2D(pool_size=2)(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(units=512, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)

    outputs = keras.layers.Dense(n_class, activation='softmax')(x)

    return keras.Model(inputs, outputs, name=f'2D_multi_channel')
