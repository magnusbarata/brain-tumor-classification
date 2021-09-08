from tensorflow import keras

def baseline_2d(input_shape, n_class=2):
    inputs = keras.Input(input_shape)

    x = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu')(inputs)
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

    return keras.Model(inputs, outputs, name='baseline')