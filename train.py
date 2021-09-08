import argparse
from datetime import datetime
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split

from datagens.vol_datagen import VolumeDatagen
from datagens.img_datagen import ImageDatagen
from models import find_model
import utils
keras.backend.clear_session()

def main(args):
    params = utils.Hyperparams(args.settings)
    utils.set_seed(params.seed)

    ## Data prep
    df = pd.read_csv(
        f'{params.data.dirname}/train_labels.csv',
        dtype={'BraTS21ID': str}
    )
    df = df.set_index('BraTS21ID').drop(['00109', '00123', '00709'])
    X, y = df.index, df.MGMT_value.values   
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, stratify=y, test_size=params.data.val_size, random_state=params.seed)

    if params.modal == '2d':      
        datagen_tr = ImageDatagen(
            X_tr, y_tr,
            batch_size=params.data.batch_size,
            image_size=params.data.image_size,
            seq_type=params.data.seq_type,
            datadir=params.data.dirname
        )
        datagen_val = ImageDatagen(
            X_val, y_val,
            batch_size=params.data.batch_size,
            image_size=params.data.image_size,
            seq_type=params.data.seq_type,
            datadir=params.data.dirname
        )
    else:
        datagen_tr = VolumeDatagen(
            X_tr, y_tr,
            batch_size=params.data.batch_size,
            volume_size=params.data.volume_size,
            seq_type=params.data.seq_type,
            datadir=params.data.dirname
        )
        datagen_val = VolumeDatagen(
            X_val, y_val,
            batch_size=params.data.batch_size,
            volume_size=params.data.volume_size,
            seq_type=params.data.seq_type,
            datadir=params.data.dirname
        )

    ## Training
    if utils.continue_training(args.exp_dir):
        pass # TODO
    else:
        model = find_model(params.model.name)(datagen_tr.x_shape[1:], datagen_tr.n_class)
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            params.train.lr, 
            decay_steps=params.train.decay_steps,
            decay_rate=params.train.decay_rate,
            staircase=True
        )
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    
    chkpt_best = keras.callbacks.ModelCheckpoint(f'{args.exp_dir}/model_best.h5', monitor='val_accuracy', save_best_only=True)
    chkpt_latest = keras.callbacks.ModelCheckpoint(f'{args.exp_dir}/model_latest.h5', monitor='loss')
    tensorboard = keras.callbacks.TensorBoard(log_dir=f'{args.exp_dir}/logs', histogram_freq=1)
    stopper = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15)
    model.fit(
        datagen_tr,
        validation_data=datagen_val,
        epochs=params.train.epoch,
        verbose=1,
        workers=6,
        use_multiprocessing=False,
        callbacks=[chkpt_best, chkpt_latest, tensorboard, stopper]
    )
    params._save(f'{args.exp_dir}/train_params.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir', nargs='?', default=datetime.today().strftime('./exps/%Y%m%d'))
    parser.add_argument('--settings', default='default_params.json', help='Training hyperparameters.')
    main(parser.parse_args())