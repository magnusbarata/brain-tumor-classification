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

def data_prep(params_data, seed):
    params_data = utils.Hyperparams(params_data)
    df = pd.read_csv(
        f'{params_data.datadir}/train_labels.csv',
        dtype={'BraTS21ID': str}
    )
    df = df.set_index('BraTS21ID').drop(['00109', '00123', '00709'])
    X, y = df.index, df.MGMT_value.values   
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, stratify=y, test_size=params_data.val_size, random_state=seed
    )

    if params_data.image_size is not None:
        datagen_tr = ImageDatagen(
            X_tr, y_tr, **params_data
        )
        datagen_val = ImageDatagen(
            X_val, y_val, **params_data
        )
    else:
        datagen_tr = VolumeDatagen(
            X_tr, y_tr, **params_data
        )
        params_data.augmentations = None
        datagen_val = VolumeDatagen(
            X_val, y_val, **params_data
        )

    return datagen_tr, datagen_val


def train(exp_dir, params, datagen_tr, datagen_val):
    if utils.continue_training(exp_dir):
        pass # TODO
    else:
        model = find_model(params.model.name)(
            datagen_tr.x_shape[1:], datagen_tr.n_class, **params.model
        )
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            params.train.lr, 
            decay_steps=params.train.decay_steps,
            decay_rate=params.train.decay_rate,
            staircase=True
        )
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    
    val_acc = 'val_pred_accuracy' if params.model.use_aux else 'val_accuracy'
    chkpt_best = keras.callbacks.ModelCheckpoint(f'{exp_dir}/model_best.h5', monitor=val_acc, save_best_only=True)
    chkpt_latest = keras.callbacks.ModelCheckpoint(f'{exp_dir}/model_latest.h5', monitor='loss')
    tensorboard = keras.callbacks.TensorBoard(log_dir=f'{exp_dir}/logs', histogram_freq=1)
    stopper = keras.callbacks.EarlyStopping(monitor=val_acc, patience=15)
    model.fit(
        datagen_tr,
        validation_data=datagen_val,
        epochs=params.train.epoch,
        verbose=1,
        workers=6,
        use_multiprocessing=False,
        callbacks=[chkpt_best, chkpt_latest, tensorboard, stopper]
    )


def main(args):
    params = utils.Hyperparams(args.settings)
    if params.ensemble:
        seq_types = ['FLAIR', 'T1w', 'T1wCE', 'T2w']
    else:
        seq_types = [params.data.seq_type]
    
    for seq_type in seq_types:
        print(f'==========  Training {seq_type}  ==========')
        utils.set_seed(params.seed)
        params.data.seq_type = seq_type
        exp_dir = args.exp_dir
        if params.ensemble:
            exp_dir += f'/{seq_type}'
        datagen_tr, datagen_val = data_prep(params.data, params.seed)
        train(exp_dir, params, datagen_tr, datagen_val)
    
    if params.ensemble:
        del params.data.seq_type
    params._save(f'{args.exp_dir}/train_params.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir', nargs='?', default=datetime.today().strftime('./exps/%Y%m%d'))
    parser.add_argument('--settings', default='default_params.json', help='Training hyperparameters.')
    main(parser.parse_args())
