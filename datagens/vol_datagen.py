import glob
import re
import os

import tensorflow as tf
import numpy as np
from skimage.transform import resize
import volumentations as volaug
from ..utils import snake_to_camel

from datagens import BaseDatagen

class VolumeDatagen(BaseDatagen):
    """3D volume data generator class.

    Args:
      samples: List of filenames.
      labels: Numpy arrays of label for each file. The labels must be integer.
        If not provided, default to `None`, which generate data without labels
        (useful for testing).
      volume_size: Tuple of `(height, width, depth)` integer representing the size of the volume 
        which will be passed to the model.
      seq_type: Sequence type. Either `FLAIR`, `T1w`, `T1wCE`, `T2w`, or `ALL`.
        If `ALL`, then all sequence types will be appended into the channel.
      datadir: Root of data directory.
      dtype: String, data type of the volume.
      augmentations: List of dictionaries. Each dictionary denotes what augmentation to apply
        and the parameters that the augmentation function accepts. For list of accepted 
        augmentations, please refer to the repo ZFTurbo/volumentations. This variable accepts
        the structure below (put it as a parameter inside `data`):
        ```
        "augmentations": [
            {
                <func_name_in_snake_case>: {
                    <param_0>: <value_0>,
                    <param_1>: <value_1>,
                    ...
                }
            },
            ...
        ]
        ```
        Don't forget to convert the parameters to JSON compatible structure.
    """
    def __init__(self, samples,
                 labels=None,
                 volume_size=(128,128,64),
                 seq_type='T1w',
                 datadir='/data',
                 dtype='float32',
                 augmentations=None,
                 **kwargs):
        super(VolumeDatagen, self).__init__(samples, **kwargs)
        self.labels = labels
        self.n_class = 0 if labels is None else len(np.unique(labels))
        self.mode = 'test' if labels is None else 'train'
        self.volume_size = volume_size
        self.seq_type = seq_type
        self.datadir = datadir
        self.dtype = dtype
        self.augmentations = augmentations
        self._set_shape()

    def _set_shape(self):
        """Deduce the shape of the generated batch"""
        item = self[0]
        if self.labels is None:
            self.x_shape = (None, *item.shape[1:])
            self.y_shape = None
        else:
            self.x_shape = (None, *item[0].shape[1:])
            self.y_shape = (None, self.n_class)
    
    def to_dataset(self):
        """Convert generator to `tf.data`"""
        if self.labels is None:
            signature = (tf.TensorSpec(self.x_shape))
        else:
            signature = (tf.TensorSpec(self.x_shape), tf.TensorSpec(self.y_shape))
        return tf.data.Dataset.from_generator(self, output_signature=signature)
    
    def __getitem__(self, idx):
        """Gets batch at position `idx`."""
        b_indices = self.indices[idx*self.batch_size : (idx + 1)*self.batch_size]
        X = np.array([self.load_vol(self.samples[index]) for index in b_indices])

        if self.labels is None:
            return X
        return X, tf.keras.utils.to_categorical(self.labels[b_indices], self.n_class)

    def get_augmentation(self):
        """Get augmentation functions.

        Returns:
          Either:
            None, if there is no augmentation specified, or
            composed augmentation functions specified in self.augmentations.
        """
        if self.augmentations is None:
            return None

        return volaug.Compose(
            [
                getattr(
                    volaug, 
                    snake_to_camel(list(augmentation.keys())[0])
                )(**list(augmentation.values())[0])
                for augmentation in self.augmentations
            ], p=1.0
        )

    def load_vol(self, case_id):
        """Loads a volume array from case ID.
        
        Returns:
          Volume array with shape `(height, width, depth, channel)` after resize.
        """
        if self.seq_type == 'ALL':
            seq_types = ['FLAIR', 'T1w', 'T1wCE', 'T2w']
        else:
            seq_types = [self.seq_type]
        
        vol = np.empty((*self.volume_size, len(seq_types)))
        aug = self.get_augmentation()
        for channel, seq_type in enumerate(seq_types):
            vol_dir = f'{self.datadir}/{self.mode}/{case_id}/{seq_type}'
            if os.path.isdir(vol_dir):
                # Stacking slices depthwise
                files = sorted(
                    glob.glob(f'{vol_dir}/*.dcm'),
                    key=lambda path: int(re.sub('\D', '', os.path.basename(path)))
                )
                s_vol = np.stack([self.get_dcm_arr(f) for f in files], axis=-1)
            else:
                s_vol = np.load(f'{vol_dir}.npy')

            # Resize volume
            vol[:,:,:,channel] = resize(
                s_vol, self.volume_size, order=1, mode='constant', anti_aliasing=True
            )

            # Augment
            if aug:
                vol[:,:,:,channel] = aug(image=vol[:,:,:,channel])['image']

        return vol.astype(self.dtype)
