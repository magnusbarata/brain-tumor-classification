import glob
import re
import os

import tensorflow as tf
import numpy as np
from skimage.transform import resize

from datagens import BaseDatagen

class VolumeDatagen(BaseDatagen):
    """3D volume data generator class.

    Args:
      samples: List of filenames.
      labels: List of label for each file. The labels must be integer.
      volume_size: Tuple of `(height, width, depth)` integer representing the size of the volume 
        which will be passed to the model.
      seq_type: Sequence type. Either `FLAIR`, `T1w`, `T1wCE`, or `T2w`.
      datadir: Root of data directory.
      dtype: String, data type of the volume.
    """
    def __init__(self, samples,
                 labels=None,
                 volume_size=(128,128,64),
                 seq_type='T1w',
                 datadir='/data',
                 dtype='float32',
                 **kwargs):
        super(VolumeDatagen, self).__init__(samples, **kwargs)
        self.labels = labels
        self.n_class = 0 if labels is None else len(np.unique(labels))
        self.mode = 'test' if labels is None else 'train'
        self.volume_size = volume_size
        self.seq_type = seq_type
        self.datadir = datadir
        self.dtype = dtype
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

    def load_vol(self, case_id):
        """Loads a volume array from case ID.
        
        Returns:
          Volume array with shape `(height, width, depth, channel)` after resize.
        """
        vol_dir = f'{self.datadir}/{self.mode}/{case_id}/{self.seq_type}'
        if os.path.isdir(vol_dir):
            # Stacking slices depthwise
            files = sorted(
                glob.glob(f'{vol_dir}/*.dcm'),
                key=lambda path: int(re.sub('\D', '', os.path.basename(path)))
            )
            vol = np.stack([self.get_dcm_arr(f) for f in files], axis=-1)
        else:
            vol = np.load(f'{vol_dir}.npy')
        vol = np.expand_dims(vol, -1)

        # Resize volume
        vol = resize(vol, self.volume_size, order=1, mode='constant', anti_aliasing=True)
        return vol.astype(self.dtype)