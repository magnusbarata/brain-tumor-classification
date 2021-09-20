import glob
import tensorflow as tf
import numpy as np
from datagens import BaseDatagen

class ImageDatagen(BaseDatagen):
    """2D image data generator class.

    Args:
      samples: List of case ID.
      labels: List of label for each case ID. The labels must be integer.
      image_size: Tuple of `(height, width)` integer representing the size of the image 
        which will be passed to the model.
      seq_type: Sequence type. Either `FLAIR`, `T1w`, `T1wCE`, or `T2w`.
      datadir: Root of data directory.
      dtype: String, data type of the image.
      aug_fns: List of functions to be applied on the image for data augmentation.
    """
    def __init__(self, samples,
                 labels=None,
                 image_size=(256,256),
                 seq_type='T1w',
                 datadir='/data',
                 dtype='float32',
                 aug_fns=None,
                 **kwargs):
        super(ImageDatagen, self).__init__(samples, **kwargs)
        self.labels = labels
        self.n_class = 0 if labels is None else len(np.unique(labels))
        self.mode = 'test' if labels is None else 'train'
        self.image_size = image_size
        self.seq_type = seq_type
        self.datadir = datadir
        self.dtype = dtype
        if aug_fns:
            assert isinstance(aug_fns, list), 'aug_fns must be a list of functions.'
        self.aug_fns = aug_fns
        self._convert_2d_dataset()
        self._set_shape()
        self.on_epoch_end()

    def _set_shape(self):
        """Deduce the shape of the generated batch"""
        item = self[0]
        if self.labels is None:
            self.x_shape = (None, *item.shape[1:])
            self.y_shape = None
        else:
            self.x_shape = (None, *item[0].shape[1:])
            self.y_shape = (None, self.n_class)
            
    def _convert_2d_dataset(self):
        """Convert case ID to filenames, labels"""
        new_samples = []
        if self.labels is None:
            for case_id in self.samples:
                files = glob.glob(f'{self.datadir}/{self.mode}/{case_id}/{self.seq_type}/*.dcm')
                new_samples.extend(files)
            self.samples = np.array(new_samples)
        else:
            new_labels = []
            for case_id,mgmt in zip(self.samples, self.labels):
                files = glob.glob(f'{self.datadir}/{self.mode}/{case_id}/{self.seq_type}/*.dcm')
                new_samples.extend(files)
                new_labels.extend([mgmt] * len(files))
            self.samples, self.labels = np.array(new_samples), np.array(new_labels)

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
        X = np.array([self.load_img(self.samples[index]) for index in b_indices])

        if self.labels is None:
            return X
        return X, tf.keras.utils.to_categorical(self.labels[b_indices], self.n_class)

    def load_img(self, fname):
        """Loads an image array from filename.

        This method supports nifti, dicom, and ordinary ('png', 'jpg', ...) image file.
        
        Returns:
          Image array of `(height, width, channel)` after resize and data augmentations.
        """
        ext = fname.split('.', maxsplit=1)[-1].lower()
        if ext == 'nii' or ext == 'nii.gz':
            img = self.get_nii_arr(fname)
        elif ext == 'dcm':
            img = self.get_dcm_arr(fname)
        else:
            img = self.get_img_arr(fname)
        
        img = img.astype(self.dtype)
        if img.ndim < 3:
            img = np.expand_dims(img, -1)

        if self.aug_fns:
            img = self.apply_augmentations(img)
        return tf.keras.preprocessing.image.smart_resize(img, self.image_size)

    def apply_augmentations(self, img):
        """Apply data augmentations to a single image array."""
        for fn in self.aug_fns:
            img = fn(img)
        return img