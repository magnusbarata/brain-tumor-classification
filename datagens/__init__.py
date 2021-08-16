"""Provides custom data generators to be fed to the model."""
from tensorflow import keras
import pydicom as dcm
import nibabel as nib
import numpy as np
import imageio

class BaseDatagen(keras.utils.Sequence):
    """Base data generator class.
    
    This class serves as a custom data generator that loads data from a list of filenames.
    You must implement the `__getitem__` method when inheriting this class.
    The instance of this class can be used (i.e. passed to `model.fit()`) as is or converted
    into `tf.Dataset` using `tf.data.Dataset.from_generator()`.

    Example:
    ```python
    # Here `files` is a list of filenames.
    datagen = BaseDatagen(files)
    dataset = tf.data.Dataset.from_generator(datagen, output_signature=(tf.TensorSpec(shape=datagen.x_shape)))
    ```
    
    Args:
      samples: List of filenames.
      batch_size: Number of elements to be provided on each batch.
      shuffle: Whether the data should be shuffled on every epoch.
      drop_remainder: Whether to drop the last batch, if the last batch has fewer
        elements than the `batch_size`.
    """
    def __init__(self, samples,
                 batch_size=32,
                 shuffle=True,
                 drop_remainder=False):
        self.samples = samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_remainder = drop_remainder
        self.on_epoch_end()

    def on_epoch_end(self):
        """Shuffle the data at the end of every epoch."""
        self.indices = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        """Number of batches."""
        if self.drop_remainder:
            return len(self.samples) // self.batch_size
        return np.ceil(len(self.samples) / self.batch_size).astype(int)

    def __getitem__(self, idx):
        """Gets batch at position `idx`."""
        raise NotImplementedError

    def __call__(self):
        """Return an iterator when the instance of this class if invoked."""
        return iter(self)

    def __iter__(self):
        """Initialize the iterator."""
        self.idx = 0
        return self

    def __next__(self):
        """Return the next element in the sequence."""
        if self.idx < len(self):
            result = self[self.idx]
            self.idx += 1
            return result
        else:
            raise StopIteration

    @staticmethod
    def get_img_arr(fname):
        """Static method to read from ordinary image file such as 'png', 'jpeg', etc.
        
        Returns:
          ndarray representation of an image.
        """
        return imageio.imread(fname)

    @staticmethod
    def get_dcm_arr(fname):
        """Static method to read dicom image.

        The pixel values of the image are converted into Hounsfield Unit (HU).
        
        Returns:
          ndarray representations of a dicom image.
        """
        ds = dcm.dcmread(fname)
        return ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept

    @staticmethod
    def get_nii_arr(fname):
        """Static method to read nifti image.
        
        Returns:
          ndarray representations of a nifti image.
        """
        return nib.load(fname).get_fdata()