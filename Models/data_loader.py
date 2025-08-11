# import os
# import sys
# import logging
# from glob import glob
# import h5py
# import numpy as np
# from multiprocessing import Pool
# from collections import Counter
# from sklearn.preprocessing import LabelEncoder, LabelBinarizer
# from sklearn.utils.class_weight import compute_class_weight

# from typing import Optional, Callable, List


# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
# )
# logger = logging.getLogger(__name__)


# def fft(x):
#     x = np.fft.rfft(x)[:, 1:]
#     x = np.abs(x) + 1
#     x = np.log10(x)
#     return x


# class DASDataLoader:
#     def __init__(
#         self,
#         data_dir: str,
#         sample_len: int,
#         transform: Callable,
#         fsize: int = 8192,
#         shift: int = 2048,
#         drop_noise: bool = True,
#         decimate: Optional[dict] = None,
#     ):
#         self.data_dir = data_dir
#         self.fsize = fsize
#         self.shift = shift
#         self.drop_noise = drop_noise
#         self.sample_len = sample_len
#         self.decimate = decimate or {}
#         self.transform = transform

#     def _parse_label(self, label):
#         samples = []
#         logger.info(f"Parsing directory structures for [{label}].")
#         for ds in glob(os.path.join(self.data_dir, label, "*.h5")):
#             logger.info(f"Parsing dataset >> {ds}")
#             with h5py.File(ds) as f:
#                 data = f["Acquisition"]["Raw[0]"]["RawData"]
#                 bmp = np.load(ds[:-2] + "npy")

#                 if label in self.decimate:
#                     wins = self.generate_windows(
#                         data, bmp, self.decimate[label])
#                 else:
#                     wins = self.generate_windows(data, bmp)
#             samples.extend(wins)
#         logger.warning(f" The number of samples for {label} is {len(samples)}")
#         return samples, [label] * len(samples)

#     def parse_dataset(self):
#         samples = []
#         labels = []

#         with Pool(12) as p:
#             results = p.map(self._parse_label, os.listdir(self.data_dir))
#             for s, l in results:
#                 samples.extend(s)
#                 labels.extend(l)

#         logger.info("Preprocessing windows")
#         self.samples = self.preprocess_windows(np.array(samples))
#         if self.drop_noise:
#             logger.info("Dropping noise")
#             self.samples, labels = self.drop_bad_samples(
#                 self.samples, samples, labels)

#         logger.info("Getting labels")
#         self.str_labels = np.array(labels)
#         self.labels = self.encode_labels(labels)
#         return self.samples, self.labels

#     def preprocess_windows(self, wins):
#         logger.info(f"Applying {self.transform} on the dataset")
#         samples = self.transform(wins)
#         if self.sample_len:
#             if len(samples.shape) == 2:
#                 samples = samples[:, : self.sample_len]
#             elif len(samples.shape) == 3:
#                 samples = samples[:, : self.sample_len]
#             else:
#                 raise ValueError
#         return samples

#     def encode_labels(self, labels: List[str]):
#         counts = Counter(labels)
#         logger.warning(f"The samples per category counts >> {counts}")

#         labels = np.array(labels).reshape(-1, 1)

#         # class weight computes requires different encoding
#         _lab = LabelEncoder().fit_transform(labels)
#         self.class_weights = compute_class_weight(
#             "balanced",
#             classes=np.unique(_lab),
#             y=_lab,
#         )
#         logger.warning(
#             f"The dataset has following class weights {self.class_weights}.")

#         self.encoder = LabelBinarizer()
#         labels = self.encoder.fit_transform(labels)
#         labels = labels.astype(np.float32)
#         logger.warning(f"LabelEncoder labels >> {self.encoder.classes_}!")
#         return labels

#     def generate_windows(self, series, bmp: np.array, decimate=1) -> List[np.array]:
#         windows = []
#         for pos, channel in np.transpose(np.where(bmp))[::decimate]:
#             windows.append(
#                 np.array(
#                     series[
#                         pos * self.shift: (pos * self.shift + self.fsize),
#                         channel,
#                     ],
#                     dtype=np.float32,
#                 )
#             )

#             if windows[-1].shape != (self.fsize,):
#                 windows.pop(-1)
#         return windows

#     @staticmethod
#     def drop_bad_samples(data: np.array, unprocessed_data: np.array, labels: List[str]):
#         def spec_cmp(x: np.array):
#             split = int(len(x) * 0.1)
#             if np.mean(x[:split]) - np.mean(x[split:]) > 0.05:
#                 return True
#             else:
#                 return False

#         labels = np.array(labels)
#         cond = np.apply_along_axis(
#             spec_cmp,
#             1,
#             fft(unprocessed_data),
#         )
#         ignored = labels == "regular"
#         cond = cond | ignored
#         data = data[cond]
#         labels = labels[cond]
#         return data, labels.tolist()


# def main():
#     decim_dict = {
#     'car': 1,
#     'longboard': 2,
#     'manipulation': 3,
#     'openclose': 1,
#     'regular': 50,
#     'running': 3,
#     'walk': 2,
#     'fence': 1,  
#     'construction': 3,  
#     }
#     parser = DASDataLoader(
#         '../data',
#         2048,
#         transform=fft,
#         fsize=8192,
#         shift=2048,
#         decimate=decim_dict,
#     )

#     x, y = parser.parse_dataset()

#     print(x, y)
#     print(f'The dataset contains {len(x)} elements')

#     return 0


# if __name__ == "__main__":
#     sys.exit(main())








import os
import logging
from glob import glob
import h5py
import numpy as np
from multiprocessing import Pool
from collections import Counter
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
from typing import Optional, Callable, List, Tuple

# Initialize logger for tracking processing steps
logger = logging.getLogger(__name__)

def fft(x: np.array) -> np.array:
    """
    Apply Fast Fourier Transform (FFT) to the input signal.

    :param x: Input 2D array where each row represents a signal window.
    :return: Transformed array with logarithmic scaling applied.
    """
    x = np.fft.rfft(x)[:, 1:]  # Compute FFT and remove the DC component
    x = np.abs(x) + 1  # Convert to absolute values and add 1 to avoid log(0)
    x = np.log10(x)  # Apply logarithmic scaling
    return x


class DASDataLoader:
    """
    A data loader for DAS (Distributed Acoustic Sensing) dataset.
    It supports loading, preprocessing, window segmentation, and labeling of the data.

    Attributes:
        data_dir (str): Path to the dataset directory containing labeled subdirectories.
        sample_len (int): Length of sample windows (trimmed if necessary).
        transform (Callable): Preprocessing function (e.g., FFT) applied to the dataset.
        fsize (int): Window size for sliding window segmentation.
        shift (int): Step size for the sliding window (overlap depends on this value).
        drop_noise (bool): Whether to drop noisy samples based on spectral analysis.
        decimate (dict): Dictionary specifying the decimation factor for each label.
    """

    def __init__(
            self,
            data_dir: str,
            sample_len: int,
            transform: Callable,
            fsize: int = 4096,
            shift: int = 2048,
            drop_noise: bool = True,
            decimate: Optional[dict] = None,
    ):
        self.data_dir = data_dir
        self.fsize = fsize
        self.shift = shift
        self.drop_noise = drop_noise
        self.sample_len = sample_len
        self.decimate = decimate or {}  # Default to empty dict if no decimation is provided
        self.transform = transform

    def _parse_label(self, label: str):
        """
        Parses a specific label directory in the dataset, extracts signal windows, and assigns labels.

        :param label: Name of the subdirectory representing a label.
        :return: A tuple containing a list of signal windows and their corresponding labels.
        """
        samples = []
        logger.info(f"Parsing dataset for label [{label}]...")

        # Iterate over all .h5 files in the label directory
        for ds in glob(os.path.join(self.data_dir, label, "*.h5")):
            logger.info(f"Processing file: {ds}")
            with h5py.File(ds) as f:
                data = f["Acquisition"]["Raw[0]"]["RawData"]  # Extract raw DAS data
                bmp = np.load(ds[:-2] + "npy")  # Load bitmap marking event positions

                # Apply decimation if specified for the label
                decimation_factor = self.decimate.get(label, 1)
                wins = self.generate_windows(data, bmp, decimation_factor)
            
            # print(wins.shape)
            samples.extend(wins)

        logger.warning(f"Total samples extracted for {label}: {len(samples)}")
        return samples, [label] * len(samples)

    def parse_dataset(self):
        """
        Parses the entire dataset, applying preprocessing, noise filtering, and encoding labels.

        :return: A tuple containing the preprocessed feature samples and corresponding labels.
        """
        samples = []
        labels = []

        # Use multiprocessing to speed up label parsing across different directories
        with Pool(12) as p:
            results = p.map(self._parse_label, os.listdir(self.data_dir))
            for s, l in results:
                samples.extend(s)
                labels.extend(l)

        logger.info("Applying preprocessing transformation to dataset...")
        self.samples = self.preprocess_windows(np.array(samples))

        # Optionally drop noisy samples
        if self.drop_noise:
            logger.info("Filtering out noisy samples...")
            self.samples, labels = self.drop_bad_samples(self.samples, samples, labels)

        # Encode labels using One-Hot Encoding
        logger.info("Encoding labels...")
        self.str_labels = np.array(labels)
        self.labels = self.encode_labels(labels)
        return self.samples, self.labels

    def preprocess_windows(self, wins: np.array) -> np.array:
        """
        Applies preprocessing to the extracted signal windows.

        :param wins: Array of extracted signal windows.
        :return: Preprocessed array with proper trimming and verification.
        """
        logger.info(f"Applying preprocessing function: {self.transform}")
        samples = self.transform(wins)

        # Trim samples to the required sample length
        if self.sample_len:
            if len(samples.shape) == 2:
                samples = samples[:, : self.sample_len]
            elif len(samples.shape) == 3:
                samples = samples[:, : self.sample_len]
            else:
                raise ValueError("Unexpected sample shape!")

        return samples

    def encode_labels(self, labels: List[str]) -> np.array:
        """
        Encodes string labels into one-hot encoding and computes class weights to handle imbalance.

        :param labels: List of string labels.
        :return: One-hot encoded label array.
        """
        counts = Counter(labels)
        logger.warning(f"Sample counts per category: {counts}")

        labels = np.array(labels).reshape(-1, 1)
        # print("Les différents labels : ", labels)
        # Compute class weights to handle class imbalance
        encoded_labels = LabelEncoder().fit_transform(labels)
        self.class_weights = compute_class_weight(
            "balanced",
            classes=np.unique(encoded_labels),
            y=encoded_labels,
        )
        logger.warning(f"Computed class weights: {self.class_weights}")

        # One-hot encode labels for training
        self.encoder = LabelBinarizer()
        labels = self.encoder.fit_transform(labels).astype(np.float32)
        logger.warning(f"Label encoding complete: {self.encoder.classes_}")

        return labels

    def generate_windows(self, series: np.array, bmp: np.array, decimate: int = 1) -> List[np.array]:
        """
        Generates signal windows based on a sliding window approach and event markers.

        :param series: 2D array containing raw signal data (time x channels).
        :param bmp: 2D array marking events (1 where an event occurs).
        :param decimate: Decimation factor to reduce the number of extracted windows.
        :return: List of extracted 1D signal windows.
        """
        windows = []
        # Iterate over bitmap locations where an event is marked (value=1)
        for pos, channel in np.transpose(np.where(bmp))[::decimate]:
            # window = series[pos * self.shift: (pos * self.shift + self.fsize), channel]
            window = series[pos * self.shift : pos * self.shift + self.fsize, channel]

            windows.append(np.array(window, dtype=np.float32))
            # print(f"Fenêtre extraite shape : {window.shape}")
            # Ensure window has correct shape
            assert windows[-1].shape == (self.fsize,), "Error in array indexing!"

        return windows

    @staticmethod
    def drop_bad_samples(data: np.array, unprocessed_data: np.array, labels: List[str]) -> Tuple[np.array, np.array]:
        """
        Filters out noisy samples based on spectral analysis.

        :param data: Preprocessed dataset.
        :param unprocessed_data: Original dataset before transformation.
        :param labels: Corresponding labels.
        :return: Tuple containing the filtered dataset and labels.
        """
        def spec_cmp(x: np.array) -> bool:
            """
            Checks if the spectral density difference in the first 10% of data is significant.

            :param x: Input spectral density array.
            :return: Boolean indicating whether the sample should be kept.
            """
            split = int(len(x) * 0.1)
            return np.mean(x[:split]) - np.mean(x[split:]) > 0.05

        labels = np.array(labels)
        
        # Apply spectral comparison on the dataset
        cond = np.apply_along_axis(spec_cmp, 1, fft(unprocessed_data))

        # Ensure 'regular' label samples are always kept
        cond |= labels == "regular"

        # Filter dataset based on noise conditions
        return data[cond], labels[cond].tolist()

