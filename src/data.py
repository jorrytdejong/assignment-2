from enum import Enum
import h5py
from glob import glob
import os

import numpy as np
import re
from tqdm import tqdm
from torch.utils.data import Dataset
import torch

def get_dataset_name(filenamewithdir):
    # print('Filename with directory:', filenamewithdir)
    return re.findall(r'/([a-zA-Z0-9_]+)_[0-9]*.h5', filenamewithdir)[0]



class DataSetType(Enum):
    CROSS = "Cross"
    INTRA = "Intra"
    
TASK_TO_LABEL = {
    "rest": 0,
    "task_motor": 1,
    "task_story_math": 2,
    "task_working_memory": 3,
}

def get_task_label(file_name: str) -> int:
    for task in TASK_TO_LABEL.keys():
        if task in file_name:
            return TASK_TO_LABEL[task]
    raise ValueError(f"Invalid task in file name: {file_name}")


def stride_downsample(matrix: np.ndarray, downsample_factor: int = 20) -> np.ndarray:
    if downsample_factor <= 1:
        return matrix
    return matrix[:, ::downsample_factor]


def block_mean_downsample(matrix: np.ndarray, downsample_factor: int = 20) -> np.ndarray:
    if downsample_factor <= 1:
        return matrix

    n_channels, n_time = matrix.shape
    n_full_blocks = n_time // downsample_factor
    downsampled_parts = []

    if n_full_blocks > 0:
        trimmed = matrix[:, : n_full_blocks * downsample_factor]
        block_means = trimmed.reshape(n_channels, n_full_blocks, downsample_factor).mean(axis=2)
        downsampled_parts.append(block_means)

    remainder_start = n_full_blocks * downsample_factor
    if remainder_start < n_time:
        tail_mean = matrix[:, remainder_start:].mean(axis=1, keepdims=True)
        downsampled_parts.append(tail_mean)

    return np.concatenate(downsampled_parts, axis=1)


def preprocess_meg(matrix: np.ndarray, downsample_factor: int = 20, mode: str = "stride") -> np.ndarray:
    matrix = matrix.astype(np.float64)

    if mode == "stride":
        matrix = stride_downsample(matrix, downsample_factor)
    elif mode == "block_mean":
        matrix = block_mean_downsample(matrix, downsample_factor)
    else:
        raise ValueError(f"Unknown preprocessing mode: {mode}")

    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)

    matrix = (matrix - mean) / std
    return matrix.astype(np.float32)


def make_window_starts(sequence_length: int, window_size: int, stride: int) -> list[int]:
    if sequence_length <= window_size:
        return [0]

    starts = list(range(0, sequence_length - window_size + 1, stride))
    final_start = sequence_length - window_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def downsampled_length(original_length: int, downsample_factor: int) -> int:
    return int(np.ceil(original_length / downsample_factor))


class DataSet(Dataset):
    def __init__(self, dataset_type: DataSetType, split: str = 'train'):
        super().__init__()
        self.dataset_type = dataset_type
        self.split = split
        
    def load(self):
        self.mean = None
        self.std = None
        self.files = []
        self.samples = []
        
        self.dataset_base = ""
        match self.dataset_type:
            case DataSetType.CROSS:
                self.dataset_base = "Cross"
            case DataSetType.INTRA:
                self.dataset_base = "Intra"
            case _:
                raise ValueError(f"Invalid dataset type: {self.dataset_type}")
                
        filenamepath = f"data/{self.dataset_base}/{self.split}"
        
        all_files = glob(os.path.join(filenamepath, "*.h5"))
        print('Found', len(all_files), 'files in folder', filenamepath)
        file_path = all_files[0]
        print('Opening file:', file_path)

        with h5py.File(file_path, 'r') as f:
            datasetname = get_dataset_name(file_path.replace('data/', ''))
            print('Dataset name:', datasetname)
            
            print(f.keys())
            matrix = f.get(datasetname)[()]
            print(type(matrix))
            print(matrix.shape)
            
    def __len__(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    def __getitem__(self, index): 
        raise NotImplementedError("Subclasses must implement this method")
            
            
class VQVAE_DataSet(DataSet):
    def __init__(self, dataset_type: DataSetType, split: str = 'train', window_size: int = 2048, window_stride: int = 2048):
        super().__init__(dataset_type, split)
        
        self.mean = None
        self.std = None
        
        self.files = []
        self.samples: list[tuple[str, str, int]] = []
        self.window_size = window_size
        self.window_stride = window_stride
        
    def load(self):
        
        self.dataset_base = ""
        match self.dataset_type:
            case DataSetType.CROSS:
                self.dataset_base = "Cross"
            case DataSetType.INTRA:
                self.dataset_base = "Intra"
            case _:
                raise ValueError(f"Invalid dataset type: {self.dataset_type}")
                
        filenamepath = f"data/{self.dataset_base}/{self.split}"
        
        all_files = glob(os.path.join(filenamepath, "*.h5"))
        if not all_files:
            raise FileNotFoundError(
                f"No .h5 files found in {filenamepath}. "
                f"Expected files like data/{self.dataset_base}/{self.split}/*.h5."
            )
        # print('Found', len(all_files), 'files in folder', filenamepath)
        # file_path = all_files[0]
        # print('Opening file:', file_path)
        
        for file_path in tqdm(all_files, desc='initializing dataset'):
            with h5py.File(file_path, 'r') as f:
                datasetname = get_dataset_name(file_path.replace('data/', ''))
                # print('Dataset name:', datasetname)
                
                dataset = f.get(datasetname)
                matrix = dataset[()]
                
                if self.mean is None:
                    self.mean = matrix.mean(axis=1)
                else:
                    self.mean += matrix.mean(axis=1)
                    
                if self.std is None:
                    self.std = matrix.std(axis=1)
                else:
                    self.std += matrix.std(axis=1)

                num_timesteps = matrix.shape[1]
                if num_timesteps <= self.window_size:
                    self.samples.append((file_path, datasetname, 0))
                else:
                    last_start = max(0, num_timesteps - self.window_size)
                    for start in range(0, last_start + 1, self.window_stride):
                        self.samples.append((file_path, datasetname, start))
                    if self.samples[-1][0] == file_path and self.samples[-1][2] != last_start:
                        self.samples.append((file_path, datasetname, last_start))
                    
        assert self.mean is not None and self.std is not None, "Mean and std must be initialized"
        self.mean /= len(all_files)
        self.std /= len(all_files)
        std_tensor = torch.as_tensor(self.std)
        std_tensor = torch.where(std_tensor == 0, torch.ones_like(std_tensor), std_tensor)
        self.std = std_tensor.numpy()
        
        self.files = all_files
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        file_path, datasetname, start = self.samples[index]
        with h5py.File(file_path, 'r') as f:
            dataset = f.get(datasetname)
            
            y = get_task_label(file_path)
            end = start + self.window_size
            window = dataset[:, start:end]
            x = torch.from_numpy((window - self.mean[:, None]) / self.std[:, None]).float().T
            
            return x, y


class MEGBaselineWindowDataset(DataSet):
    def __init__(
        self,
        dataset_type: DataSetType,
        split: str = 'train',
        downsample_factor: int = 20,
        window_size: int = 1024,
        window_stride: int = 512,
        preprocess_mode: str = "stride",
        return_file_index: bool = False,
    ):
        super().__init__(dataset_type, split)
        self.downsample_factor = downsample_factor
        self.window_size = window_size
        self.window_stride = window_stride
        self.preprocess_mode = preprocess_mode
        self.return_file_index = return_file_index
        self.files: list[str] = []
        self.samples: list[tuple[int, int]] = []
        self.cache: dict[int, np.ndarray] = {}

    def load(self):
        match self.dataset_type:
            case DataSetType.CROSS:
                self.dataset_base = "Cross"
            case DataSetType.INTRA:
                self.dataset_base = "Intra"
            case _:
                raise ValueError(f"Invalid dataset type: {self.dataset_type}")

        filenamepath = f"data/{self.dataset_base}/{self.split}"
        self.files = sorted(glob(os.path.join(filenamepath, "*.h5")))
        if not self.files:
            raise FileNotFoundError(
                f"No .h5 files found in {filenamepath}. "
                f"Expected files like data/{self.dataset_base}/{self.split}/*.h5."
            )

        self.samples = []
        for file_idx, file_path in enumerate(tqdm(self.files, desc='initializing baseline dataset')):
            with h5py.File(file_path, 'r') as f:
                datasetname = get_dataset_name(file_path.replace('data/', ''))
                matrix = f.get(datasetname)[()]

            reduced_time = downsampled_length(matrix.shape[1], self.downsample_factor)
            for start in make_window_starts(reduced_time, self.window_size, self.window_stride):
                self.samples.append((file_idx, start))

    def __len__(self) -> int:
        return len(self.samples)

    def _get_processed_file(self, file_idx: int) -> np.ndarray:
        if file_idx not in self.cache:
            file_path = self.files[file_idx]
            with h5py.File(file_path, 'r') as f:
                datasetname = get_dataset_name(file_path.replace('data/', ''))
                matrix = f.get(datasetname)[()]
            self.cache[file_idx] = preprocess_meg(
                matrix,
                downsample_factor=self.downsample_factor,
                mode=self.preprocess_mode,
            )
        return self.cache[file_idx]

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        file_idx, start = self.samples[index]
        file_path = self.files[file_idx]
        matrix = self._get_processed_file(file_idx)

        end = start + self.window_size
        window = matrix[:, start:end]
        if window.shape[1] < self.window_size:
            padding = self.window_size - window.shape[1]
            window = np.pad(window, ((0, 0), (0, padding)), mode="constant")

        x = torch.from_numpy(window).float().T
        y = get_task_label(file_path)
        if self.return_file_index:
            return x, y, file_idx
        return x, y
