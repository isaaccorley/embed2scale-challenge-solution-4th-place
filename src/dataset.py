import glob
import os
import random
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset
from torchvision import transforms

#  SSL4EO-S12 v1.1 stats
S2L1C_MEAN_TRAIN = [
    2607.345,
    2393.068,
    2320.225,
    2373.963,
    2562.536,
    3110.071,
    3392.832,
    3321.154,
    3583.77,
    1838.712,
    1021.753,
    3205.112,
    2545.798,
]
S2L1C_STD_TRAIN = [
    786.523,
    849.702,
    875.318,
    1143.578,
    1126.248,
    1161.98,
    1273.505,
    1246.79,
    1342.755,
    576.795,
    45.626,
    1340.347,
    1145.036,
]

S2L2A_MEAN_TRAIN = [
    1793.243,
    1924.863,
    2184.553,
    2340.936,
    2671.402,
    3240.082,
    3468.412,
    3563.244,
    3627.704,
    3711.071,
    3416.714,
    2849.625,
]
S2L2A_STD_TRAIN = [
    1160.144,
    1201.092,
    1219.943,
    1397.225,
    1400.035,
    1373.136,
    1429.17,
    1485.025,
    1447.836,
    1652.703,
    1471.002,
    1365.307,
]

S1GRD_MEAN_TRAIN = [-12.577, -20.265]
S1GRD_STD_TRAIN = [5.179, 5.872]

S2RGB_MEAN_TRAIN = [100.708, 87.489, 61.932]
S2RGB_STD_TRAIN = [68.550, 47.647, 40.592]

# Mean and standard devation for the challenge data per https://github.com/DLR-MF-DAS/embed2scale-challenge-supplement/issues/6
# Note that these are different from the SSL4EO-S12 v1.1 moments.
S1GRD_MEAN = [-11.834, -19.243]
S1GRD_STD = [4.305, 5.479]

S2L1C_MEAN = [
    1635.299,
    1402.885,
    1289.505,
    1281.272,
    1534.981,
    2272.474,
    2630.972,
    2587.956,
    2889.274,
    976.031,
    20.369,
    2109.307,
    1350.051,
]
S2L1C_STD = [
    1123.963,
    1187.2,
    1128.715,
    1322.882,
    1285.925,
    1250.079,
    1325.492,
    1294.318,
    1343.536,
    636.232,
    27.82,
    1023.855,
    834.773,
]

S2L2A_MEAN = [
    802.067,
    917.472,
    1130.01,
    1210.515,
    1587.985,
    2355.781,
    2650.339,
    2787.571,
    2860.466,
    2921.651,
    2221.172,
    1549.952,
]
S2L2A_STD = [
    1563.348,
    1604.422,
    1553.908,
    1622.652,
    1593.07,
    1523.264,
    1556.785,
    1618.961,
    1532.417,
    1653.532,
    1183.218,
    1025.306,
]


def collate_fn(batch):
    if isinstance(batch, dict) or isinstance(batch, torch.Tensor):
        # Single sample
        return batch
    elif isinstance(batch, list) and isinstance(batch[0], torch.Tensor):
        # Concatenate tensors along sample dim
        return torch.concat(batch, dim=0)
    elif isinstance(batch, list) and isinstance(batch[0], dict):
        if "file_name" in batch[0]:
            file_names = [sample["file_name"] for sample in batch]
            data = [sample["data"] for sample in batch]
            if isinstance(data[0], torch.Tensor):
                data = torch.concat(data, dim=0)
            elif isinstance(data[0], dict):
                data = {m: torch.concat([b[m] for b in data], dim=0) for m in data[0].keys()}
            return {"data": data, "file_name": file_names}
        else:
            # Concatenate each modality tensor along sample dim
            return {m: torch.concat([b[m] for b in batch], dim=0) for m in batch[0].keys()}


class SSL4EOS12Dataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        split_file: str | Path = None,
        modalities: list = None,
        transform: transforms.Compose | None = None,
        concat: bool = False,
        single_timestamp: bool = False,
        num_timestamps: int = 4,
        num_batch_samples: int | None = None,
    ):
        """
        Dataset class for the SSL4EOS12 V1.1 dataset.
        :param data_dir: Path to data directory of the selected split.
        :param split_file: optional, txt file with list of zarr.zip file name. Reduces initialization time.
        :param modalities: list of modalities folders, defaults to ['S2L1C', 'S2L2A', 'S1GRD'].
        :param transform: tranform function that processes a dict or numpy array (if concat=True).
        :param concat: Concatenate all modalities along the band dimension.
        :param single_timestamp: Loads a single timestamp instead of all four timestamps.
        :param num_batch_samples: Subsample samples in zarr files, e.g. if GPU memory is not sufficient.
        """
        self.data_dir = Path(data_dir)
        self.modalities = modalities or ["S2L1C", "S2L2A", "S1GRD"]
        self.transform = transform
        self.concat = concat
        self.num_batch_samples = num_batch_samples

        if split_file is not None:
            with open(split_file) as f:
                self.samples = f.read().splitlines()
        else:
            self.samples = sorted(os.listdir(self.data_dir / self.modalities[0]))
            self.samples = [f for f in self.samples if f.endswith(".zarr.zip")]

        self.single_timestamp = single_timestamp
        self.num_timestamps = num_timestamps
        if single_timestamp:
            # Repeat samples to include all timestamps in the dataset
            self.samples = np.repeat(self.samples, num_timestamps)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        :param idx: Index of zarr.zip file.
        :return: dict of modalities or tensor (if concat=True) with dims [B, T, C, H, W] or [B, C, H, W]
            (if single_timestamp=True).
        """
        data = {}
        # Load numpy values for each modality from zarr.zip files
        for modality in self.modalities:
            ds = xr.open_zarr(self.data_dir / modality / self.samples[idx])
            if self.single_timestamp:
                # Select a single timestamp
                ds = ds.isel(time=idx % self.num_timestamps)
            data[modality] = ds.bands.values

        num_samples = data[self.modalities[0]].shape[0]
        if self.num_batch_samples is not None and self.num_batch_samples != num_samples:
            # Subsample samples
            selected = random.sample(list(range(num_samples)), k=self.num_batch_samples)
            for modality in self.modalities:
                data[modality] = data[modality][selected]

        # Save band dims in case of dict outputs
        num_band_dims = {m: data[m].shape[-3] for m in self.modalities}
        band_dims_idx = {
            m: n for m, n in zip(self.modalities, [0] + np.cumsum(list(num_band_dims.values())).tolist(), strict=False)
        }

        # Concatenate along band dim for transform and convert to Tensor
        data = torch.Tensor(np.concatenate(list(data.values()), axis=-3))

        if self.transform is not None:
            data = self.transform(data)

        if not self.concat:
            # Split up modality data and return as dict
            data = {m: data[..., band_dims_idx[m] : band_dims_idx[m] + num_band_dims[m], :, :] for m in self.modalities}

        return data


class E2SChallengeDataset(Dataset):
    def __init__(
        self,
        data_path: str = None,
        transform=None,
        modalities: list[str] = None,
        dataset_name: str = "bands",
        seasons: int = 4,
        randomize_seasons: bool = False,
        concat: bool = True,
        output_file_name: bool = False,
    ):
        """Dataset class for the embed2scale challenge data

        Parameters
        ----------
        data_path : str, path-like
            Path to challenge data. Assumes that under data_path there are 3 subfolders, named after the modalities.
        transform : torch.Compose
            Transformations to apply to the data
        modalities : list[str]
            List of modalities to include. Should correpond to the subfolders under data_path.
        dataset_name : str
            Name of dataset in zarr archive. Use 'bands' here. Defaults to 'bands'.
        seasons : int
            Number of seasons to load. Must be integer between 1 and 4. Default is 4.
        randomize_seasons : bool
            Toggle randomized order of seasons. If True, the order of the seasons will be randomized. Default is False.
        concat : bool
            Toggle concatenating the modalities along the channel dimension. Default is True.
        output_file_name : bool
            Toggle output of the file name.

        Returns
        -------
        torch.Tensor or dict
            If output_file_name=False, outputs a torch.Tensor.
            If output_file_name=True, outputs a dictionary with fields 'data' and 'file_name'. 'data' is a torch.Tensor if concat=True and a dict with one field per modality, each containing a torch.Tensor if False. 'file_name' is the id of the loaded file.
        """

        self.data_path = data_path
        self.transform = transform
        self.modalities = modalities
        self.dataset_name = dataset_name
        assert isinstance(seasons, int) and (1 <= seasons <= 4), "Number of seasons must be integer between 1 and 4."

        self.seasons = seasons
        self.randomize_seasons = randomize_seasons
        if not randomize_seasons:
            self.possible_seasons = list(range(seasons))
        else:
            self.possible_seasons = list(range(4))
        assert len(modalities) > 0, "No modalities provided."
        self.concat = concat
        self.output_file_name = output_file_name

        self.samples = glob.glob(os.path.join(data_path, modalities[0], "*.zarr.zip"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = self.samples[idx]
        file_name = os.path.splitext(os.path.basename(sample_path))[0].replace(".zarr", "")
        if self.randomize_seasons:
            seasons = [
                self.possible_seasons[ind]
                for ind in torch.randperm(len(self.possible_seasons)).tolist()[: self.seasons]
            ]
        else:
            seasons = self.possible_seasons
        sample_paths = [sample_path] + [
            sample_path.replace(self.modalities[0] + os.sep, modality + os.sep) for modality in self.modalities[1:]
        ]
        data = {}

        for modality, sample_path in zip(self.modalities, sample_paths, strict=False):
            season_index = xr.DataArray(seasons, dims="time")
            data[modality] = xr.open_zarr(sample_path).isel(time=season_index)[self.dataset_name].values

        n_bands_per_modality = {m: d.shape[-3] for m, d in data.items()}
        start_ind_of_modality = {
            m: n
            for m, n in zip(
                self.modalities, [0] + np.cumsum(list(n_bands_per_modality.values())).tolist(), strict=False
            )
        }

        # Concatenate data
        data = np.concatenate(list(data.values()), axis=-3)
        data = data.astype(np.float32)  # uint16 before, but that type is not accepted by from_numpy()
        data = torch.from_numpy(data)

        # Transform
        if self.transform is not None:
            data = self.transform(data)

        if not self.concat:
            data = {
                m: data[..., start_ind_of_modality[m] : start_ind_of_modality[m] + n_bands_per_modality[m], :, :]
                for m in self.modalities
            }

        if self.output_file_name:
            return {"data": data, "file_name": file_name}
        else:
            return data


class NumpyDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform

        self.filenames = [fn.replace(".npy", "") for fn in sorted(os.listdir(self.root)) if fn.endswith(".npy")]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fn = os.path.join(self.root, self.filenames[idx] + ".npy")
        data = np.load(fn)
        data = torch.from_numpy(data).float()
        if self.transform:
            data = self.transform(data)
        return {"data": data, "file_name": self.filenames[idx]}
