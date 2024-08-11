import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

class WeatherDataset(Dataset):
    def __init__(self, data_paths):
        self.data_paths = data_paths
        self.valid_paths = self.filter_valid_files()

    def filter_valid_files(self):
        valid_paths = []
        for file_path in self.data_paths:
            with h5py.File(file_path, 'r') as f:
                if 'rates.crop' in f.keys():
                    valid_paths.append(file_path)
                else:
                    print(f"Skipping file: {file_path} (no 'rates.crop' key)")
        print(f"Filtered dataset: {len(valid_paths)} valid files found.")
        return valid_paths

    def __len__(self):
        return len(self.valid_paths)

    def __getitem__(self, idx):
        file_path = self.valid_paths[idx]
        print(f"Processing file: {file_path}")
        with h5py.File(file_path, 'r') as f:
            data = f['rates.crop'][:]

            lead_times = f.get('lead_times', np.array([0]))
            hrrr_input_2496 = f.get('hrrr_input_2496', np.zeros((617, 624, 624)))
            hrrr_stale_state = f.get('hrrr_stale_state', np.zeros((1, 624, 624)))
            input_2496 = f.get('input_2496', np.zeros((39, 624, 624)))
            input_4996 = f.get('input_4996', np.zeros((17, 624, 624)))
            hrrr_target = f.get('hrrr_target', np.zeros((617, 128, 128)))

            precipitation_targets = {
                'mrms_rate': f.get('precipitation_targets/mrms_rate', np.zeros((512, 512))),
                'mrms_accumulation': f.get('precipitation_targets/mrms_accumulation', np.zeros((512, 512)))
            }

            surface_targets = {
                'omo_temperature': f.get('surface_targets/omo_temperature', np.zeros((128, 128))),
                'omo_dew_point': f.get('surface_targets/omo_dew_point', np.zeros((128, 128))),
                'omo_wind_speed': f.get('surface_targets/omo_wind_speed', np.zeros((128, 128))),
                'omo_wind_component_x': f.get('surface_targets/omo_wind_component_x', np.zeros((128, 128))),
                'omo_wind_component_y': f.get('surface_targets/omo_wind_component_y', np.zeros((128, 128)))
            }

            return {
                'lead_times': torch.tensor(lead_times, dtype=torch.long).squeeze(),
                'hrrr_input_2496': torch.tensor(hrrr_input_2496, dtype=torch.float32),
                'hrrr_stale_state': torch.tensor(hrrr_stale_state, dtype=torch.float32),
                'input_2496': torch.tensor(input_2496, dtype=torch.float32),
                'input_4996': torch.tensor(input_4996, dtype=torch.float32),
                'precipitation_targets': {k: torch.tensor(v, dtype=torch.long) for k, v in precipitation_targets.items()},
                'surface_targets': {k: torch.tensor(v, dtype=torch.float32) for k, v in surface_targets.items()},
                'hrrr_target': torch.tensor(hrrr_target, dtype=torch.float32)
            }

def get_data_paths(data_dir):
    data_paths = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".h5"):
                data_paths.append(os.path.join(root, file))
    return data_paths

# Date Path
train_data_dir = "/home/cdp/anaconda3/envs/jiaqi/metnet3-pytorch-main/train_data"
val_data_dir = "/home/cdp/anaconda3/envs/jiaqi/metnet3-pytorch-main/val_data"

train_data_paths = get_data_paths(train_data_dir)
val_data_paths = get_data_paths(val_data_dir)

# Dateset
train_dataset = WeatherDataset(train_data_paths)
val_dataset = WeatherDataset(val_data_paths)

# Batch size and number workers
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
