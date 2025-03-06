import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

def get_file_path(root_dir, dataset_type, preprocess_method, split, cf_number):
    """
    Construct the file path based on the FI2010 file naming and folder structure.
    
    Expected structure:
      {root_dir}/{dataset_type}/
         {prefix}.{dataset_type}_{preprocess_method}/
             {dataset_type}_{preprocess_method}_{split}/
                 {FilePrefix}_Dst_{dataset_type}_{FilePreprocess}_CF_{cf_number}.txt
                 
    Where:
      - prefix is "1" for Zscore, "2" for MinMax, "3" for DecPre.
      - FilePrefix is "Train" for Training and "Test" for Testing.
      - FilePreprocess uses "ZScore" (capital S) for Zscore, and remains the same for others.
    
    Example for Auction Zscore Training CF_1:
      ./Experiments/dataset/BenchmarkDatasets/Auction/1.Auction_Zscore/Auction_Zscore_Training/Train_Dst_Auction_ZScore_CF_1.txt
    """
    prefix_map = {"Zscore": "1", "MinMax": "2", "DecPre": "3"}
    folder_preprocess = {"Zscore": "Zscore", "MinMax": "MinMax", "DecPre": "DecPre"}
    file_preprocess = {"Zscore": "ZScore", "MinMax": "MinMax", "DecPre": "DecPre"}
    
    if dataset_type not in ["Auction", "NoAuction"]:
        raise ValueError("dataset_type must be 'Auction' or 'NoAuction'.")
    if preprocess_method not in prefix_map:
        raise ValueError("preprocess_method must be 'Zscore', 'MinMax', or 'DecPre'.")
    if split not in ["Training", "Testing"]:
        raise ValueError("split must be 'Training' or 'Testing'.")
    if not (1 <= cf_number <= 9):
        raise ValueError("cf_number must be between 1 and 9.")
    
    folder = f"{prefix_map[preprocess_method]}.{dataset_type}_{folder_preprocess[preprocess_method]}"
    file_prefix = "Train" if split == "Training" else "Test"
    filename = f"{file_prefix}_Dst_{dataset_type}_{file_preprocess[preprocess_method]}_CF_{cf_number}.txt"
    
    full_path = os.path.join(root_dir,
                             dataset_type,
                             folder,
                             f"{dataset_type}_{folder_preprocess[preprocess_method]}_{split}",
                             filename)
    return full_path

def load_fi2010_dataset(file_path):
    """
    Load the FI2010 dataset from a text file.
    
    The file contains 149 rows:
      - Rows 0 to 143: 144 features per sample.
      - Rows 144 to 148: 5 label rows (for 5 classification tasks; 1: up, 2: stationary, 3: down).
      
    Since each column in the file represents one sample, np.loadtxt returns an array of shape (149, num_samples).
    We transpose it to obtain a shape of (num_samples, 149), then split into features and labels.
    
    Returns:
      features: NumPy array of shape (num_samples, 144)
      labels:   NumPy array of shape (num_samples, 5)
    """
    data = np.loadtxt(file_path)
    # Transpose comme ça chaque échantillon devient une ligne
    data = data.T  # now shape is (num_samples, 149)
    features = data[:, :144]
    labels = data[:, 144:149]
    return features, labels

class FI2010Dataset(Dataset):
    """
    PyTorch Dataset for the FI2010 dataset.
    
    It loads one or more text files (each containing both features and labels),
    transposes them so that each sample is a row, and concatenates all samples.
    Each sample is returned as a tuple (features, labels), with features as a float tensor
    and labels as a long tensor.
    """
    def __init__(self, file_paths, transform=None):
        """
        Parameters:
          file_paths (list of str): List of full paths to FI2010 dataset files.
          transform (callable, optional): Optional transform to apply on features.
        """
        self.transform = transform
        features_list = []
        labels_list = []
        for fp in tqdm(file_paths, total=len(file_paths), desc="Loading files"):
            try:
                feat, lab = load_fi2010_dataset(fp)
                features_list.append(feat)
                labels_list.append(lab)
            except Exception as e:
                print(f"Error loading file {fp}: {e}")
        if not features_list:
            raise ValueError("No files were loaded. Check your file paths and structure.")
        self.features = np.concatenate(features_list, axis=0)
        self.labels = np.concatenate(labels_list, axis=0)
    
    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.transform:
            x = self.transform(x)
        return x, y

# ------------------------------------------------------------------------------
# Create DataLoader objects for training and testing
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    root_path = "../../dataset/BenchmarkDatasets"
    
    training_file_paths = []
    testing_file_paths = []
    
    for cf in range(1, 10):
        try:
            train_fp = get_file_path(root_path, "Auction", "Zscore", "Training", cf)
            test_fp  = get_file_path(root_path, "Auction", "Zscore", "Testing", cf)
            training_file_paths.append(train_fp)
            testing_file_paths.append(test_fp)
        except Exception as e:
            print(f"Error constructing file path for CF {cf}: {e}")
    
    train_dataset = FI2010Dataset(training_file_paths)
    test_dataset = FI2010Dataset(testing_file_paths)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print("Training dataset features shape:", train_dataset.features.shape)
    print("Training dataset labels shape:", train_dataset.labels.shape)
    print("Testing dataset features shape:", test_dataset.features.shape)
    print("Testing dataset labels shape:", test_dataset.labels.shape)
    
    for batch in train_loader:
        inputs, labels = batch
        print("Batch inputs shape:", inputs.shape)
        print("Batch labels shape:", labels.shape)
        break
