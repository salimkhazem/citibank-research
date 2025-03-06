import torch 
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from sklearn.model_selection import KFold
from dataset import get_file_path, FI2010Dataset

class FI2010DataLoader:
    """
    DataLoader for the FI2010 dataset supporting both standard and cross-validation splits.
    
    The FI2010 dataset has two main configurations:
    1. Standard split: Train on specific folds, test on others
    2. Cross-validation: Use the 9 folds with the anchored CV method as described in the dataset
    """
    
    def __init__(self, 
                 root_path,
                 dataset_type="Auction",  # "Auction" or "NoAuction"
                 preprocess_method="Zscore", # "Zscore", "MinMax", or "DecPre"
                 batch_size=32,
                 shuffle_train=True,
                 num_workers=4,
                 pin_memory=True):
        """
        Initialize the DataLoader with dataset configuration.
        
        Args:
            root_path (str): Root directory containing the FI2010 dataset
            dataset_type (str): "Auction" or "NoAuction"
            preprocess_method (str): "Zscore", "MinMax", or "DecPre" 
            batch_size (int): Batch size for DataLoader
            shuffle_train (bool): Whether to shuffle training data
            num_workers (int): Number of worker processes for data loading
            pin_memory (bool): Whether to pin memory in DataLoader for faster GPU transfer
        """
        self.root_path = root_path
        self.dataset_type = dataset_type
        self.preprocess_method = preprocess_method
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Verify dataset structure
        self._verify_dataset_structure()
    
    def _verify_dataset_structure(self):
        """Verify that the dataset files exist with expected structure"""
        try:
            # Check if at least fold 1 exists to validate dataset structure
            train_path = get_file_path(self.root_path, self.dataset_type, 
                                      self.preprocess_method, "Training", 1)
            test_path = get_file_path(self.root_path, self.dataset_type, 
                                     self.preprocess_method, "Testing", 1)
            
            if not os.path.exists(train_path):
                raise FileNotFoundError(f"Training file not found at: {train_path}")
            if not os.path.exists(test_path):
                raise FileNotFoundError(f"Testing file not found at: {test_path}")
                
            print(f"Dataset structure verified for {self.dataset_type} with {self.preprocess_method} normalization")
        except Exception as e:
            raise RuntimeError(f"Dataset structure verification failed: {e}")
    
    def get_standard_loaders(self, train_folds, test_folds):
        """
        Get standard train and test DataLoaders using specific folds.
        
        Args:
            train_folds (list): List of fold numbers to use for training (1-9)
            test_folds (list): List of fold numbers to use for testing (1-9)
            
        Returns:
            tuple: (train_loader, test_loader)
        """
        # Validate fold numbers
        for fold in train_folds + test_folds:
            if not (1 <= fold <= 9):
                raise ValueError(f"Fold number {fold} is invalid. Must be between 1 and 9.")
        
        # Construct file paths for training
        train_file_paths = []
        for cf in train_folds:
            train_path = get_file_path(self.root_path, self.dataset_type, 
                                     self.preprocess_method, "Training", cf)
            train_file_paths.append(train_path)
        
        # Construct file paths for testing
        test_file_paths = []
        for cf in test_folds:
            test_path = get_file_path(self.root_path, self.dataset_type, 
                                    self.preprocess_method, "Testing", cf)
            test_file_paths.append(test_path)
        
        # Create datasets
        train_dataset = FI2010Dataset(train_file_paths)
        test_dataset = FI2010Dataset(test_file_paths)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        print(f"Train dataset: {len(train_dataset)} samples from folds {train_folds}")
        print(f"Test dataset: {len(test_dataset)} samples from folds {test_folds}")
        
        return train_loader, test_loader
    
    def get_anchored_cross_validation_loaders(self, k=9):
        """
        Get data loaders for anchored cross-validation as described in the FI2010 dataset.
        
        In anchored cross-validation for this dataset:
        - Fold 1 contains 1 day training, 1 day testing
        - Fold 2 contains 2 days training (fold 1 train+test), 1 day testing
        - Fold 3 contains 3 days training (fold 1+2 train+test), 1 day testing
        - And so on...
        
        Args:
            k (int): Number of folds (default: 9)
            
        Returns:
            list: List of (train_loader, test_loader) tuples for each fold
        """
        if k != 9:
            print(f"Warning: Dataset is designed with 9 folds, but {k} was requested.")
            k = min(k, 9)
        
        cv_loaders = []
        
        for fold in range(1, k+1):
            # For anchored cross-validation:
            # - Train data includes all previous days (folds 1 to fold-1 train+test data)
            # - Test data is just the current fold's test data

            # Get training data (all previous days)
            train_file_paths = []
            for prev_fold in range(1, fold):
                # Add previous fold's training data
                train_path = get_file_path(self.root_path, self.dataset_type, 
                                         self.preprocess_method, "Training", prev_fold)
                train_file_paths.append(train_path)
                
                # Add previous fold's testing data
                test_path = get_file_path(self.root_path, self.dataset_type, 
                                        self.preprocess_method, "Testing", prev_fold)
                train_file_paths.append(test_path)
            
            # Add current fold's training data
            train_path = get_file_path(self.root_path, self.dataset_type, 
                                     self.preprocess_method, "Training", fold)
            train_file_paths.append(train_path)
            
            # Get testing data (just current fold)
            test_file_paths = [get_file_path(self.root_path, self.dataset_type, 
                                           self.preprocess_method, "Testing", fold)]
            
            # Create datasets
            train_dataset = FI2010Dataset(train_file_paths)
            test_dataset = FI2010Dataset(test_file_paths)
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.batch_size,
                shuffle=self.shuffle_train,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
            
            test_loader = DataLoader(
                test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
            
            print(f"Fold {fold} - Train: {len(train_dataset)} samples, Test: {len(test_dataset)} samples")
            cv_loaders.append((train_loader, test_loader))
        
        return cv_loaders
    
    def get_leave_one_fold_out_cv_loaders(self):
        """
        Get data loaders for leave-one-fold-out cross-validation.
        
        Each fold uses 8 folds for training and 1 fold for testing.
        
        Returns:
            list: List of (train_loader, test_loader) tuples for each fold
        """
        cv_loaders = []
        
        for test_fold in range(1, 10):
            # Training folds are all folds except the test fold
            train_folds = [i for i in range(1, 10) if i != test_fold]
            
            # Get loaders for this fold
            train_loader, test_loader = self.get_standard_loaders(train_folds, [test_fold])
            cv_loaders.append((train_loader, test_loader))
            
            print(f"Leave-one-fold-out CV - Fold {test_fold} - "
                  f"Training on folds {train_folds}, Testing on fold {test_fold}")
        
        return cv_loaders


# Usage example
if __name__ == "__main__":
    root_path = "../../dataset/BenchmarkDatasets"
    
    # Create data loader
    fi2010_loader = FI2010DataLoader(
        root_path=root_path,
        dataset_type="Auction",
        preprocess_method="Zscore",
        batch_size=32
    )
    
    print("\n--- Standard DataLoader ---")
    # Example 1: Standard train/test split (training on folds 1-7, testing on folds 8-9)
    train_loader, test_loader = fi2010_loader.get_standard_loaders(
        train_folds=[1, 2, 3, 4, 5, 6, 7],
        test_folds=[8, 9]
    )
    
    # Check shapes of loaded data
    for inputs, labels in train_loader:
        print(f"Batch features shape: {inputs.shape}")
        print(f"Batch labels shape: {labels.shape}")
        break
    
    print("\n--- Anchored Cross-Validation ---")
    # Example 2: Anchored cross-validation
    cv_loaders = fi2010_loader.get_anchored_cross_validation_loaders()
    print(f"Number of CV folds: {len(cv_loaders)}")
    
    # Example for the first fold
    train_loader, test_loader = cv_loaders[0]
    for inputs, labels in train_loader:
        print(f"First fold training batch shape: {inputs.shape}")
        break
    
    print("\n--- Leave-One-Fold-Out Cross-Validation ---")
    # Example 3: Leave-one-fold-out cross-validation
    locv_loaders = fi2010_loader.get_leave_one_fold_out_cv_loaders()
    print(f"Number of LOCV folds: {len(locv_loaders)}")