from collections import Counter
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Sampler:
    @staticmethod
    def oversample_balance(scalar_features: pd.DataFrame, multidim_features: List[Dict],
                           y: np.ndarray) -> Tuple[pd.DataFrame, List[Dict], np.ndarray]:
        """
        Oversamples the minority classes to match the majority class.

        Args:
            scalar_features (pd.DataFrame): the scalar feature data.
            multidim_features (List[Dict]): the multidimensional feature data.
            y (np.ndarray): The target data.

        Returns:
            Tuple[pd.DataFrame, List[Dict], np.ndarray]: Balanced scalar_features, multidim_features, and y.
        """
        class_counts = Counter(y)
        max_count = max(class_counts.values())

        balanced_indices = []
        for class_label, count in class_counts.items():
            class_indices = np.where(y == class_label)[0]
            balanced_indices.extend(np.random.choice(class_indices, max_count, replace=True))

        return Sampler._resample(scalar_features, multidim_features, y, balanced_indices)

    @staticmethod
    def undersample_balance(scalar_features: pd.DataFrame, multidim_features: List[Dict],
                            y: np.ndarray) -> Tuple[pd.DataFrame, List[Dict], np.ndarray]:
        """
        Undersamples the majority classes to match the minority class.

        Args:
            scalar_features (pd.DataFrame): the scalar feature data.
            multidim_features (List[Dict]): the multidimensional feature data.
            y (np.ndarray): The target data.

        Returns:
            Tuple[pd.DataFrame, List[Dict], np.ndarray]: Balanced scalar_feature, multidim_features, and y.
        """
        class_counts = Counter(y)
        min_count = min(class_counts.values())

        balanced_indices = []
        for class_label, count in class_counts.items():
            class_indices = np.where(y == class_label)[0]
            balanced_indices.extend(np.random.choice(class_indices, min_count, replace=False))

        return Sampler._resample(scalar_features, multidim_features, y, balanced_indices)

    @staticmethod
    def combined_balance(scalar_features: pd.DataFrame, multidim_features: List[Dict], y: np.ndarray,
                         target_ratio: float = 0.5) -> Tuple[pd.DataFrame, List[Dict], np.ndarray]:
        """
        Combines oversampling and undersampling to balance the dataset.

        Args:
            scalar_features (pd.DataFrame): the scalar feature data.
            multidim_features (List[Dict]): the multidimensional feature data.
            y (np.ndarray): The target data.
            target_ratio (float): The desired ratio of the most common class to target.

        Returns:
            Tuple[pd.DataFrame, List[Dict], np.ndarray]: Balanced scalar_features, multidim_features, and y.
        """

        class_counts = Counter(y)
        majority_class = max(class_counts, key=class_counts.get)
        majority_count = class_counts[majority_class]
        target_count = int(majority_count * target_ratio)

        balanced_indices = []
        for class_label, count in class_counts.items():
            class_indices = np.where(y == class_label)[0]
            if count > target_count:
                balanced_indices.extend(np.random.choice(class_indices, target_count, replace=False))
            else:
                balanced_indices.extend(np.random.choice(class_indices, target_count, replace=True))

        return Sampler._resample(scalar_features, multidim_features, y, balanced_indices)

    @staticmethod
    def split_data(scalar_features: pd.DataFrame, multidim_features: List[Dict], y: np.ndarray,
                   test_size: float = 0.2, random_state: int = None) \
            -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict], List[Dict], np.ndarray, np.ndarray]:
        """
        Perform a train-test split on the provided data (scalar_features, multidim_features, and y).

        Args:
            scalar_features (pd.DataFrame): the scalar feature data.
            multidim_features (List[Dict]): the multidimensional feature data.
            y (np.ndarray): The target data.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): The seed value for randomness.

        Returns:
            Tuple containing:
            - scalar_features_train (pd.DataFrame): Training set for scalar_features
            - scalar_features_test (pd.DataFrame): Test set for scalar_features
            - multidim_features_train (List[Dict]): Training set for multidim_features
            - multidim_features_test (List[Dict]): Test set for multidim_features
            - y_train (np.ndarray): Training set for y
            - y_test (np.ndarray): Test set for y
        """
        if len(scalar_features) != len(multidim_features) or len(scalar_features) != len(y):
            raise ValueError("scalar_features, multidim_features, and y must have the same length")

        indices = np.arange(len(y))
        train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state)

        scalar_features_train, scalar_features_test = scalar_features.iloc[train_indices], scalar_features.iloc[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        multidim_features_train, multidim_features_test = [multidim_features[i] for i in train_indices], [multidim_features[i] for i in test_indices]

        return scalar_features_train, scalar_features_test, multidim_features_train, multidim_features_test, y_train, y_test

    @staticmethod
    def _resample(scalar_features: pd.DataFrame, multidim_features: List[Dict], y: np.ndarray,
                  indices: List[int]) -> Tuple[pd.DataFrame, List[Dict], np.ndarray]:
        """
        Resamples the data based on given indices.

        Args:
            scalar_features (pd.DataFrame): the scalar feature data.
            multidim_features (List[Dict]): the multidimensional feature data.
            y (np.ndarray): The target data.
            indices (List[int]): The indices to use for resampling.

        Returns:
            Tuple[pd.DataFrame, List[Dict], np.ndarray]: Resampled scalar_features, multidim_features, and y.
        """
        return (
            scalar_features.iloc[indices].reset_index(drop=True),
            [multidim_features[i] for i in indices],
            y[indices]
        )
