import pandas as pd
import numpy as np
import torch
import os

from torch.utils.data import Dataset, DataLoader


class MultiViewFrameDataset(Dataset):
    """Multi-view surveillance dataset of horses in their box."""

    def __init__(self, data_selection_path, views, image_size, seq_length,
                 train_subjects, test_subjects):
        """
        Args:
        data_dir_path: str (path to directory containing the extracted frames)
        views: [int] (which viewpoints to include in the dataset, e.g. all [0,1,2,3])
                      The viewpoints are indexed starting from "front left" (FL=0) and
                      then clock-wise -- FR=1, BR=2, BL=3. Front being the side facing
                      the corridor, and R/L as defined from inside of the box.)
        image_size: (int, int)
        seq_length: int (1 for single frames)
        train_subjects: [str] (list of subject names)
        test_subjects: [str] (list of subject names)
        """
        self.data_dir_path = data_dir_path
        self.views = views
        self.image_size = image_size
        self.seq_length = seq_length
        self.train_subjects = train_subjects
        self.test_subjects = test_subjects


