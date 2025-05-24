from torch.utils.data import Dataset
from utils.npytar import NpyTarReader
import os
import numpy as np

class ShapeNet(Dataset):
    def __init__(self, filename, input_shape = (1, 32, 32, 32)):
        super().__init__()
        self.is_tar = filename.endswith('.tar')
        
        if self.is_tar:
            reader = NpyTarReader(filename)
            self.xc = np.zeros((reader.length(), ) + input_shape, dtype = np.float32)
            reader.reopen() 
            for ix, (x, name) in enumerate(reader):
                self.xc[ix] = x.astype(np.float32)
        else:
            files = os.listdir(filename)
            self.xc = np.zeros((len(files), ) + input_shape, dtype = np.float32)
            self.reader = None

    def __len__(self):
        return len(self.xc)

    def __getitem__(self, index):
        x = self.xc[index]
        return 3.0 * x - 1.0 # normalize to [-1, 2] for matching the VAE
        # return x