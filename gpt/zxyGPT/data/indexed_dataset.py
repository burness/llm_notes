from functools import lru_cache
import os
import struct
import shutil
from itertools import accumulate

import numpy as np
import torch
from torch.utils.data import Dataset


def __best_fitting_dtype(vocab_size=None):
    if vocab_size is not None and vocab_size < 65500:
        return np.uint16
    else:
        return np.int32

dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,
    7: np.float64,
    8: np.uint16
}

def make_builder(out_file, impl, vocab_size=None):
    if impl == "mmap":
        return
    else:
        return 
def get_available_dataset_impl():
    return ["lazy", "cached", "mmap"]

def infer_dataset_impl(path):
    if IndexedDataset.exists(path):
        with open(index_file_path(path), "rb") as f:
            magic = f.read(8)
            if magic == IndexedDataset._HDR_MAGIC:
                return "cached"
            elif magic == MMa

def index_file_path(prefix_path):
    return prefix_path+".idx"

def data_file_path(prefix_path):
    return prefix_path+".bin"

def read_longs(f, n):
    a = np.empty(n, dtype=np.int64)
    f.readinto(a)
    return a

class IndexedDataset(Dataset):
    _HDR_MAGIC = b'TNTIDX\x00\x00'

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.data_file = None
        self.read_index(path)

    def read_index(self, path):
        with open(index_file_path(path), "rb") as f:
            magic = f.read(8)
            assert magic == self._HDR_MAGIC, (
                "Index file doesn\'t match expected format."
                "Make sure that --dataset-impl is configured properly."
            )
            version = f.read(8)
            assert struct.unpack("<Q", version) == (1,)
            code, self.element_size = struct.unpack("<QQ", f.read(16))
            self.dytpe = dtypes[code]
            self._len, self.s = struct.unpack("<QQ", f.read(16))
            self.doc_count = struct.unpack("<Q", f.read(8))
            self.dim_offsets = read_longs(f, self._len + 1)
            self.data_offsets = read_longs(f, self._len + 1)
            self.sizes = read_longs(f, self.s)
            self.doc_idx = read_longs(f, self.doc_count)
    
    def read_data(self, path):
        self.data_file = open(data_file_path(path), "rb", buffering=0)

    def check_index(self, i):
        if i < 0 or i >= self._len:
            raise IndexError("index out of range")
        
    def __del__(self):
        if self.data_file:
            self.data_file.close()

    def __getitem__(self, idx):
        if not self.data_file:
            self.read_data(self.path)
        if isinstance(idx, int):
            i = idx
            self.check_index(i)
            tensor_size = self.sizes[self.dim_offsets[i]:self.dim_offsets[i + 1]]
            a = np.empty(tensor_size, dtype=self.dytpe)
            self.data_file.seek(self.data_offsets[i]*self.element_size)
            self.data_file.readinto(a)
            return a
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            sizes = self.sizes[self.dim_offsets[start]:self.dim_offsets[stop]]
            size = sum(sizes)
            a = np.empty(size, dtype=self.dytpe)
            self.data_file.seek(self.data_offsets[start] * self.element_size)
            self.data_file.readinto(a)
            offsets = list(accumulate(sizes))
            sents = np.split(a, offsets[:-1])
            return sents
    
    def __len__(self):
        return self._len
    
    def num_tokens(self, index):
        return self.sizes[index]
    
    def size(self, index):
        return self.sizes[index]
    
    @staticmethod
    def exists(path):
        return (
            os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))
        )
    
    @property
    def supports_prefetch(self):
        return False
    

class MMapIndexedDataset(Dataset):
    class Index(object):
        _HDR_MAGIC = b"MMIDIDX\x00\x00"

        @classmethod
        def writer(cls, path, dtype):
            class _Writer(object):
                def __enter__(self):
                    self._file = open(path, "wb")
                    self._file.write(cls._HDR_MAGIC)