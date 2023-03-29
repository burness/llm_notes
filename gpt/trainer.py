import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
from gpt.utils import CfgNode as CN

class Trainer:
    @staticmethod
    def get_default_config():
        c = CN()
        c.device = "auto"
        c.num_workers = 4
        c.max_iters = None
        c.batch_size = 64
        c.learning_rate = 3e-4
        c.betas = (0.9, 0.95)
        c.weight_decay = 0.1
        c.grad_norm_clip = 1.0
        return c
    
    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device
        
        self.model = self.model.to(self.device)
        print("running on device", self.device)
        
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent:str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent:str, callback):
        self.callbacks[onevent] = [callback]

    def tr