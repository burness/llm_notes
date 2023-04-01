import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
from zxyGPT.utils import CfgNode as CN

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

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        self.optimizer = model.configure_optimizers(config)

        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch
            logits, self.loss = model(x, y)
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.trigger_callbacks("on_batch_end")
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break