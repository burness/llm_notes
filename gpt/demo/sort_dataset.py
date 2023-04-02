import torch
import pickle
from torch.utils.data import Dataset

class SortDataset(Dataset):
    def __init__(self, split, length=6, num_digits=3):
        assert split in {"train", "test"}
        self.split = split
        self.length = length
        self.num_digits = num_digits

    def __len__(self):
        return 10000
    
    def get_vocab_size(self):
        return self.num_digits
    
    def get_block_size(self):
        return self.length*2 -1
    
    def __getitem__(self, idx):
        while True:
            inp = torch.randint(self.num_digits, size = (self.length, ), dtype=torch.long)
            if torch.rand(1).item() < 0.5:
                if inp.unique().nelement() > self.length // 2:
                    continue
            h = hash(pickle.dumps(inp.tolist()))
            inp_split = "test" if h % 4 == 0 else "train"
            if inp_split == self.split:
                break
        sol = torch.sort(inp)[0]
        cat = torch.cat((inp, sol), dim=0)

        x = cat[:-1].clone()
        y = cat[1:].clone()
        y[:self.length-1] = -1
        return x,y
    
if __name__ == "__main__":
    sort_dataset = SortDataset("train")
    for x, y in sort_dataset:
        print(x, y)