import os
import torch
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoader:
    
    def __init__(self, B, T, split, step=None, config=None):
        self.B = B
        self.T = T
        assert split in {"train", "val"}

        data_root = "data/datasets/FW_tokenized_20B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        print(f"found {len(shards)} shards for split {split}")

        self.current_shard = 0
        self.current_position = 0

        if step is not None:
            position = config.total_batch_size * step
            self.current_shard = position // config.shard_size
            self.current_position = position - (self.current_shard * config.shard_size)

        self.tokens = load_tokens(self.shards[self.current_shard])

    def reset(self):
        self.current_shard = 0
        self.current_position = 0
        self.tokens = load_tokens(self.shards[self.current_shard])

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T+1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)

        self.current_position += B*T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T

        return x, y
