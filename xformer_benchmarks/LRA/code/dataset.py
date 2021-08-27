# Credits: Nystromformer repo
# https://github.com/mlpen/Nystromformer

import logging
import pickle

import torch
from torch.utils.data.dataset import Dataset

logging.getLogger().setLevel(logging.INFO)


class LRADataset(Dataset):
    def __init__(self, file_path, seq_len = 1024):
        with open(file_path, "rb") as f:
            self.examples = pickle.load(f)

        self.seq_len = seq_len
        logging.info(f"Loaded {file_path}... size={len(self.examples)}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.create_inst(self.examples[i], self.seq_len)

    @staticmethod
    def create_inst(inst, seq_len = 1024):
        output = {"input_ids_0": torch.tensor(inst["input_ids_0"], dtype=torch.long)[:seq_len]}
        output["mask_0"] = (output["input_ids_0"] != 0).float()

        if "input_ids_1" in inst:
            output["input_ids_1"] = torch.tensor(inst["input_ids_1"], dtype=torch.long)[:seq_len]
            output["mask_1"] = (output["input_ids_1"] != 0).float()
        output["label"] = torch.tensor(inst["label"], dtype=torch.long)
        return output
