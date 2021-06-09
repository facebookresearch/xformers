# Credits: Nystromformer repo
# https://github.com/mlpen/Nystromformer

import logging
import pickle

import torch
from torch.utils.data.dataset import Dataset

logging.getLogger().setLevel(logging.INFO)


class LRADataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, "rb") as f:
            self.examples = pickle.load(f)

        logging.info(f"Loaded {file_path}... size={len(self.examples)}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.create_inst(self.examples[i])

    @staticmethod
    def create_inst(inst):
        output = {"input_ids_0": torch.tensor(inst["input_ids_0"], dtype=torch.long)}
        output["mask_0"] = (output["input_ids_0"] != 0).float()

        if "input_ids_1" in inst:
            output["input_ids_1"] = torch.tensor(inst["input_ids_1"], dtype=torch.long)
            output["mask_1"] = (output["input_ids_1"] != 0).float()
        output["label"] = torch.tensor(inst["label"], dtype=torch.long)
        return output
