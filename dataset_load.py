import argparse
import os
import sys
import traceback
from dataclasses import dataclass
from socket import gethostname
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
# import wandb
import yaml
from datasets import load_from_disk
from datasets import table as ds_table
from sklearn.model_selection import KFold
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import AutoConfig
from transformers.models.bert.configuration_bert import BertConfig

from model import Seq2SeqTransformerWithNotes
from utils.bert_embeddings import MosaicBertForEmbeddingGenerationHF
from utils.eval import mapk, recallTop
from utils.train import (
    WarmupStableDecay,
    create_mask,
    create_source_mask,
    enforce_reproducibility,
    generate_square_subsequent_mask,
)
from utils.utils import (
    get_paths,
    load_data,
)

class ForcastWithNotes(Dataset):
    def __init__(
        self, source_sequences, target_sequences, hospital_ids, tokenized_notes
    ):
        self.source_sequences = source_sequences
        self.target_sequences = target_sequences
        self.hospital_ids = hospital_ids
        self.tokenized_notes = load_from_disk(tokenized_notes)

    def __len__(self):
        return len(self.source_sequences)

    def __getitem__(self, idx):
        hospital_ids = self.hospital_ids[idx]
        hospital_ids_lens = len(hospital_ids)

        return {
            "source_sequences": torch.tensor(self.source_sequences[idx]),
            "target_sequences": torch.tensor(self.target_sequences[idx]),
            "tokenized_notes": self.tokenized_notes[hospital_ids],
            "hospital_ids_lens": hospital_ids_lens,
        }

# 2) override it so any old‐path gets rewritten
def _patched_mm(path):
        # replace the old absolute prefix with your current one:
        path = path.replace(
            "/home/sifal.klioui/final_tokenized_reindex_hadm",
            "/home/rayane.aliouane/PatientTrajectoryForecasting/final_tokenized_reindex_hadm",
        )
        return _orig_mm(path)

if __name__ == "__main__":
    # 1) stash the original loader
    _orig_mm = ds_table._memory_mapped_arrow_table_from_file

    ds_table._memory_mapped_arrow_table_from_file = _patched_mm

    # 3) now load — the monkey‐patch will transparently fix every shard‐path
    dataset_obj = torch.load("final_dataset/dataset.pth")

