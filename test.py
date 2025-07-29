import os
import torch
import torch.distributed as dist
from datetime import timedelta

def setup(rank, world_size):
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]
    print(f"Process {rank}/{world_size} - MASTER_ADDR: {master_addr}, MASTER_PORT: {master_port}")
    
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=1800)
    )
    torch.cuda.set_device(rank % torch.cuda.device_count())

def cleanup():
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = int(os.environ["SLURM_NTASKS"])
    rank = int(os.environ["SLURM_PROCID"])
    setup(rank, world_size)
    try:
        # Your training code here
        print(f"Rank {rank} training complete")
    finally:
        cleanup()