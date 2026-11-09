import os, sys, torch, torch.distributed as dist
os.environ.setdefault("USE_LIBUV", "0")
def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    backend = "gloo" if sys.platform == "win32" else "nccl"
    dist.init_process_group(backend=backend)
    dev = torch.device(f"cuda:{local_rank}")
    x = torch.randn(2048, 2048, device=dev)
    for i in range(50):
        x = x @ x
        x = x / (x.norm() + 1e-6)
    dist.all_reduce(x.detach().sum().cpu())  # gloo-friendly sync via cpu scalar
    torch.cuda.synchronize()
    print(f"rank={rank} local_rank={local_rank} device={dev} current={torch.cuda.current_device()} name={torch.cuda.get_device_name(local_rank)} ok", flush=True)
    dist.destroy_process_group()
if __name__ == "__main__":
    main()
