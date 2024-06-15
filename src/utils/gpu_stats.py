import torch


def get_gpu_memory():
    memory_usage = {}
    for i in range(torch.cuda.device_count()):
        memory_usage[f'GPU {i}'] = f'{torch.cuda.memory_allocated(i) / (1024 ** 2):.2f} MB'  # Memory in MB
    return memory_usage
