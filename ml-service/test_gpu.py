import torch

def check_m1_gpu():
    print(f"PyTorch Version: {torch.__version__}")
    
    # MPS (Metal Performance Shaders) 
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        print("\nm1 gpu is ready.")
        print("ready to train.")
        
        # GPU calculate
        x = torch.ones(1, device=mps_device)
        print(f"Test Tensor on GPU: {x}")
    else:
        print("\nmps not found")

if __name__ == "__main__":
    check_m1_gpu()