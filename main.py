import torch
import argparse
import os
from model import initialize_Unet3D
from model_2 import initialize_Unet3D_2
from model_3 import initialize_Unet3D_3
from train import train
from utils import customize_seed
from train_boundary_loss import train_with_boundary_loss

def main(args):
    """
    Main function that initializes and trains the Unet model.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """

    # assert torch.cuda.device_count() >= 2, "Requires at least 2 GPUs"

    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using {torch.cuda.device_count()} GPUs")
    
    #model intialization
    model = initialize_Unet3D_2(device, out_channels=6) 

    #mutli-GPU training setup, packagin the model into DataParallel
    # model = torch.nn.DataParallel(model, device_ids=[0, 1])      #using GPU 0, 1; but can later specify in command using 'CUDA_VISIBLE_DEVICES=5,6'
    # model = model.to(device)

    #model training entrance
    # train_with_boundary_loss(model, args, device)
    train(model, args, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training script for anatomical landmark segmentation with 3D U-Net."
    )

    # Reproducibility settings
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for reproducibility"
    )
    
    # Training parameters
    parser.add_argument(
        "--batch_size", type=int, default=2,
        help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=4000,
        help="Number of epochs for training"
    )
    parser.add_argument(
        "--experiment_name", type=str, default='unet3d_hip_segmentation_Yuyang',
        help="Name of the experiment directory"
    )

    args = parser.parse_args()

    # Fix randomness
    customize_seed(args.seed)

    main(args)     # Main function call