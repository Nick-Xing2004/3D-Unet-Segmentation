import torch
import argparse
import os
from model import initialize_Unet3D
from train import train
from utils import customize_seed

def main(args):
    """
    Main function that initializes and trains the Unet model.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    #model intialization
    model = initialize_Unet3D(device, in_channels=7)         #now in_channels = 7 (with the positional encodings)

    #model training entrance
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
        "--epochs", type=int, default=200,
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