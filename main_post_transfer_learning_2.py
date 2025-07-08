import torch
import argparse
import os
from model import initialize_Unet3D
from model_2 import initialize_Unet3D_2
from model_3 import initialize_Unet3D_3
from train import train
from utils import customize_seed

#the pipeline entrance after transfer learning    (both encoder & decoder weights copied from pretrained model)

def main(args):
    """
    Main function that initializes and trains the Unet model.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    device = "cuda:4" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("employing transfer learning: loading pretrained model weights for encoder and decoder, but not the final conv layer")
    
    #target model intialization
    target_model = initialize_Unet3D_2(device, out_channels=6)

    #transfer learning model loading
    pretrained_model = initialize_Unet3D_2(device, out_channels=5)
    #load the trained weights
    pretrained_model.load_state_dict(torch.load('Unet_3D_Yuyang_pretraining.pth'))

    pretrained_dict = pretrained_model.state_dict()
    target_dict = target_model.state_dict()
    #loading all layers' weights except the final conv layer
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in target_dict and not k.startswith("final_conv")}
    target_dict.update(pretrained_dict)
    target_model.load_state_dict(target_dict)
    
    #model training entrance
    train(target_model, args, device)

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
        "--epochs", type=int, default=100,
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