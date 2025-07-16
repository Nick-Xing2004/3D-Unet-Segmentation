import argparse
from utils import customize_seed
from best_model_validation_set_visualization import visualize_best_model_validation_set

def main(args):
    visualize_best_model_validation_set(args)
    print('🗺️visualization work done!')

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