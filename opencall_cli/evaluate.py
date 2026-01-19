import sys
import os
pro_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pro_dir)

import argparse
import torch
import json
import numpy as np
import random
from opencall.utils.util import model_eval, init, get_dataset_in_one_dir, log_func


def get_parser():
    parser = argparse.ArgumentParser(description='Model Evaluation Script')
    
    # Model related
    parser.add_argument(
        "--model_dir", type=str, required=True,
        help="Directory containing the model config.toml file"
    )
    parser.add_argument(
        "--weight_path", type=str, required=True,
        help="Path to the model weights file (.tar)"
    )
    
    # Data related
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Directory containing the validation data"
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=16,
        help="Batch size for validation"
    )
    parser.add_argument(
        "--val_size", type=int, default=20000,
        help="Number of validation samples"
    )
    parser.add_argument(
        "--tokenization", type=str, default="kmer",
        help="Tokenization method"
    )
    
    # Device and precision
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device to run evaluation on (e.g., 'cuda:0', 'cpu')"
    )
    parser.add_argument(
        "--use_half", action="store_true", default=True,
        help="Use half precision (FP16) for inference"
    )
    
    # Output
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save evaluation results (defaults to model_dir)"
    )
    parser.add_argument(
        "--output_name", type=str, default="evaluation_results",
        help="Name for the output JSON file"
    )
    
    # Other
    parser.add_argument(
        "--seed", type=int, default=25,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--limit_train_size", type=int, default=0,
        help="Limit training size (for compatibility with data loading)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Training batch size (for compatibility with data loading)"
    )
    parser.add_argument(
        "--encoder_only", action="store_true", default=False,
        help="Use encoder-only mode"
    )
    parser.add_argument(
        "--check_sparsity", action="store_true", default=False,
        help="Check and report model sparsity (useful for pruned models)"
    )
    parser.add_argument(
        "--sparse_layers", type=str, default=None,
        help="Comma-separated list of layer names to check sparsity (e.g., 'encoder.0.conv.weight,encoder.1.conv.weight')"
    )
    
    return parser


def calculate_model_sparsity(model, layer_names=None):
    """
    Calculate the sparsity of the model.
    
    Args:
        model: The model to calculate sparsity for
        layer_names: Optional list of layer names to check. If None, check all parameters.
    
    Returns:
        tuple: (layer_sparsity_dict, total_sparsity)
    """
    total_params = 0
    total_zeros = 0
    layer_sparsity = {}
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if layer_names is None or name in layer_names:
                num_params = param.numel()
                num_zeros = (param == 0).sum().item()
                
                if num_params > 0:
                    layer_sparsity[name] = num_zeros / num_params
                    total_params += num_params
                    total_zeros += num_zeros
    
    total_sparsity = total_zeros / total_params if total_params > 0 else 0.0
    return layer_sparsity, total_sparsity


def main():
    # Parse arguments
    args = get_parser().parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Initialize
    device = args.device
    if device.startswith('cuda'):
        device_id = int(device.split(':')[1]) if ':' in device else 0
        torch.cuda.set_device(device_id)
        init(args.seed, device_id, deterministic=True)
    else:
        init(args.seed, device, deterministic=True)
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.model_dir
    os.makedirs(output_dir, exist_ok=True)
    
    log_path = os.path.join(output_dir, "evaluation.log")
    
    # Log evaluation parameters
    msg = "{} {} {}".format("=" * 20, "START EVALUATION", "=" * 20)
    log_func(msg, log_path)
    log_func("Evaluation Parameters:", log_path)
    for key, value in vars(args).items():
        log_func("  {}: {}".format(key, value), log_path)
    
    # Check if model files exist
    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
    
    config_path = os.path.join(args.model_dir, "config.toml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    if not os.path.exists(args.weight_path):
        raise FileNotFoundError(f"Weight file not found: {args.weight_path}")
    
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    
    # Determine device for evaluation
    device_for_eval = device_id if device.startswith('cuda') else device
    
    # Initialize sparsity variables
    layer_sparsity = {}
    total_sparsity = 0.0
    
    # Check model sparsity if requested
    if args.check_sparsity:
        from opencall.utils.util import network
        import toml
        
        log_func("Checking model sparsity...", log_path)
        
        # Load model to check sparsity
        config = toml.load(config_path)
        temp_device = device_for_eval if device.startswith('cuda') else 'cpu'
        temp_model = network(config_path).to(temp_device)
        temp_model.load_state_dict(torch.load(args.weight_path, map_location=device))
        
        # Parse sparse layer names if provided
        sparse_layer_names = None
        if args.sparse_layers:
            sparse_layer_names = [name.strip() for name in args.sparse_layers.split(',')]
        
        layer_sparsity, total_sparsity = calculate_model_sparsity(temp_model, sparse_layer_names)
        
        log_func("", log_path)
        log_func("{} {} {}".format("=" * 20, "MODEL SPARSITY", "=" * 20), log_path)
        log_func(f"Total Sparsity: {total_sparsity:.4f} ({total_sparsity*100:.2f}%)", log_path)
        log_func("Layer-wise Sparsity:", log_path)
        for layer_name, sparsity in sorted(layer_sparsity.items()):
            log_func(f"  {layer_name}: {sparsity:.4f} ({sparsity*100:.2f}%)", log_path)
        log_func("{} {} {}".format("=" * 40, "", "=" * 0), log_path)
        log_func("", log_path)
        
        # Clean up
        del temp_model
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    
    log_func("Loading validation data...", log_path)
    
    # Load validation data
    # We need to load dataset using the same function as training
    _, valid_loader = get_dataset_in_one_dir(args, dist=False, encoder_only=args.encoder_only)
    
    log_func(f"Validation dataset size: {len(valid_loader.dataset)}", log_path)
    log_func(f"Number of validation batches: {len(valid_loader)}", log_path)
    
    # Run evaluation
    log_func("Starting evaluation...", log_path)
    
    res = model_eval(
        dataloader=valid_loader,
        model_dir=args.model_dir,
        weight_path=args.weight_path,
        is_half=args.use_half,
        device=device_for_eval,
    )
    
    # Extract results
    mean_acc = res[0]
    median_acc = res[1]
    duration = res[2]
    samples_per_sec = res[3]
    bases_per_sec = res[4]
    val_chunks_num = res[5]
    
    # Log results
    log_func("", log_path)
    log_func("{} {} {}".format("=" * 20, "EVALUATION RESULTS", "=" * 20), log_path)
    log_func("Mean Accuracy:       {:.2f}%".format(mean_acc), log_path)
    log_func("Median Accuracy:     {:.2f}%".format(median_acc), log_path)
    log_func("Time Taken:          {:.2f}s".format(duration), log_path)
    log_func("Samples/sec:         {:.2E}".format(samples_per_sec), log_path)
    log_func("Bases/sec:           {:.2f}".format(bases_per_sec), log_path)
    log_func("Validation Chunks:   {:.0f}".format(val_chunks_num), log_path)
    log_func("{} {} {}".format("=" * 20, "FINISHED", "=" * 20), log_path)
    
    # Save results to JSON
    results_dict = {
        'mean_accuracy': round(mean_acc, 2),
        'median_accuracy': round(median_acc, 2),
        'duration_seconds': round(duration, 2),
        'samples_per_second': round(samples_per_sec, 2),
        'bases_per_second': round(bases_per_sec, 2),
        'validation_chunks_num': int(val_chunks_num),
        'model_dir': args.model_dir,
        'weight_path': args.weight_path,
        'data_dir': args.data_dir,
    }
    
    # Add sparsity information if checked
    if args.check_sparsity:
        results_dict['total_sparsity'] = round(total_sparsity, 4)
        results_dict['layer_sparsity'] = {k: round(v, 4) for k, v in layer_sparsity.items()}
    
    output_json_path = os.path.join(output_dir, f"{args.output_name}.json")
    with open(output_json_path, 'w') as json_file:
        json.dump(results_dict, json_file, indent=4)
    
    log_func(f"Results saved to: {output_json_path}", log_path)
    
    # Print summary to console
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Mean Accuracy:       {mean_acc:.2f}%")
    print(f"Median Accuracy:     {median_acc:.2f}%")
    print(f"Time Taken:          {duration:.2f}s")
    print(f"Samples/sec:         {samples_per_sec:.2E}")
    print(f"Bases/sec:           {bases_per_sec:.2f}")
    print(f"Validation Chunks:   {val_chunks_num:.0f}")
    print(f"\nResults saved to: {output_json_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

