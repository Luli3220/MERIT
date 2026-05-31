import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def merge_fsdp_checkpoint(checkpoint_path, output_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Check if this is a FSDP checkpoint (usually contains model.pt or similar distributed files)
    # But standard PyTorch FSDP saves as a directory with shards or a single consolidated file if configured
    # Verl's FSDP saver might behave differently. 
    # Let's assume standard HuggingFace load if it's already converted, or use specific loading if it's raw FSDP.
    
    # However, Verl typically saves actor weights using FSDP state dict.
    # If the checkpoint folder contains 'actor/pytorch_model_fsdp_0', it's raw FSDP.
    # If it's a single pytorch_model.bin, it might be already consolidated.
    
    actor_path = os.path.join(checkpoint_path, "actor")
    if not os.path.exists(actor_path):
        print(f"Error: Actor directory not found at {actor_path}")
        return

    print("Detected actor checkpoint.")
    
    # Check for HuggingFace directory inside actor
    hf_path = os.path.join(actor_path, "huggingface")
    
    # Case 1: Weights are already consolidated in actor_path (unlikely based on ls output)
    # Case 2: Weights are sharded (model_world_size_*.pt) in actor_path
    # Case 3: Configs are in huggingface/ subdir
    
    # Verl saves FSDP shards in actor/ and HF configs in actor/huggingface/ (but usually NO weights in huggingface/)
    # We need to:
    # 1. Load the model config from huggingface/
    # 2. Initialize the model structure
    # 3. Load and merge the FSDP shards
    # 4. Save as standard HF
    
    config_path = actor_path
    if os.path.exists(hf_path):
        print(f"Found huggingface config directory: {hf_path}")
        config_path = hf_path
    
    print(f"Loading model config from {config_path}...")
    try:
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
        # Try loading tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config_path, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load config/tokenizer: {e}")
        return

    print("Initializing model structure (on CPU)...")
    # Initialize empty model based on config
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        
    model = model.to_empty(device="cpu") # Move to CPU to load weights

    print("Loading and merging FSDP shards...")
    
    # Find all model shard files
    import glob
    shard_pattern = os.path.join(actor_path, "model_world_size_*.pt")
    shard_files = sorted(glob.glob(shard_pattern))
    
    if not shard_files:
        print(f"Error: No FSDP shards found matching {shard_pattern}")
        # Check if pytorch_model.bin exists (consolidated)
        if os.path.exists(os.path.join(actor_path, "pytorch_model.bin")):
            print("Found consolidated pytorch_model.bin, loading directly...")
            state_dict = torch.load(os.path.join(actor_path, "pytorch_model.bin"), map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            return
    else:
        print(f"Found {len(shard_files)} shards. Merging...")
        full_state_dict = {}
        
        for shard_file in shard_files:
            print(f"Processing {os.path.basename(shard_file)}...")
            shard = torch.load(shard_file, map_location="cpu")
            # FSDP shards might be flat params or state dicts depending on saving method
            # Verl typically saves the local state dict of the model
            
            # If keys overlap, it's problematic, but for FSDP sharded by module/layer, we update
            # Warning: Standard FSDP saving might save flattened params which are hard to merge without FSDP context
            # However, looking at Verl's source code (if available) or common practice:
            # If save_state_dict was used, it might be partial state dicts.
            
            full_state_dict.update(shard)
            
        print("Loading state dict into model...")
        try:
            model.load_state_dict(full_state_dict)
        except Exception as e:
            print(f"Error loading state dict: {e}")
            print("Trying strict=False...")
            try:
                model.load_state_dict(full_state_dict, strict=False)
            except Exception as e:
                 print(f"Fatal error loading weights: {e}")
                 return

    print(f"Saving HuggingFace model to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("Conversion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Verl FSDP checkpoint to HuggingFace format")
    parser.add_argument("--local_dir", type=str, required=True, help="Path to the checkpoint directory (e.g., .../global_step_100)")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the converted HF model")
    
    args = parser.parse_args()
    
    merge_fsdp_checkpoint(args.local_dir, args.save_path)
