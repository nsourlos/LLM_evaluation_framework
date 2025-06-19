import torch
import transformers
from ..utils.paths import get_file_paths
from ..config import excel_file_name

_, _, custom_cache_dir, _, _ = get_file_paths(excel_file_name)

def get_model(model_name, commercial_api_providers, custom_cache_dir=custom_cache_dir):
    """Given a model name, return the loaded model, tokenizer, and pipeline"""

    if not any(provider in model_name for provider in commercial_api_providers): #For Hugging Face models
        model_HF="/".join(model_name.split('/')[1:])
        pipeline=initialize_model(model_HF, custom_cache_dir)

    #Returns below variables if defined, and returns None for any that are not.
    model = locals().get('model', None)
    tokenizer = locals().get('tokenizer', None)
    pipeline = locals().get('pipeline', None)

    return model, tokenizer, pipeline

def initialize_model(model_id, custom_cache_dir=custom_cache_dir):
    # # Check if mps acceleration is available (For MacOS)
    # device = "mps" if torch.backends.mps.is_available() else "cpu"
    # print(f"Using device {device}")
    # model.to(device)
    # transformers.set_seed(42) #Tried for reproducibility but didn't work

    pipeline = transformers.pipeline( 
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16, "cache_dir":custom_cache_dir},
            # trust_remote_code=True,
            device_map="auto" #Use 'cuda' if one GPU available (works with 32GB VRAM for 7B models) - 'auto' the alternative for distributed over all available GPUs
        )
    return pipeline 