from huggingface_hub import snapshot_download
import os

model_id = ["meta-llama/Llama-3.1-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.1", 
            "mistralai/Mixtral-8x7B-Instruct-v0.1"
            ]

cache_dir = "/mnt/lustre/users/inf/js2042/models/cache"

hf_token = os.getenv("HF_TOKEN")
if hf_token is None:
    raise ValueError("Please set the HF_TOKEN environment variable with your Hugging Face token.")

for model in model_ids:
    print(f"Downloading {model_id} to {cache_dir}...")

    snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        token=hf_token,
        local_files_only=False,
        resume_download=True,
        ignore_patterns=["*.safetensors", "*.json"],
    )
    print(f"Model {model_id} downloaded successfully to {cache_dir}.")