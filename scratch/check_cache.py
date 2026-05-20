import torch
import os

path = r"d:\Code\Github-Code\EnglishApp\EnglishApp-API-FastAPI\reference_cache.pt"
if os.path.exists(path):
    try:
        data = torch.load(path, map_location="cpu")
        print(f"Success: Loaded {len(data)} items from cache.")
    except Exception as e:
        print(f"Error: Cache file corrupted: {e}")
else:
    print("Cache file does not exist.")
