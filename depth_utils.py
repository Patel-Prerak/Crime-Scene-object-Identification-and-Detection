
import torch
from transformers import pipeline
from PIL import Image
import numpy as np

# Global cache for the pipeline to avoid reloading
_DEPTH_PIPE = None

def get_depth_pipeline():
    global _DEPTH_PIPE
    if _DEPTH_PIPE is None:
        try:
            # Using Depth Anything Small for speed/performance balance
            # If this fails, we can fall back to DPT
            _DEPTH_PIPE = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")
        except Exception as e:
            print(f"Error loading Depth Anything: {e}. Falling back to Intel/dpt-large.")
            _DEPTH_PIPE = pipeline(task="depth-estimation", model="Intel/dpt-large")
    return _DEPTH_PIPE

def estimate_depth(image):
    """
    Estimates depth from a PIL Image.
    Returns:
        depth_map (PIL.Image): Grayscale depth map.
        depth_array (np.array): Raw depth values.
    """
    pipe = get_depth_pipeline()
    
    # Inference
    # pipeline returns a dict with 'depth' (PIL Image)
    result = pipe(image)
    depth_map = result["depth"]
    
    # Convert to numpy for advanced usage if needed
    depth_array = np.array(depth_map)
    
    return depth_map, depth_array
