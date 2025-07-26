import os
import glob
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, MTCNN

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_CACHE = os.path.join(SCRIPT_DIR, "face_embeddings_facenet.pkl")
# You can experiment with batch size. Larger sizes might be faster if you have enough VRAM.
BATCH_SIZE = 64 

# --- Image transformation for pre-aligned images ---
# This remains the same. It's the standard transform for the model.
aligned_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- NEW: Custom Dataset for efficient loading ---
# This class will read image paths and load them.
class FaceImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            # Open image and convert to RGB
            img = Image.open(path).convert('RGB')
            # Apply transforms if they are provided (for pre-aligned mode)
            if self.transform:
                img = self.transform(img)
            return img, path
        except Exception as e:
            # If an image is corrupt or can't be opened, return None
            print(f"Warning: Could not load image {path}. Skipping. Error: {e}")
            return None, path

# --- NEW: Custom Collate Function for DataLoader ---
# This function gathers the data from the Dataset and prepares it for the model.
# It also safely skips any corrupt images that failed to load in __getitem__.
def collate_fn(batch):
    # Filter out None values returned by the Dataset for corrupt images
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return None, None
    
    images, paths = zip(*batch)
    
    # If images are tensors (from pre-aligned mode), stack them.
    # If they are PIL Images (for MTCNN mode), they remain as a list.
    if isinstance(images[0], torch.Tensor):
        images = torch.stack(images, 0)
        
    return images, paths

# --- UPDATED: Main function using the new DataLoader ---
def generate_and_cache_embeddings(image_paths, progress_callback=None, use_mtcnn=True):
    """
    Generates facial embeddings using an efficient DataLoader to parallelize
    image loading and preprocessing.
    """
    current_mode = 'mtcnn' if use_mtcnn else 'prealigned'
    
    # Cache loading logic remains unchanged
    if os.path.exists(EMBEDDINGS_CACHE):
        print(f"Loading embeddings from cache: {EMBEDDINGS_CACHE}")
        try:
            with open(EMBEDDINGS_CACHE, 'rb') as f:
                cached_data = pickle.load(f)
                if isinstance(cached_data, dict) and 'mode' in cached_data:
                    if cached_data['mode'] == current_mode:
                        print(f"Cache is valid for '{current_mode}' mode.")
                        return cached_data.get('embeddings', {}), cached_data.get('rejected', {})
                    else:
                        print(f"Cache mode ('{cached_data['mode']}') does not match current mode ('{current_mode}'). Regenerating.")
                else:
                    print("Old or invalid cache format detected. Regenerating embeddings.")
        except Exception as e:
            print(f"Could not load or parse cache file. Regenerating embeddings. Error: {e}")

    all_embeddings = {}
    rejected_files = {'corrupt': [], 'processing_error': []}
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"--- Running on device: {device} ---")

    # --- Initialize Models ---
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    mtcnn = None
    if use_mtcnn:
        print("Building MTCNN face detection model...")
        # Note: We are setting keep_all=True here to align detections with paths,
        # but will only process the first valid face found per image.
        mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=device, keep_all=True 
        )
        print("MTCNN model built.")
    
    # --- MAJOR CHANGE: Use DataLoader for Efficiency ---
    # Don't apply a transform if using MTCNN, as it works on PIL Images.
    transform_to_use = aligned_transform if not use_mtcnn else None
    dataset = FaceImageDataset(image_paths, transform=transform_to_use)
    
    # Set num_workers > 0 to enable parallel loading.
    # A good starting point is 4, but you can tune this for your machine.
    # Set pin_memory=True to potentially speed up data transfer to the GPU.
    data_loader = DataLoader(dataset, 
                             batch_size=BATCH_SIZE, 
                             shuffle=False, 
                             num_workers=4,
                             collate_fn=collate_fn,
                             pin_memory=True)

    total_images = len(image_paths)
    processed_count = 0
    print(f"Generating embeddings for up to {total_images} images using DataLoader...")

    # --- The New, Efficient Processing Loop ---
    for batch_images, batch_paths in tqdm(data_loader, desc="Generating Embeddings"):
        # If the whole batch was corrupt, collate_fn returns None
        if batch_images is None:
            # Record corrupt paths if any (though collate_fn handles this)
            if batch_paths: rejected_files['corrupt'].extend(batch_paths)
            processed_count += len(batch_paths) if batch_paths else BATCH_SIZE
            continue

        try:
            faces_tensor = None
            valid_paths = []

            if use_mtcnn:
                # MTCNN processes a batch of PIL images
                with torch.no_grad():
                    # The output is a list of tensors (or None if no face is found)
                    faces_batch_detected = mtcnn(batch_images)
                
                # Filter out images where no face was detected and align paths
                valid_faces = []
                for i, face_tensor in enumerate(faces_batch_detected):
                    # We only process the first face found in an image
                    if face_tensor is not None:
                        # Since keep_all=True, it might be a list of faces. Take the first.
                        if isinstance(face_tensor, list) and len(face_tensor) > 0:
                            valid_faces.append(face_tensor[0])
                            valid_paths.append(batch_paths[i])
                        # Or it's a single tensor
                        elif torch.is_tensor(face_tensor):
                             valid_faces.append(face_tensor)
                             valid_paths.append(batch_paths[i])
                        else:
                            rejected_files['processing_error'].append(batch_paths[i])
                    else:
                        rejected_files['processing_error'].append(batch_paths[i])
                
                if valid_faces:
                    faces_tensor = torch.stack(valid_faces).to(device)

            else: # Pre-aligned mode
                faces_tensor = batch_images.to(device)
                valid_paths = list(batch_paths)

            # Generate embeddings for the valid faces in the batch
            if faces_tensor is not None and len(faces_tensor) > 0:
                with torch.no_grad():
                    embeddings = resnet(faces_tensor).detach().cpu().numpy()
                for i, path in enumerate(valid_paths):
                    all_embeddings[path] = embeddings[i]

        except Exception as e:
            print(f"\nAn error occurred processing a batch: {e}")
            rejected_files['processing_error'].extend(batch_paths)
        
        # Update progress
        processed_count += len(batch_paths)
        if progress_callback:
            progress_callback(processed_count, total_images)

    # Caching logic remains the same
    print(f"Saving {len(all_embeddings)} embeddings to cache: {EMBEDDINGS_CACHE}")
    data_to_cache = {
        'mode': current_mode,
        'embeddings': all_embeddings,
        'rejected': rejected_files
    }
    with open(EMBEDDINGS_CACHE, 'wb') as f:
        pickle.dump(data_to_cache, f)
        
    return all_embeddings, rejected_files


def get_image_paths(folder):
    """Gets all common image file paths from a directory and removes duplicates."""
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    paths = []
    # The recursive search is kept as it's useful for subdirectories
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
    
    # --- FIX: Remove duplicate paths found on case-insensitive systems ---
    unique_paths = list(set(paths))
    
    print(f"Found {len(unique_paths)} unique images in {folder}")
    return unique_paths

def cluster_embeddings(embeddings_dict, epsilon, min_samples):
    """Clusters embeddings using DBSCAN with configurable parameters."""
    if not embeddings_dict: return None, None
    paths = list(embeddings_dict.keys())
    embeddings = np.array(list(embeddings_dict.values()))
    
    print(f"Clustering {len(paths)} faces with DBSCAN (eps={epsilon}, min_samples={min_samples})...")
    # Use n_jobs=-1 to use all available CPU cores for clustering
    clt = DBSCAN(metric="cosine", eps=epsilon, min_samples=min_samples, n_jobs=-1)
    clt.fit(embeddings)
    
    labels = clt.labels_
    print("Grouping file paths by cluster...")
    clusters = {}
    for path, label in zip(paths, labels):
        label = int(label)
        if label not in clusters: clusters[label] = []
        clusters[label].append(path)
    return clusters, labels
