import os
import glob
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from PIL import Image

import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, MTCNN

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_CACHE = os.path.join(SCRIPT_DIR, "face_embeddings_facenet.pkl")
BATCH_SIZE = 64

# --- Image transformation for pre-aligned images ---
# The InceptionResnetV1 model expects 160x160 standardized images.
# FIX: Replaced the custom standardization with the correct, standard PyTorch transform.
# This maps image tensors from a [0, 1] range to a [-1, 1] range.
aligned_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def generate_and_cache_embeddings(image_paths, progress_callback=None, use_mtcnn=True):
    """
    Generates facial embeddings, now with correct standardization and a "smarter" cache
    that is aware of the processing mode.
    """
    current_mode = 'mtcnn' if use_mtcnn else 'prealigned'
    
    # FIX: Caching logic now checks if the cache mode matches the current run mode.
    if os.path.exists(EMBEDDINGS_CACHE):
        print(f"Loading embeddings from cache: {EMBEDDINGS_CACHE}")
        try:
            with open(EMBEDDINGS_CACHE, 'rb') as f:
                cached_data = pickle.load(f)
                
                # New, robust cache format check
                if isinstance(cached_data, dict) and 'mode' in cached_data:
                    if cached_data['mode'] == current_mode:
                        print(f"Cache is valid for '{current_mode}' mode.")
                        return cached_data.get('embeddings', {}), cached_data.get('rejected', {})
                    else:
                        print(f"Cache mode ('{cached_data['mode']}') does not match current mode ('{current_mode}'). Regenerating.")
                else:
                    # Handle old cache format or invalid dict
                    print("Old or invalid cache format detected. Regenerating embeddings for safety.")

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
        mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=device, keep_all=False
        )
        print("MTCNN model built.")
    
    total_images = len(image_paths)
    print(f"Generating embeddings for {total_images} images...")

    for i in tqdm(range(0, total_images, BATCH_SIZE), desc="Generating Embeddings"):
        batch_paths = image_paths[i : i + BATCH_SIZE]
        
        image_batch_pil = []
        valid_paths_in_batch = []
        for path in batch_paths:
            try:
                img = Image.open(path).convert('RGB')
                image_batch_pil.append(img)
                valid_paths_in_batch.append(path)
            except Exception:
                rejected_files['corrupt'].append(path)
                continue
        
        if not image_batch_pil:
            if progress_callback:
                progress_callback(i + len(batch_paths), total_images)
            continue

        try:
            faces_tensor = None
            if use_mtcnn:
                faces_batch_detected = mtcnn(image_batch_pil)
                valid_faces = [face for face in faces_batch_detected if face is not None]
                if valid_faces:
                    faces_tensor = torch.stack(valid_faces).to(device)
            else:
                transformed_batch = [aligned_transform(img) for img in image_batch_pil]
                faces_tensor = torch.stack(transformed_batch).to(device)

            if faces_tensor is not None and len(faces_tensor) > 0:
                with torch.no_grad():
                    embeddings = resnet(faces_tensor).detach().cpu().numpy()
                for j, path in enumerate(valid_paths_in_batch):
                    all_embeddings[path] = embeddings[j]
            else:
                rejected_files['processing_error'].extend(valid_paths_in_batch)

        except Exception as e:
            print(f"\nAn error occurred processing a batch: {e}")
            rejected_files['processing_error'].extend(valid_paths_in_batch)
        
        if progress_callback:
            progress_callback(i + len(batch_paths), total_images)

    print(f"Saving {len(all_embeddings)} embeddings to cache: {EMBEDDINGS_CACHE}")
    # FIX: Save data in the new, more robust dictionary format
    data_to_cache = {
        'mode': current_mode,
        'embeddings': all_embeddings,
        'rejected': rejected_files
    }
    with open(EMBEDDINGS_CACHE, 'wb') as f:
        pickle.dump(data_to_cache, f)
        
    return all_embeddings, rejected_files


def get_image_paths(folder):
    """Gets all common image file paths from a directory."""
    extensions = ('*.jpg', '*.jpeg', '*.png')
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return paths

def cluster_embeddings(embeddings_dict, epsilon, min_samples):
    """Clusters embeddings using DBSCAN with configurable parameters."""
    if not embeddings_dict: return None, None
    paths = list(embeddings_dict.keys())
    embeddings = np.array(list(embeddings_dict.values()))
    
    print(f"Clustering {len(paths)} faces with DBSCAN (eps={epsilon}, min_samples={min_samples})...")
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
