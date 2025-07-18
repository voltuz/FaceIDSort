# Face Identity Sorter

A desktop application for sorting and clustering large collections of photos based on facial identity. This tool uses deep learning to generate facial embeddings and the DBSCAN algorithm to group similar faces together, allowing for efficient management and organization of images.

The application provides a graphical user interface (GUI) built with PyQt6 to visualize clusters, inspect individual images, and save the sorted results.


## Features

- **High-Performance Face Clustering**: Utilizes the pre-trained FaceNet model (InceptionResnetV1) for accurate facial embedding generation.
- **Efficient Clustering Algorithm**: Employs DBSCAN to group faces without needing to specify the number of clusters beforehand.
- **Two Processing Modes**:
    1.  **Pre-aligned Mode (Fast)**: Directly processes images that are already cropped to faces.
    2.  **Detection Mode (Slower)**: Uses an integrated MTCNN to automatically detect and crop faces from full images.
- **Interactive GUI**:
    - Visualize all discovered clusters.
    - Inspect and curate clusters with a lazy-loading image viewer for handling thousands of images smoothly.
    - Select/deselect entire clusters for batch processing.
- **Configurable Parameters**: Adjust DBSCAN's `epsilon` and `min_samples` parameters directly from the UI to fine-tune clustering sensitivity.
- **Project Management**: Save and load your curation progress (which images are approved/selected in each cluster) to a project file.
- **Responsive Layout**: The cluster display automatically adjusts its layout based on the window size to avoid horizontal scrolling.

## Setup and Installation

This guide will walk you through setting up the project and installing all necessary dependencies.

### Prerequisites

- Python 3.8+
- An environment manager like `venv` (recommended) or `conda`.

### Installation Steps

1.  **Clone the Repository**

    First, clone this repository to your local machine.
    ```bash
    git clone [https://github.com/your-username/face-sorter.git](https://github.com/your-username/face-sorter.git)
    cd face-sorter
    ```

2.  **Create a Virtual Environment**

    It is highly recommended to create a virtual environment to keep the project's dependencies isolated.
    ```bash
    # For venv
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install PyTorch**

    This project depends on PyTorch. For the best performance (especially with a GPU), you should install it manually before the other requirements. Visit the [official PyTorch website](https://pytorch.org/get-started/locally/) to find the correct installation command for your specific OS and CUDA version.

    *Example for Linux/Windows with CUDA 11.8:*
    ```bash
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
    ```
    *Example for a CPU-only installation:*
    ```bash
    pip install torch torchvision torchaudio
    ```

4.  **Install All Other Dependencies**

    Once PyTorch is installed, you can install the rest of the required packages from the `requirements.txt` file with a single command.
    ```bash
    pip install -r requirements.txt
    ```

## How to Use

1.  **Run the application:**
    ```bash
    python sorter_app.py
    ```

2.  **Select an Image Folder**: Click the "Select Image Folder" button to choose the directory containing your images.

3.  **Choose a Processing Mode**:
    - If your images are already cropped faces, use the **"Process Pre-aligned Face Crops"** mode for the best performance.
    - If your images are regular photos, use the **"Detect Faces in Full Images"** mode.

4.  **Start Clustering**: Click the "Start Clustering" button. The progress bar will show the status of the embedding generation.

5.  **Review and Curate**:
    - Once clustering is complete, the main window will populate with the discovered clusters.
    - Click **"View / Edit Selection"** on any cluster to open the inspector.
    - In the inspector, click on images to deselect them if they don't belong. Confirm your selection to save the changes.

6.  **Save Results**:
    - Select the clusters you wish to save by checking the box on their group title.
    - Click **"Save Selected Clusters"** and choose a destination folder. The approved images from the selected clusters will be copied into subfolders (e.g., `cluster_0`, `cluster_1`, etc.).

7.  **Save/Load Project (Optional)**:
    - Use the **"Save Project"** button to save the state of your checked clusters and approved images.
    - To resume your work later, run the clustering on the same folder first, then click **"Load Project"** to restore your previous selections.
