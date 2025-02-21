# /// script
# requires-python = "~=3.12.0"
# dependencies = [
#    "numpy~=2.2.0",
#    "torch~=2.6.0",
#    "pillow~=11.1.0",
#    "scikit-learn~=1.6.0",
#    "transformers~=4.49.0",
# ]
# ///


"""
This script:
- Loads images from a directory.
- For each image:
    - Computes an embedding.
    - Saves the image path and embedding to an SQLite database.
- Retrieves all embeddings from the database.
- Clusters the embeddings using DBSCAN with cosine metric.
- Prints the cluster groupings for each image.

Usage:
$ uv run ./c__hh_ml_code.py

That's right, no need for a venv.
Donations to <https://docs.astral.sh/uv/>  🙂
"""

import logging
import pickle
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import DBSCAN
from transformers import CLIPModel, CLIPProcessor

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s [%(module)s-%(funcName)s()::%(lineno)d] %(message)s',
    datefmt='%d/%b/%Y %H:%M:%S',
)
log = logging.getLogger(__name__)


## setup stuff ------------------------------------------------------

## image-dir path ---------------------------------------------------
relative_images_dir_path = './jpeg_images/'
IMAGES_DIR_PATH: Path = Path(relative_images_dir_path).resolve()

## db path ----------------------------------------------------------
relative_db_path: str = '../image_embeddings.db'
DB_PATH: Path = Path(relative_db_path).resolve()

## use apple gpu if available ---------------------------------------
## (mps overview: <https://chatgpt.com/share/67b7e078-76e4-8006-9d62-858497781336>)
if torch.backends.mps.is_available():
    log.info('using mps')
    device = torch.device('mps')
else:
    log.info('using cpu')
    device = torch.device('cpu')

## load model -------------------------------------------------------
"""
Overview of CLIPModel, CLIPProcessor, transformers, and torch:
<https://chatgpt.com/share/67b5d81e-ce68-8006-a244-5bf35cd1cebb>
"""
model_name = 'openai/clip-vit-base-patch32'
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)


## main function, calling others below ------------------------------
def main() -> None:
    """
    Manages overall processing.
    Called by dundermain.
    """
    ## setup db connection ------------------------------------------
    conn: sqlite3.Connection = setup_database(DB_PATH)

    ## clear embeddings table --------------------------------------
    cursor: sqlite3.Cursor = conn.cursor()
    cursor.execute('DELETE FROM embeddings')
    conn.commit()

    ## load image-paths ---------------------------------------------
    image_paths: list[Path] = load_image_paths(IMAGES_DIR_PATH)

    ## process each image -------------------------------------------
    for i, image_path in enumerate(image_paths):
        ## get filename from path -----------------------------------
        filename = image_path.name
        log.info(f'processing, ``{filename}``')
        try:
            ## load image -------------------------------------------
            image: Image.Image = load_and_preprocess_image(image_path)
            ## compute embedding ------------------------------------
            embedding: np.ndarray = get_image_embedding(image)
            ## save embedding ---------------------------------------
            save_embedding(conn, filename, embedding)
            if i > 50:
                break
        except Exception as e:
            log.info(f'Error processing {image_path}: {e}')

    ## retrieve all embeddings from db ------------------------------
    data: list[tuple[int, str, np.ndarray]] = load_all_embeddings(conn)
    if not data:
        log.info('No embeddings found in the database.')
        return

    ## prep embeddings and image-paths for clustering ---------------
    """
    Collects all individual embeddings into one array, useful for batch processing 
    or feeding data into machine learning models.
    """
    ids, paths, embeddings = tuple(zip(*data))
    embeddings_np: np.ndarray = np.stack(embeddings, axis=0)

    ## cluster embeddings -------------------------------------------
    """
    "eps" (epsilon) is a key parameter for the DBSCAN clustering algorithm.

    Definition:

    eps defines the maximum distance between two data points for them to be
    considered as neighbors. In other words, it determines how "close"
    points need to be to be grouped in the same cluster.

    Impact on Clustering:

    A smaller eps value means that only very nearby points will be grouped
    together, which might lead to more clusters and potentially some points
    being labeled as noise (or outliers).

    A larger eps value allows points that are farther apart to be grouped
    together, potentially merging distinct clusters into one.

    Experimenting with different eps values or using techniques like the
    k-distance graph can help in finding a more appropriate value.
    """
    eps_val: float = 0.1
    labels: np.ndarray = cluster_embeddings(embeddings_np, eps=eps_val, min_samples=1)

    ## print cluster-grouping for each image ------------------------
    log.info('\n\nCluster groupings:')
    for img_id, path, label in zip(ids, paths, labels):
        log.info(f'filename, ``{path}``; cluster: ``{label}``')

    ## close db connection ------------------------------------------
    conn.close()

    return

    ## end def main()


def setup_database(db_path: str) -> sqlite3.Connection:
    """
    Create (if not exists) and connect to the SQLite database.
    """
    conn: sqlite3.Connection = sqlite3.connect(db_path)
    cursor: sqlite3.Cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
    """)
    conn.commit()
    return conn


def load_image_paths(images_dir_path: Path) -> list[Path]:
    """
    Loads all image paths from the given directory.
    """
    if not images_dir_path.exists():
        raise FileNotFoundError(f'Images directory not found: {images_dir_path}')
    image_paths: list[Path] = list(images_dir_path.glob('*.jpg'))
    ## sort image paths ---------------------------------------------
    image_paths.sort()
    # log.info(f'image_paths: ``{image_paths}``')
    log.info(f'Loaded ``{len(image_paths)}`` image paths from ``{images_dir_path}``.')
    return image_paths


def load_and_preprocess_image(image_path: str) -> Image.Image:
    """
    Loads image from the given path and returns a PIL Image.
    """
    if not image_path.exists():
        raise FileNotFoundError(f'image_path not found: ``{image_path}``')
    img: Image.Image = Image.open(image_path).convert('RGB')
    return img


def get_image_embedding(image: Image.Image) -> np.ndarray:
    """
    Generates an embedding for the given image using CLIP.
    Returns a numpy array of embeddings.
    """
    inputs: dict[str, Any] = processor(images=image, return_tensors='pt')
    # Move input tensors to the correct device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get image features (embeddings) from the CLIP model
    with torch.no_grad():
        image_features: torch.Tensor = model.get_image_features(**inputs)

    # Normalize the features (commonly done for similarity tasks)
    image_features: torch.Tensor = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

    # Move tensor to CPU and convert to numpy
    embedding: np.ndarray = image_features.cpu().numpy().squeeze()
    # log.info(f'embedding shape: ``{embedding.shape}``')
    # log.info(f'embedding: ``{embedding}``')
    return embedding


def save_embedding(conn: sqlite3.Connection, image_path: str, embedding: np.ndarray) -> None:
    """
    Saves image-path and its embedding (serialized as a BLOB) to the database.
    """
    cursor: sqlite3.Cursor = conn.cursor()
    ## serialize the numpy array using pickle -----------------------
    embedding_blob: bytes = pickle.dumps(embedding)
    cursor.execute(
        """
        INSERT INTO embeddings (image_path, embedding)
        VALUES (?, ?)
    """,
        (str(image_path), embedding_blob),
    )
    conn.commit()
    return


def load_all_embeddings(conn: sqlite3.Connection) -> list[tuple[int, str, np.ndarray]]:
    """
    Retrieves all embeddings and associated image paths from the database.
    Returns a list of (id, image_path, embedding) tuples.

    Note:
    - Embeddings are loaded into memory because SQLite doesn't support native vector operations
      like similarity or nearest-neighbor search on stored binary objects.
    - By loading embeddings into memory, we can use the DBSCAN algorithm to cluster them in the next step.
    - For large datasets, we should use a database that supports vector operations.
    """
    cursor: sqlite3.Cursor = conn.cursor()
    cursor.execute('SELECT id, image_path, embedding FROM embeddings')
    rows: list[tuple[int, str, bytes]] = cursor.fetchall()
    result: list[tuple[int, str, np.ndarray]] = []
    for row in rows:
        emb: np.ndarray = pickle.loads(row[2])
        result.append((row[0], row[1], emb))
    return result


def cluster_embeddings(embeddings: np.ndarray, eps: float = 0.1, min_samples: int = 1) -> np.ndarray:
    """
    Clusters the embeddings using DBSCAN with cosine metric.

    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a clustering algorithm
    that groups together points that are closely packed

    Parameters:
    - eps: Maximum distance between two samples for them to be considered as in the same neighborhood.
    - min_samples: Minimum number of samples in a neighborhood for a point to be considered a core point.
    - metric: The metric to use when calculating distance between instances in a feature array.

    Returns the cluster labels, an array where each element is an integer representing the cluster
    to which the corresponding embedding belongs.
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels: np.ndarray = clustering.fit_predict(embeddings)
    log.info(f'cluster labels, ``{labels}``')
    return labels


if __name__ == '__main__':
    main()
