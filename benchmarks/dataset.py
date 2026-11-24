import numpy as np

from models.models import DIMENSIONS


def generate_dataset(
    size: int,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    vectors = rng.standard_normal(
        size=(size, DIMENSIONS),
        dtype=np.float32,
    )

    # Normalización L2
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)

    return vectors / norms


def generate_batches(
    vectors: np.ndarray,
    batch_size: int,
):
    for start in range(0, len(vectors), batch_size):
        end = min(start + batch_size, len(vectors))

        yield [
            {
                "id": i + 1,
                "text": f"Documento {i + 1}",
                "embedding": vectors[i].tolist(),
            }
            for i in range(start, end)
        ]
