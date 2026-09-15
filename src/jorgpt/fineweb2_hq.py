from datasets import IterableDataset, load_dataset

DATASET_NAME = "epfml/FineWeb2-HQ"
DATASET_CONFIG = "spa_Latn"
DATASET_SPLIT = "train"

def load_fineweb2_hq_spanish() -> IterableDataset:
    """Carga FineWeb2-HQ en español mediante streaming."""

    return load_dataset(DATASET_NAME,DATASET_CONFIG,split=DATASET_SPLIT,streaming=True, )
