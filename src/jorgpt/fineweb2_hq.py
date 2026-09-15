from datasets import IterableDataset, load_dataset

DATASET_NAME = "epfml/FineWeb2-HQ"
DATASET_CONFIG = "spa_Latn"
DATASET_SPLIT = "train"
DATASET_REVISION = "c0c06e94fd3a44ae9e802b2b0fc533817601eb5e"

def load_fineweb2_hq_spanish() -> IterableDataset:
    """Carga FineWeb2-HQ en español mediante streaming."""

    return load_dataset(DATASET_NAME,DATASET_CONFIG,split=DATASET_SPLIT,revision=DATASET_REVISION,streaming=True, )
