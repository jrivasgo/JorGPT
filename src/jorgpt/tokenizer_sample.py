import hashlib
import json
from pathlib import Path

from jorgpt.fineweb2_hq import {DATASET_CONFIG,DATASET_NAME,DATASET_REVISION,DATASET_SPLIT,)

from jorgpt.tokenizer_data import (DEFAULT_MIN_CHARS,DEFAULT_SEED,DEFAULT_SHUFFLE_BUFFER,iter_tokenizer_texts,)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024,) -> str:
    """Calcula el SHA-256 de un archivo sin cargarlo entero en memoria."""

    if chunk_size <= 0:
        raise ValueError("chunk_size debe ser mayor que 0.")

    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def write_tokenizer_sample(output_path: str | Path,target_chars: int,seed: int = DEFAULT_SEED,shuffle_buffer: int = DEFAULT_SHUFFLE_BUFFER,min_chars: int = DEFAULT_MIN_CHARS,) -> dict[str, object]:
    """Escribe una muestra reproducible de textos para entrenar el tokenizer."""

    if target_chars <= 0:
        raise ValueError("target_chars debe ser mayor que 0.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_chars = 0
    documents = 0

    texts = iter_tokenizer_texts(seed=seed,shuffle_buffer=shuffle_buffer,min_chars=min_chars,)

    with output_path.open("w",encoding="utf-8",newline="\n",) as file:
        for text in texts:
            separator = "\n\n" if documents > 0 else ""

            file.write(separator)
            file.write(text)

            total_chars += len(separator) + len(text)
            documents += 1

            if total_chars >= target_chars:
                break

    if total_chars < target_chars:
        raise RuntimeError("El dataset se agotó antes de alcanzar target_chars.")

    file_bytes = output_path.stat().st_size
    file_sha256 = sha256_file(output_path)

    manifest = {
        "manifest_version": 1,
        "sample_file": output_path.name,
        "dataset": {
            "name": DATASET_NAME,
            "config": DATASET_CONFIG,
            "split": DATASET_SPLIT,
            "revision": DATASET_REVISION,
        },
        "sampling": {
            "seed": seed,
            "shuffle_buffer": shuffle_buffer,
            "min_chars": min_chars,
            "target_chars": target_chars,
            "normalization": "NFC",
            "strip_whitespace": True,
            "encoding": "utf-8",
            "newline": "\n",
            "document_separator": "\n\n",
        },
        "result": {
            "documents": documents,
            "characters": total_chars,
            "bytes": file_bytes,
            "sha256": file_sha256,
        },
    }

    manifest_path = output_path.with_suffix(".manifest.json")

    with manifest_path.open("w",encoding="utf-8",newline="\n",) as file:
        json.dump(manifest,file,ensure_ascii=False,indent=2,sort_keys=True,)
        file.write("\n")

    return manifest
