from pathlib import Path

from jorgpt.tokenizer_data import (DEFAULT_MIN_CHARS,DEFAULT_SEED,DEFAULT_SHUFFLE_BUFFER,iter_tokenizer_texts,)


def write_tokenizer_sample(output_path: str | Path,target_chars: int,seed: int = DEFAULT_SEED,shuffle_buffer: int = DEFAULT_SHUFFLE_BUFFER,min_chars: int = DEFAULT_MIN_CHARS,) -> dict[str, int]:
    """Escribe una muestra reproducible de textos para entrenar el tokenizer."""

    if target_chars <= 0:
        raise ValueError("target_chars debe ser mayor que 0.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_chars = 0
    documents = 0

    texts = iter_tokenizer_texts(seed=seed,shuffle_buffer=shuffle_buffer,min_chars=min_chars,)

    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for text in texts:
            separator = "\n\n" if documents > 0 else ""

            file.write(separator)
            file.write(text)

            total_chars += len(separator) + len(text)
            documents += 1

            if total_chars >= target_chars:
                break

    return {"documents": documents,"characters": total_chars,}
