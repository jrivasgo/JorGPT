import unicodedata
from collections.abc import Iterator

from jorgpt.fineweb2_hq import load_fineweb2_hq_spanish


DEFAULT_SEED = 12
DEFAULT_SHUFFLE_BUFFER = 10_000
DEFAULT_MIN_CHARS = 128


def normalize_text(text: str) -> str:
    """Normaliza un texto usando Unicode NFC."""

    return unicodedata.normalize("NFC", text).strip()


def iter_tokenizer_texts(
    seed: int = DEFAULT_SEED,
    shuffle_buffer: int = DEFAULT_SHUFFLE_BUFFER,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> Iterator[str]:
    """Genera textos españoles válidos para entrenar el tokenizer."""

    dataset = load_fineweb2_hq_spanish()

    dataset = dataset.shuffle(
        seed=seed,
        buffer_size=shuffle_buffer,
    )

    for document in dataset:
        text = document.get("text")

        if not isinstance(text, str):
            continue

        text = normalize_text(text)

        if len(text) < min_chars:
            continue

        yield text
