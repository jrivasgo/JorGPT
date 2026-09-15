import jorgpt.tokenizer_data as tokenizer_data


def test_normalize_text_uses_nfc_and_strip():
    text = "  Cafe\u0301  "

    normalized = tokenizer_data.normalize_text(text)

    assert normalized == "Café"


def test_iter_tokenizer_texts_filters_invalid_documents(monkeypatch):
    valid_text = "  Cafe\u0301 " + (" texto" * 30)

    documents = [{"text": None},{"text": "demasiado corto"},{"text": valid_text},]

    class FakeDataset:
        def shuffle(self, seed, buffer_size):
            return self

        def __iter__(self):
            return iter(documents)

    monkeypatch.setattr(tokenizer_data,"load_fineweb2_hq_spanish",lambda: FakeDataset(),)

    texts = list(tokenizer_data.iter_tokenizer_texts())

    assert len(texts) == 1
    assert texts[0].startswith("Café")
    assert len(texts[0]) >= 128


def test_iter_tokenizer_texts_uses_seed_and_buffer(monkeypatch):
    received = {}

    class FakeDataset:
        def shuffle(self, seed, buffer_size):
            received["seed"] = seed
            received["buffer_size"] = buffer_size
            return self

        def __iter__(self):
            return iter([])

    monkeypatch.setattr(tokenizer_data,"load_fineweb2_hq_spanish",lambda: FakeDataset(),)

    list(tokenizer_data.iter_tokenizer_texts())

    assert received["seed"] == 12
    assert received["buffer_size"] == 10_000
