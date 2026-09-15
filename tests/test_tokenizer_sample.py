import json

import pytest

import jorgpt.tokenizer_sample as tokenizer_sample


def test_sha256_file(tmp_path):
    path = tmp_path / "sample.bin"
    path.write_bytes(b"JorGPT")

    result = tokenizer_sample.sha256_file(path)

    assert (result== "02208429d06f4ddbfa652244e493c452395e3cda72adc127bec779e2b7911b47")


def test_write_tokenizer_sample_creates_sample_and_manifest(tmp_path,monkeypatch,):
    texts = ["Café","España","Este documento no debería llegar a escribirse.",]

    def fake_iter_tokenizer_texts(seed, shuffle_buffer, min_chars):
        yield from texts

    monkeypatch.setattr(tokenizer_sample,"iter_tokenizer_texts",fake_iter_tokenizer_texts,)

    output_path = tmp_path / "sample.txt"

    manifest = tokenizer_sample.write_tokenizer_sample(output_path,target_chars=12,)

    manifest_path = tmp_path / "sample.manifest.json"

    assert output_path.exists()
    assert manifest_path.exists()

    assert output_path.read_text(encoding="utf-8") == "Café\n\nEspaña"

    assert manifest["result"]["documents"] == 2
    assert manifest["result"]["characters"] == 12
    assert manifest["result"]["bytes"] == output_path.stat().st_size

    assert (manifest["result"]["sha256"]== tokenizer_sample.sha256_file(output_path))

    saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert saved_manifest == manifest
    assert manifest["sampling"]["seed"] == 12
    assert manifest["sampling"]["min_chars"] == 128
    assert manifest["dataset"]["revision"] == tokenizer_sample.DATASET_REVISION


def test_write_tokenizer_sample_rejects_invalid_target(tmp_path):
    output_path = tmp_path / "sample.txt"

    with pytest.raises(ValueError):tokenizer_sample.write_tokenizer_sample(output_path,target_chars=0,)


def test_write_tokenizer_sample_detects_exhausted_dataset(tmp_path,monkeypatch,):
    def fake_iter_tokenizer_texts(seed, shuffle_buffer, min_chars):
        yield "Texto demasiado pequeño para alcanzar el objetivo."

    monkeypatch.setattr(tokenizer_sample,"iter_tokenizer_texts",fake_iter_tokenizer_texts,)

    output_path = tmp_path / "sample.txt"

    with pytest.raises(RuntimeError):
        tokenizer_sample.write_tokenizer_sample(output_path,target_chars=10_000,)
