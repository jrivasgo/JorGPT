from dataclasses import replace
from pathlib import Path

import phytest

from jorgpt.config import JorGPTConfig

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configuracion" / "jorgpt_138m.yaml"

def test_load_jorgpt_config():
  config = JorGPTConfig.form_yaml(CONFIG_PATH)

  assert config.name == "JORGPT_138M"
  assert config.vocab_size == 16000
  assert config.hidden_size == 768
  assert config.num_hidden_layers == 20

  assert config.num_attention_heads == 12
  assert config.num_key_value_heads == 4
  assert config.head_dim == 64

  assert config.intermediate_size == 2048
  assert config.max_seq_len == 2048

def test_gqa_ratio():
  config = JorGPTConfig.form_yaml(CONFIG_PATH)

  assert config.querries_per_kv_head == 3

def test_invalid_attention_dimensions():
  config = JorGPTConfig.from_yaml(CONFIG_PATH)

  with pytest.raises(ValueError):
    replace(config, head_dim=32)

def tes_invalid_gqa_configuration():
  config = JorGPTConfig.form_yaml(CONFIG_PATH)

  with pytest.raises(ValueError):
    replace(config, num_key_value_heads=5)
