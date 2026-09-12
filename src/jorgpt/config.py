from dataclasses import dataclass
from pathlib import Path

import yaml

@dataclass(frozen=True)
class JorGPTConfig:
  """Configuramos la arquitectura de nuestro modelo, JorGPT."""

  name: str

  vocab_size: int
  hidden_size: int
  num_hidden_layers: int

  num_attention_heads: int
  num_key_value_heads: int
  head_dim: int

  intermediate_size: int
  max_seq_len: int

  rope_theta: float
  rms_norm_eps: float

  tie_word_embeddings: bool

  attention_bias: bool
  mlp_bias: bool
  lm_head_bias: bool

  dropout: float
  initializer_range: float

  use_gqa: bool
  use_rope: bool
  use_swiglu: bool

  def __post_init__(self) -> None:
  """Comprueba que tenga sentido"""

   if self.vocab_size <=0:
     raise ValueError("El tamaño de la variable vocab_size debe ser mayor que 0.")

   if self.vocab_size >= 65536:
            raise ValueError("vocab_size debe ser menor que 65536 para usar uint16.")

   if self.hidden_size <= 0:
            raise ValueError("hidden_size debe ser mayor que 0.")

   if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers debe ser mayor que 0.")

   if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads debe ser mayor que 0.")

   if self.num_key_value_heads <= 0:
            raise ValueError("num_key_value_heads debe ser mayor que 0.")

   if self.head_dim <= 0:
            raise ValueError("head_dim debe ser mayor que 0.")

   if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size debe ser igual a num_attention_heads * head_dim.")
      
   if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads debe ser divisible entre num_key_value_heads.")

   if self.intermediate_size <= 0:
            raise ValueError("intermediate_size debe ser mayor que 0.")

   if self.max_seq_len <= 0:
            raise ValueError("max_seq_len debe ser mayor que 0.")

   if self.rope_theta <= 0:
            raise ValueError("rope_theta debe ser mayor que 0.")

   if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps debe ser mayor que 0.")

   if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout debe estar entre 0.0 y 1.0.")

   if self.initializer_range <= 0:
            raise ValueError("initializer_range debe ser mayor que 0.")
  @property
  def queries_per_kv_head(self) -> int:
        """Número de cabezas Query que comparten cada cabeza Key/Value."""

        return self.num_attention_heads // self.num_key_value_heads

  @classmethod
  def from_yaml(cls, path: str | Path) -> "JorGPTConfig":
        """Carga una configuración de JorGPT desde un archivo YAML."""

        path = Path(path)

        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        if not isinstance(data, dict) or "model" not in data:
            raise ValueError("El archivo YAML debe contener una sección 'model'.")

        model_config = data["model"]

        if not isinstance(model_config, dict):
            raise ValueError("La sección 'model' debe contener una configuración.")

        return cls(**model_config)
