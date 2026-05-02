import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

@dataclass
class PathConfig:
    """Manages project paths."""
    # Base paths
    project_root: Path = Path(__file__).resolve().parent.parent.parent
    
    # Data paths
    data_dir: Path = field(init=False)
    data_raw: Path = field(init=False)
    data_gold: Path = field(init=False)
    data_synthetic: Path = field(init=False)
    
    # Output paths
    output_dir: Path = field(init=False)
    
    def __post_init__(self):
        self.data_dir = self.project_root / "data"
        self.data_raw = self.data_dir / "raw"
        self.data_gold = self.data_dir / "gold"
        self.data_synthetic = self.data_dir / "synthetic"
        self.output_dir = self.project_root / "checkpoints"
        
        # Subdirectories
        self.data_raw_pdfs = self.data_raw / "pdfs"
        
        # Ensure directories exist
        for p in [self.data_raw, self.data_gold, self.data_synthetic, self.output_dir, self.data_raw_pdfs]:
            p.mkdir(parents=True, exist_ok=True)

@dataclass
class ModelConfig:
    """Configuration for the NMT Model."""
    model_name: str = "facebook/nllb-200-3.3B"
    max_source_length: int = 128
    max_target_length: int = 128
    
    # Training
    batch_size: int = 12 #8 #2
    accumulate_grad_batches: int = 3 #4 #16
    learning_rate: float = 5e-5 #2e-4
    weight_decay: float = 0.01
    max_epochs: int = 20
    val_check_interval: float = 0.5
    precision: str = "bf16-mixed"
    num_workers: int = 4
    seed: int = 42
    
    # LoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    
    log_every_n_steps: int = 10

@dataclass
class GenConfig:
    """Configuration for Synthetic Data Generation (LLM).

    Backend is selected automatically by get_generator() based on model_id prefix:
      "ollama/<name>"  → OllamaGenerator  (e.g. "ollama/qwen2.5:32b")
      anything else    → HFChatGenerator  (e.g. "Qwen/Qwen2.5-32B-Instruct")
    """
    model_id: str = "ollama/qwen2.5:32b"
    ollama_base_url: str = "http://localhost:11434"

    # HuggingFace-only options (ignored by OllamaGenerator)
    load_in_4bit: bool = True
    load_in_8bit: bool = False

    # Generation params
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True
    num_variants: int = 1   # sentences generated per term (each uses a different doc-type prompt)
    num_workers: int = 4    # parallel threads for Ollama (overridden dynamically by model size)
    cloud_num_workers: int = 8  # parallel threads for cloud APIs (rate-limit bound, not VRAM bound)

@dataclass
class WandbConfig:
    """Configuration for WandB Logging."""
    enabled: bool = True
    project: str = "leo-nmt"
    name: str = "nllb-finetune"
    mode: str = "online" # "online", "offline", "disabled"

@dataclass
class SpiderConfig:
    """Configuration for Competitor Spider."""
    target_urls: List[str] = field(default_factory=lambda: [
        "https://alpac.it/prodotti",
        "https://www.incovar.it/produzione-cassonetti-coibentati",
        "https://europlastik.it/cassonetti-coibentati",
        "https://www.defaverisrl.it/prodotti",
        "https://www.trelleborg.com/en/seals-and-profiles/products-and-solutions",
        "https://www.deventer-profile.com/en/products",
        "https://www.hella.info/it/finestre-e-facciata",
        "https://www.gealan.de/en/systems",
        "https://www.velux.it/prodotti",
    ])

@dataclass
class Config:
    """Central Configuration Singleton."""
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    gen: GenConfig = field(default_factory=GenConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    spider: SpiderConfig = field(default_factory=SpiderConfig)

    def __post_init__(self):
        # Dynamically separate checkpoints by the active model
        model_safe_name = self.model.model_name.replace("/", "_")
        self.paths.output_dir = self.paths.project_root / f"checkpoints_{model_safe_name}"
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)

# Global Instance
conf = Config()
