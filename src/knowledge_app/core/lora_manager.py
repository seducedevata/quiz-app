import logging
import os
from pathlib import Path
from typing import List, Optional

try:
    from peft import PeftModel, PeftConfig, get_peft_model
except ImportError:  # Graceful degradation if peft isn't installed
    PeftModel = None  # type: ignore
    PeftConfig = None  # type: ignore
    get_peft_model = None  # type: ignore

logger = logging.getLogger(__name__)


class LoraManager:
    """Centralized LoRA adapter management service.

    Responsibilities:
    - Discover available LoRA adapters on disk.
    - Provide adapter metadata for UI selection.
    - Load an adapter onto a base model for inference.
    """

    def __init__(self, adapters_root: str = "lora_adapters_mistral"):
        self.adapters_root = Path(adapters_root)
        if not self.adapters_root.exists():
            logger.warning(f"LoRA adapters directory not found: {self.adapters_root}")
        logger.info(f"LoraManager initialized with root: {self.adapters_root}")

    # ------------------------------------------------------------------
    # Discovery helpers
    # ------------------------------------------------------------------

    def list_adapters(self) -> List[str]:
        """Return a list of available adapter names."""
        if not self.adapters_root.exists():
            return []
        adapters = [d.name for d in self.adapters_root.iterdir() if d.is_dir()]
        logger.info(f"Found {len(adapters)} LoRA adapters: {adapters}")
        return adapters

    def adapter_path(self, adapter_name: str) -> Path:
        """Get the full path for a given adapter name."""
        return self.adapters_root / adapter_name

    def has_adapter(self, adapter_name: str) -> bool:
        return self.adapter_path(adapter_name).exists()

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def load_adapter(self, base_model, adapter_name: str):
        """Load a LoRA adapter onto a base model.

        Parameters
        ----------
        base_model : transformers.PreTrainedModel
            The base model to attach the adapter to.
        adapter_name : str
            The name (folder) of the adapter to load.

        Returns
        -------
        transformers.PreTrainedModel
            The base model with the LoRA weights applied, or the base model
            unchanged if loading fails.
        """
        if PeftModel is None:
            logger.error("peft library not available. Cannot load LoRA adapters.")
            return base_model

        adapter_dir = self.adapter_path(adapter_name)
        if not adapter_dir.exists():
            logger.error(f"LoRA adapter not found: {adapter_dir}")
            return base_model

        try:
            logger.info(f"🔗 Loading LoRA adapter '{adapter_name}'...")
            model = PeftModel.from_pretrained(base_model, adapter_dir)
            logger.info("✅ LoRA adapter loaded successfully")
            return model
        except Exception as e:
            logger.error(f"❌ Failed to load LoRA adapter '{adapter_name}': {e}")
            return base_model