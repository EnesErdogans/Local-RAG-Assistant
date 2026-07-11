"""
Singleton wrapper around FoundryLocalManager.
Both embeddings.py and llm_client.py import from here so the manager
is initialised exactly once regardless of import order.
"""

import threading
from foundry_local_sdk import FoundryLocalManager, Configuration

_init_lock = threading.Lock()


def get_manager() -> FoundryLocalManager:
    with _init_lock:
        if FoundryLocalManager.instance is None:
            config = Configuration(app_name="local-rag-assistant")
            FoundryLocalManager.initialize(config)
    return FoundryLocalManager.instance


def get_or_load_model(alias: str):
    """Return a loaded model for the given alias, downloading it first if needed."""
    manager = get_manager()
    model = manager.catalog.get_model(alias)

    if model is None:
        raise RuntimeError(
            f"Model '{alias}' not found in the Foundry Local catalog.\n"
            "Check the alias spelling or run:  foundry-local catalog list"
        )

    if not model.is_cached:
        print(f"[Foundry] Downloading '{alias}' (first time only)...")

        def _progress(pct: float) -> None:
            bar = int(pct / 5)
            print(f"\r  [{'#' * bar}{' ' * (20 - bar)}] {pct:.1f}%", end="", flush=True)

        model.download(progress_callback=_progress)
        print()

    if not model.is_loaded:
        print(f"[Foundry] Loading '{alias}' into memory...")
        model.load()

    return model
