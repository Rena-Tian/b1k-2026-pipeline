"""Load a FAST action tokenizer directory without going through ``AutoProcessor``.

``UniversalActionProcessor.from_pretrained(dir)`` breaks under transformers>=5 (the auto-tokenizer
path can no longer instantiate the ``PreTrainedTokenizerFast`` backend from this layout), while
``PreTrainedTokenizerFast.from_pretrained(dir)`` still works. So construct the processor by hand from
``processor_config.json`` + the BPE tokenizer.
"""
import importlib.util
import json
from pathlib import Path

from transformers import PreTrainedTokenizerFast


def load_fast_processor(tokenizer_dir: str | Path):
    tokenizer_dir = Path(tokenizer_dir)
    processor_file = tokenizer_dir / "processing_action_tokenizer.py"
    if not processor_file.exists():
        raise FileNotFoundError(f"Tokenizer processing file not found at {processor_file}")
    spec = importlib.util.spec_from_file_location("processing_action_tokenizer", processor_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cfg = json.loads((tokenizer_dir / "processor_config.json").read_text())
    kwargs = {k: cfg[k] for k in ("scale", "vocab_size", "min_token", "action_dim", "time_horizon") if k in cfg}
    bpe = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir))
    return module.UniversalActionProcessor(bpe, **kwargs)
