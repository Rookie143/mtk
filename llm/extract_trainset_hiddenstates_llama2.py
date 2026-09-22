from contextlib import redirect_stderr, redirect_stdout
import os

import torch

from feature_protocol import FeatureProtocol, native_extract, set_determinism
from llama2_memory_extract import extract as memory_extract


ATTACK_FILES = ["JailJudge_1.json", "ijp_1.json", "nanogcg_1.json", "autodan_1.json", "drattack_1.json", "pair_1.json", "pap_gpt3.5_1.json", "pap_gpt4_1.json", "pap_llama2_1.json", "saa_1.json", "tap_1.json", "zulu_1.json"]


def extract_trainset_hiddenstates(
    seed, device, dtype, batch_size, benign_train_set_list, malicious_train_set_list,
    rebuild_cache=False, force_rebuild_cache=False, k=10,
):
    protocol = FeatureProtocol(
        "llama2", "llama2", ATTACK_FILES,
        benign_train_set_list, malicious_train_set_list,
        "last_token_native_llama2_chat_template",
        "last_token_native_llama2_chat_template",
    )
    set_determinism(42)
    rank_path = protocol.rank_cache_path(seed, k)
    saved = None
    if not force_rebuild_cache and not rebuild_cache:
        saved = protocol.load_rank_cache(seed, dtype, k)
        if saved is not None:
            return saved
    if saved is None:
        with open(os.devnull, "w", encoding="utf-8") as sink:
            with redirect_stdout(sink), redirect_stderr(sink):
                model, tokenizer = protocol.model_and_tokenizer(device, dtype)
                extract = memory_extract if batch_size == 16 else native_extract
                protocol.prepare_test_features(device, dtype, batch_size, force_rebuild_cache, model, tokenizer, extract)
                protocol.prepare_training_features(seed, device, dtype, batch_size, force_rebuild_cache, model, tokenizer, extract)
                protocol.prepare_ranks(seed, device, k)
        del model, tokenizer
    saved = torch.load(rank_path, map_location="cpu", weights_only=False)
    protocol.validate_rank(saved, seed, dtype, k)
    return saved
