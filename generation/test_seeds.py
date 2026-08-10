import random

from generation.seeds import RBAC_KIND_WEIGHTS, WORKLOAD_KIND_WEIGHTS, sample_seed


def test_deterministic_given_same_rng_state():
    a = sample_seed(random.Random(123))
    b = sample_seed(random.Random(123))
    assert a == b


def test_rbac_mode_only_yields_rbac_kinds():
    rng = random.Random(7)
    for _ in range(50):
        seed = sample_seed(rng, mode="rbac")
        assert seed.kind in RBAC_KIND_WEIGHTS


def test_hard_negative_mode_only_yields_workload_kinds():
    rng = random.Random(7)
    for _ in range(50):
        seed = sample_seed(rng, mode="hard-negative")
        assert seed.kind in WORKLOAD_KIND_WEIGHTS


def test_base_mode_has_variety_of_kinds():
    rng = random.Random(7)
    kinds = {sample_seed(rng, mode="base").kind for _ in range(200)}
    assert len(kinds) >= 5


def test_prompt_constraints_mentions_key_fields():
    seed = sample_seed(random.Random(1))
    text = seed.to_prompt_constraints()
    assert seed.domain in text
    assert seed.service_name in text
    assert seed.kind in text


def test_sequence_of_seeds_is_diverse():
    rng = random.Random(99)
    seeds = [sample_seed(rng) for _ in range(100)]
    service_names = {s.service_name for s in seeds}
    assert len(service_names) > 50
