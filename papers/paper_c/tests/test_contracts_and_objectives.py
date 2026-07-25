from copy import deepcopy
import math
from pathlib import Path

import pytest

from paper_c.contracts import (
    ContractError,
    load_config,
    output_path,
    validate_config,
)
from paper_c.objectives import (
    assert_step_zero_identity,
    dpo_logratio_loss,
    dpo_loss,
    pair_ce_loss,
    signed_margin,
)


def config():
    return load_config("config/study.json")


def test_candidate_and_smoke_configs_are_valid():
    assert load_config("config/study.json")["stage2"]["max_steps"] == 200
    assert load_config("config/smoke.json")["stage2"]["max_steps"] == 2


def test_config_rejects_unknown_fields_and_objective_confound():
    candidate = deepcopy(config())
    candidate["mystery"] = True
    with pytest.raises(ContractError, match="unknown"):
        validate_config(candidate)
    candidate = deepcopy(config())
    candidate["stage2"]["dropout"] = 0.1
    with pytest.raises(ContractError, match="dropout"):
        validate_config(candidate)
    candidate = deepcopy(config())
    candidate["stage2"]["objectives"] = ["verdict_ce", "dpo", "pair_ce"]
    with pytest.raises(ContractError, match="objectives"):
        validate_config(candidate)


def test_output_paths_cannot_escape_workspace():
    assert output_path("artifacts/example").is_relative_to(output_path("."))
    with pytest.raises(ContractError, match="escapes"):
        output_path("../../outside-paper-c")
    with pytest.raises(ContractError, match="escapes"):
        output_path(Path("/tmp/paper-c-forbidden"))


def test_exact_scalar_objectives():
    margin = 1.3
    reference = -0.4
    assert math.isclose(pair_ce_loss(margin, 1.0), math.log1p(math.exp(-margin)))
    assert math.isclose(dpo_loss(reference, reference, 0.1), math.log(2.0))
    assert math.isclose(
        dpo_loss(margin, reference, 0.1),
        dpo_logratio_loss(margin, 0.0, reference, 0.0, 0.1),
        abs_tol=1e-12,
    )
    assert signed_margin(1.0, 3.0, 1) == 2.0
    assert signed_margin(1.0, 3.0, 0) == -2.0


def test_step_zero_preflight_is_fail_closed():
    result = assert_step_zero_identity([0.1, -0.2], [0.1, -0.2], beta=0.1, atol=1e-6)
    assert math.isclose(result["mean_dpo_loss"], math.log(2.0))
    with pytest.raises(ContractError, match="reference margin"):
        assert_step_zero_identity([0.2], [0.1], beta=0.1, atol=1e-3)

