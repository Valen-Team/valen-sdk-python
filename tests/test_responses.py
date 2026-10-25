from types import SimpleNamespace
import numpy as np
import pytest

from valen_sdk.responses import answer
from valen_sdk.protocol import to_internal, to_public


@pytest.mark.parametrize("temperature", [1.0, 0.5, 2.0, 1e-50])
def test_probabilities_are_finite_normalized_and_ordered(temperature):
    q = SimpleNamespace(kind="choice", keys=["red", "blue"], descriptions=["Red", "Blue"])
    result = answer(q, np.array([2.0, 0.0]), temperature)
    assert result["choice"] == "red"
    assert sum(result["probabilities"].values()) == pytest.approx(1)
    assert all(np.isfinite(p) for p in result["probabilities"].values())


@pytest.mark.parametrize("temperature", [0, -1, float("inf"), float("nan"), True])
def test_invalid_temperature(temperature):
    with pytest.raises(ValueError):
        answer(SimpleNamespace(keys=["a"], kind="choice"), [1.0], temperature)


def test_public_boolean_candidate_collision_roundtrip():
    body = {"input": "Test", "questions": [{"type": "choice", "name": "test", "instructions": "Choose",
            "choices": [{"value": True, "description": "Boolean"}, {"value": "true", "description": "String"}]}]}
    request = to_internal(body)
    assert list(request["questions"]["test"]["criteria"]) == ["bool:true", "true"]
    response = {"answers": {"test": {"probabilities": {"bool:true": .8, "true": .2}, "confidence": .6}}}
    assert to_public(body, response)["answers"][0]["choice"] is True
