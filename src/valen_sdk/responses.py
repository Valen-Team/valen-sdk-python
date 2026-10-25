"""NumPy implementation of the existing Valen response formulas."""
import math
import numpy as np


def answer(question, logits, temperature=1.0):
    if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Calibration temperature must be finite and positive")
    scores = np.asarray(logits, dtype=np.float64)
    if scores.ndim != 1 or len(scores) != len(question.keys) or not np.isfinite(scores).all():
        raise ValueError("Expected one finite logit per candidate")
    with np.errstate(over="ignore", under="ignore"):
        probs = np.exp((scores - scores.max()) / temperature)
    probs = (probs / probs.sum()).tolist()
    total = sum(probs)
    probs = [p / total for p in probs]
    result = {"type": question.kind}
    if question.kind == "noul":
        result["noul"] = probs[question.keys.index("true")]
        return result
    result["probabilities"] = dict(zip(question.keys, probs))
    mode = max(range(len(probs)), key=probs.__getitem__)
    if question.kind == "choice":
        result["choice"] = question.keys[mode]
        result["confidence"] = 1.0 if len(probs) == 1 else max(0.0, (max(probs) - 1 / len(probs)) / (1 - 1 / len(probs)))
    else:
        result["score"] = sum(i * p for i, p in enumerate(probs))
        result["legend"] = dict(zip(question.keys, question.descriptions))
        distance = sum(p * abs(i - mode) for i, p in enumerate(probs))
        uniform_mad = sum(abs(i - (len(probs) - 1) / 2) for i in range(len(probs))) / len(probs)
        result["confidence"] = max(0.0, 1.0 - distance / uniform_mad)
    return result
