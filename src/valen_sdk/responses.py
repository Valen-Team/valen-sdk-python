"""Existing Valen response formulas, shared by local logits and LLM probabilities."""
import math


def answer(question, logits, temperature=1.0):
    import numpy as np

    if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Calibration temperature must be finite and positive")
    scores = np.asarray(logits, dtype=np.float64)
    if scores.ndim != 1 or len(scores) != len(question.keys) or not np.isfinite(scores).all():
        raise ValueError("Expected one finite logit per candidate")
    with np.errstate(over="ignore", under="ignore"):
        probs = np.exp((scores - scores.max()) / temperature)
    return from_probabilities(question, (probs / probs.sum()).tolist())


def from_probabilities(question, probs):
    """Typed answer from one probability per candidate, in candidate order."""
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


def box(logits, distances, grid):
    """bbox answer from the grounding graph: per-token logits and edge distances in cell units."""
    import numpy as np

    rows, cols = grid
    cy, cx = np.meshgrid((np.arange(rows, dtype=np.float32) + 0.5) / rows,
                         (np.arange(cols, dtype=np.float32) + 0.5) / cols, indexing="ij")
    cx, cy = cx.reshape(-1), cy.reshape(-1)
    d = np.asarray(distances, dtype=np.float32) * np.asarray([1 / cols, 1 / rows, 1 / cols, 1 / rows], dtype=np.float32)
    boxes = np.stack((cx - d[:, 0], cy - d[:, 1], cx + d[:, 2], cy + d[:, 3]), -1).clip(0, 1)
    scores = np.asarray(logits, dtype=np.float32)
    probabilities = np.exp(scores - scores.max())
    probabilities /= probabilities.sum()
    index = int(probabilities.argmax())
    best = boxes[index]
    inside = (cx >= best[0]) & (cx <= best[2]) & (cy >= best[1]) & (cy <= best[3])
    confidence = float(probabilities[inside].sum()) if inside.any() else float(probabilities[index])
    x1, y1, x2, y2 = (round(float(v), 4) for v in best)
    return {"type": "bbox", "bbox": [x1, y1, x2, y2], "point": [round((x1 + x2) / 2, 4), round((y1 + y2) / 2, 4)],
            "confidence": round(min(max(confidence, 0.0), 1.0), 4)}
