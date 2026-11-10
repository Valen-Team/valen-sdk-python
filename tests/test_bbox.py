import json
import math

import httpx
import numpy as np
import pytest

from valen_sdk import Bbox, Valen, ValenInputError, ValenResponseError
from valen_sdk.onnx import block_attention_mask
from valen_sdk.protocol import to_internal, to_public
from valen_sdk.responses import box
from valen_sdk.schema import candidates
from valen_sdk.validation import validate_request, validate_response

ANSWER = {"bbox": [0.25, 0.3, 0.8, 0.9], "point": [0.525, 0.6], "confidence": 0.9}
USAGE = {"input_tokens": 10, "output_tokens": 0}


def test_decisions_protocol_roundtrip():
    body = {"input": "Test", "questions": [
        {"name": "mug", "type": "bbox", "instructions": "The mug.", "image": 1},
        {"name": "cup", "type": "bbox", "instructions": "The cup."},
        {"name": "full", "type": "predicate", "instructions": "Full?"}]}
    request = to_internal(body)
    assert request["questions"] == {
        "mug": {"type": "bbox", "instructions": "The mug.", "image": 1},
        "cup": {"type": "bbox", "instructions": "The cup.", "image": 0},
        "full": {"type": "noul", "instructions": "Full?"}}
    assert candidates(request["questions"]["mug"]) == []
    response = {"model": "Valen", "usage": USAGE, "answers": {
        "mug": {"type": "bbox", **ANSWER}, "cup": {"type": "bbox", **ANSWER}, "full": {"type": "noul", "noul": 0.8}}}
    public = to_public(body, response)
    assert public["answers"] == [{"name": "mug", "type": "bbox", **ANSWER}, {"name": "cup", "type": "bbox", **ANSWER},
                                 {"name": "full", "type": "predicate", "probability": 0.8}]
    assert validate_response(body, public) == public


@pytest.mark.parametrize("image", [-1, 1.0, True, "0"])
def test_invalid_image_index(image):
    question = {"type": "bbox", "instructions": "The mug.", "image": image}
    with pytest.raises(ValueError, match="image"):
        to_internal({"input": "Test", "questions": [question]})
    with pytest.raises(ValueError, match="image"):
        candidates(question)
    for body in ({"input": "Test", "questions": [question]}, {"state": "Test", "questions": {"mug": question}}):
        with pytest.raises(ValenInputError) as error:
            validate_request(body)
        assert error.value.field.endswith(".image")


@pytest.mark.parametrize("change", [{"bbox": [0.1, 0.2, 0.3]}, {"bbox": [0.1, 0.2, 0.3, 1.5]}, {"point": [0.5]},
                                    {"point": None}, {"confidence": 2}, {"bbox": [0.1, 0.2, 0.3, float("nan")]}])
def test_response_validation(change):
    public = {"input": "Test", "questions": [{"name": "mug", "type": "bbox", "instructions": "The mug."}]}
    internal = {"state": "Test", "questions": {"mug": Bbox(instructions="The mug.")}}
    validate_request(public)
    validate_request(internal)
    good = {"type": "bbox", **ANSWER}
    validate_response(public, {"model": "Valen", "usage": USAGE, "answers": [{"name": "mug", **good}]})
    validate_response(internal, {"model": "Valen", "usage": USAGE, "answers": {"mug": good}})
    bad = {**good, **change}
    for request, answers in ((public, [{"name": "mug", **bad}]), (internal, {"mug": bad})):
        with pytest.raises(ValenResponseError) as error:
            validate_response(request, {"model": "Valen", "usage": USAGE, "answers": answers})
        assert error.value.field == "answers.mug." + next(iter(change))


def test_remote_passes_bbox_through():
    requests = []

    def handle(request):
        requests.append(request)
        public = request.url.path.endswith("decisions")
        answer = {"type": "bbox", **ANSWER}
        return httpx.Response(200, json={"model": "Valen", "usage": USAGE,
                                         "answers": [{"name": "mug", **answer}] if public else {"mug": answer}})

    with Valen.remote("https://example.test/v1", transport=httpx.MockTransport(handle)) as client:
        result = client.decisions(input="Test", questions=[{"name": "mug", "type": "bbox", "instructions": "The mug."}])
        assert result.by_name("mug").bbox == ANSWER["bbox"] and result.by_name("mug").point == ANSWER["point"]
        result = client.system_one("Test", {"mug": Bbox(instructions="The mug.", image=1)})
        assert result.bboxes["mug"].confidence == 0.9
    assert json.loads(requests[1].content)["questions"]["mug"] == {"type": "bbox", "instructions": "The mug.", "image": 1}


def reference_decode(logits, distances, grid):
    """Scalar port of GroundingHead.forward (after the regression) and decode in valen/modeling/qwen/grounding.py."""
    rows, cols = grid
    centres = [((c + 0.5) / cols, (r + 0.5) / rows) for r in range(rows) for c in range(cols)]
    clamp = lambda v: min(max(v, 0.0), 1.0)
    boxes = [[clamp(cx - d[0] / cols), clamp(cy - d[1] / rows), clamp(cx + d[2] / cols), clamp(cy + d[3] / rows)]
             for (cx, cy), d in zip(centres, distances)]
    top = max(logits)
    weights = [math.exp(z - top) for z in logits]
    probabilities = [w / sum(weights) for w in weights]
    index = max(range(len(logits)), key=probabilities.__getitem__)
    x1, y1, x2, y2 = boxes[index]
    inside = [p for p, (cx, cy) in zip(probabilities, centres) if x1 <= cx <= x2 and y1 <= cy <= y2]
    confidence = sum(inside) if inside else probabilities[index]
    x1, y1, x2, y2 = (round(v, 4) for v in boxes[index])
    return {"type": "bbox", "bbox": [x1, y1, x2, y2], "point": [round((x1 + x2) / 2, 4), round((y1 + y2) / 2, 4)],
            "confidence": round(clamp(confidence), 4)}


def test_decode_matches_reference():
    grid = (3, 4)
    rng = np.random.default_rng(0)
    for _ in range(50):
        logits = rng.normal(0, 3, 12)
        distances = np.log1p(np.exp(rng.normal(0, 1.5, (12, 4)))) + 0.5
        assert box(logits, distances, grid) == reference_decode(logits.tolist(), distances.tolist(), grid)


def test_decode_by_hand():
    # 2 x 2 grid; token 3 (bottom right, centre 0.75, 0.75) wins.
    logits = [0.0, 0.0, 0.0, math.log(5.0)]
    distances = [[0.5] * 4] * 3 + [[1.2, 0.6, 0.9, 0.5]]
    # x1 = 0.75 - 1.2 / 2, y1 = 0.75 - 0.6 / 2, x2 = min(0.75 + 0.9 / 2, 1), y2 = min(0.75 + 0.5 / 2, 1);
    # centres inside: tokens 2 (0.25, 0.75) and 3, with probabilities 1/8 and 5/8.
    assert box(logits, distances, (2, 2)) == {"type": "bbox", "bbox": [0.15, 0.45, 1.0, 1.0],
                                              "point": [0.575, 0.725], "confidence": 0.75}
    # A box that contains no centre reports the winning token's probability.
    result = box([0, 0, 0, 0, math.log(4.0)], [[0.6] * 4] * 4 + [[0.5] * 4], (1, 5))
    assert result["bbox"] == [0.8, 0.0, 1.0, 1.0] and result["confidence"] == 0.5
    assert box([1000.0, -1000.0], [[0.5] * 4] * 2, (1, 2))["confidence"] == 1.0


def visible(mask):
    return (mask == 0).astype(int).tolist()


def test_bbox_blocks_mask_keeps_bbox_questions_apart():
    # state, state | glue | Q1 | Q2 (bbox) | Q3 | glue | readouts 1, 2, 3
    segments = np.array([-1, -1, 0, 1, 2, 3, 0, 1, 2, 3])
    # forward_kwargs maps decision blocks 1 and 3 to 1 and leaves the bbox block 2 alone; segment_mask then
    # lets a state token see the state, and any other token see the state, the glue and its own block.
    state = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    glue = [1, 1, 1, 0, 0, 0, 1, 0, 0, 0]
    decision = [1, 1, 1, 1, 0, 1, 1, 1, 0, 1]
    bbox = [1, 1, 1, 0, 1, 0, 1, 0, 1, 0]
    assert visible(block_attention_mask("bbox_blocks", segments, 2, (2,))) == [
        state, state, glue, decision, bbox, decision, glue, decision, bbox, decision]
    # Without bbox questions every question shares block 1.
    merged = [1] * 10
    assert visible(block_attention_mask("bbox_blocks", segments, 2)) == [
        state, state, glue, merged, merged, merged, glue, merged, merged, merged]
    # Two bbox questions are isolated from each other and from the decision question.
    one = [1, 1, 1, 1, 0, 0, 1, 1, 0, 0]
    two = [1, 1, 1, 0, 1, 0, 1, 0, 1, 0]
    three = [1, 1, 1, 0, 0, 1, 1, 0, 0, 1]
    assert visible(block_attention_mask("bbox_blocks", segments, 2, (2, 3))) == [
        state, state, glue, one, two, three, glue, one, two, three]
    assert visible(block_attention_mask("question_blocks", segments, 2)) == [
        state, state, glue, one, two, three, glue, one, two, three]


def test_bbox_blocks_mask_when_first_question_is_bbox():
    # The reference keeps bbox block 1 as 1 and also merges the decision blocks into 1, so they share a block.
    segments = np.array([-1, -1, 0, 1, 2, 3, 0, 1, 2, 3])
    state = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    glue = [1, 1, 1, 0, 0, 0, 1, 0, 0, 0]
    merged = [1] * 10
    assert visible(block_attention_mask("bbox_blocks", segments, 2, (1,))) == [
        state, state, glue, merged, merged, merged, glue, merged, merged, merged]
    # Bbox questions 1 and 3 with decision question 2: block 2 joins block 1, block 3 stays alone.
    first = [1, 1, 1, 1, 1, 0, 1, 1, 1, 0]
    third = [1, 1, 1, 0, 0, 1, 1, 0, 0, 1]
    mask = block_attention_mask("bbox_blocks", segments, 2, (1, 3))
    assert visible(mask) == [state, state, glue, first, first, third, glue, first, first, third]
    assert mask.dtype == np.float32 and mask.min() == np.finfo(np.float32).min


def test_llm_backend_rejects_bbox():
    client = Valen.llm("openai", "test", api_key="test", transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    with pytest.raises(ValueError, match="Unsupported question type: bbox"):
        client.decisions(input="Test", questions=[{"name": "mug", "type": "bbox", "instructions": "The mug."}])
