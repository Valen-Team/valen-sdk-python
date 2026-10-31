"""Request-aware response validation shared by all public backends."""
import math

from .errors import ValenResponseError


def validate_response(request, response):
    model = response.get("model") if isinstance(response, dict) else None

    def fail(path, message):
        raise ValenResponseError(f"{path}: {message}", field=path, model=model)

    def number(value, path, low=None, high=None):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            fail(path, "expected a finite number")
        if (low is not None and value < low) or (high is not None and value > high):
            fail(path, "number is out of range")

    if not isinstance(response, dict):
        fail("response", "expected an object")
    if not isinstance(response.get("model"), str) or not response["model"]:
        fail("model", "expected a nonempty model identifier")
    usage = response.get("usage")
    if not isinstance(usage, dict):
        fail("usage", "expected an object")
    for key in ("input_tokens", "output_tokens"):
        value = usage.get(key)
        if value is not None and (type(value) is not int or value < 0):
            fail("usage." + key, "expected a nonnegative integer or null")
    public = "input" in request
    questions = request.get("questions", [] if public else {})
    if public:
        expected = {q.get("name", str(i)): q for i, q in enumerate(questions)}
        answers = response.get("answers")
        if not isinstance(answers, list):
            fail("answers", "expected a list")
        actual = {}
        for answer in answers:
            if not isinstance(answer, dict) or not isinstance(answer.get("name"), str):
                fail("answers", "expected named answer objects")
            name = answer["name"]
            if name in actual:
                fail(f"answers.{name}", "duplicate answer")
            actual[name] = answer
    else:
        expected = questions
        actual = response.get("answers")
        if not isinstance(actual, dict):
            fail("answers", "expected an object")
    if set(actual) != set(expected):
        fail("answers", f"question names differ (missing={sorted(set(expected)-set(actual))}, unknown={sorted(set(actual)-set(expected))})")
    if public:
        response["answers"] = [actual[name] for name in expected]
    for name, question in expected.items():
        q = question.model_dump() if hasattr(question, "model_dump") else question
        answer, path = actual[name], f"answers.{name}"
        kind = q["type"]
        if not isinstance(answer, dict) or answer.get("type") != kind:
            fail(path + ".type", f"expected {kind}")
        if kind in {"predicate", "noul"}:
            key = "probability" if public else "noul"
            number(answer.get(key), path + "." + key, 0, 1)
            continue
        criteria = q.get("criteria", {})
        keys = ([c["value"] for c in q["choices"]] if kind == "choice" else
                [c["label"] for c in q["levels"]]) if public else (
                    list(criteria) if kind == "choice" else [str(i) for i in range(len(criteria))])
        probs = answer.get("probabilities")
        if public:
            if not isinstance(probs, list):
                fail(path + ".probabilities", "expected a list")
            if any(not isinstance(p, dict) for p in probs):
                fail(path + ".probabilities", "expected candidate objects")
            labels = [p.get("value" if kind == "choice" else "label") for p in probs if isinstance(p, dict)]
            values = [p.get("probability") for p in probs if isinstance(p, dict)]
        else:
            if not isinstance(probs, dict):
                fail(path + ".probabilities", "expected an object")
            labels, values = list(probs), list(probs.values())
        # bool and string candidates must remain distinct.
        identity = lambda v: (type(v).__name__, str(v))
        if len(labels) != len(keys) or sorted(map(identity, labels)) != sorted(map(identity, keys)):
            fail(path + ".probabilities", "candidate labels differ from the request")
        for p in values:
            number(p, path + ".probabilities", 0, 1)
        if not math.isclose(sum(values), 1, abs_tol=1e-5, rel_tol=0):
            fail(path + ".probabilities", "probabilities must sum to 1")
        number(answer.get("confidence"), path + ".confidence", 0, 1)
        if kind == "choice":
            if identity(answer.get("choice")) not in list(map(identity, keys)):
                fail(path + ".choice", "unknown candidate")
        else:
            number(answer.get("score"), path + ".score", 0, len(keys)-1)
    return response


def validate_request(request):
    """Structural checks only: endpoint/model limits remain backend-specific."""
    from .errors import ValenInputError

    def fail(path, message):
        raise ValenInputError(f"{path}: {message}", field=path)

    public = "input" in request
    if public and not isinstance(request.get("input"), (str, list)):
        fail("input", "expected text or messages")
    if not public and not isinstance(request.get("state"), (str, dict)):
        fail("state", "expected text or an object")
    questions = request.get("questions")
    if public:
        if not isinstance(questions, list):
            fail("questions", "expected a list")
        entries = []
        for index, question in enumerate(questions):
            if not isinstance(question, dict):
                fail(f"questions.{index}", "expected an object")
            entries.append((question.get("name", str(index)), question))
    else:
        if not isinstance(questions, dict):
            fail("questions", "expected an object")
        entries = list(questions.items())
    names = set()
    for name, question in entries:
        if not isinstance(name, str) or not name or name in names:
            fail("questions", "question names must be unique nonempty strings")
        names.add(name)
        q = question.model_dump() if hasattr(question, "model_dump") else question
        path = f"questions.{name}"
        if not isinstance(q, dict):
            fail(path, "expected an object")
        if not isinstance(q.get("instructions"), str) or not q["instructions"].strip():
            fail(path + ".instructions", "expected nonempty text")
        kind = q.get("type")
        if kind == ("predicate" if public else "noul"):
            continue
        if kind not in {"choice", "score"}:
            fail(path + ".type", "unsupported question type; use raw predict for extensions")
        if public:
            candidates = q.get("choices" if kind == "choice" else "levels")
            if not isinstance(candidates, list) or not candidates:
                fail(path, "expected candidate objects")
            if any(not isinstance(c, dict) for c in candidates):
                fail(path, "expected candidate objects")
            labels = [c.get("value" if kind == "choice" else "label") for c in candidates]
            descriptions = [c.get("description") for c in candidates]
        else:
            criteria = q.get("criteria")
            if kind == "choice":
                if not isinstance(criteria, dict) or not criteria:
                    fail(path + ".criteria", "expected named candidates")
                labels, descriptions = list(criteria), list(criteria.values())
            else:
                if not isinstance(criteria, list):
                    fail(path + ".criteria", "expected ordered descriptions")
                labels, descriptions = [str(i) for i in range(len(criteria))], criteria
        allowed = (str, bool) if public and kind == "choice" else (str,)
        if any(type(label) not in allowed or label == "" for label in labels):
            fail(path, "invalid candidate labels")
        if len({(type(label).__name__, str(label)) for label in labels}) != len(labels):
            fail(path, "duplicate candidate labels")
        if kind == "score" and len(labels) < 2:
            fail(path, "Score requires at least two levels")
        if any(not isinstance(d, str) or not d.strip() for d in descriptions):
            fail(path, "candidate descriptions must be nonempty text")
