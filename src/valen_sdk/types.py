"""Public request helpers and typed response objects."""
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, model_validator


class Choice(BaseModel):
    type: Literal["choice"] = "choice"
    instructions: str
    criteria: dict[str, str]


class Noul(BaseModel):
    type: Literal["noul"] = "noul"
    instructions: str


Predicate = Noul


class Score(BaseModel):
    type: Literal["score"] = "score"
    instructions: str
    criteria: list[str]


class Usage(BaseModel):
    model_config = ConfigDict(extra="allow")
    input_tokens: int | None = None
    output_tokens: int | None = None


class ChoiceProbability(BaseModel):
    value: StrictStr | StrictBool
    probability: float = Field(ge=0, le=1)


class LevelProbability(BaseModel):
    label: str
    value: int
    probability: float = Field(ge=0, le=1)


class ChoiceAnswer(BaseModel):
    name: str
    type: Literal["choice"]
    choice: StrictStr | StrictBool
    probabilities: list[ChoiceProbability]
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)


class PredicateAnswer(BaseModel):
    name: str
    type: Literal["predicate"]
    probability: float = Field(ge=0, le=1)


class ScoreAnswer(BaseModel):
    name: str
    type: Literal["score"]
    score: float = Field(allow_inf_nan=False)
    probabilities: list[LevelProbability]
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)


class DecisionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    answers: list[ChoiceAnswer | PredicateAnswer | ScoreAnswer]
    usage: Usage
    internal_usage: dict[str, Any] | None = None

    def by_name(self, name):
        for answer in self.answers:
            if answer.name == name:
                return answer
        raise KeyError(f"Unknown answer {name!r}; available: {[a.name for a in self.answers]}")


class SystemOneAnswer(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["choice", "noul", "score"]
    choice: str | None = None
    noul: float | None = None
    score: float | None = None
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    legend: dict[str, str] | None = None

    @model_validator(mode="after")
    def required_answer_fields(self):
        import math
        required = {"choice": ("choice", "probabilities", "confidence"),
                    "noul": ("noul",), "score": ("score", "probabilities", "confidence", "legend")}[self.type]
        for field in required:
            if getattr(self, field) is None:
                raise ValueError(f"{self.type} answer requires {field}")
        for field in ("noul", "score", "confidence"):
            value = getattr(self, field)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{field} must be finite")
        return self


class SystemOneResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    answers: dict[str, SystemOneAnswer]
    usage: Usage
    internal_usage: dict[str, Any] | None = None

    @property
    def choices(self):
        return {key: answer for key, answer in self.answers.items() if answer.type == "choice"}

    @property
    def nouls(self):
        return {key: answer for key, answer in self.answers.items() if answer.type == "noul"}

    @property
    def scores(self):
        return {key: answer for key, answer in self.answers.items() if answer.type == "score"}
