"""Public request helpers and typed response objects."""
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr


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
    input_tokens: int = 0
    output_tokens: int = 0


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
    confidence: float


class PredicateAnswer(BaseModel):
    name: str
    type: Literal["predicate"]
    probability: float = Field(ge=0, le=1)


class ScoreAnswer(BaseModel):
    name: str
    type: Literal["score"]
    score: float
    probabilities: list[LevelProbability]
    confidence: float


class DecisionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    answers: list[ChoiceAnswer | PredicateAnswer | ScoreAnswer]
    usage: Usage
    internal_usage: dict[str, Any] | None = None

    def by_name(self, name):
        return next(answer for answer in self.answers if answer.name == name)


class SystemOneAnswer(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["choice", "noul", "score"]
    choice: str | None = None
    noul: float | None = None
    score: float | None = None
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    legend: dict[str, str] | None = None


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
