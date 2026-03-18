"""
llm_core.py

Handles everything LLM-related: building the prompt, calling Ollama,
validating the response, and handing back a clean ExperimentProposal.

The Pydantic layer is non-negotiable here. The LLM will eventually return
something malformed — you want a ValidationError with a useful message,
not a KeyError that blows up halfway through a training run.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

import requests
from pydantic import BaseModel, Field, field_validator

from config import (
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
    OLLAMA_TIMEOUT,
    GOAL_F1,
    SEARCH_SPACE,
)

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ExperimentProposal:
    """What the LLM produces each iteration — the agent's next move."""
    rationale:            str        # diagnosis of what happened so far
    architecture:         str        # e.g. "EfficientNetV2-S, dropout=0.3"
    optimizer:            str
    learning_rate:        float
    batch_size:           int
    epochs:               int
    scheduler:            str        # CosineAnnealing | StepLR | OneCycleLR
    augmentations:        list[str]
    class_weights:        bool
    freeze_backbone:      bool
    unfreeze_after_epoch: int
    hypothesis:           str        # what the agent thinks will happen


class ExperimentProposalSchema(BaseModel):
    """
    Pydantic schema for the raw JSON the LLM returns.

    Keeps the validation logic separate from the dataclass so we can
    fail loudly before we ever touch ExperimentProposal.
    """
    rationale:            str
    architecture:         str
    optimizer:            str = Field(..., pattern="^(AdamW|SGD)$")
    learning_rate:        float
    batch_size:           int
    epochs:               int
    scheduler:            str
    augmentations:        list[str]
    class_weights:        bool
    freeze_backbone:      bool
    unfreeze_after_epoch: int
    hypothesis:           str

    @field_validator("scheduler")
    @classmethod
    def scheduler_must_be_valid(cls, v: str) -> str:
        valid = {"CosineAnnealing", "StepLR", "OneCycleLR"}
        if v not in valid:
            raise ValueError(f"scheduler must be one of {sorted(valid)}, got '{v}'")
        return v

    @field_validator("learning_rate")
    @classmethod
    def lr_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"learning_rate must be positive, got {v}")
        return v

    @field_validator("batch_size")
    @classmethod
    def batch_size_sane(cls, v: int) -> int:
        if v not in (32, 64):
            raise ValueError(f"batch_size must be 32 or 64 for this GPU, got {v}")
        return v

    @field_validator("epochs")
    @classmethod
    def epochs_in_range(cls, v: int) -> int:
        if not (1 <= v <= 30):
            raise ValueError(f"epochs out of expected range [1, 30], got {v}")
        return v

    @field_validator("augmentations")
    @classmethod
    def augmentations_must_be_valid(cls, v: list[str]) -> list[str]:
        valid = set(SEARCH_SPACE["augmentations"])
        bad = [a for a in v if a not in valid]
        if bad:
            raise ValueError(f"unknown augmentation(s): {bad}. Valid: {sorted(valid)}")
        return v

    @field_validator("unfreeze_after_epoch")
    @classmethod
    def unfreeze_epoch_valid(cls, v: int) -> int:
        if v not in (0, 3, 5):
            raise ValueError(f"unfreeze_after_epoch must be 0, 3, or 5. Got {v}")
        return v

    def to_dataclass(self) -> ExperimentProposal:
        return ExperimentProposal(**self.model_dump())


# ── Prompt construction ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert ML engineer running a fine-tuning experiment loop on a medical \
image classification dataset. Your job is to analyze previous training runs and \
propose the next experiment.

Rules:
- Output ONLY a valid JSON object. No explanation, no preamble, no markdown fences.
- Every field in the schema is required.
- "rationale" should explain what went wrong (or right) in previous runs and why \
you're making these specific changes.
- "hypothesis" should state what you predict will happen — be specific about the \
expected val_f1 and why.
- batch_size must be 32 or 64.
- optimizer must be "AdamW" or "SGD".
- scheduler must be one of: "CosineAnnealing", "StepLR", "OneCycleLR".
- augmentations must be a subset of: {augmentations}.
- unfreeze_after_epoch must be one of: 0, 3, 5.

JSON schema:
{{
  "rationale":            "<string — diagnosis of previous runs>",
  "architecture":         "<string — e.g. EfficientNetV2-S, dropout=0.3>",
  "optimizer":            "<AdamW|SGD>",
  "learning_rate":        <float>,
  "batch_size":           <32|64>,
  "epochs":               <int>,
  "scheduler":            "<CosineAnnealing|StepLR|OneCycleLR>",
  "augmentations":        ["<aug1>", ...],
  "class_weights":        <true|false>,
  "freeze_backbone":      <true|false>,
  "unfreeze_after_epoch": <0|3|5>,
  "hypothesis":           "<string — predicted outcome and reasoning>"
}}
""".format(augmentations=SEARCH_SPACE["augmentations"])


def _build_user_prompt(context: str) -> str:
    return (
        f"Target: val_f1 >= {GOAL_F1}\n\n"
        f"Experiment history:\n{context}\n\n"
        "Propose the next experiment as JSON."
    )


# ── LLM client ────────────────────────────────────────────────────────────────

class LLMCore:
    """
    Thin wrapper around Ollama's /api/chat endpoint.

    Only public method you need: propose(context) → ExperimentProposal.
    Everything else is plumbing.
    """

    def __init__(
        self,
        host:  str = OLLAMA_HOST,
        model: str = OLLAMA_MODEL,
    ) -> None:
        self.host  = host.rstrip("/")
        self.model = model
        self._url  = f"{self.host}/api/chat"

    # ── public ────────────────────────────────────────────────────────────────

    def propose(self, context: str) -> ExperimentProposal:
        """
        Given a formatted history of past runs, ask the LLM for the next
        experiment proposal. Validates with Pydantic before returning.

        Raises:
            RuntimeError: if Ollama call fails or the response can't be parsed
                          after retries.
        """
        raw = self._call(context)
        return self._parse(raw)

    def ping(self) -> bool:
        """Quick health check — returns True if Ollama is up and the model is loaded."""
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            return any(self.model in m for m in models)
        except requests.RequestException:
            return False

    # ── private ───────────────────────────────────────────────────────────────

    def _call(self, context: str, retries: int = 2) -> str:
        """
        POST to Ollama and return the raw message content string.
        Retries on connection errors — not on bad JSON (that's a parsing problem).
        """
        payload = {
            "model":  self.model,
            "stream": False,
            "keep_alive": 0,  # unload model from VRAM immediately after response
            "options": {"temperature": OLLAMA_TEMPERATURE},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _build_user_prompt(context)},
            ],
        }

        last_exc: Optional[Exception] = None
        for attempt in range(1, retries + 2):
            try:
                logger.debug("Ollama call attempt %d/%d", attempt, retries + 1)
                r = requests.post(
                    self._url,
                    json=payload,
                    timeout=OLLAMA_TIMEOUT,
                )
                r.raise_for_status()
                content = r.json()["message"]["content"]
                logger.debug("Raw LLM response:\n%s", content)
                return content

            except requests.RequestException as e:
                last_exc = e
                logger.warning("Ollama request failed (attempt %d): %s", attempt, e)

        raise RuntimeError(
            f"Ollama unreachable after {retries + 1} attempts: {last_exc}"
        )

    def _parse(self, raw: str) -> ExperimentProposal:
        """
        Extract the JSON block from the LLM's response and validate it.

        The model sometimes wraps JSON in ```json fences or adds a one-liner
        before the object — strip that before parsing.
        """
        cleaned = self._extract_json(raw)
        if cleaned is None:
            raise RuntimeError(
                f"Could not find a JSON object in LLM response:\n{raw[:500]}"
            )

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"LLM returned invalid JSON: {e}\nRaw content:\n{cleaned[:500]}"
            ) from e

        try:
            schema = ExperimentProposalSchema(**data)
        except Exception as e:
            raise RuntimeError(
                f"ExperimentProposal validation failed: {e}\nParsed data:\n{data}"
            ) from e

        return schema.to_dataclass()

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """
        Pull the first {...} block out of the response.
        Handles ```json fences, leading text, trailing commentary.
        """
        text = re.sub(r"```(?:json)?", "", text).strip()

        start = text.find("{")
        if start == -1:
            return None

        depth  = 0
        in_str = False
        escape = False

        for i, ch in enumerate(text[start:], start=start):
            if escape:
                escape = False
                continue
            if ch == "\\" and in_str:
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

        return None


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    llm = LLMCore()

    print(f"Checking Ollama at {llm.host} with model '{llm.model}'...")
    if not llm.ping():
        print(
            f"\n[ERROR] Model '{llm.model}' not found in Ollama.\n"
            f"Run:  ollama pull {llm.model}\n"
            f"Then: ollama serve"
        )
        sys.exit(1)

    print("Ollama is up. Sending a test prompt...\n")

    test_context = """
Run 1:
  Proposal:   AdamW, lr=0.001, batch=32, epochs=10, no class weights, backbone frozen
  Scheduler:  CosineAnnealing
  Augments:   RandomHorizontalFlip, RandomRotation
  Hypothesis: Expect ~0.78 F1 as a baseline
  Result:     best_val_f1=0.7812, status=done
  Verdict:    Hypothesis roughly correct. Normal recall was 0.61 — clear sign of
              class imbalance causing the model to over-predict Pneumonia.
""".strip()

    try:
        proposal = llm.propose(test_context)
        print("=" * 60)
        print("Proposal parsed successfully:")
        print(f"  rationale:    {proposal.rationale[:80]}...")
        print(f"  optimizer:    {proposal.optimizer}")
        print(f"  lr:           {proposal.learning_rate}")
        print(f"  batch_size:   {proposal.batch_size}")
        print(f"  epochs:       {proposal.epochs}")
        print(f"  scheduler:    {proposal.scheduler}")
        print(f"  augments:     {proposal.augmentations}")
        print(f"  class_weights:{proposal.class_weights}")
        print(f"  freeze:       {proposal.freeze_backbone}")
        print(f"  unfreeze at:  epoch {proposal.unfreeze_after_epoch}")
        print(f"  hypothesis:   {proposal.hypothesis[:80]}...")
        print("=" * 60)
        print("\nPhase 1 smoke test passed.")
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)