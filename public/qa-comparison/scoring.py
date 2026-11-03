import math
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    answer: str = Field(min_length=1)
    verdict: Literal["yes", "no", "value", "unknown"]
    value: float | int | None
    sources: list[str]

    @model_validator(mode="after")
    def consistent(self):
        if self.verdict == "value":
            if self.value is None or isinstance(self.value, bool) or not math.isfinite(self.value):
                raise ValueError("value verdict requires a finite number")
        elif self.value is not None:
            raise ValueError("non-value verdict requires null")
        return self

def check_fields(answer, case, source_ids):
    expected = case["expected"]
    match = answer.verdict == expected["verdict"]
    if expected["verdict"] == "value":
        match = match and answer.value == expected["value"]
    return {"structured_match": int(match),
            "source_ids_valid": int(bool(answer.sources) and set(answer.sources) <= source_ids)}

def summarize(rows):
    n = len(rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    out = {"questions": n, "generation_successes": len(ok),
           "generation_failures": n-len(ok),
           "structured_match_rate": sum(r.get("structured_match", 0) for r in rows)/n,
           "source_id_valid_rate": sum(r.get("source_ids_valid", 0) for r in rows)/n}
    if ok:
        out["generation_latency_mean_s"] = sum(r["latency_s"] for r in ok)/len(ok)
    for metric in ("faithfulness", "relevancy"):
        vals = [r[metric]["score"] for r in rows if metric in r and r[metric].get("score") is not None]
        out[metric+"_coverage"] = len(vals)/n
        if vals:
            out[metric+"_mean"] = sum(vals)/len(vals)
    for kind in sorted({r["category"] for r in rows}):
        part = [r for r in rows if r["category"] == kind]
        out["match_"+kind] = sum(r.get("structured_match",0) for r in part)/len(part)
    return out
