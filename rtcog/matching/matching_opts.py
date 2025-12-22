from dataclasses import dataclass

@dataclass(frozen=True)
class MatchingOpts:
    match_method: str
    match_start: int
    vols_noaction: int