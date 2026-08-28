"""Extract <question>/<hint> blocks from Challenger rollouts."""
import regex as re


_Q_RE = re.compile(r"<question>(.*?)</question>", re.DOTALL)
_H_RE = re.compile(r"<hint>(.*?)</hint>", re.DOTALL)

# Reject placeholder-only echoes of the Challenger prompt template.
_PLACEHOLDER_RE = re.compile(
    r"\[\s*your\s+(question|hint|problem|answer)\b|"
    r"\byour\s+(question|hint|problem)\s+here\b",
    re.IGNORECASE,
)


def _is_placeholder(s: str) -> bool:
    return bool(_PLACEHOLDER_RE.search(s))


def extract_qh(text: str) -> tuple[str, str]:
    qs = _Q_RE.findall(text)
    hs = _H_RE.findall(text)
    if not (qs and hs):
        return "", ""
    q = qs[-1].strip()
    h = hs[-1].strip()
    if not q or not h:
        return "", ""
    if _is_placeholder(q) or _is_placeholder(h):
        return "", ""
    return q, h
