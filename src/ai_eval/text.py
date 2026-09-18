"""Text normalization utilities."""

_CURLY = {
    "\u2019": "'",
    "\u2018": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u00a0": " ",
}


def normalize(text: str) -> str:
    """Lowercase, strip, and convert typographic quotes/dashes to ASCII."""
    for curly, straight in _CURLY.items():
        text = text.replace(curly, straight)
    return text.lower().strip()


def truncate(text: str, max_len: int = 400) -> str:
    """Truncate for display, preserving word boundaries."""
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "..."