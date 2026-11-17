"""Turn source text into tokens, without executing any Python code."""
from dataclasses import dataclass
import re


class LanguageError(ValueError):
    """A readable error in NumLang source."""


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    column: int


_PATTERN = re.compile(r"(?P<number>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)|(?P<name>[A-Za-z_][A-Za-z_0-9]*)|(?P<op>[+\-*/^()])")


def tokenize(source: str) -> list[Token]:
    tokens = []
    pos = 0
    while pos < len(source):
        if source[pos].isspace():
            pos += 1
            continue
        match = _PATTERN.match(source, pos)
        if match is None:
            raise LanguageError(f"column {pos + 1}: unexpected character {source[pos]!r}")
        tokens.append(Token(match.lastgroup, match.group().lower(), pos + 1))
        pos = match.end()
    tokens.append(Token("end", "", len(source) + 1))
    return tokens
