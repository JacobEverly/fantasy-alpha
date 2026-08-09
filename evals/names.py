"""Player-name normalization shared by label builders and DraftBench.

nflverse and ADP sources disagree on suffixes, punctuation, and formal vs.
nickname first names ("Mike Vick" vs "Michael Vick"). norm_name canonicalizes
both sides so (norm_name, position) joins work across sources.
"""
from __future__ import annotations

import re
import unicodedata

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

_FIRST_NAME_SETS = [
    ["mike", "michael"], ["chris", "christopher", "kris"], ["matt", "matthew"],
    ["rob", "robert", "bob", "bobby"], ["will", "william", "bill", "billy"],
    ["dan", "daniel", "danny"], ["dave", "david"], ["jim", "james", "jimmy"],
    ["joe", "joseph", "joey"], ["josh", "joshua"], ["ben", "benjamin"],
    ["alex", "alexander"], ["zach", "zachary", "zack"], ["jon", "jonathan"],
    ["steve", "steven", "stephen"], ["tony", "anthony"], ["drew", "andrew", "andy"],
    ["nick", "nicholas"], ["pat", "patrick"], ["greg", "gregory"],
    ["ken", "kenneth", "kenny"], ["jeff", "jeffrey"], ["tom", "thomas", "tommy"],
    ["gabe", "gabriel"], ["cam", "cameron"], ["mitch", "mitchell"],
    ["ron", "ronald", "ronnie"], ["ray", "raymond"], ["sam", "samuel"],
    ["ted", "theodore"], ["dj", "d j"], ["aj", "a j"], ["cj", "c j"],
]
_FIRST_CANON = {v: s[0] for s in _FIRST_NAME_SETS for v in s}
ALIASES = {"beaniewells": "chriswells"}


def norm_name(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    tokens = re.sub(r"[^a-z0-9 ]", "", ascii_name.lower().replace(".", "")).split()
    while tokens and tokens[-1] in SUFFIXES:
        tokens.pop()
    if tokens:
        tokens[0] = _FIRST_CANON.get(tokens[0], tokens[0])
    key = "".join(tokens)
    return ALIASES.get(key, key)
