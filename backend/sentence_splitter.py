from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

_WHITESPACE_RE = re.compile(r"\s+")
_TERMINAL_CLOSERS = "\"')]}”’"
_TOKEN_STRIP_CHARS = "\"'([{<)]}>.,!?;:”“’"
_ABBREVIATION_RE = re.compile(r"([A-Za-z][A-Za-z.]*)\.$")
_INITIAL_OR_ACRONYM_RE = re.compile(r"(?:[A-Za-z]\.){1,}$")
_COMMON_ABBREVIATIONS = {
    "adj.",
    "adm.",
    "al.",
    "approx.",
    "asst.",
    "assoc.",
    "ave.",
    "brig.",
    "bros.",
    "capt.",
    "cmdr.",
    "col.",
    "corp.",
    "cpl.",
    "dr.",
    "etc.",
    "e.g.",
    "est.",
    "gen.",
    "gov.",
    "hon.",
    "i.e.",
    "inc.",
    "jr.",
    "lt.",
    "maj.",
    "messrs.",
    "mlle.",
    "mme.",
    "mr.",
    "mrs.",
    "ms.",
    "mt.",
    "no.",
    "p.m.",
    "ph.d.",
    "prof.",
    "rep.",
    "rev.",
    "sen.",
    "sgt.",
    "sr.",
    "st.",
    "supt.",
    "vs.",
}
_COMMON_SENTENCE_STARTERS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "he",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "my",
    "on",
    "our",
    "please",
    "she",
    "that",
    "the",
    "their",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "was",
    "we",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "would",
    "you",
    "your",
}


@dataclass(frozen=True)
class SentenceFragment:
    sentence_index: int
    text: str


class SentenceSplitter(Protocol):
    def split(self, text: str) -> list[SentenceFragment]:
        """Return normalized sentences in original order."""


class EnglishSentenceSplitter:
    def split(self, text: str) -> list[SentenceFragment]:
        normalized_text = normalize_whitespace(text)
        if not normalized_text:
            return []

        sentences: list[str] = []
        sentence_start = 0
        cursor = 0
        length = len(normalized_text)

        while cursor < length:
            current_char = normalized_text[cursor]
            if current_char not in ".!?":
                cursor += 1
                continue

            sentence_end = self._consume_terminal(normalized_text, cursor)
            next_index = sentence_end + 1
            while next_index < length and normalized_text[next_index].isspace():
                next_index += 1

            if self._is_boundary(
                text=normalized_text,
                sentence_start=sentence_start,
                terminal_index=cursor,
                terminal_end=sentence_end,
                next_index=next_index,
            ):
                sentence_text = normalized_text[sentence_start : sentence_end + 1].strip()
                if sentence_text:
                    sentences.append(sentence_text)
                sentence_start = next_index
                cursor = next_index
                continue

            cursor = sentence_end + 1

        trailing_text = normalized_text[sentence_start:].strip()
        if trailing_text:
            sentences.append(trailing_text)

        return [
            SentenceFragment(sentence_index=index, text=sentence_text)
            for index, sentence_text in enumerate(sentences)
        ]

    def _consume_terminal(self, text: str, terminal_index: int) -> int:
        terminal_end = terminal_index
        while terminal_end + 1 < len(text) and text[terminal_end + 1] in ".!?":
            terminal_end += 1
        while terminal_end + 1 < len(text) and text[terminal_end + 1] in _TERMINAL_CLOSERS:
            terminal_end += 1
        return terminal_end

    def _is_boundary(
        self,
        *,
        text: str,
        sentence_start: int,
        terminal_index: int,
        terminal_end: int,
        next_index: int,
    ) -> bool:
        terminal_char = text[terminal_index]
        if next_index >= len(text):
            return True

        next_token = _next_token(text, next_index)
        if not next_token:
            return True

        if terminal_char in "!?":
            return True

        if self._is_decimal_point(text, terminal_index):
            return False

        candidate = _terminal_candidate(text[sentence_start : terminal_index + 1])
        if candidate and candidate.lower() in _COMMON_ABBREVIATIONS:
            return False

        stripped_next_token = next_token.strip(_TOKEN_STRIP_CHARS)
        if not stripped_next_token:
            return True

        if stripped_next_token[0].islower():
            return False

        normalized_next_token = stripped_next_token.lower()
        if candidate and _INITIAL_OR_ACRONYM_RE.fullmatch(candidate):
            return normalized_next_token in _COMMON_SENTENCE_STARTERS

        if terminal_end == terminal_index and text[terminal_index] == ".":
            return True

        return True

    def _is_decimal_point(self, text: str, terminal_index: int) -> bool:
        if terminal_index <= 0 or terminal_index >= len(text) - 1:
            return False
        return text[terminal_index - 1].isdigit() and text[terminal_index + 1].isdigit()


_DEFAULT_SPLITTER = EnglishSentenceSplitter()


def get_sentence_splitter() -> SentenceSplitter:
    return _DEFAULT_SPLITTER


def split_sentences(text: str) -> list[SentenceFragment]:
    return get_sentence_splitter().split(text)


def normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _next_token(text: str, start_index: int) -> str:
    end_index = start_index
    while end_index < len(text) and not text[end_index].isspace():
        end_index += 1
    return text[start_index:end_index]


def _terminal_candidate(text: str) -> str | None:
    match = _ABBREVIATION_RE.search(text)
    if match is None:
        return None
    return match.group(1) + "."
