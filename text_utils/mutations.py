from __future__ import annotations

import random
import re
import string
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache

import pysbd


class Script(str, Enum):
    LATIN = "latin"
    CYRILLIC = "cyrillic"
    ARABIC = "arabic"
    HEBREW = "hebrew"
    DEVANAGARI = "devanagari"
    BENGALI = "bengali"


SCRIPT_LETTER_POOLS = {
    Script.LATIN: (string.ascii_lowercase, True),
    Script.CYRILLIC: ("абвгдежзийклмнопрстуфхцчшщэюя", True),
    Script.ARABIC: ("ابتثجحخدذرزسشصضطظعغفقكلمنهوي", False),
    Script.HEBREW: ("אבגדהוזחטיכלמנסעפצקרשת", False),
    Script.DEVANAGARI: ("अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह", False),
    Script.BENGALI: ("অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ", False),
}
SCRIPT_DIGIT_POOLS = {Script.LATIN, Script.CYRILLIC, Script.ARABIC}
PYSBD_SUPPORTED = {
    "en", "hi", "mr", "bg", "es", "ru", "ar", "am", "hy", "fa",
    "ur", "pl", "zh", "nl", "da", "fr", "it", "el", "my", "ja", "de", "kk",
}

_WORD_RE = re.compile(r"\b[^\W\d_]{2,}\b", flags=re.UNICODE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|[\r\n]+")


@lru_cache(maxsize=32)
def _sentence_segmenter(language: str) -> pysbd.Segmenter:
    return pysbd.Segmenter(language=language, clean=False)


def _detect_script_from_text(text: str) -> Script | None:
    counts: dict[Script, int] = {script: 0 for script in Script}
    for ch in text:
        if not ch.isalpha():
            continue
        codepoint = ord(ch)
        if "a" <= ch.lower() <= "z":
            counts[Script.LATIN] += 1
        elif 0x0400 <= codepoint <= 0x052F:
            counts[Script.CYRILLIC] += 1
        elif 0x0600 <= codepoint <= 0x06FF or 0x0750 <= codepoint <= 0x077F or 0x08A0 <= codepoint <= 0x08FF:
            counts[Script.ARABIC] += 1
        elif 0x0590 <= codepoint <= 0x05FF:
            counts[Script.HEBREW] += 1
        elif 0x0900 <= codepoint <= 0x097F:
            counts[Script.DEVANAGARI] += 1
        elif 0x0980 <= codepoint <= 0x09FF:
            counts[Script.BENGALI] += 1

    detected = max(counts, key=counts.get, default=None)
    if detected is None or counts.get(detected, 0) <= 0:
        return None
    return detected


def _pysbd_language_for_text(text: str, lang: str | None = None) -> str:
    if lang:
        normalized = lang.split("_", 1)[0].lower()
        if normalized in PYSBD_SUPPORTED:
            return normalized

    script = _detect_script_from_text(text)
    if script == Script.CYRILLIC:
        return "ru"
    if script == Script.ARABIC:
        return "ar"
    if script == Script.HEBREW:
        return "ar"
    if script == Script.DEVANAGARI:
        return "hi"
    if script == Script.BENGALI:
        return "hi"
    return "en"

_OCR_MAP: dict[str, set[str]] = {
    "0": {"o", "O"},
    "1": {"l", "I", "i"},
    "3": {"e", "E"},
    "4": {"a", "A"},
    "5": {"s", "S"},
    "6": {"b", "G"},
    "7": {"t", "T"},
    "8": {"B"},
    "@": {"a", "A"},
    "$": {"s", "S"},
    "\"": {"'", "’", "‘", "“", "”"},
    "9": {"g", "q"},
}
_OCR_CHAR_MAP: dict[str, list[str]] = {}
for replacement, originals in _OCR_MAP.items():
    for ch in originals:
        _OCR_CHAR_MAP.setdefault(ch, []).append(replacement)

_UNICODE_ACCENT_VARIANTS: dict[str, list[str]] = {
    "a": ["á", "à", "â", "ä", "ã", "å", "ā", "ă", "ą"],
    "b": ["ḃ", "ƀ", "ɓ"],
    "c": ["ç", "ć", "č"],
    "d": ["ď", "đ", "ḋ", "ḍ"],
    "e": ["é", "è", "ê", "ë", "ē", "ė", "ę"],
    "f": ["ƒ"],
    "g": ["ğ", "ĝ", "ġ", "ģ"],
    "h": ["ħ", "ḥ"],
    "i": ["í", "ì", "î", "ï", "ī", "į"],
    "j": ["ĵ"],
    "k": ["ķ", "ḱ"],
    "l": ["ĺ", "ļ", "ľ", "ł"],
    "m": ["ṃ"],
    "n": ["ñ", "ń", "ņ", "ň"],
    "o": ["ó", "ò", "ô", "ö", "õ", "ō", "ő"],
    "p": ["ṕ"],
    "r": ["ŕ", "ř", "ṛ"],
    "s": ["ś", "š", "ş", "ș"],
    "t": ["ť", "ţ", "ṭ", "ŧ"],
    "u": ["ú", "ù", "û", "ü", "ū", "ű", "ŭ", "ũ"],
    "v": ["ṽ"],
    "w": ["ŵ", "ẁ", "ẃ"],
    "y": ["ý", "ÿ", "ŷ"],
    "z": ["ź", "ž", "ż", "ẓ"],
}
_KEYBOARD_MAP: dict[str, list[str]] = {
    "q": ["w", "a"],
    "w": ["q", "e", "s"],
    "e": ["w", "r", "d"],
    "r": ["e", "t", "f"],
    "t": ["r", "y", "g"],
    "y": ["t", "u", "h"],
    "u": ["y", "i", "j"],
    "i": ["u", "o", "k"],
    "o": ["i", "p", "l"],
    "p": ["o", "l"],
    "a": ["q", "s", "z"],
    "s": ["a", "d", "w", "x"],
    "d": ["s", "f", "e", "c"],
    "f": ["d", "g", "r", "v"],
    "g": ["f", "h", "t", "b"],
    "h": ["g", "j", "y", "n"],
    "j": ["h", "k", "u", "m"],
    "k": ["j", "l", "i"],
    "l": ["k", "o", "p"],
    "z": ["a", "x"],
    "x": ["z", "c", "s"],
    "c": ["x", "v", "d"],
    "v": ["c", "b", "f"],
    "b": ["v", "n", "g"],
    "n": ["b", "m", "h"],
    "m": ["n", "j"],
}


@lru_cache(maxsize=16_384)
def _is_punctuation_token(token: str) -> bool:
    text = token[1:] if token.startswith("▁") else token
    return len(text) > 0 and all(unicodedata.category(c).startswith("P") for c in text)


def augment_boundary(tokens: list[str], strip_punct: bool) -> list[str]:
    if strip_punct and tokens:
        return [t for t in tokens if not _is_punctuation_token(t)]
    return tokens


def split_sentences(text: str, *, lang: str | None = None) -> list[str]:
    """Split text with pysbd first, then fall back to the regex backup."""
    language = _pysbd_language_for_text(text, lang=lang)
    segments = [segment.strip() for segment in _sentence_segmenter(language).segment(text) if segment and segment.strip()]
    if len(segments) >= 2:
        return segments
    return [segment.strip() for segment in _SENTENCE_SPLIT_RE.split(text) if segment and segment.strip()]


@dataclass(frozen=True)
class MutationConfig:
    keep_original: bool = True
    boundary_strip_prob: float = 0.08
    sentence_mutation_prob: float = 0.12
    sentence_casing_prob: float = 0.18
    word_casing_prob: float = 0.20
    spacing_noise_prob: float = 0.12
    char_noise_prob: float = 0.12
    accent_strip_prob: float = 0.10
    format_noise_prob: float = 0.10
    script_letter_prob: float = 0.0
    script_digit_prob: float = 0.0
    sentence_uppercase_prob: float = 0.35
    sentence_lowercase_prob: float = 0.35
    word_uppercase_prob: float = 0.35
    word_lowercase_prob: float = 0.35
    word_titlecase_prob: float = 0.30
    merge_word_prob: float = 0.50
    split_word_prob: float = 0.50
    ocr_char_prob: float = 0.34
    keyboard_char_prob: float = 0.33
    unicode_accent_char_prob: float = 0.33
    max_sentence_edits: int = 1
    max_word_edits: int = 1
    safe_accent_strip_langs: set[str] = field(default_factory=lambda: {"en", "es", "fr", "it", "nl", "pt"})
    script_letter_pools: dict[Script, tuple[str, bool]] = field(default_factory=lambda: SCRIPT_LETTER_POOLS)
    script_digit_pools: set[Script] = field(default_factory=lambda: set(SCRIPT_DIGIT_POOLS))


class TextMutator:
    """Apply light sentence- and character-level tweet mutations."""

    def __init__(self, config: MutationConfig | None = None) -> None:
        self.config = config or MutationConfig()

    def augment(self, text: str, *, rng: random.Random, lang: str | None = None) -> list[str]:
        variants: list[str] = []
        if self.config.keep_original:
            variants.append(text)

        if rng.random() < self.config.boundary_strip_prob:
            stripped = self._strip_terminal_punctuation(text)
            if stripped and stripped != text:
                variants.append(stripped)

        if rng.random() < self.config.sentence_mutation_prob:
            sentence_variant = self._mutate_sentence_structure(text, rng=rng, lang=lang)
            if sentence_variant and sentence_variant != text:
                variants.append(sentence_variant)

        if rng.random() < self.config.sentence_casing_prob:
            sentence_variant = self._apply_sentence_casing(
                text,
                rng=rng,
                uppercase_prob=self.config.sentence_uppercase_prob,
                lowercase_prob=self.config.sentence_lowercase_prob,
            )
            if sentence_variant and sentence_variant != text:
                variants.append(sentence_variant)

        if rng.random() < self.config.word_casing_prob:
            word_variant = self._apply_random_word_casing(
                text,
                rng=rng,
                lang=lang,
                uppercase_prob=self.config.word_uppercase_prob,
                lowercase_prob=self.config.word_lowercase_prob,
                titlecase_prob=self.config.word_titlecase_prob,
            )
            if word_variant and word_variant != text:
                variants.append(word_variant)

        if rng.random() < self.config.spacing_noise_prob:
            spacing_variant = self._apply_random_spacing_noise(
                text,
                rng=rng,
                lang=lang,
                merge_prob=self.config.merge_word_prob,
                split_prob=self.config.split_word_prob,
            )
            if spacing_variant and spacing_variant != text:
                variants.append(spacing_variant)

        if rng.random() < self.config.char_noise_prob:
            char_variant = self._apply_random_char_noise(
                text,
                rng=rng,
                lang=lang,
                prob=1.0,
            )
            if char_variant and char_variant != text:
                variants.append(char_variant)

        if rng.random() < self.config.accent_strip_prob:
            accent_variant = self._apply_random_accent_stripping(
                text,
                rng=rng,
                lang=lang,
                prob=1.0,
            )
            if accent_variant and accent_variant != text:
                variants.append(accent_variant)

        if rng.random() < self.config.format_noise_prob:
            format_variant = self._add_formatting_noise(text, rng=rng, artifact_prob=1.0)
            if format_variant and format_variant != text:
                variants.append(format_variant)

        if lang is not None and rng.random() < self.config.script_letter_prob:
            letter_variant = self._inject_random_letter_into_sentence(
                text,
                rng=rng,
                lang=lang,
                prob=1.0,
            )
            if letter_variant and letter_variant != text:
                variants.append(letter_variant)

        if lang is not None and rng.random() < self.config.script_digit_prob:
            digit_variant = self._inject_random_digit_into_sentence(
                text,
                rng=rng,
                lang=lang,
                prob=1.0,
            )
            if digit_variant and digit_variant != text:
                variants.append(digit_variant)

        return [variant for variant in dict.fromkeys(variants) if variant.strip()]

    def _mutate_sentence_structure(self, text: str, *, rng: random.Random, lang: str | None = None) -> str:
        sentences = split_sentences(text, lang=lang)
        if len(sentences) < 2:
            return text

        mutated = list(sentences)
        for _ in range(max(1, self.config.max_sentence_edits)):
            action = rng.choice(["swap", "drop", "duplicate"])
            if action == "swap" and len(mutated) >= 2:
                i, j = sorted(rng.sample(range(len(mutated)), 2))
                mutated[i], mutated[j] = mutated[j], mutated[i]
            elif action == "drop" and len(mutated) > 1:
                del mutated[rng.randrange(len(mutated))]
            elif action == "duplicate" and mutated:
                idx = rng.randrange(len(mutated))
                mutated.insert(idx, mutated[idx])
        return " ".join(mutated)

    def _apply_sentence_casing(
        self,
        sentence: str,
        *,
        rng: random.Random,
        uppercase_prob: float,
        lowercase_prob: float,
    ) -> str:
        total_prob = max(0.0, uppercase_prob) + max(0.0, lowercase_prob)
        if total_prob <= 0 or rng.random() >= total_prob:
            return sentence
        if rng.random() * total_prob < uppercase_prob:
            return sentence.upper()
        return sentence.lower()

    def _apply_random_word_casing(
        self,
        sentence: str,
        *,
        rng: random.Random,
        lang: str | None,
        uppercase_prob: float,
        lowercase_prob: float,
        titlecase_prob: float,
    ) -> str:
        total_prob = max(0.0, uppercase_prob) + max(0.0, lowercase_prob) + max(0.0, titlecase_prob)
        if total_prob <= 0 or rng.random() >= total_prob:
            return sentence
        matches = list(_WORD_RE.finditer(sentence))
        if not matches:
            return sentence
        match = rng.choice(matches)
        word = match.group(0)
        if len(word) < 3:
            return sentence
        if lang is not None and self._lang_to_script(lang) not in {Script.LATIN, None}:
            return sentence
        roll = rng.random() * total_prob
        if roll < uppercase_prob:
            replacement = word.upper()
        elif roll < uppercase_prob + lowercase_prob:
            replacement = word.lower()
        else:
            replacement = word[:1].upper() + word[1:].lower()
        if replacement == word:
            return sentence
        return f"{sentence[:match.start()]}{replacement}{sentence[match.end():]}"

    def _apply_random_spacing_noise(
        self,
        sentence: str,
        *,
        rng: random.Random,
        lang: str | None,
        merge_prob: float,
        split_prob: float,
    ) -> str:
        total_prob = max(0.0, merge_prob) + max(0.0, split_prob)
        if total_prob <= 0 or rng.random() >= total_prob:
            return sentence
        if lang is not None and self._lang_to_script(lang) not in {Script.LATIN, None}:
            return sentence

        if rng.random() < (merge_prob / total_prob if total_prob else 0.0):
            matches = list(re.finditer(r"\b[^\W\d_]{2,}\s+[^\W\d_]{2,}\b", sentence, flags=re.UNICODE))
            if not matches:
                return sentence
            match = rng.choice(matches)
            return f"{sentence[:match.start()]}{match.group(0).replace(' ', '', 1)}{sentence[match.end():]}"

        matches = list(re.finditer(r"\b[^\W\d_]{4,}\b", sentence, flags=re.UNICODE))
        if not matches:
            return sentence
        match = rng.choice(matches)
        word = match.group(0)
        split_at = rng.randint(2, len(word) - 2)
        replacement = f"{word[:split_at]} {word[split_at:]}"
        return f"{sentence[:match.start()]}{replacement}{sentence[match.end():]}"

    def _apply_random_char_noise(
        self,
        sentence: str,
        *,
        rng: random.Random,
        lang: str | None,
        prob: float,
    ) -> str:
        if prob <= 0 or rng.random() >= prob:
            return sentence
        if lang is not None and self._lang_to_script(lang) not in {Script.LATIN, None}:
            return sentence

        mutated = sentence
        for _ in range(max(1, self.config.max_word_edits)):
            matches = list(_WORD_RE.finditer(mutated))
            if not matches:
                break
            match = rng.choice(matches)
            word = match.group(0)
            if len(word) < 2:
                continue

            chars = list(word)
            mode = rng.choices(
                ["ocr", "accent", "keyboard"],
                weights=[
                    self.config.ocr_char_prob,
                    self.config.unicode_accent_char_prob,
                    self.config.keyboard_char_prob,
                ],
                k=1,
            )[0]

            if mode == "ocr":
                positions = [idx for idx, ch in enumerate(chars) if ch in _OCR_CHAR_MAP or ch.lower() in _OCR_CHAR_MAP]
                if not positions:
                    continue
                idx = rng.choice(positions)
                ch = chars[idx]
                replacements = _OCR_CHAR_MAP.get(ch) or _OCR_CHAR_MAP.get(ch.lower()) or []
                if not replacements:
                    continue
                replacement = rng.choice(replacements)
                if ch.isupper():
                    replacement = replacement.upper()
                chars[idx] = replacement
            elif mode == "accent":
                positions = [idx for idx, ch in enumerate(chars) if ch.lower() in _UNICODE_ACCENT_VARIANTS]
                if not positions:
                    continue
                idx = rng.choice(positions)
                ch = chars[idx]
                variants = _UNICODE_ACCENT_VARIANTS.get(ch.lower(), [])
                if not variants:
                    continue
                replacement = rng.choice(variants)
                if ch.isupper():
                    replacement = replacement.upper()
                chars[idx] = replacement
            else:
                positions = [idx for idx, ch in enumerate(chars) if ch.lower() in _KEYBOARD_MAP]
                if not positions:
                    continue
                idx = rng.choice(positions)
                ch = chars[idx]
                variants = _KEYBOARD_MAP.get(ch.lower(), [])
                if not variants:
                    continue
                replacement = rng.choice(variants)
                if ch.isupper():
                    replacement = replacement.upper()
                chars[idx] = replacement

            mutated = f"{mutated[:match.start()]}{''.join(chars)}{mutated[match.end():]}"

        return mutated

    def _apply_random_accent_stripping(
        self,
        sentence: str,
        *,
        rng: random.Random,
        lang: str | None,
        prob: float,
    ) -> str:
        if prob <= 0 or rng.random() >= prob:
            return sentence
        if lang is not None and lang not in self.config.safe_accent_strip_langs:
            return sentence
        stripped = self._strip_latin_accents(sentence)
        return stripped if stripped != sentence else sentence

    def _add_formatting_noise(self, sentence: str, *, rng: random.Random, artifact_prob: float) -> str:
        if artifact_prob <= 0 or rng.random() >= artifact_prob:
            return sentence

        pattern = rng.choice(["wrap", "trail"])
        if pattern == "wrap":
            prefix, suffix = rng.choice([
                ("(", ")"),
                ("[", "]"),
                ("\"", "\""),
                ("“", "”"),
                ("«", "»"),
                ("‘", "’"),
            ])
            return f"{prefix}{sentence}{suffix}"
        if pattern == "trail":
            return f"{sentence}{rng.choice([':', ';', '...', '?!', ' |', ' | |', ',', '!!'])}"
        return sentence

    def _strip_latin_accents(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text)
        return "".join(ch for ch in normalized if not unicodedata.combining(ch))

    def _strip_terminal_punctuation(self, text: str) -> str:
        end = len(text)
        while end > 0 and unicodedata.category(text[end - 1]).startswith("P"):
            end -= 1
        return text[:end].rstrip()

    def _inject_random_letter_into_sentence(self, sentence: str, *, rng: random.Random, lang: str, prob: float) -> str:
        if prob <= 0 or rng.random() >= prob:
            return sentence
        script = self._lang_to_script(lang)
        if script not in self.config.script_letter_pools:
            return sentence

        parts = sentence.split()
        if len(parts) < 2:
            return sentence

        pool, make_upper = self.config.script_letter_pools[script]
        letter = rng.choice(pool)
        if make_upper and rng.random() < 0.5:
            letter = letter.upper()

        insert_at = rng.randint(1, len(parts) - 1)
        parts = parts[:insert_at] + [letter] + parts[insert_at:]
        return " ".join(parts)

    def _inject_random_digit_into_sentence(self, sentence: str, *, rng: random.Random, lang: str, prob: float) -> str:
        if prob <= 0 or rng.random() >= prob:
            return sentence
        script = self._lang_to_script(lang)
        if script not in self.config.script_digit_pools:
            return sentence

        parts = sentence.split()
        if len(parts) < 2:
            return sentence

        insert_at = rng.randint(1, len(parts) - 1)
        parts = parts[:insert_at] + [rng.choice(string.digits)] + parts[insert_at:]
        return " ".join(parts)

    def _lang_to_script(self, lang: str) -> Script | None:
        lower = lang.lower()
        if lower.startswith(("en", "es", "fr", "it", "nl", "pt")):
            return Script.LATIN
        if lower.startswith(("ru", "uk", "bg", "sr", "mk")):
            return Script.CYRILLIC
        if lower.startswith(("ar", "fa", "ur")):
            return Script.ARABIC
        if lower.startswith(("he",)):
            return Script.HEBREW
        if lower.startswith(("hi", "mr", "ne", "sa")):
            return Script.DEVANAGARI
        if lower.startswith(("bn",)):
            return Script.BENGALI
        return None
