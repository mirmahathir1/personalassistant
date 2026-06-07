"""Local Bengali-to-English text translation."""

from __future__ import annotations

import os

import argostranslate.package
import argostranslate.settings
import argostranslate.translate

# Force Argos Translate's lightweight MiniSBD sentence splitter. The default
# can select a stanza-based splitter that is broken with current stanza releases
# for this model path (KeyError: 'packages').
os.environ.setdefault("ARGOS_CHUNK_TYPE", "MINISBD")
argostranslate.settings.chunk_type = argostranslate.settings.ChunkType.MINISBD

FROM_CODE = "bn"
TO_CODE = "en"
_ready = False


def _has_translation() -> bool:
    languages = argostranslate.translate.get_installed_languages()
    from_lang = next((lang for lang in languages if lang.code == FROM_CODE), None)
    to_lang = next((lang for lang in languages if lang.code == TO_CODE), None)
    if from_lang is None or to_lang is None:
        return False
    return from_lang.get_translation(to_lang) is not None


def ensure_package() -> None:
    """Make sure the bn->en translation model is installed, downloading once."""
    global _ready
    if _ready:
        return
    if _has_translation():
        _ready = True
        return

    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()
    match = next(
        (
            package
            for package in available
            if package.from_code == FROM_CODE and package.to_code == TO_CODE
        ),
        None,
    )
    if match is None:
        raise RuntimeError(
            f"No Argos Translate package available for {FROM_CODE}->{TO_CODE}."
        )
    argostranslate.package.install_from_path(match.download())
    _ready = True


def translate_bangla_to_english(text: str) -> str:
    if not text.strip():
        return ""
    ensure_package()
    return argostranslate.translate.translate(text, FROM_CODE, TO_CODE).strip()


def warm_up_bangla_to_english() -> None:
    """Install the bn->en model up front so input turns do not pay the download."""
    ensure_package()
