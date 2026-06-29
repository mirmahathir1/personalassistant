"""Emoji decoration for assistant replies.

After a reply is generated, each sentence has a 50% chance of getting an emoji
appended right before its terminal punctuation (e.g. the full stop). The emoji
is chosen by embedding the sentence and picking the emoji whose short text
description has the closest (cosine) embedding to the sentence.

Embeddings reuse the project's existing Ollama `nomic-embed-text` mechanism: the
caller passes an `embed_fn(list[str]) -> list[np.ndarray] | None` that returns
L2-normalised vectors (so cosine similarity is a plain dot product), matching
`retrieval.ConversationIndex._embed`.
"""

from __future__ import annotations

import random
import re

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is a hard dep elsewhere
    np = None


# A WhatsApp-style emoji set. Each entry is (emoji, short text description used
# to embed it). Descriptions are kept short and plain so the embedding reflects
# the emoji's core meaning rather than incidental words.
EMOJIS: list[tuple[str, str]] = [
    ("😀", "grinning happy smile"),
    ("😃", "smiling with open mouth, happy"),
    ("😄", "laughing happy grin"),
    ("😁", "beaming grin, pleased"),
    ("😆", "laughing hard, hilarious"),
    ("😅", "nervous laugh, relieved sweat"),
    ("🤣", "rolling on the floor laughing"),
    ("😂", "tears of joy, crying laughing"),
    ("🙂", "slight gentle smile"),
    ("🙃", "upside down, silly sarcasm"),
    ("😉", "wink, playful joke"),
    ("😊", "warm friendly smile, blushing"),
    ("😇", "innocent angel halo"),
    ("🥰", "loving smile with hearts, adore"),
    ("😍", "heart eyes, in love, infatuated"),
    ("🤩", "star struck, amazed excited"),
    ("😘", "blowing a kiss, affection"),
    ("😋", "yummy tasty, savoring food"),
    ("😛", "tongue out, teasing playful"),
    ("🤪", "zany goofy wild face"),
    ("🤔", "thinking, pondering, considering"),
    ("🤨", "raised eyebrow, skeptical doubt"),
    ("😐", "neutral blank expression"),
    ("😑", "expressionless, unamused"),
    ("🙄", "rolling eyes, annoyed disdain"),
    ("😬", "grimace, awkward tension"),
    ("😏", "smirk, smug confident"),
    ("😴", "sleeping, tired, asleep"),
    ("😪", "sleepy drowsy tired"),
    ("😌", "relieved content calm"),
    ("😔", "sad pensive, dejected"),
    ("😟", "worried concerned"),
    ("😕", "confused, slightly unhappy"),
    ("🙁", "frowning, slightly sad"),
    ("😣", "persevering, struggling effort"),
    ("😖", "confounded, frustrated distress"),
    ("😫", "tired weary, exhausted"),
    ("😩", "weary, fed up, distressed"),
    ("🥺", "pleading puppy eyes, begging"),
    ("😢", "crying, single tear, sad"),
    ("😭", "loudly sobbing, bawling"),
    ("😤", "huffing, triumphant frustration"),
    ("😠", "angry, mad"),
    ("😡", "furious, enraged, red angry"),
    ("🤬", "cursing, swearing, very angry"),
    ("😱", "screaming in fear, shocked"),
    ("😨", "fearful, anxious scared"),
    ("😰", "anxious sweat, nervous fear"),
    ("😥", "sad but relieved, disappointed"),
    ("😓", "downcast sweat, hard work worry"),
    ("🤗", "hugging, warm embrace"),
    ("🤭", "giggling, oops hand over mouth"),
    ("🤫", "shushing, quiet secret"),
    ("😶", "no mouth, speechless silent"),
    ("😲", "astonished, surprised gasp"),
    ("😳", "flushed, embarrassed shocked"),
    ("🥳", "partying, celebration, hooray"),
    ("😎", "cool, sunglasses, confident"),
    ("🤓", "nerd, studious geek"),
    ("😷", "sick, wearing a mask, ill"),
    ("🤒", "sick with a thermometer, fever"),
    ("🤕", "injured, bandaged head, hurt"),
    ("🤮", "vomiting, disgusted sick"),
    ("🥵", "hot, overheated, sweating"),
    ("🥶", "freezing cold, frozen"),
    ("👍", "thumbs up, approval, good, yes"),
    ("👎", "thumbs down, disapproval, no, bad"),
    ("👏", "clapping, applause, well done"),
    ("🙌", "raising hands, praise celebration"),
    ("🙏", "thank you, please, prayer hands"),
    ("💪", "strong muscle, power effort"),
    ("👌", "okay, perfect, fine"),
    ("✌️", "peace, victory"),
    ("🤝", "handshake, agreement, deal"),
    ("👋", "waving hello goodbye"),
    ("❤️", "red heart, love"),
    ("💔", "broken heart, heartbreak"),
    ("💖", "sparkling heart, love affection"),
    ("🔥", "fire, hot, amazing, lit"),
    ("⭐", "star, excellent favorite"),
    ("✨", "sparkles, magic, special, new"),
    ("🎉", "party popper, congratulations"),
    ("🎊", "confetti celebration"),
    ("💯", "hundred points, perfect, total"),
    ("🤷", "shrug, dunno, whatever"),
    ("🎁", "gift present"),
    ("🍀", "luck, lucky clover"),
    ("☀️", "sun, sunny, bright weather"),
    ("🌧️", "rain, rainy weather"),
    ("⚡", "lightning, energy, fast"),
    ("🌈", "rainbow, hope, colorful"),
    ("🍕", "pizza, food, hungry"),
    ("☕", "coffee, drink, morning"),
    ("🍺", "beer, drink, cheers"),
    ("🎂", "birthday cake"),
    ("🎵", "music note, song, melody"),
    ("⚽", "soccer football sport"),
    ("🏆", "trophy, winner, champion"),
    ("💰", "money, cash, rich"),
    ("💡", "idea, lightbulb, insight"),
    ("📚", "books, study, reading, learning"),
    ("✅", "check mark, done, correct, yes"),
    ("❌", "cross mark, wrong, no, error"),
    ("⏰", "alarm clock, time, deadline"),
    ("🚀", "rocket, launch, fast progress"),
    ("👀", "eyes, looking, watching, curious"),
    ("💤", "sleep, zzz, tired bored"),
    ("🤦", "facepalm, disbelief, frustration"),
    ("🥱", "yawning, bored, tired"),
]


# Terminal sentence punctuation. The emoji is inserted just before this marker.
_SENTENCE_RE = re.compile(r"(.*?[.!?]+)(\s+|$)", re.DOTALL)

# Probability that any given sentence gets an emoji.
_EMOJI_CHANCE = 0.5

# Lazily-built matrix of emoji-description embeddings (rows aligned with EMOJIS).
_emoji_matrix = None  # type: ignore[assignment]


def _ensure_emoji_matrix(embed_fn) -> bool:
    """Embed every emoji description once, caching the result. Returns success."""
    global _emoji_matrix
    if _emoji_matrix is not None:
        return True
    if np is None:
        return False
    vecs = embed_fn([desc for _e, desc in EMOJIS])
    if not vecs or len(vecs) != len(EMOJIS):
        return False
    _emoji_matrix = np.vstack(vecs)
    return True


def _best_emoji(sentence_vec) -> str:
    """Return the emoji whose description is closest (cosine) to the sentence."""
    sims = _emoji_matrix @ sentence_vec  # both sides are L2-normalised -> cosine
    return EMOJIS[int(np.argmax(sims))][0]


def decorate(text: str, embed_fn, rng: random.Random | None = None) -> str:
    """Append a fitting emoji before the terminal punctuation of ~50% of sentences.

    `embed_fn(list[str]) -> list[np.ndarray] | None` must return L2-normalised
    vectors (the project's `nomic-embed-text` embeddings). On any failure
    (no numpy, embeddings unavailable, no sentences) the original text is
    returned unchanged.
    """
    if not text or np is None:
        return text
    if not _ensure_emoji_matrix(embed_fn):
        return text

    roll = (rng or random).random

    # Split into (sentence-including-its-terminator, trailing-whitespace) pairs.
    matches = list(_SENTENCE_RE.finditer(text))
    if not matches:
        return text

    # Decide which sentences get an emoji, then batch-embed only those.
    chosen = []  # (match_index, sentence_body, terminator_len)
    for mi, m in enumerate(matches):
        seg = m.group(1)  # body + terminal punctuation, no trailing space
        body = seg.rstrip(".!?")
        term = seg[len(body):]
        if not body.strip() or not term:
            continue
        if roll() < _EMOJI_CHANCE:
            chosen.append((mi, body, len(term)))

    if not chosen:
        return text

    sent_vecs = embed_fn([body for _mi, body, _t in chosen])
    if not sent_vecs or len(sent_vecs) != len(chosen):
        return text

    # Map each chosen match index to the emoji to insert before its terminator.
    inserts = {}
    for (mi, _body, term_len), vec in zip(chosen, sent_vecs):
        inserts[mi] = (_best_emoji(vec), term_len)

    # Rebuild the string, inserting emojis before the terminal punctuation.
    out = []
    last_end = 0
    for mi, m in enumerate(matches):
        out.append(text[last_end:m.start()])
        seg = m.group(1)
        trailing = m.group(2)
        if mi in inserts:
            emoji, term_len = inserts[mi]
            out.append(seg[:-term_len])
            out.append(" " + emoji)
            out.append(seg[-term_len:])
        else:
            out.append(seg)
        out.append(trailing)
        last_end = m.end()
    out.append(text[last_end:])
    return "".join(out)
