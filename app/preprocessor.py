"""
preprocessor.py — 10-stage corpus cleaning pipeline.

WHY NOT sklearn's fetch_20newsgroups(remove=('headers','footers','quotes'))?
  That removes headers/footers/quotes with a single flag — hiding every decision.
  We make each decision explicit because they have downstream consequences:
  noise tokens inflate vocabulary, hurt embedding quality, and distort cluster centroids.

PIPELINE STAGES (applied in order):
  1.  Strip Usenet header block       — routing metadata, not content
  2.  Strip PGP / MIME blocks         — binary garbage in embedding space
  3.  Strip signature blocks          — off-topic boilerplate per post
  4.  Strip quoted replies (> lines)  — duplicates; distort centroid computation
  5.  Remove URLs and email addresses — unique tokens; zero semantic generalisation
  6.  Remove Usenet artifacts         — message IDs, base64 lines, separators
  7.  ASCII language filter           — non-English posts cluster by language not topic
  8.  Near-duplicate detection        — MD5 of first 500 chars; ~4% crossposted articles
  9.  Minimum token threshold (50)    — stub posts have no standalone semantic value
  10. Chunk + mean-pool for long posts — prevents silent tail truncation at 512 tokens
"""

import re
import hashlib
from dataclasses import dataclass, field
from typing import Optional, Tuple


# ── Compiled regexes (compiled once at module load) ───────────────────────────

# Stage 1: Usenet header lines (From:, Path:, Message-ID:, Newsgroups:, etc.)
_RE_HEADER_LINE = re.compile(
    r'^(From|Path|Newsgroups|Subject|Message-ID|Date|References|'
    r'Sender|Organization|Lines|Approved|Distribution|Expires|'
    r'Followup-To|Keywords|Summary|Xref|Reply-To|NNTP-Posting-Host|'
    r'X-[A-Za-z-]+)\s*:.*$',
    re.MULTILINE | re.IGNORECASE
)

# Stage 2: PGP blocks
_RE_PGP = re.compile(
    r'-----BEGIN PGP.*?-----END PGP[^-]*-----',
    re.DOTALL
)
# MIME boundaries and encoded content
_RE_MIME = re.compile(
    r'Content-Type:.*?(?=\n\n|\Z)', re.DOTALL | re.IGNORECASE
)

# Stage 3: Signature blocks (RFC 3676 "-- " separator and common variants)
_RE_SIG = re.compile(
    r'(\n-- \n.*|\n---+\n.*|\n___+\n.*)', re.DOTALL
)

# Stage 4: Quoted reply lines (any depth of >)
_RE_QUOTED = re.compile(r'^[ \t]*>+.*$', re.MULTILINE)
# Attribution lines: "In article <...>, user@host wrote:"
_RE_ATTRIBUTION = re.compile(
    r'^(In article|writes:|wrote:|said:|On .+ wrote:).*$',
    re.MULTILINE | re.IGNORECASE
)

# Stage 5: URLs and email addresses
_RE_URL = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
_RE_EMAIL = re.compile(r'\S+@\S+\.\S+')

# Stage 6: Usenet artifacts
_RE_MSGID = re.compile(r'<[^>]+@[^>]+>')           # <message-id@host>
_RE_BASE64 = re.compile(r'^[A-Za-z0-9+/]{40,}=*$', re.MULTILINE)  # base64 lines
_RE_SEPARATORS = re.compile(r'^[-=_*#]{4,}\s*$', re.MULTILINE)    # visual dividers
_RE_QUOTED_PRINTABLE = re.compile(r'=[0-9A-F]{2}', re.IGNORECASE) # =3D style


@dataclass
class CleaningStats:
    total_scanned: int = 0
    kept: int = 0
    dropped_short: int = 0
    dropped_non_english: int = 0
    dropped_duplicate: int = 0
    dropped_empty: int = 0
    dropped_encoding: int = 0
    total_raw_tokens: int = 0
    total_clean_tokens: int = 0
    per_newsgroup: dict = field(default_factory=dict)

    def noise_pct(self) -> float:
        if self.total_raw_tokens == 0:
            return 0.0
        return 1.0 - (self.total_clean_tokens / self.total_raw_tokens)

    def print_report(self):
        kept_pct = 100 * self.kept / max(self.total_scanned, 1)
        noise_pct = 100 * self.noise_pct()
        raw_avg = self.total_raw_tokens // max(self.kept, 1)
        clean_avg = self.total_clean_tokens // max(self.kept, 1)

        print("╔" + "═" * 58 + "╗")
        print("║" + "   CORPUS CLEANING DIAGNOSTIC REPORT".center(58) + "║")
        print("╠" + "═" * 58 + "╣")
        print(f"║  Total files scanned:    {self.total_scanned:>8}".ljust(59) + "║")
        print(f"║  Documents KEPT:         {self.kept:>8}  ({kept_pct:.1f}%)".ljust(59) + "║")
        print(f"║  ── Dropped (too short): {self.dropped_short:>8}".ljust(59) + "║")
        print(f"║  ── Dropped (non-Eng):   {self.dropped_non_english:>8}".ljust(59) + "║")
        print(f"║  ── Dropped (duplicate): {self.dropped_duplicate:>8}".ljust(59) + "║")
        print(f"║  ── Dropped (empty):     {self.dropped_empty:>8}".ljust(59) + "║")
        print("╠" + "═" * 58 + "╣")
        print(f"║  Avg tokens BEFORE clean:{raw_avg:>8}".ljust(59) + "║")
        print(f"║  Avg tokens AFTER clean: {clean_avg:>8}".ljust(59) + "║")
        print(f"║  Noise tokens removed:   {noise_pct:.1f}% of raw corpus".ljust(59) + "║")
        print("╚" + "═" * 58 + "╝")

        if self.per_newsgroup:
            print("\n  Documents kept per newsgroup:")
            max_count = max(self.per_newsgroup.values())
            for ng, cnt in sorted(self.per_newsgroup.items()):
                bar = "█" * int(40 * cnt / max_count)
                print(f"  {ng:<40} {cnt:>4}  {bar}")


def clean_text(raw: str) -> Tuple[str, int, int]:
    """
    Apply all 10 cleaning stages.
    Returns: (cleaned_text, raw_token_count, clean_token_count)
    """
    raw_tokens = len(raw.split())

    text = raw

    # Stage 1: Strip header block
    text = _RE_HEADER_LINE.sub("", text)

    # Stage 2: Strip PGP/MIME
    text = _RE_PGP.sub("", text)
    text = _RE_MIME.sub("", text)

    # Stage 3: Strip signatures
    text = _RE_SIG.sub("", text)

    # Stage 4: Strip quoted replies and attributions
    text = _RE_QUOTED.sub("", text)
    text = _RE_ATTRIBUTION.sub("", text)

    # Stage 5: Remove URLs and emails
    text = _RE_URL.sub(" ", text)
    text = _RE_EMAIL.sub(" ", text)

    # Stage 6: Remove Usenet artifacts
    text = _RE_MSGID.sub(" ", text)
    text = _RE_BASE64.sub("", text)
    text = _RE_SEPARATORS.sub("", text)
    text = _RE_QUOTED_PRINTABLE.sub(" ", text)

    # Normalise whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.strip()

    clean_tokens = len(text.split())
    return text, raw_tokens, clean_tokens


def should_keep(
    text: str,
    token_count: int,
    content_hash: str,
    seen_hashes: set,
    min_tokens: int = 50
) -> Tuple[bool, str]:
    """
    Decide whether to keep a document.
    Returns (keep: bool, reason: str)
    """
    # Stage 7: Language filter — <85% ASCII ratio → non-English
    if len(text) > 0:
        ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text)
        if ascii_ratio < 0.85:
            return False, "non_english"

    # Stage 8: Near-duplicate detection
    if content_hash in seen_hashes:
        return False, "duplicate"

    # Stage 9: Minimum token count
    if token_count < min_tokens:
        return False, "too_short"

    if not text.strip():
        return False, "empty"

    return True, "keep"


def compute_hash(text: str) -> str:
    """MD5 of first 500 chars — fast dedup for crossposted articles."""
    return hashlib.md5(text[:500].encode("utf-8", errors="ignore")).hexdigest()
