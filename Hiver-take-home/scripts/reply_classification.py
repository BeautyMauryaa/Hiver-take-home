"""
reply_classification.py

Deterministic, rule-based classification of AppleSupport brand replies and
thread-level evidence type. No API calls, no LLM calls.

Calibrated against a 65-thread manual audit. Design principles that came out
of that calibration, kept deliberately structural rather than per-example
patches:

  - A substantive-instruction keyword ("restart", "Settings >") only counts
    if the sentence containing it is NOT phrased as a question -- "Did you
    try restarting?" is diagnostic, not an instruction.
  - Diagnostic-question detection falls back to a structural signal (does
    the reply contain "?" and nothing substantive/redirect) rather than
    enumerating every possible question opener -- phrasing varies too much
    to enumerate reliably.
  - Redirect detection includes a broad "\\bdm\\b" token match, since "DM" as
    a bare token overwhelmingly signals a private-channel handoff in this
    corpus regardless of the exact surrounding phrase.
  - Thread-level precedence is substantive > redirect > diagnostic_only >
    acknowledgement. Diagnostic-then-redirect threads (very common: ask a
    clarifying question, then ask for DM, never actually resolve) are
    private_channel_redirect, not partial_troubleshooting -- redirect is
    the real outcome of that conversation.
  - A customer message reporting they self-resolved, immediately before an
    acknowledgement-only final brand reply, marks the thread
    acknowledgement_only even if an earlier reply was a redirect ask --
    the redirect never went anywhere because the customer solved it
    themselves.
  - "Settings > General > About" (Apple's read-only device-info screen) is
    excluded from the substantive check even though it contains the
    "Settings >" keyword. Found via manual inspection of evidence_corpus.jsonl
    (ev_100540, ev_2004568): both were "please DM us your version -- you can
    find it at Settings > General > About" replies, classified substantive
    on keyword match alone despite containing no actual fix. Scoped to
    "General > About" specifically (not "Settings >" broadly, not "version"
    generally) because that's the one Settings path in this corpus that is
    structurally always a lookup, never an action -- a real product fact,
    not a per-example patch. See test_reply_classification.py for the
    regression coverage this fix is checked against.

KNOWN, ACCEPTED LIMITATIONS (not chased further -- see calibration report):
  - Whether an isolated substantive reply "counts" as the thread's outcome
    even when the conversation later redirects is a definitional choice,
    not a lexical question -- this module treats ANY substantive reply as
    sufficient for visible_resolution, by design, because that matches what
    the evidence-unit corpus actually needs (extractable real guidance),
    not "did this specific conversation end well."
"""
import re

SUBSTANTIVE_PATTERNS = [
    r"updated? to\s+(the\s+)?(latest\s+)?(ios|version)",
    r"\brestart\b",
    r"settings\s*(&gt;|>)",
    r"\bpress\b",
    r"\btap\b",
    r"here'?s what you can do",
    r"here are (the |our )?steps",
    r"look here for our steps",
    r"follow these steps",
    r"\bwork.?around\b",
    r"backup your (device|iphone|data)",
    r"\btry\b\s+(?!anything\b|everything\b|something\b|nothing\b)\w+ing\b",
    r"has steps to (address|fix)",
    r"steps to address",
    r"check (this|the) article",
    r"here is how to",
]

# a declarative "here's WHERE to look" pointer, not an instruction to act --
# excluded from counting as substantive even though it may contain a
# substantive keyword like "Settings >"
INFORMATIONAL_POINTER_PATTERNS = [
    r"can be found",
    r"\bfound (in|under)\b",
]

# a conditional status-check ("if you are updated to X", "if it's on the
# latest version") -- asking WHETHER something is already true, not
# instructing the customer to do it
CONDITIONAL_STATUS_CHECK_PATTERNS = [
    r"if (you are|it is|it'?s|you'?ve)\b.{0,20}(updated?|on)\b",
    r"letting us know if you are",
]

# Settings > General > About is Apple's device-info screen (model, serial
# number, iOS/software version) -- it has no toggles or fixes on it. In this
# corpus, a sentence that points here is structurally always "go look up
# your version/model" (usually followed by a DM handoff), never an actual
# troubleshooting action, unlike other Settings paths ("Settings > Bluetooth,
# disable it", "Settings > General > Accessibility > ... > Auto-Brightness")
# which name a real feature to change. This is why the exclusion is scoped
# to "General > About" specifically rather than to "Settings >" broadly or
# to the word "version" alone -- it's a structural fact about where actual
# fixes live in Apple's UI, not a keyword ban.
DEVICE_INFO_LOOKUP_PATTERNS = [
    r"settings\s*(&gt;|>)\s*general\s*(&gt;|>)\s*about\b",
]

REDIRECT_PATTERNS = [
    r"\bdm us\b",
    r"\bdirect message\b",
    r"send us a dm",
    r"meet (us|you) in (a )?dm",
    r"join us in dm",
    r"shoot us a dm",
    r"let'?s (move|go|take this|meet up) .*\bdm\b",
    r"reach out .*\bdm\b",
    r"\bprivate message\b",
    r"send us a (private|direct) message",
    r"online sales support",
    r"apple tv experts",
    r"support via twitter in english",
    r"support is available in english",
    r"our .*(experts|team) (here|below)",
    r"\bvia (a |an )?dm\b",
    r"\bin (a )?dm\b",
    r"\bdm\b",
]

DIAGNOSTIC_PATTERNS = [
    r"which (iphone|ios|version|model)",
    r"what'?s going on",
    r"what exactly is happening",
    r"tell us (more|what)",
    r"are you (getting|seeing|receiving|experiencing|referring)",
    r"have you tried",
    r"do you (have|see|recall|know if)",
    r"is your .* updated",
    r"which .* are you (using|currently|running)",
]

ACKNOWLEDGEMENT_PATTERNS = [
    r"glad to hear",
    r"happy to hear",
    r"have a great (rest of your day|day)",
    r"we'?re here for you if you need us",
]

SELF_RESOLUTION_PATTERNS = [
    r"\bfixed it\b",
    r"figured out",
    r"solved it",
    r"\bsorted\b",
    r"found (it|out)",
    r"works now",
]

NON_ENGLISH_SIGNAL_PATTERNS = [
    r"support via twitter in english",
    r"support is available in english",
    r"twitter support is available in english",
]


def _matches_any(text_lower, patterns):
    return any(re.search(p, text_lower) for p in patterns)


def _split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p for p in parts if p.strip()]


def _is_question(sentence):
    s = sentence.strip()
    if s.endswith("?"):
        return True
    return bool(re.match(r"^(do|does|did|are|is|have|has|would|could|can|will)\b", s.lower()))


def classify_brand_reply(text):
    if not isinstance(text, str) or not text.strip():
        return "unclassified"

    sentences = _split_sentences(text)
    for sent in sentences:
        s_lower = sent.lower()
        if _is_question(sent):
            continue
        if _matches_any(s_lower, INFORMATIONAL_POINTER_PATTERNS):
            continue
        if _matches_any(s_lower, CONDITIONAL_STATUS_CHECK_PATTERNS):
            continue
        if _matches_any(s_lower, DEVICE_INFO_LOOKUP_PATTERNS):
            continue
        if _matches_any(s_lower, SUBSTANTIVE_PATTERNS):
            return "substantive"

    t = text.lower()
    if _matches_any(t, REDIRECT_PATTERNS):
        return "redirect"
    if _matches_any(t, DIAGNOSTIC_PATTERNS):
        return "diagnostic_only"
    if "?" in text:
        return "diagnostic_only"
    if _matches_any(t, ACKNOWLEDGEMENT_PATTERNS):
        return "acknowledgement"
    return "unclassified"


def is_non_english_signal(brand_texts):
    for t in brand_texts:
        if isinstance(t, str) and _matches_any(t.lower(), NON_ENGLISH_SIGNAL_PATTERNS):
            return True
    return False


def _customer_self_resolved(turns):
    brand_indices = [i for i, t in enumerate(turns) if t.get("role") == "brand"]
    if not brand_indices:
        return False
    last_brand_idx = brand_indices[-1]
    for i in range(last_brand_idx):
        t = turns[i]
        if t.get("role") == "customer" and isinstance(t.get("text"), str):
            if _matches_any(t["text"].lower(), SELF_RESOLUTION_PATTERNS):
                return True
    return False


def classify_thread_evidence_type(brand_labels, brand_texts, turns=None):
    if is_non_english_signal(brand_texts):
        return "noise_other"

    if turns is not None and _customer_self_resolved(turns):
        brand_indices = [i for i, t in enumerate(turns) if t.get("role") == "brand"]
        if brand_indices:
            last_label = brand_labels[-1] if brand_labels else None
            if last_label in ("acknowledgement", "unclassified"):
                return "acknowledgement_only"

    if "substantive" in brand_labels:
        return "visible_resolution"
    if "redirect" in brand_labels:
        return "private_channel_redirect"
    if "diagnostic_only" in brand_labels:
        return "partial_troubleshooting"
    if brand_labels and all(l == "acknowledgement" for l in brand_labels):
        return "acknowledgement_only"
    return "noise_other"
