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

KNOWN, ACCEPTED LIMITATIONS (not chased further -- see calibration report):
  - A declarative sentence like "This can be found in Settings > General >
    About" (an informational pointer to WHERE something is) is lexically
    indistinguishable from a real instruction to change a setting. This
    causes a small, known residual false-positive rate on "substantive".
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
# substantive keyword like "Settings >". Covers the small closed set of
# location-reference verbs actually used in this corpus ("found", "see",
# "locate", "find", "appears" + "under") as opposed to action/instruction
# verbs ("tap", "go to", "enable", "reset", "turn off"), so it cannot match
# a genuine instruction sentence.
INFORMATIONAL_POINTER_PATTERNS = [
    r"can be found",
    r"\bfound (in|under)\b",
    r"\b(see|find|locate|appears)\b.{0,20}\bunder\b",
]

# a conditional status-check ("if you are updated to X", "if it's on the
# latest version") -- asking WHETHER something is already true, not
# instructing the customer to do it
CONDITIONAL_STATUS_CHECK_PATTERNS = [
    r"if (you are|it is|it'?s|you'?ve)\b.{0,20}(updated?|on)\b",
    r"letting us know if you are",
]

# a sentence that references "version" together with a Settings-path but
# contains NO actual action verb is a lookup/reporting reference ("tell us
# what version you're running under Settings > General > About"), not an
# instruction -- regardless of which specific lookup verb is used (see,
# find, locate, running, confirm, using...). This generalizes past the
# enumerated-verb approach above, which keeps missing new verb variants.
# Structural, not a DM check: fires independent of whether DM is mentioned,
# and never fires on a sentence containing an action verb, so it cannot
# affect "real fix + trailing DM footer" cases.
ACTION_VERB_PATTERNS = [
    r"\breset\b", r"\brestart\b", r"\bturn (on|off)\b", r"\benable\b",
    r"\bdisable\b", r"\btoggle\b", r"\btap\b", r"\bpress\b", r"\bgo to\b",
    r"\bchange\b", r"\bupdate\b", r"\bbackup\b",
]


def _is_version_lookup_reference(sentence_lower):
    has_version = "version" in sentence_lower
    has_settings_path = bool(re.search(r"settings\s*(&gt;|>)", sentence_lower))
    has_action_verb = any(re.search(p, sentence_lower) for p in ACTION_VERB_PATTERNS)
    return has_version and has_settings_path and not has_action_verb

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
        if _is_version_lookup_reference(s_lower):
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