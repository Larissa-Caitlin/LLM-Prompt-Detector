import re
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from config import MODEL_PATH, CONFIDENCE_THRESHOLD
from logger import logger

# ── Rule-based patterns ──────────────────────────────────────────────
INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"you are now",
    r"pretend you have no",
    r"disregard your",
    r"act as (a |an )?(different|new|unrestricted)",
    r"jailbreak",
    r"dan mode",
    r"do anything now",
    r"no restrictions",
    r"forget your (previous |prior )?instructions",
    r"your new instructions",
    r"system prompt",
    r"override (all )?instructions",
]

HARMFUL_PATTERNS = [
    r"how to hack into (my|a|someone|their|his|her|our)",
    r"how to hack",
    r"how to attack",
    r"how to break into",
    r"how to steal",
    r"how to make (a )?(bomb|weapon|malware|virus|ransomware|trojan|keylogger)",
    r"how to bypass security",
    r"how to ddos",
    r"how to phish",
    r"how to exploit",
    r"how to crack (a )?password",
    r"how to hack into",
    r"hack into (a |my |their |someone)",
    r"create (a )?(malware|virus|ransomware|trojan|keylogger|botnet)",
    r"write (a )?(malware|virus|ransomware|exploit|keylogger)",
    r"make (a )?(malware|virus|ransomware|trojan|keylogger)",
]

SAFE_WHITELIST = [
    r"^what is ",
    r"^how does ",
    r"^explain ",
    r"^tell me about ",
    r"^define ",
    r"^what are ",
    r"^how do (i|you|we) ",
    r"^why is ",
    r"^when did ",
    r"^who is ",
    r"^how did ",
    r"^what was ",
    r"^how (do i|can i) install ",
    r"^how (do i|can i) use ",
    r"^how (do i|can i) create ",
    r"^how (do i|can i) make ",
    r"^what does ",
    r"^can you explain ",
    r"^could you explain ",
]


# ── Load model once at startup ───────────────────────────────────────
from config import MODEL_PATH, CONFIDENCE_THRESHOLD


tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model     = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ── Layer 1: rule check ──────────────────────────────────────────────
def rule_check(prompt):
    prompt_lower = prompt.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, prompt_lower):
            return True, pattern
    return False, None

# ── Layer 2: ML check ────────────────────────────────────────────────
def ml_check(prompt):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs  = torch.softmax(logits, dim=1)[0]

    injection_score = probs[1].item()
    return injection_score > CONFIDENCE_THRESHOLD, round(injection_score, 4)

# ── Combined detector ────────────────────────────────────────────────
def detect(prompt):
    if not prompt.strip():
        return {
            "blocked":    False,
            "layer":      None,
            "reason":     "Empty prompt",
            "confidence": 0.0
        }

    # Layer 1
    flagged, rule = rule_check(prompt)
    if flagged:
        return {
            "blocked":    True,
            "layer":      "Rule filter",
            "reason":     f"Matched: {rule}",
            "confidence": 1.0
        }

    # Layer 2
    blocked, score = ml_check(prompt)
    if blocked:
        return {
            "blocked":    True,
            "layer":      "ML classifier",
            "reason":     "Adversarial prompt detected",
            "confidence": score
        }

    return {
        "blocked":    False,
        "layer":      None,
        "reason":     "No threat detected",
        "confidence": round(1 - score, 4)
    }