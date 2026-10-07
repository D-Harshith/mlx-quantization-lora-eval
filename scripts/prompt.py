LABELS = ["bug", "feature", "question"]

SYSTEM = """You are a GitHub issue triage assistant. Classify the issue into exactly one label:

bug - something is broken, crashes, or behaves differently from what is documented or expected
feature - a request for new functionality or an improvement to existing behavior
question - the author is asking for help, usage guidance, or clarification

Reply with one word only: bug, feature, or question."""

def build_messages(case):
    """Turn one eval case into chat messages for the model."""
    user = f"Title: {case['title']}\n\nBody:\n{case['body']}"
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]

def parse_label(text):
    """Map the model's raw reply to a label, or 'invalid' if it isn't one."""
    words = text.strip().lower().split()
    first = words[0].strip(".,:;!*\"'`") if words else ""
    return first if first in LABELS else "invalid"
