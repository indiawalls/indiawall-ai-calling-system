"""
Text Normalization for Hindi Speech Synthesis (Kokoro-82M).
Converts numbers, emails, phone digits, markdown, and symbols into natural spoken Hindi phrases.
Prevents phonetic glitches and robotic artifacts in Kokoro TTS.
"""

import re

HINDI_DIGIT_WORDS = {
    "0": "शून्य", "1": "एक", "2": "दो", "3": "तीन", "4": "चार",
    "5": "पाँच", "6": "छह", "7": "सात", "8": "आठ", "9": "नौ",
}

COMMON_ABBREVIATIONS = [
    (r"\bsq\.?\s*ft\.?\b", " स्क्वायर फीट "),
    (r"\bsqft\b", " स्क्वायर फीट "),
    (r"\bft\.?\b", " फीट "),
    (r"\bNCR\b", " एनसीआर "),
    (r"\bHQ\b", " हेडक्वार्टर "),
    (r"\bkm\.?\b", " किलोमीटर "),
    (r"\bapprox\.?\b", " लगभग "),
    (r"\bRs\.?\s*", " रुपये "),
    (r"₹\s*", " रुपये "),
    (r"%", " प्रतिशत "),
    (r"&", " और "),
    (r"\+", " प्लस "),
]


def normalize_text_for_hindi_tts(text: str) -> str:
    """
    Clean and normalize input text for pristine Hindi TTS generation:
    - Strips markdown formatting (*, #, _, ~, `, |)
    - Expands phone numbers into spaced Hindi digits
    - Expands emails into spoken words
    - Replaces symbols (%, ₹, +, /) with Hindi words
    - Cleans excessive punctuation to prevent TTS clipping
    """
    if not text:
        return ""

    # 1. Strip Markdown syntax (*bold*, _italic_, # heading, `code`, - bullet)
    text = re.sub(r"[\*\_#`~|>]+", " ", text)
    text = re.sub(r"^\s*[-•*]\s*", "", text, flags=re.MULTILINE)

    # 2. Normalize Email
    def _replace_email(m):
        domain = m.group(2).replace(".", " डॉट ")
        user = m.group(1).replace("indiawallsofficial", "इंडिया वाल्स ऑफिशियल")
        return f"{user} एट {domain}".replace("gmail", "जीमेल").replace("com", "कॉम")

    text = re.sub(r"([a-zA-Z0-9_.+-]+)@([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", _replace_email, text)

    # 3. Normalize 10-digit phone numbers
    def _replace_phone(match):
        digits = re.sub(r"\D", "", match.group())
        if len(digits) == 12 and digits.startswith("91"):
            digits = digits[2:]
        if len(digits) == 10:
            words = [HINDI_DIGIT_WORDS.get(d, d) for d in digits]
            return " ".join(words[:5]) + ", " + " ".join(words[5:])
        return match.group()

    text = re.sub(r"(\+?91[\-\s]?)?[6-9][0-9\s\-]{8,12}[0-9]", _replace_phone, text)

    # 4. Normalize common abbreviations and units
    for pattern, replacement in COMMON_ABBREVIATIONS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # 5. Normalize slash as "प्रति" (e.g., रुपये/फीट -> रुपये प्रति फीट)
    text = re.sub(r"([a-zA-Z\u0900-\u097F]+)/([a-zA-Z\u0900-\u097F]+)", r"\1 प्रति \2", text)

    # 6. Normalize trailing / duplicate punctuation marks (e.g., "??" -> "?", "..." -> "।")
    text = re.sub(r"\.{2,}", "।", text)
    text = re.sub(r"[?!]{2,}", "?", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text
