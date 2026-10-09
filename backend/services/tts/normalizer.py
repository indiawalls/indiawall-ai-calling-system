"""
Text Normalization for Hindi Speech Synthesis (Kokoro-82M).
Converts numbers, emails, and phone digits into natural spoken Hindi phrases.
"""

import re

HINDI_DIGIT_WORDS = {
    "0": "शून्य", "1": "एक", "2": "दो", "3": "तीन", "4": "चार",
    "5": "पाँच", "6": "छह", "7": "सात", "8": "आठ", "9": "नौ",
}


def normalize_text_for_hindi_tts(text: str) -> str:
    """
    Convert raw 10-digit phone numbers and emails into individual spoken Hindi digit words
    with pause commas so TTS speaks 'सात आठ दो...' instead of 'सात अरब बयासी करोड़'.
    """
    if not text:
        return ""

    # 1. Normalize Email
    def _replace_email(m):
        domain = m.group(2).replace(".", " डॉट ")
        user = m.group(1).replace("indiawallsofficial", "इंडिया वाल्स ऑफिशियल")
        return f"{user} एट {domain}".replace("gmail", "जीमेल").replace("com", "कॉम")

    text = re.sub(r"([a-zA-Z0-9_.+-]+)@([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", _replace_email, text)

    # 2. Normalize 10-digit phone numbers (with optional +91, dashes or spaces)
    def _replace_phone(match):
        digits = re.sub(r"\D", "", match.group())
        if len(digits) == 12 and digits.startswith("91"):
            digits = digits[2:]
        if len(digits) == 10:
            words = [HINDI_DIGIT_WORDS.get(d, d) for d in digits]
            return " ".join(words[:5]) + ", " + " ".join(words[5:])
        return match.group()

    text = re.sub(r"(\+?91[\-\s]?)?[6-9][0-9\s\-]{8,12}[0-9]", _replace_phone, text)
    return text
