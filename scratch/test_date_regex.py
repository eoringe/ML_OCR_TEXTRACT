import re

DATE_PATTERNS = [
    r'\b(?:0?[1-9]|[12]\d|3[01])([-/])(?:0?[1-9]|1[012])\1\d{2,4}\b',
    r'\b\d{4}([-/])(?:0?[1-9]|1[012])\1(?:0?[1-9]|[12]\d|3[01])\b',
    r'\b(?:0?[1-9]|[12]\d|3[01])\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b',
    r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(?:0?[1-9]|[12]\d|3[01]),?\s+\d{2,4}\b',
]

def pre_clean(text):
    text_cleaned = re.sub(r'(\d{1,2})\s*[?|\\\/]\s*(\d{1,2})', r'\1/\2', text)
    text_cleaned = re.sub(r'(\d{1,2})\s*[?|\\\/]\s*(\d{2,4})', r'\1/\2', text_cleaned)
    return text_cleaned

line = 'UF3i/3-41/ 12/144'
cleaned = pre_clean(line)
print(f"Cleaned line: '{cleaned}'")

for idx, pattern in enumerate(DATE_PATTERNS):
    matches = list(re.finditer(pattern, cleaned, re.IGNORECASE))
    print(f"Pattern {idx} matches:")
    for m in matches:
        print(f"  Match: '{m.group(0)}'")
