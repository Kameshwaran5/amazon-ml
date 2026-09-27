"""
Deterministic, country-agnostic normalization module for Business Entity Resolution.
Implements:
- Unicode NFKC normalization, lowercasing, unidecode transliteration
- Legal suffix equivalence dictionary (US, India, France/EU agnostic)
- Address abbreviation dictionary
- Regex extraction of building/unit numbers and postal codes
- Degrading gracefully to empty/None without country gating
"""

import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple
import jellyfish
import unidecode


# Legal suffix mapping: maps variant to canonical token
LEGAL_SUFFIX_MAP = {
    "corp": "corporation",
    "corporation": "corporation",
    "inc": "incorporated",
    "incorporated": "incorporated",
    "ltd": "limited",
    "limited": "limited",
    "pvt": "private",
    "private": "private",
    "llc": "llc",
    "l.l.c": "llc",
    "co": "company",
    "company": "company",
    "sa": "sa",
    "sarl": "sarl",
    "sas": "sas",
    "gmbh": "gmbh",
    "plc": "plc",
    "holdings": "holdings",
    "enterprises": "enterprises",
    "services": "services",
    "technologies": "technologies",
    "associates": "associates",
    "consulting": "consulting",
    "solutions": "solutions",
}

# Regex to match legal suffixes at word boundaries (especially at the end of names)
_LEGAL_SUFFIX_REGEX = re.compile(
    r"\b(private limited|pvt ltd|pvt\.?\s*ltd\.?|private ltd\.?|"
    r"corporation|corp\.?|incorporated|inc\.?|limited|ltd\.?|"
    r"l\.?l\.?c\.?|llc|l\.?l\.?p\.?|llp|company|co\.?|"
    r"s\.?a\.?r\.?l\.?|sarl|s\.?a\.?s\.?u\.?|sasu|s\.?a\.?s\.?|sas|s\.?a\.?|sa|"
    r"e\.?u\.?r\.?l\.?|eurl|sci|snc|et fils|fils|cie|gmbh|plc|p\.?l\.?c\.?)\b",
    re.IGNORECASE,
)

# Address abbreviation mappings: word boundary regexes
ADDRESS_ABBREV_MAP = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "bd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "pkwy": "parkway",
    "cir": "circle",
    "hwy": "highway",
    "apt": "apartment",
    "ste": "suite",
    "fl": "floor",
    "bldg": "building",
    "hno": "house",
    "h.no": "house",
    "str": "street",
}

_ADDRESS_REPLACEMENTS = [
    (re.compile(r"\b" + re.escape(abbr) + r"\b", re.IGNORECASE), full)
    for abbr, full in ADDRESS_ABBREV_MAP.items()
]

# Regex for postal codes (5-6 digits: US 5-digit, India 6-digit, France 5-digit)
_POSTAL_CODE_REGEX = re.compile(r"\b([1-9]\d{4,5})\b")

# Regex for unit / apartment / flat / suite numbers
_UNIT_NUMBER_REGEX = re.compile(
    r"\b(?:apt|apartment|suite|ste|unit|flat|plot|h\.?no|door|#)\s*[:.-]?\s*([a-z0-9\-]+)\b",
    re.IGNORECASE,
)

# Regex for leading building numbers: e.g. "1795 Westchester Drive"
_LEADING_BUILDING_NUM_REGEX = re.compile(r"^\s*([0-9]+[a-z]?)\b", re.IGNORECASE)

# All numbers in a string
_ALL_NUMBERS_REGEX = re.compile(r"\b\d+\b")


def clean_text_basic(text: Optional[str]) -> str:
    """
    Unicode NFKC normalization, lowercase, transliterate via unidecode,
    normalize ampersands and punctuation/whitespace.
    """
    if not text:
        return ""
    # Unicode NFKC
    text = unicodedata.normalize("NFKC", text)
    # Transliterate accents / non-ASCII cleanly (crucial for French e.g. café -> cafe, diacritics)
    text = unidecode.unidecode(text)
    # Lowercase
    text = text.lower()
    # Replace & with and
    text = text.replace("&", " and ")
    # Replace common punctuation with space except alphanumeric
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_business_name(name: Optional[str]) -> Tuple[str, str, Optional[str]]:
    """
    Normalize business name.
    Returns:
    - normalized_name: cleaned name
    - core_name: name with legal suffixes removed (for matching across suffix variants)
    - primary_phonetic: Metaphone code of the first significant word (for phonetic blocking)
    """
    raw_name_str = name or ""
    # Strip web domain extensions before punctuation stripping (e.g. lafdetailinggroup.com -> lafdetailinggroup)
    raw_no_domain = re.sub(r"\.(com|org|net|in|co\.in|fr|io|biz|info|edu|gov)\b", " ", raw_name_str, flags=re.IGNORECASE)
    cleaned = clean_text_basic(raw_no_domain)
    if not cleaned:
        return "", "", None

    # Also handle space-separated domain tokens e.g. "lafdetailinggroup com"
    cleaned_no_domain = re.sub(r"\b(com|org|net|in|co in|fr|io|biz|info|edu|gov)\b", " ", cleaned)
    cleaned_no_domain = re.sub(r"^www\s+", " ", cleaned_no_domain).strip()

    # Strip legal suffixes from end or boundaries for core_name
    core = _LEGAL_SUFFIX_REGEX.sub(" ", cleaned_no_domain)
    core = re.sub(r"^\b(the|mr|mrs|ms|dr)\b\s*", "", core, flags=re.IGNORECASE)
    core = re.sub(r"\b(and|et|and sons?)\b\s*$", "", core, flags=re.IGNORECASE)
    core = re.sub(r"\s+", " ", core).strip()
    if not core:
        core = cleaned

    tokens = core.split()
    primary_word = tokens[0] if tokens else ""
    # Jellyfish metaphone
    try:
        primary_phonetic = jellyfish.metaphone(primary_word) if primary_word else None
    except Exception:
        primary_phonetic = None

    return cleaned, core, primary_phonetic


def normalize_address(address: Optional[str]) -> Tuple[str, Optional[str], Optional[str], Set[str]]:
    """
    Normalize business address.
    Returns:
    - normalized_address: cleaned and expanded address
    - postal_code: 5 or 6 digit postal code if found, else None
    - unit_number: extracted unit/flat/suite/building number or None
    - all_numbers: set of all numeric strings in the address
    """
    cleaned = clean_text_basic(address)
    if not cleaned:
        return "", None, None, set()

    # Address abbreviation expansions
    expanded = cleaned
    for pattern, repl in _ADDRESS_REPLACEMENTS:
        expanded = pattern.sub(repl, expanded)
    expanded = re.sub(r"\s+", " ", expanded).strip()

    # Postal code extraction
    postal_code = None
    postal_match = _POSTAL_CODE_REGEX.search(cleaned)
    if postal_match:
        postal_code = postal_match.group(1)

    # Unit number extraction
    unit_number = None
    unit_match = _UNIT_NUMBER_REGEX.search(cleaned)
    if unit_match:
        unit_number = unit_match.group(1).lower()
    else:
        leading_match = _LEADING_BUILDING_NUM_REGEX.search(cleaned)
        if leading_match:
            unit_number = leading_match.group(1).lower()

    # All numbers present in address
    all_numbers = set(_ALL_NUMBERS_REGEX.findall(cleaned))

    return expanded, postal_code, unit_number, all_numbers


def extract_name_representations(raw_name: Optional[str]) -> Dict[str, any]:
    """
    Stage 2 expanded parallel representations for a business name:
    - raw: original string
    - normalized: NFKC + lowercase + unidecode + punctuation/whitespace normalized
    - alnum_only: alphanumeric characters only
    - compact: spaces removed
    - token_sorted: tokens sorted alphabetically
    - core_name: legal suffixes stripped
    - phonetic_code: metaphone code of first significant word
    """
    raw = raw_name or ""
    normalized, core, phon = normalize_business_name(raw)
    compact = normalized.replace(" ", "")
    alnum_only = re.sub(r"[^a-z0-9]", "", normalized)
    tokens = normalized.split()
    token_sorted = " ".join(sorted(tokens)) if tokens else ""

    return {
        "raw": raw,
        "normalized": normalized,
        "alnum_only": alnum_only,
        "compact": compact,
        "token_sorted": token_sorted,
        "core_name": core,
        "phonetic_code": phon,
    }


def extract_address_representations(raw_address: Optional[str]) -> Dict[str, any]:
    """
    Stage 2 expanded parallel representations for a business address:
    - raw: original string
    - normalized: abbreviations expanded, NFKC + unidecode + lowercase
    - alnum_only: alphanumeric characters only
    - compact: spaces removed
    - postal_code: extracted 5-6 digit postal code or None
    - unit_number: extracted unit/flat/suite or None
    - all_numbers: set of all numeric strings
    - street_tokens: list of cleaned street/locality words
    """
    raw = raw_address or ""
    normalized, postal, unit, all_nums = normalize_address(raw)
    compact = normalized.replace(" ", "")
    alnum_only = re.sub(r"[^a-z0-9]", "", normalized)
    street_tokens = [w for w in normalized.split() if len(w) >= 3 and not w.isdigit()]

    return {
        "raw": raw,
        "normalized": normalized,
        "alnum_only": alnum_only,
        "compact": compact,
        "postal_code": postal,
        "unit_number": unit,
        "all_numbers": all_nums,
        "street_tokens": street_tokens,
    }

