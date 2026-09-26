import re
import unicodedata
import anyascii
import pandas as pd

# Multi-region business legal suffixes and abbreviations (US, India, France)
LEGAL_TERMS = {
    r"\bcorp\b": "corporation",
    r"\binc\b": "incorporated",
    r"\bltd\b": "limited",
    r"\bpvt\b": "private",
    r"\bpvt\s+ltd\b": "private limited",
    r"\bllc\b": "limited liability company",
    r"\bllp\b": "limited liability partnership",
    r"\bco\b": "company",
    r"\b&\b": "and",
    # French entity types
    r"\bsarl\b": "sarl",
    r"\bsas\b": "sas",
    r"\bsasu\b": "sasu",
    r"\beurl\b": "eurl",
    r"\bsci\b": "sci",
    r"\bsa\b": "sa",
}

# Multi-region address abbreviations (US, India, France)
ADDRESS_TERMS = {
    r"\bst\b": "street",
    r"\brd\b": "road",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bln\b": "lane",
    r"\bdr\b": "drive",
    r"\bapt\b": "apartment",
    r"\bste\b": "suite",
    r"\bfl\b": "floor",
    r"\bno\b": "number",
    # French address terms
    r"\bbd\b": "boulevard",
    r"\bav\b": "avenue",
    r"\ball\b": "allee",
    r"\bpl\b": "place",
    r"\bch\b": "chemin",
}

# Suffixes to strip when computing core business name
LEGAL_STRIP_REGEX = re.compile(
    r"\b(corporation|incorporated|limited|private|llc|llp|company|sarl|sas|sasu|eurl|sci|sa|corp|inc|ltd|pvt|co)\b",
    flags=re.IGNORECASE,
)

LEGAL_REGEXES = [(re.compile(p, re.IGNORECASE), repl) for p, repl in LEGAL_TERMS.items()]
ADDRESS_REGEXES = [(re.compile(p, re.IGNORECASE), repl) for p, repl in ADDRESS_TERMS.items()]
AMP_REGEX = re.compile(r"&")
PUNCT_REGEX = re.compile(r"[^\w\s]")
WHITESPACE_REGEX = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """
    Normalizes unicode via anyascii phonetic transliteration, lowercases,
    expands ampersands, and strips punctuation.
    Never erases Indian or French scripts!
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    # Transliterate non-ASCII (Hindi/Devanagari, accented French, etc.) to Latin
    text = anyascii.anyascii(text)
    text = text.lower()
    text = AMP_REGEX.sub(" and ", text)
    text = PUNCT_REGEX.sub(" ", text)
    text = WHITESPACE_REGEX.sub(" ", text).strip()
    return text


def clean_business_name(name: str) -> str:
    """Normalize business name and expand legal entity suffixes."""
    norm = normalize_text(name)
    for regex, replacement in LEGAL_REGEXES:
        norm = regex.sub(replacement, norm)
    return WHITESPACE_REGEX.sub(" ", norm).strip()


def strip_legal_suffixes(name: str) -> str:
    """Strips common legal suffixes to isolate the core brand name."""
    core = LEGAL_STRIP_REGEX.sub(" ", name)
    return WHITESPACE_REGEX.sub(" ", core).strip()


def clean_address(address: str) -> str:
    """Normalize address components and expand street abbreviations."""
    norm = normalize_text(address)
    for regex, replacement in ADDRESS_REGEXES:
        norm = regex.sub(replacement, norm)
    return WHITESPACE_REGEX.sub(" ", norm).strip()


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Adds cleaned text columns and core brand name for matching."""
    df = df.copy()
    col_name = "business_name" if "business_name" in df.columns else "name"
    col_addr = "business_address" if "business_address" in df.columns else "address"

    df["clean_name"] = df[col_name].fillna("").apply(clean_business_name)
    df["core_name"] = df["clean_name"].apply(strip_legal_suffixes)
    df["clean_address"] = df[col_addr].fillna("").apply(clean_address)
    df["clean_country"] = df["country"].fillna("").str.strip().str.upper()
    df["combined_text"] = df["clean_name"] + " " + df["clean_address"]
    return df
