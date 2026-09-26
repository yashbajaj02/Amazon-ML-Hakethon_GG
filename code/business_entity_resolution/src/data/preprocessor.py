import re
import unicodedata
import pandas as pd

# Common business legal suffixes and abbreviations
LEGAL_TERMS = {
    r"\bcorp\b": "corporation",
    r"\binc\b": "incorporated",
    r"\bltd\b": "limited",
    r"\bpvt\b": "private",
    r"\bllc\b": "limited liability company",
    r"\bco\b": "company",
    r"\b&\b": "and",
}

# Common address abbreviations
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
}

LEGAL_REGEXES = [(re.compile(p), repl) for p, repl in LEGAL_TERMS.items()]
ADDRESS_REGEXES = [(re.compile(p), repl) for p, repl in ADDRESS_TERMS.items()]
AMP_REGEX = re.compile(r"&")
PUNCT_REGEX = re.compile(r"[^\w\s]")
WHITESPACE_REGEX = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize unicode, lowercase, expand ampersands, and remove punctuation."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
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


def clean_address(address: str) -> str:
    """Normalize address components and expand street abbreviations."""
    norm = normalize_text(address)
    for regex, replacement in ADDRESS_REGEXES:
        norm = regex.sub(replacement, norm)
    return WHITESPACE_REGEX.sub(" ", norm).strip()


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Adds cleaned text columns for matching."""
    df = df.copy()
    df["clean_name"] = df["business_name"].fillna("").apply(clean_business_name)
    df["clean_address"] = df["business_address"].fillna("").apply(clean_address)
    df["clean_country"] = df["country"].fillna("").str.strip().str.upper()
    df["combined_text"] = df["clean_name"] + " " + df["clean_address"]
    return df
