import panphon
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from importlib.resources import files

# ==============================================================================
# PATCH PANPHON UTF-8 (WINDOWS FIX)
# ==============================================================================
import panphon.featuretable as _pf

def _read_bases_utf8(self, fn: str, weights):
    spec_to_int = {"+": 1, "0": 0, "-": -1}

    with files("panphon").joinpath(fn).open("r", encoding="utf-8") as f:
        df = pd.read_csv(f)

    df["ipa"] = df["ipa"].apply(self.normalize)

    feature_names = list(df.columns[1:])

    df[feature_names] = df[feature_names].map(
        lambda x: spec_to_int[x]
    )

    segments = [
        (
            row["ipa"],
            _pf.Segment(
                feature_names,
                row[1:].to_dict(),
                weights=weights
            )
        )
        for (_, row) in df.iterrows()
    ]

    seg_dict = dict(segments)

    return segments, seg_dict, feature_names

_pf.FeatureTable._read_bases = _read_bases_utf8

# ==============================================================================
# INIT
# ==============================================================================
ft = panphon.FeatureTable()

# ==============================================================================
# PHONEME INVENTORY
# ==============================================================================

VOWELS = [
    # High front
    "i", "iː", "ɪ",

    # Mid front
    "e", "eɪ", "ɛ",

    # Low/front-open
    "æ", "a",

    # Open/back
    "ɑ", "ɑː", "ɒ",

    # Mid/back rounded
    "ɔ", "ɔː", "o", "oʊ",

    # High back
    "u", "uː", "ʊ",

    # Central vowels
    "ə", "ɐ", "ʌ",

    # Rhotic vowels
    "ə˞", "ɜ˞", "ɹ",

    # Common diphthongs
    "aɪ", "aʊ", "ɔɪ"
]

CONSONANTS = [
    # Stops
    "p", "b",
    "t", "d",
    "k", "ɡ",

    # Fricatives
    "f", "v",
    "θ", "ð",
    "s", "z",
    "ʃ", "ʒ",
    "h",

    # Nasals
    "m", "n", "ŋ",

    # Liquids
    "l", "r",

    # Glides
    "j", "w",

    # Affricates
    "tʃ", "dʒ"
]

# ==============================================================================
# VECTOR EXTRACTION
# ==============================================================================

def get_vector(phone):
    vec = ft.word_to_vector_list(phone)

    if not vec:
        return None

    return np.array([
        1.0 if x == '+' else
        -1.0 if x == '-' else
        0.0
        for x in vec[0]
    ], dtype=float)

# ==============================================================================
# BUILD MATRIX
# ==============================================================================

def build_similarity_matrix(phonemes):
    matrix = pd.DataFrame(
        index=phonemes,
        columns=phonemes,
        dtype=float
    )

    for p1 in phonemes:
        for p2 in phonemes:

            v1 = get_vector(p1)
            v2 = get_vector(p2)

            if v1 is None or v2 is None:
                similarity = 0.0
            else:
                similarity = cosine_similarity(
                    [v1],
                    [v2]
                )[0][0]

            similarity = round(float(similarity), 3)

            matrix.loc[p1, p2] = similarity

    return matrix

# ==============================================================================
# GENERATE MATRICES
# ==============================================================================

print("Generating vowel similarity matrix...")
vowel_matrix = build_similarity_matrix(VOWELS)

print("Generating consonant similarity matrix...")
consonant_matrix = build_similarity_matrix(CONSONANTS)

# ==============================================================================
# SAVE
# ==============================================================================

vowel_matrix.to_csv(
    "vowel_similarity_matrix.csv",
    encoding="utf-8"
)
consonant_matrix.to_csv(
    "consonant_similarity_matrix.csv",
    encoding="utf-8"
)

vowel_matrix.to_string(
    buf="vowel_similarity_matrix.txt",
    float_format="{:.3f}".format,
)
consonant_matrix.to_string(
    buf="consonant_similarity_matrix.txt",
    float_format="{:.3f}".format,
)

print("Saved:")
print("- vowel_similarity_matrix.csv")
print("- vowel_similarity_matrix.txt")
print("- consonant_similarity_matrix.csv")
print("- consonant_similarity_matrix.txt")