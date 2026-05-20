import os, sys
# Ensure project root is in PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
# Disable heavy model initialization
os.environ['DISABLE_MODEL_INIT'] = '1'

from app.services.scoring_service import normalize_ph, IPA_RE
import unicodedata

def test_case(word, ph_raw):
    user_ph = list(IPA_RE.findall(unicodedata.normalize("NFC", ph_raw)))
    # Dummy expected phonemes (same as user for demonstration)
    ph_exp = user_ph
    # Normalization
    ph_exp_norm, _ = normalize_ph(ph_exp)
    user_ph_norm, _ = normalize_ph(user_ph)
    print(f"Word: {word}\n  Raw: {ph_raw}\n  Tokens: {user_ph}\n  Normalized: {user_ph_norm}\n")

# Sample test cases focusing on rhotic handling
samples = [
    ("bar", "bɑːɹ"),
    ("brother", "bɹʌðɚ"),
    ("red", "ɹɛd"),
    ("agree", "ɐɡɹiː"),
    ("car", "kɑːɹ"),
]
for w, raw in samples:
    test_case(w, raw)
