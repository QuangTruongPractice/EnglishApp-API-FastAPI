import os
import sys
sys.path.append(os.getcwd())

# Import patches first to patch torch.load before other imports
import app.core.patches

import torch
import numpy as np
import whisperx
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
from app.services.scoring_service import _espeak_ipa, IPA_RE
import unicodedata

# Manually load the models to check output
_ph_name = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(_ph_name)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(_ph_name)
model = Wav2Vec2ForCTC.from_pretrained(_ph_name)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

test_files = [
    ("abroad", "audio/abroad.wav"),
    ("abstract", "audio/abstract.wav"),
    ("across", "audio/across.wav"),
    ("afraid", "audio/afraid.wav"),
    ("agree", "audio/agree.wav"),
    ("bar", "audio/bar.wav"),
    ("brother", "audio/brother.wav"),
]

output_lines = []

for word, path in test_files:
    if not os.path.exists(path):
        output_lines.append(f"File {path} does not exist.\n")
        continue
        
    # Load audio
    audio_arr = whisperx.load_audio(path)
    
    # Extract phonemes via model
    inp = feature_extractor(audio_arr, sampling_rate=16000, return_tensors="pt")
    if device == "cuda":
        inp = {k: v.to(device) for k, v in inp.items()}
    with torch.inference_mode():
        logits = model(**inp).logits
    pred_ids = torch.argmax(logits, dim=-1)[0]
    pred_phonemes = tokenizer.decode(pred_ids)
    
    # Extract phonemes via espeak
    espeak_raw, espeak_tokens = _espeak_ipa(word)
    
    output_lines.append(f"Word: {word}\n")
    output_lines.append(f"  Espeak raw: {espeak_raw}\n")
    output_lines.append(f"  Espeak tokens: {espeak_tokens}\n")
    output_lines.append(f"  Model output: {pred_phonemes}\n")
    output_lines.append(f"  Model tokens: {tuple(IPA_RE.findall(unicodedata.normalize('NFC', pred_phonemes)))}\n")
    output_lines.append("-" * 40 + "\n")

with open("scratch/test_phoneme_output.txt", "w", encoding="utf-8") as f:
    f.writelines(output_lines)
print("Done testing!")
