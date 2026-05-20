from transformers import Wav2Vec2CTCTokenizer

_ph_name = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(_ph_name)
vocab = tokenizer.get_vocab()

turned_r = "\u0279"

with open("scratch/vocab_output.txt", "w", encoding="utf-8") as f:
    f.write(f"Is 'r' in vocab? {'r' in vocab}\n")
    f.write(f"Is 'ɹ' in vocab? {turned_r in vocab}\n")
    f.write("\nAll keys in vocab containing 'r' or 'ɹ':\n")
    for k in sorted(vocab.keys()):
        if 'r' in k or turned_r in k:
            f.write(f"  {k}: {vocab[k]}\n")
print("Done writing to scratch/vocab_output.txt")
