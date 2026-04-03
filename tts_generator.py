import os
from gtts import gTTS
from pydub import AudioSegment

def text_to_wav(input_file='word.txt', output_dir='audio'):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Read the input file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    # Split words by comma or newline and strip whitespace
    words = [word.strip() for word in content.replace(',', '\n').split('\n') if word.strip()]

    print(f"Found {len(words)} words to process: {', '.join(words)}")

    for word in words:
        try:
            # 1. Generate speech using gTTS (Google Text-to-Speech)
            tts = gTTS(text=word, lang='en')
            
            # 2. Save as temporary MP3
            temp_mp3 = f"{word}_temp.mp3"
            tts.save(temp_mp3)

            # 3. Convert MP3 to WAV using pydub
            # Note: Requires ffmpeg to be installed on your system
            audio = AudioSegment.from_mp3(temp_mp3)
            
            # Sanitize filename (remove special chars if any)
            safe_word = "".join([c for c in word if c.isalnum() or c in (' ', '-', '_')]).strip()
            wav_filename = os.path.join(output_dir, f"{safe_word}.wav")
            
            audio.export(wav_filename, format="wav")
            print(f"Successfully generated: {wav_filename}")

            # 4. Remove temporary MP3
            os.remove(temp_mp3)
            
        except Exception as e:
            print(f"Error processing '{word}': {e}")

if __name__ == "__main__":
    # Ensure gTTS is installed: pip install gTTS
    # Ensure pydub is installed: pip install pydub (already in requirements.txt)
    # Ensure ffmpeg is installed on your system
    
    text_to_wav()
