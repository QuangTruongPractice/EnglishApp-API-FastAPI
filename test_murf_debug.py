import os
import requests
from dotenv import load_dotenv

load_dotenv()

MURF_API_KEY = os.getenv("MURF_API_KEY")

def test_murf_stream():
    url = "https://global.api.murf.ai/v1/speech/stream"
    headers = {
        "api-key": MURF_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "voice_id": "Matthew",
        "text": "Hello! How can I assist you today? Please let me know your issue, and I'll do my best to help.",
        "locale": "en-US",
        "model": "FALCON",
        "format": "MP3",
        "sampleRate": 24000,
        "channelType": "MONO"
    }
    
    print(f"Testing URL: {url}")
    print(f"Payload: {payload}")
    
    try:
        response = requests.post(url, json=payload, headers=headers, stream=True)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Success! Receiving stream...")
            with open("test_audio.mp3", "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            print("Saved to test_audio.mp3")
        else:
            print(f"Error Response: {response.text}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_murf_stream()
