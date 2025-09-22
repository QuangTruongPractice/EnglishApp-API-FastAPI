from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import whisperx
import torch
from typing import List, Dict, Optional
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import gc
import warnings
import os
import difflib
from fuzzywuzzy import fuzz
import tempfile
import uvicorn
from pydantic import BaseModel
from pyngrok import ngrok
import yt_dlp
from datetime import datetime
import logging
import string
import nltk
from nltk.corpus import words as nltk_words

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words', quiet=True)

warnings.filterwarnings("ignore")
os.environ['PYTORCH_LIGHTNING_UTILITIES_WARNINGS'] = 'ignore'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Free AI Pronunciation Service", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])

UPLOAD_FOLDER = 'temp_downloads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class PronunciationResponse(BaseModel):
    success: bool
    transcription: str
    expected_text: str
    fluency: int
    accuracy: int
    total_score: int
    pronunciation_errors: List[Dict]
    pronunciation_feedback: List[str]
    speech_rate: float
    duration: float
    word_count: int
    processing_time: float
    error: Optional[str] = None

class VideoProcessRequest(BaseModel):
    url: str

class VideoProcessResponse(BaseModel):
    success: bool
    data: Optional[Dict] = None
    error: Optional[str] = None

class FreeIntelligentFeedbackGenerator:
    def __init__(self):
        self.english_words = set(word.lower() for word in nltk_words.words())

        self.positive_messages = {
            'excellent': "Xuất sắc! Phát âm rất rõ ràng.",
            'good': "Rất tốt! Phát âm đã cải thiện nhiều.",
            'fair': "Không tệ! Hãy tiếp tục luyện tập.",
            'needs_improvement': "Cố gắng tốt! Hãy luyện từng từ chậm rãi."
        }

    def generate_intelligent_feedback(self, expected_text: str, spoken_text: str,
                                      word_segments: List[Dict], fluency_score: int,
                                      accuracy_score: int, speech_rate: float,
                                      duration: float) -> Dict:
        # Clean texts for comparison
        expected_clean = self._clean_text(expected_text)
        spoken_clean = self._clean_text(spoken_text)

        # Analyze errors
        error_analysis = self._analyze_pronunciation_errors(expected_clean, spoken_clean)

        # Generate concise feedback
        feedback_data = {
            "overall_feedback": self._generate_overall_feedback(accuracy_score, fluency_score),
            "pronunciation_errors": error_analysis["errors"][:3],  # Max 3 errors
            "positive_aspects": self._identify_positive_aspects(accuracy_score, fluency_score, speech_rate, duration),
            "improvement_suggestions": self._generate_improvement_suggestions(error_analysis, accuracy_score,
                                                                              fluency_score, speech_rate),
            "next_steps": self._generate_next_steps(accuracy_score, fluency_score)
        }

        return {"success": True, "feedback": feedback_data}

    def _clean_text(self, text: str) -> str:
        """Remove punctuation and normalize text"""
        translator = str.maketrans('', '', string.punctuation)
        cleaned = text.lower().translate(translator)
        return ' '.join(cleaned.split())

    def _analyze_pronunciation_errors(self, expected_text: str, spoken_text: str) -> Dict:
        """Analyze pronunciation errors - keep only major ones"""
        expected_words = expected_text.split()
        spoken_words = spoken_text.split()

        errors = []
        matcher = difflib.SequenceMatcher(None, expected_words, spoken_words)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace' and len(errors) < 3:  # Limit to 3 errors
                for idx in range(min(i2 - i1, j2 - j1)):
                    expected_word = expected_words[i1 + idx] if i1 + idx < len(expected_words) else ""
                    spoken_word = spoken_words[j1 + idx] if j1 + idx < len(spoken_words) else ""

                    if expected_word and spoken_word:
                        similarity = fuzz.ratio(expected_word, spoken_word)

                        if similarity < 60:  # Only major errors
                            errors.append({
                                "type": "substitution",
                                "expected_word": expected_word,
                                "spoken_word": spoken_word,
                                "feedback": f"'{expected_word}' → '{spoken_word}' cần sửa",
                                "severity": "high" if similarity < 40 else "medium"
                            })

        return {"errors": errors, "total_errors": len(errors)}

    def _generate_overall_feedback(self, accuracy_score: int, fluency_score: int) -> str:
        """Generate short overall feedback"""
        total_score = (accuracy_score + fluency_score) / 2

        if total_score >= 90:
            category = 'excellent'
        elif total_score >= 75:
            category = 'good'
        elif total_score >= 60:
            category = 'fair'
        else:
            category = 'needs_improvement'

        base_message = self.positive_messages[category]
        return f"{base_message} ({int(total_score)}/100)"

    def _identify_positive_aspects(self, accuracy_score: int, fluency_score: int,
                                   speech_rate: float, duration: float) -> List[str]:
        """Identify what the user did well - keep it concise"""
        positives = []

        if accuracy_score >= 85:
            positives.append("Phát âm rất chính xác!")
        elif accuracy_score >= 70:
            positives.append("Phần lớn từ đều đúng.")
        elif accuracy_score >= 50:
            positives.append("Cố gắng tốt!")

        if fluency_score >= 85:
            positives.append("Tốc độ tự nhiên!")
        elif fluency_score >= 70:
            positives.append("Nhịp độ ổn.")

        if 2.0 <= speech_rate <= 4.0:
            positives.append("Tốc độ phù hợp.")

        # Always include at least one positive
        if not positives:
            positives.append("Đã dũng cảm thử thách!")

        return positives[:2]  # Limit to top 2

    def _generate_improvement_suggestions(self, error_analysis: Dict, accuracy_score: int,
                                          fluency_score: int, speech_rate: float) -> List[str]:
        """Generate short improvement suggestions"""
        suggestions = []

        # Speed suggestions
        if speech_rate > 5.0:
            suggestions.append("Nói chậm hơn để rõ ràng.")
        elif speech_rate < 1.5:
            suggestions.append("Có thể nói nhanh hơn.")

        # Error-specific suggestions
        total_errors = error_analysis.get("total_errors", 0)
        if total_errors > 2:
            suggestions.append("Luyện từng từ riêng biệt.")
        elif accuracy_score < 70:
            suggestions.append("Nghe và bắt chước người bản ngữ.")
        elif fluency_score < 70:
            suggestions.append("Đọc to để tăng độ trôi chảy.")
        else:
            suggestions.append("Bạn cần luyện tập nhiều hơn.")

        return suggestions[:2]  # Max 2 suggestions

    def _generate_next_steps(self, accuracy_score: int, fluency_score: int) -> List[str]:
        """Generate concise next steps"""
        steps = []

        if accuracy_score < fluency_score:
            steps.append("Ưu tiên luyện phát âm chính xác.")
        else:
            steps.append("Luyện nói trôi chảy hơn.")

        steps.append("Luyện 10-15 phút mỗi ngày.")

        return steps


# Utility functions (unchanged)
def cleanup_temp_file(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logger.error(f"Failed to cleanup {file_path}: {e}")


def format_video_data(download_result, transcribe_result):
    video_data = {
        "videoId": download_result['video_id'],
        "title": download_result['title'],
        "youtubeUrl": download_result['youtube_url'],
        "duration": download_result.get('duration'),
        "language": transcribe_result['language'],
        "status": "PROCESSED",
        "segmentsCount": len(transcribe_result['segments'])
    }

    subtitles_data = []
    for segment in transcribe_result['segments']:
        subtitle = {
            "segmentId": segment['segment_id'],
            "startTime": float(segment['start']),
            "endTime": float(segment['end']),
            "originalText": segment['text'],
            "confidence": float(segment.get('confidence', 0.0))
        }
        subtitles_data.append(subtitle)

    return {"video": video_data, "subtitles": subtitles_data}


# Enhanced Pronunciation Scorer với SHORT Intelligent Feedback
class PronunciationScorer:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, 'initialized'):
            return

        self.device = "cpu"
        self.compute_type = "int8"

        self.transcribe_model = whisperx.load_model("tiny.en", device=self.device, compute_type=self.compute_type)
        self.align_model, self.align_metadata = whisperx.load_align_model(language_code="en", device=self.device)

        # Initialize SHORT feedback generator
        self.feedback_generator = FreeIntelligentFeedbackGenerator()
        self.initialized = True

    def process_audio(self, audio_path: str, expected_text: str = "") -> Dict:
        start_time = time.time()

        try:
            # Load and transcribe audio
            audio = whisperx.load_audio(audio_path)
            result = self.transcribe_model.transcribe(audio, batch_size=16, language="en")

            if not result.get("segments"):
                return {"error": "No speech detected", "success": False}

            # Align for word-level timestamps
            try:
                aligned_result = whisperx.align(result["segments"], self.align_model, self.align_metadata, audio,
                                                self.device, return_char_alignments=False)
                word_segments = aligned_result.get("word_segments", [])
            except:
                word_segments = self._extract_words_simple(result["segments"])

            if not word_segments:
                word_segments = self._extract_words_simple(result["segments"])

            # Calculate metrics in parallel
            duration = result["segments"][-1]["end"] if result["segments"] else 0
            with ThreadPoolExecutor(max_workers=3) as executor:
                fluency_future = executor.submit(self._calculate_fluency, word_segments, duration)
                accuracy_future = executor.submit(self._calculate_accuracy, word_segments, expected_text)

                fluency = fluency_future.result()
                accuracy = accuracy_future.result()

            spoken_text = " ".join([w.get("word", "") for w in word_segments])
            total_score = int((fluency * 0.35) + (accuracy * 0.65))
            speech_rate = len(word_segments) / duration if duration > 0 else 0

            # Generate SHORT intelligent feedback
            feedback_result = self.feedback_generator.generate_intelligent_feedback(
                expected_text, spoken_text, word_segments, fluency, accuracy, speech_rate, duration
            )

            if feedback_result["success"]:
                ai_feedback = feedback_result["feedback"]
                pronunciation_errors = ai_feedback.get("pronunciation_errors", [])

                # Compile SHORT feedback
                pronunciation_feedback = []
                if ai_feedback.get("overall_feedback"):
                    pronunciation_feedback.append(ai_feedback["overall_feedback"])

                # Add positive aspects (max 2)
                pronunciation_feedback.extend(ai_feedback.get("positive_aspects", [])[:2])

                # Add improvement suggestions (max 2)
                pronunciation_feedback.extend(ai_feedback.get("improvement_suggestions", [])[:2])

                # Add next steps (max 1)
                next_steps = ai_feedback.get("next_steps", [])
                if next_steps:
                    pronunciation_feedback.append(next_steps[0])

                # Remove empty feedback and limit total to 5 lines
                pronunciation_feedback = [f for f in pronunciation_feedback if f.strip()][:5]
            else:
                # Fallback SHORT feedback
                pronunciation_errors = []
                pronunciation_feedback = [
                    f"Điểm: {total_score}/100",
                    "Cố gắng tốt!",
                    "Tiếp tục luyện tập."
                ]

            return {
                "success": True,
                "transcription": spoken_text,
                "expected_text": expected_text,
                "fluency": fluency,
                "accuracy": accuracy,
                "total_score": total_score,
                "pronunciation_errors": pronunciation_errors,
                "pronunciation_feedback": pronunciation_feedback,
                "speech_rate": speech_rate,
                "duration": duration,
                "word_count": len(word_segments),
                "processing_time": time.time() - start_time
            }

        except Exception as e:
            return {"error": f"Error: {str(e)}", "success": False}
        finally:
            if 'audio' in locals():
                del audio
            gc.collect()

    def _extract_words_simple(self, segments: List[Dict]) -> List[Dict]:
        word_segments = []
        for segment in segments:
            words = segment["text"].strip().split()
            if not words:
                continue
            duration = segment["end"] - segment["start"]
            word_duration = duration / len(words)
            for i, word in enumerate(words):
                start_time = segment["start"] + (i * word_duration)
                word_segments.append({
                    "word": word,
                    "start": start_time,
                    "end": start_time + word_duration,
                    "score": segment.get("score", 0.8)
                })
        return word_segments

    def _calculate_fluency(self, word_segments: List[Dict], duration: float) -> int:
        if not word_segments or duration <= 0:
            return 0
        if len(word_segments) < 2:
            return 85

        speech_rate = len(word_segments) / duration
        if 2.0 <= speech_rate <= 4.5:
            return 95
        elif 1.5 <= speech_rate <= 5.5:
            return 85
        elif 1.0 <= speech_rate <= 6.0:
            return 75
        else:
            return 65

    def _calculate_accuracy(self, word_segments: List[Dict], expected_text: str) -> int:
        """Enhanced accuracy calculation - ignores punctuation"""
        if not expected_text:
            return 90

        # Remove punctuation for comparison
        translator = str.maketrans('', '', string.punctuation)

        expected_words = expected_text.lower().translate(translator).split()
        spoken_words = [w["word"].lower().translate(translator).strip() for w in word_segments]
        spoken_words = [w for w in spoken_words if w]  # Remove empty strings

        if not expected_words:
            return 0

        total_matches = 0
        for expected_word in expected_words:
            best_match_score = 0
            for spoken_word in spoken_words:
                fuzz_score = fuzz.ratio(expected_word, spoken_word)

                # Phonetic similarity for better matching
                try:
                    phonetic_similarity = self._phonetic_similarity(expected_word, spoken_word)
                    combined_score = max(fuzz_score, phonetic_similarity * 100)
                except:
                    combined_score = fuzz_score

                if combined_score > best_match_score:
                    best_match_score = combined_score

            # Accept 70% similarity
            if best_match_score > 70:
                total_matches += 1

        return min(100, int((total_matches / len(expected_words)) * 100))

    def _phonetic_similarity(self, word1: str, word2: str) -> float:
        """Simple phonetic similarity calculation"""
        try:
            from jellyfish import soundex
            soundex1 = soundex(word1) if word1 else ""
            soundex2 = soundex(word2) if word2 else ""
            return 1.0 if soundex1 == soundex2 and soundex1 else 0.0
        except:
            # Fallback to edit distance
            import difflib
            return difflib.SequenceMatcher(None, word1, word2).ratio()


# YouTube Processor Class (unchanged)
class YouTubeProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if torch.cuda.is_available() else "int8"

    def download_audio(self, youtube_url):
        strategies = ['ios', 'android', 'web_simple']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        for strategy in strategies:
            try:
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': os.path.join(UPLOAD_FOLDER, f'{timestamp}_%(title)s.%(ext)s'),
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'wav',
                        'preferredquality': '192',
                    }],
                    'no_warnings': True,
                    'socket_timeout': 30,
                    'retries': 1,
                }

                # Strategy-specific configs
                if strategy == 'ios':
                    ydl_opts.update({
                        'user_agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15',
                        'extractor_args': {'youtube': {'player_client': ['ios']}}
                    })
                elif strategy == 'android':
                    ydl_opts.update({
                        'user_agent': 'Mozilla/5.0 (Linux; Android 13) AppleWebKit/537.36',
                        'extractor_args': {'youtube': {'player_client': ['android']}}
                    })

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(youtube_url, download=False)
                    title = info.get('title', 'Unknown Title')
                    video_id = info.get('id', 'unknown')
                    duration = info.get('duration', 0)
                    language = info.get('language', 'en')

                    ydl.download([youtube_url])

                    # Find downloaded file
                    audio_file = None
                    for file in os.listdir(UPLOAD_FOLDER):
                        if file.startswith(timestamp) and file.endswith('.wav'):
                            audio_file = os.path.join(UPLOAD_FOLDER, file)
                            break

                    if audio_file and os.path.exists(audio_file) and os.path.getsize(audio_file) > 1000:
                        return {
                            'success': True,
                            'audio_file': audio_file,
                            'title': title,
                            'video_id': video_id,
                            'duration': duration,
                            'language': language,
                            'youtube_url': youtube_url
                        }

            except Exception as e:
                logger.error(f"Strategy {strategy} failed: {str(e)}")
                continue

        return {'success': False, 'error': 'Unable to download video'}

    def transcribe_audio(self, audio_file):
        try:
            model = whisperx.load_model("base", device=self.device, compute_type=self.compute_type)

            audio = whisperx.load_audio(audio_file)
            result = model.transcribe(audio, batch_size=16)
            detected_language = result.get("language", "en")

            segments = result.get("segments", [])

            # Align segments
            try:
                model_a, metadata = whisperx.load_align_model(language_code=detected_language, device=self.device)
                result_aligned = whisperx.align(segments, model_a, metadata, audio, self.device,
                                                return_char_alignments=False)
                segments = result_aligned.get("segments", segments)
                del model_a
            except Exception as e:
                logger.warning(f"Alignment failed: {e}")

            del model
            gc.collect()

            # Format segments
            valid_segments = [
                {
                    'start': seg.get('start', 0),
                    'end': seg.get('end', 0),
                    'text': seg.get('text', '').strip(),
                    'confidence': seg.get('confidence', 0.0),
                    'segment_id': i
                }
                for i, seg in enumerate(segments) if seg.get('text', '').strip()
            ]

            if not valid_segments:
                return {'success': False, 'error': 'No transcribable content found'}

            return {'success': True, 'segments': valid_segments, 'language': detected_language}

        except Exception as e:
            return {'success': False, 'error': str(e)}


# Global instances
scorer = None
youtube_processor = None


@app.on_event("startup")
async def startup_event():
    global scorer, youtube_processor
    logger.info("Initializing SHORT feedback services...")
    scorer = PronunciationScorer()
    youtube_processor = YouTubeProcessor()
    logger.info("SHORT feedback services initialized!")


# Endpoints
@app.get("/")
async def root():
    return {"message": "FREE AI Pronunciation Service - Short Feedback", "status": "healthy", "version": "3.1.0"}


@app.post("/score-pronunciation")
async def score_pronunciation(
        audio_file: UploadFile = File(...),
        expected_text: str = Form(...)
):
    try:
        if not scorer:
            return {"success": False, "error": "Service not initialized"}

        logger.info(f"Received file: {audio_file.filename}, Content-Type: {audio_file.content_type}")

        # Basic validation
        if not audio_file or not audio_file.filename:
            return {"success": False, "error": "No file provided"}

        if not expected_text or not expected_text.strip():
            return {"success": False, "error": "Expected text is required"}

        # Validate file extension
        allowed_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}
        filename = audio_file.filename.lower()
        file_extension = None

        for ext in allowed_extensions:
            if filename.endswith(ext):
                file_extension = ext
                break

        if not file_extension:
            return {"success": False, "error": f"Unsupported format. Allowed: {', '.join(allowed_extensions)}"}

        temp_audio_path = None
        try:
            logger.info("Reading audio file content...")
            content = await audio_file.read()

            if len(content) == 0:
                return {"success": False, "error": "Empty file"}

            if len(content) > 10 * 1024 * 1024:  # 10MB limit
                return {"success": False, "error": "File too large. Max: 10MB"}

            logger.info(f"File size: {len(content)} bytes")

            # Create temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension, mode='wb') as temp_file:
                temp_file.write(content)
                temp_file.flush()
                temp_audio_path = temp_file.name

            logger.info(f"Processing audio with SHORT feedback: {temp_audio_path}")

            # Process audio with SHORT feedback
            result = scorer.process_audio(temp_audio_path, expected_text.strip())

            # Cleanup temp file
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
                temp_audio_path = None

            # Ensure all values are JSON serializable
            response = {
                "success": bool(result.get("success", False)),
                "transcription": str(result.get("transcription", "")),
                "expected_text": str(expected_text.strip()),
                "fluency": int(result.get("fluency", 0)),
                "accuracy": int(result.get("accuracy", 0)),
                "total_score": int(result.get("total_score", 0)),
                "pronunciation_errors": list(result.get("pronunciation_errors", [])),
                "pronunciation_feedback": [str(f) for f in result.get("pronunciation_feedback", [])],
                "speech_rate": float(result.get("speech_rate", 0.0)),
                "duration": float(result.get("duration", 0.0)),
                "word_count": int(result.get("word_count", 0)),
                "processing_time": float(result.get("processing_time", 0.0)),
                "error": str(result.get("error")) if result.get("error") else None
            }

            logger.info("Processing completed successfully with SHORT feedback")
            return response

        except Exception as processing_error:
            logger.error(f"Processing error: {str(processing_error)}")

            # Cleanup temp file on error
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass

            return {
                "success": False,
                "error": f"Audio processing failed: {str(processing_error)}",
                "transcription": "",
                "expected_text": str(expected_text),
                "fluency": 0,
                "accuracy": 0,
                "total_score": 0,
                "pronunciation_errors": [],
                "pronunciation_feedback": ["Có lỗi, vui lòng thử lại."],
                "speech_rate": 0.0,
                "duration": 0.0,
                "word_count": 0,
                "processing_time": 0.0
            }

    except Exception as outer_error:
        logger.error(f"Outer error: {str(outer_error)}")
        return {
            "success": False,
            "error": f"Request failed: {str(outer_error)}",
            "transcription": "",
            "expected_text": "",
            "fluency": 0,
            "accuracy": 0,
            "total_score": 0,
            "pronunciation_errors": [],
            "pronunciation_feedback": ["Lỗi hệ thống, thử lại."],
            "speech_rate": 0.0,
            "duration": 0.0,
            "word_count": 0,
            "processing_time": 0.0
        }


@app.post("/process-video", response_model=VideoProcessResponse)
async def process_video(request: VideoProcessRequest):
    if not youtube_processor:
        raise HTTPException(status_code=500, detail="Service not initialized")

    try:
        # Download audio
        download_result = youtube_processor.download_audio(request.url)
        if not download_result['success']:
            raise HTTPException(status_code=400, detail=download_result['error'])

        # Transcribe audio
        transcribe_result = youtube_processor.transcribe_audio(download_result['audio_file'])

        # Cleanup
        cleanup_temp_file(download_result['audio_file'])

        if not transcribe_result['success']:
            raise HTTPException(status_code=500, detail=transcribe_result['error'])

        # Format response
        formatted_data = format_video_data(download_result, transcribe_result)

        return VideoProcessResponse(success=True, data=formatted_data)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cleanup-temp")
async def cleanup_temp_files():
    try:
        files_removed = 0
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                files_removed += 1
        return {"success": True, "files_removed": files_removed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Setup ngrok tunnel
    public_url = ngrok.connect(8000, domain="satyr-dashing-officially.ngrok-free.app")
    print("Public URL:", public_url)

    # Start server
    uvicorn.run(app, host="0.0.0.0", port=8000)