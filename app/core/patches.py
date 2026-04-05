import os
import sys
import torch
import torchaudio
import imageio_ffmpeg
import logging
import warnings
from types import ModuleType

# 0. Silence noisy logs and warnings
logging.basicConfig(level=logging.WARNING)
for logger_name in ["whisperx", "faster_whisper", "pytorch_lightning", "speechbrain", "transformers"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=UserWarning, module="inspect")
warnings.filterwarnings("ignore", message=".*speechbrain.*deprecated.*")
warnings.filterwarnings("ignore", message=".*Lightning automatically upgraded.*")
os.environ['PYTORCH_LIGHTNING_UTILITIES_WARNINGS'] = '0'

# 1. Windows Subprocess Mocks (Prevent k2/speechbrain crashes)
if "k2" not in sys.modules:
    sys.modules["k2"] = ModuleType("k2")

sb_mods = [
    "speechbrain.integrations.k2_fsa",
    "speechbrain.integrations.huggingface",
    "speechbrain.integrations.huggingface.wordemb",
    "speechbrain.integrations.nlp",
    "speechbrain.integrations.numba",
    "speechbrain.integrations.numba.transducer_loss"
]
for mod in sb_mods:
    if mod not in sys.modules:
        sys.modules[mod] = ModuleType(mod)

# 2. FFmpeg PATH Registration
_FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
_FFMPEG_DIR = os.path.dirname(_FFMPEG_EXE)
if _FFMPEG_DIR not in os.environ.get("PATH", ""):
    os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + _FFMPEG_DIR

# 3. Torch Compatibility Patch (Allow non-weights_only loading)
_orig_load = torch.load
def _patched_load(*a, **kw):
    kw["weights_only"] = False
    return _orig_load(*a, **kw)
torch.load = _patched_load

# 4. Torchaudio AudioMetaData compatibility
if not hasattr(torchaudio, "AudioMetaData"):
    try:
        from torchaudio.backend.common import AudioMetaData
        torchaudio.AudioMetaData = AudioMetaData
    except ImportError:
        from dataclasses import dataclass
        @dataclass
        class _AM:
            sample_rate: int = 0
            num_frames: int = 0
            num_channels: int = 0
            bits_per_sample: int = 0
            encoding: str = ""
        torchaudio.AudioMetaData = _AM

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

print("[PATCHES] Environment stabilization applied.")
