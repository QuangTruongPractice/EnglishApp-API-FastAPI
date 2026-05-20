import os
import sys
import torch
import torch.serialization
import torchaudio
import imageio_ffmpeg
import logging
import warnings
from types import ModuleType

# Sửa lỗi DLL initialization failed (WinError 1114)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["KMP_BLOCKTIME"] = "0"

# Vá lỗi bảo mật Transformers (Yêu cầu Torch 2.6)
# Chúng ta vá trực tiếp vào module để cho phép nạp model ngay cả khi Torch < 2.6
try:
    import transformers.utils.import_utils as trans_utils
    trans_utils.check_torch_load_is_safe = lambda: None
except (ImportError, AttributeError):
    pass

# Cho phép nạp các cấu trúc dữ liệu của Omegaconf (Cần cho Pyannote/WhisperX trong Torch 2.6)
try:
    from omegaconf.listconfig import ListConfig
    from omegaconf.dictconfig import DictConfig
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([ListConfig, DictConfig])
except ImportError:
    pass

# ==========================================
# 1. TẮT CÁC CẢNH BÁO NHIỄU
# ==========================================
logging.basicConfig(level=logging.WARNING)
for logger_name in ["whisperx", "faster_whisper", "pytorch_lightning", "speechbrain", "transformers"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=UserWarning, module="inspect")
warnings.filterwarnings("ignore", message=".*speechbrain.*deprecated.*")
warnings.filterwarnings("ignore", message=".*Lightning automatically upgraded.*")
os.environ['PYTORCH_LIGHTNING_UTILITIES_WARNINGS'] = '0'

# ==========================================
# 2. GIẢ LẬP MODULE (TRÁNH CRASH TRÊN WINDOWS)
# ==========================================
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

# ==========================================
# 3. ĐĂNG KÝ FFMPEG VÀO PATH
# ==========================================
try:
    _FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
    _FFMPEG_DIR = os.path.dirname(_FFMPEG_EXE)
    if _FFMPEG_DIR not in os.environ.get("PATH", ""):
        os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + _FFMPEG_DIR
except:
    pass

# ==========================================
# 4. PATCH TƯƠNG THÍCH TORCH & AUDIO
# ==========================================
_orig_load = torch.load
def _patched_load(*a, **kw):
    # Ép buộc weights_only=False để nạp được các model cũ/phức tạp trong Torch 2.6
    kw["weights_only"] = False
    return _orig_load(*a, **kw)
torch.load = _patched_load

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

