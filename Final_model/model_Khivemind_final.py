import io
import os
import tempfile
import warnings

import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from PIL import Image
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# 1. 설정값
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_instrument_cnn_model.keras")

IMG_HEIGHT = 224
IMG_WIDTH = 224

SR = 32768
DURATION = 1.5
N_FFT = 4096
HOP_LENGTH = 128
N_MELS = 126

CLASS_INDEX_TO_NAME = {
    0: "Bee",
    1: "Hornet",
    2: "Normal",
}

# 2. 모델 로드
model = tf.keras.models.load_model(MODEL_PATH)

# 하이패스필터 적용
def highpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist

    b, a = butter(order, normal_cutoff, btype='high', analog=False)

    if len(data) <= max(len(a), len(b)) * 3:
        return data

    filtered_data = filtfilt(b, a, data)
    return filtered_data


# 3. 오디오 로드
def load_audio_from_bytes(audio_bytes: bytes, sr=SR, duration=DURATION):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        y, loaded_sr = librosa.load(tmp_path, sr=sr, mono=True)
        target_length = int(loaded_sr * duration)

        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode="constant")
        else:
            y = y[:target_length]

        return y, loaded_sr
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
# 4. 스펙트로그램 이미지 생성
def make_enhanced_spectrogram_array_from_waveform(
    y,
    sr,
    emphasize_ranges=None,
    band_weights=None,
    peak_boost=True,
    peak_boost_strength=1.25,
    peak_prominence_ratio=0.05,
    non_target_attenuation=0.85,
):
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )

    mel_freqs = librosa.mel_frequencies(n_mels=N_MELS, fmin=0, fmax=sr / 2)
    enhanced_spec = mel_spec.copy()

    if emphasize_ranges is not None:
        enhanced_spec *= non_target_attenuation
        for fmin, fmax, gain in emphasize_ranges:
            idx = np.where((mel_freqs >= fmin) & (mel_freqs < fmax))[0]
            if len(idx) > 0:
                enhanced_spec[idx, :] = mel_spec[idx, :] * gain

    if band_weights is not None:
        for fmin, fmax, weight in band_weights:
            idx = np.where((mel_freqs >= fmin) & (mel_freqs < fmax))[0]
            if len(idx) > 0:
                enhanced_spec[idx, :] *= weight

    if peak_boost:
        freq_profile = np.mean(enhanced_spec, axis=1)
        if np.max(freq_profile) > 0:
            peaks, _ = find_peaks(
                freq_profile,
                prominence=np.max(freq_profile) * peak_prominence_ratio
            )
            for p in peaks:
                left = max(0, p - 1)
                right = min(len(freq_profile), p + 2)
                enhanced_spec[left:right, :] *= peak_boost_strength

    enhanced_spec = np.maximum(enhanced_spec, 1e-10)
    mel_spec_db = librosa.power_to_db(enhanced_spec, ref=np.max)

    fig = plt.figure(figsize=(4, 4))
    librosa.display.specshow(
        mel_spec_db,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis=None,
        y_axis=None,
        cmap="viridis"
    )
    plt.axis("off")
    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))

    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# 5. 예측 함수
def predict_audio_bytes(audio_bytes: bytes) -> dict:
    y, sr = load_audio_from_bytes(audio_bytes, sr=SR, duration=DURATION)

    # 1) Normal 후보: 필터 없음
    y_normal = y.copy()

    # 2) Hornet 후보: cutoff 90
    y_hornet = highpass_filter(y.copy(), cutoff=90, fs=sr)

    # 3) Bee 후보: cutoff 150
    y_bee = highpass_filter(y.copy(), cutoff=150, fs=sr)

    common_kwargs = {
        "emphasize_ranges": [(90, 300, 1.3), (300, 1200, 1.15)],
        "band_weights": [(90,150,1.4), (150,300,1.3), (300,600,1.2), (600,1200,1.1), (1200,4000,0.9)],
        "peak_boost": True,
        "peak_boost_strength": 1.25,
        "peak_prominence_ratio": 0.05,
        "non_target_attenuation": 0.85,
    }

    x_normal = make_enhanced_spectrogram_array_from_waveform(y_normal, sr, **common_kwargs)
    x_hornet = make_enhanced_spectrogram_array_from_waveform(y_hornet, sr, **common_kwargs)
    x_bee = make_enhanced_spectrogram_array_from_waveform(y_bee, sr, **common_kwargs)

    p_normal = model.predict(x_normal, verbose=0)[0]
    p_hornet = model.predict(x_hornet, verbose=0)[0]
    p_bee = model.predict(x_bee, verbose=0)[0]

    # 클래스 인덱스는 실제 학습 순서에 맞게 수정
    idx_bee = 0
    idx_hornet = 1
    idx_normal = 2

    candidate_scores = {
        "Bee": float(p_bee[idx_bee]),
        "Hornet": float(p_hornet[idx_hornet]),
        "Normal": float(p_normal[idx_normal]),
    }

    predicted_class = max(candidate_scores, key=candidate_scores.get)
    confidence = candidate_scores[predicted_class]

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "candidate_scores": candidate_scores,
    }