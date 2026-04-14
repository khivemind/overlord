import io
import os
import tempfile
import warnings

import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from PIL import Image

warnings.filterwarnings("ignore")

# 서버 환경에서 matplotlib GUI 안 쓰게 설정
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# 1. 설정값
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_instrument_cnn_model.keras")

IMG_HEIGHT = 224
IMG_WIDTH = 224

SR = 32768
DURATION = 1.5
N_FFT = 4096
HOP_LENGTH = 128
N_MELS = 126

# 중요:
# 아래 클래스 순서는 네 notebook에서 출력된
# train_generator.class_indices 기준으로 꼭 맞춰야 함.
# flow_from_directory는 보통 알파벳 순서로 잡는다.
#
# 예:
# {'Bee': 0, 'Hornet': 1, 'Normal': 2}
#
# 내 노트북 출력값 확인 후 수정
CLASS_INDEX_TO_NAME = {
    0: "Bee",
    1: "Hornet",
    2: "Normal",
}


# =========================
# 2. 모델 로드
# =========================
model = tf.keras.models.load_model(MODEL_PATH)


# =========================
# 3. 오디오 로드
# =========================
def load_audio_from_bytes(audio_bytes: bytes, sr: int = SR, duration: float = DURATION):
    """
    업로드된 오디오 바이트를 librosa로 읽고,
    학습 때와 동일하게 길이를 맞춘다.
    """
    if not audio_bytes:
        raise ValueError("비어 있는 오디오 데이터입니다.")

    # librosa는 파일 경로를 가장 안정적으로 처리하므로 임시 파일 사용
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        y, loaded_sr = librosa.load(tmp_path, sr=sr, mono=True)

        if y is None or len(y) == 0:
            raise ValueError("오디오 로드 결과가 비어 있습니다.")

        target_length = int(loaded_sr * duration)

        if len(y) < target_length:
            pad_width = target_length - len(y)
            y = np.pad(y, (0, pad_width), mode="constant")
        else:
            y = y[:target_length]

        return y, loaded_sr

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# =========================
# 4. 스펙트로그램 이미지 생성
# =========================
def make_spectrogram_image_array(audio_bytes: bytes) -> np.ndarray:
    """
    네 notebook의 save_spectogram_image 흐름처럼
    mel spectrogram을 이미지로 만든 뒤
    모델 입력용 numpy 배열로 변환한다.
    """
    y, sr = load_audio_from_bytes(audio_bytes, sr=SR, duration=DURATION)

    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # notebook처럼 그림 저장 -> 다시 이미지 입력 형태로 맞춤
    fig = plt.figure(figsize=(4, 4))
    librosa.display.specshow(
        mel_spec_db,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis=None,
        y_axis=None,
        cmap="viridis",
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
    img_array = np.expand_dims(img_array, axis=0)  # (1, H, W, C)

    return img_array


# =========================
# 5. 예측 함수
# =========================
def predict_audio_bytes(audio_bytes: bytes) -> dict:
    """
    업로드된 오디오를 받아 클래스와 확률을 반환
    """
    img_array = make_spectrogram_image_array(audio_bytes)

    probs = model.predict(img_array, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_INDEX_TO_NAME[pred_idx]

    probabilities = {
        CLASS_INDEX_TO_NAME[i]: float(probs[i]) for i in range(len(probs))
    }

    return {
        "predicted_class": pred_label,
        "confidence": float(probs[pred_idx]),
        "probabilities": probabilities,
    }