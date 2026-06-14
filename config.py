import os

# Режим MOCK_MODE: True для работы с заглушками на слабом ПК без GPU, 
# False для полноценного инференса моделей на мощном ПК
MOCK_MODE = os.getenv("MOCK_MODE", "True").lower() in ("true", "1", "yes")

# Пытаемся безопасно импортировать torch для поддержки легкого окружения
try:
    import torch
    HAS_TORCH = True
    DEVICE = 0 if torch.cuda.is_available() else -1
    DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    HAS_TORCH = False
    DEVICE = -1
    DEVICE_STR = "cpu (torch не установлен)"

# Выбранные предобученные модели
# Для ASR (Распознавание речи)
ASR_MODEL_NAME = "openai/whisper-tiny"  # tiny быстрее загружается и меньше весит

# Для анализа эмоций по тексту (Sentiment/Emotion на русском)
# Заменили на открытую и стабильную публичную модель от blanchefort
TEXT_MODEL_NAME = "blanchefort/rubert-base-by-sentiment"

# Для анализа эмоций по аудио (Speech Emotion Recognition)
AUDIO_MODEL_NAME = "harshit345/xlsr-wav2vec-speech-emotion-recognition"

# --- СЕТЕВЫЕ НАСТРОЙКИ (Tailscale / VPN) ---
# Порт для запуска FastAPI + Gradio сервера
PORT = 7860

# Зарегистрированные вычислительные узлы (Воркеры) в сети VPN.
# Сюда можно динамически добавлять IP-адреса других ПК через веб-интерфейс.
# По умолчанию узел считает самого себя известным
KNOWN_NODES = ["127.0.0.1:7860"]

# Маршрутизация вычислений: "local" или адрес воркера (например, "100.111.22.33:7860")
# Позволяет перенаправлять тяжелые расчеты на GPU-серверы
ROUTING = {
    "asr": "local",
    "text": "local",
    "audio": "local"
}

print(f"[CONFIG] Загружен с параметрами: MOCK_MODE={MOCK_MODE}, DEVICE={DEVICE_STR}, ROUTING={ROUTING}")
