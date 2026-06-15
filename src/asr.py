import config
import random

class SpeechToText:
    def __init__(self):
        if config.MOCK_MODE:
            print("[ASR] [MOCK] Загрузка модели заглушки...")
            self.model = None
        else:
            import gigaam
            # Безопасно сопоставляем имя устройства для torch/gigaam
            device = "cuda" if "cuda" in getattr(config, "DEVICE_STR", "") else "cpu"
            print(f"[ASR] Загрузка модели {config.ASR_MODEL_NAME} на device={device}...")
            self.model = gigaam.load_model(config.ASR_MODEL_NAME, device=device, fp16_encoder=(device == "cuda"))
            print("[ASR] Готово.")

    def transcribe(self, audio_path):
        if not audio_path:
            return ""
        
        if config.MOCK_MODE:
            mock_texts = [
                "Здравствуйте! Я звоню по поводу блокировки карты, мне кажется, с неё списали деньги без моего ведома. Я очень переживаю!",
                "Добрый день. Я хотел бы узнать условия по вашему новому кредиту. Подскажите процентную ставку.",
                "Да вы издеваетесь?! Я уже полчаса жду ответа оператора! Сколько можно меня переключать?",
                "Алло, здравствуйте. Да, спасибо, всё понятно. До свидания.",
                "Нет, мне не интересны ваши предложения. Хватит мне звонить, пожалуйста."
            ]
            # Выберем детерминированно или случайно? 
            # Для демо-интерфейса случайный выбор делает приложение «живым»
            return random.choice(mock_texts)
        
        res = self.model.transcribe(audio_path)
        # GigaAM v3 возвращает объект TranscriptionResult, извлекаем из него текст
        if hasattr(res, "text"):
            return res.text
        return str(res)

