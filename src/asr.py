import config
import random

class SpeechToText:
    def __init__(self):
        if config.MOCK_MODE:
            print("[ASR] [MOCK] Загрузка модели заглушки...")
            self.pipe = None
        else:
            from transformers import pipeline
            print(f"[ASR] Загрузка модели {config.ASR_MODEL_NAME} на device={config.DEVICE}...")
            # Загружаем пайплайн ASR
            self.pipe = pipeline("automatic-speech-recognition", model=config.ASR_MODEL_NAME, device=config.DEVICE)
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
        
        result = self.pipe(audio_path)
        return result.get("text", "")
