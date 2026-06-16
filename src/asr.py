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
            return random.choice(mock_texts)
        
        res = self.model.transcribe(audio_path)
        if hasattr(res, "text"):
            return res.text
        return str(res)

    def transcribe_with_timestamps(self, audio_path):
        """
        Распознает весь аудиофайл целиком и возвращает словарь с полным текстом
        и списком слов с их таймкодами (start, end).
        """
        if not audio_path:
            return {"text": "", "words": []}

        if config.MOCK_MODE:
            text = self.transcribe(audio_path)
            # Генерируем искусственные таймкоды для слов в Mock-режиме
            words = []
            split_words = text.split()
            w_dur = 4.0 / max(len(split_words), 1)
            for i, w in enumerate(split_words):
                words.append({
                    "word": w.strip(".,!?"),
                    "start": round(i * w_dur, 2),
                    "end": round((i + 1) * w_dur, 2)
                })
            return {"text": text, "words": words}

        # Реальный инференс GigaAM
        res = self.model.transcribe(audio_path)
        text = res.text if hasattr(res, "text") else str(res)
        
        words = []
        if hasattr(res, "words") and res.words:
            for w in res.words:
                words.append({
                    "word": getattr(w, "word", ""),
                    "start": getattr(w, "start_time", 0.0),
                    "end": getattr(w, "end_time", 0.0)
                })
        else:
            # Надежный откат: аппроксимируем таймкоды на основе длины аудио
            try:
                import soundfile as sf
                info = sf.info(audio_path)
                duration = info.duration
            except Exception:
                duration = 5.0
                
            split_words = text.split()
            if split_words:
                w_dur = duration / len(split_words)
                for i, w in enumerate(split_words):
                    words.append({
                        "word": w.strip(".,!?"),
                        "start": round(i * w_dur, 2),
                        "end": round((i + 1) * w_dur, 2)
                    })
        return {"text": text, "words": words}

    def transcribe_batch(self, audio_paths):
        """Пакетное распознавание списка аудиофайлов."""
        if not audio_paths:
            return []
        if config.MOCK_MODE:
            return [self.transcribe(p) for p in audio_paths]

        try:
            results = self.model.transcribe(audio_paths)
            texts = []
            for res in results:
                if hasattr(res, "text"):
                    texts.append(res.text)
                else:
                    texts.append(str(res))
            return texts
        except Exception as e:
            print(f"[ASR] Ошибка пакетного распознавания: {e}. Переключаемся на последовательное.")
            return [self.transcribe(p) for p in audio_paths]

    def transcribe_batch_with_timestamps(self, audio_paths):
        """Пакетное распознавание с таймкодами слов."""
        if not audio_paths:
            return []
        # Последовательный запуск без перезапуска модели
        return [self.transcribe_with_timestamps(p) for p in audio_paths]


