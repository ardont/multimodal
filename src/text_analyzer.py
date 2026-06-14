import config

class TextEmotionAnalyzer:
    def __init__(self):
        if config.MOCK_MODE:
            print("[Text AI] [MOCK] Загрузка текстового анализатора...")
            self.pipe = None
        else:
            from transformers import pipeline
            print(f"[Text AI] Загрузка модели {config.TEXT_MODEL_NAME} на device={config.DEVICE}...")
            # top_k=None возвращает оценки для всех классов
            self.pipe = pipeline("text-classification", model=config.TEXT_MODEL_NAME, device=config.DEVICE, top_k=None)
            print("[Text AI] Готово.")

    def analyze(self, text):
        if not text or not text.strip():
            return {"stress": 0.0, "neutral": 1.0}
            
        if config.MOCK_MODE:
            # Реалистичные эвристики для текстового заглушечного режима
            text_lower = text.lower()
            if "блокиров" in text_lower or "списали" in text_lower or "переживаю" in text_lower:
                return {"stress": 0.75, "neutral": 0.25}
            elif "издеваетесь" in text_lower or "ждат" in text_lower or "сколько можно" in text_lower:
                return {"stress": 0.90, "neutral": 0.10}
            elif "условия" in text_lower or "кредит" in text_lower or "процент" in text_lower:
                return {"stress": 0.15, "neutral": 0.85}
            elif "хватит" in text_lower or "не интересны" in text_lower:
                return {"stress": 0.60, "neutral": 0.40}
            else:
                return {"stress": 0.10, "neutral": 0.90}
        
        # Инференс реальной модели
        outputs = self.pipe(text)
        # outputs имеет вид: [[{'label': 'negative', 'score': 0.8}, {'label': 'neutral', 'score': 0.15}, ...]]
        # или [{'label': 'negative', 'score': 0.8}, ...]
        if isinstance(outputs[0], list):
            outputs = outputs[0]
            
        scores = {item['label'].upper(): item['score'] for item in outputs}
        
        # Маппинг различных вариантов лейблов (NEGATIVE, ANGER, FEAR, SADNESS -> stress)
        # Поддерживаем как стандартный sentiment (NEGATIVE, NEUTRAL, POSITIVE), так и многоклассовые эмоции
        negative_keys = ['NEGATIVE', 'NEG', 'LABEL_2', 'ANGER', 'FEAR', 'SADNESS', 'DISGUST']
        neutral_keys = ['NEUTRAL', 'NEU', 'LABEL_0', 'POSITIVE', 'POS', 'LABEL_1', 'JOY', 'SURPRISE']
        
        stress_score = sum(scores.get(k, 0.0) for k in negative_keys)
        neutral_score = sum(scores.get(k, 0.0) for k in neutral_keys)
        
        # Нормализуем, чтобы сумма была 1.0 (на всякий случай)
        total = stress_score + neutral_score
        if total > 0:
            stress_score /= total
            neutral_score /= total
            
        return {"stress": float(stress_score), "neutral": float(neutral_score)}
