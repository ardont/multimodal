from .asr import SpeechToText
from .text_analyzer import TextEmotionAnalyzer
from .audio_analyzer import AudioEmotionAnalyzer
from . import network
import config

class MultimodalPipeline:
    def __init__(self):
        # Инициализируем локальные модели.
        # Они лениво подгружаются или создают заглушки в зависимости от config.MOCK_MODE.
        self.asr = SpeechToText()
        self.text_ai = TextEmotionAnalyzer()
        self.audio_ai = AudioEmotionAnalyzer()

    def run_analysis(self, audio_path):
        if not audio_path:
            return {
                "transcription": "",
                "text_stress": 0.0,
                "audio_stress": 0.0,
                "final_stress": 0.0,
                "features": {}
            }
            
        # 1. Шаг ASR (Распознавание речи)
        asr_route = config.ROUTING.get("asr", "local")
        if asr_route == "local":
            print("[Pipeline] Выполнение ASR локально...")
            text = self.asr.transcribe(audio_path)
        else:
            print(f"[Pipeline] Перенаправление ASR на удаленный узел: {asr_route}...")
            text = network.remote_asr(asr_route, audio_path)
            
        # 2. Шаг Анализа Текста
        text_route = config.ROUTING.get("text", "local")
        if text_route == "local":
            print("[Pipeline] Выполнение анализа текста локально...")
            text_res = self.text_ai.analyze(text)
        else:
            print(f"[Pipeline] Перенаправление анализа текста на удаленный узел: {text_route}...")
            text_res = network.remote_text_analysis(text_route, text)
            
        # 3. Шаг Анализа Звука
        audio_route = config.ROUTING.get("audio", "local")
        if audio_route == "local":
            print("[Pipeline] Выполнение анализа звука локально...")
            audio_res = self.audio_ai.analyze(audio_path)
        else:
            print(f"[Pipeline] Перенаправление анализа звука на удаленный узел: {audio_route}...")
            audio_res = network.remote_audio_analysis(audio_route, audio_path)
            
        # 4. Формула слияния результатов (Late Fusion)
        final_stress = (0.4 * text_res.get("stress", 0.0)) + (0.6 * audio_res.get("stress", 0.0))
        
        return {
            "transcription": text,
            "text_stress": round(text_res.get("stress", 0.0), 2),
            "audio_stress": round(audio_res.get("stress", 0.0), 2),
            "final_stress": round(final_stress, 2),
            "features": audio_res.get("features", {})
        }
