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
                "features": {},
                "segments": []
            }
            
        import numpy as np
        import os
        from .diarizer import diarize_audio
        
        # Сначала извлекаем общие акустические характеристики аудиофайла
        total_features = self.audio_ai.extract_acoustic_features(audio_path)
        
        if config.MOCK_MODE:
            # Заглушечный режим с имитацией реального диалога двух спикеров
            segments = diarize_audio(audio_path)
            
            mock_dialogue = [
                ("Оператор (Спикер А)", "Добрый день! Газпромбанк, меня зовут Александр. Чем я могу вам помочь?", 0.10, 0.05),
                ("Клиент (Спикер Б)", "Здравствуйте. У меня заблокировали перевод средств. Я очень переживаю, там важный платеж!", 0.65, 0.45),
                ("Оператор (Спикер А)", "Понимаю ваше беспокойство, извините. Пожалуйста, не волнуйтесь, мы сейчас всё решим.", 0.15, 0.10),
                ("Клиент (Спикер Б)", "Да сколько можно решать?! Я уже полчаса жду! Это бред какой-то, верните мои деньги!", 0.90, 0.85),
                ("Оператор (Спикер А)", "Приношу глубочайшие извинения за задержку. Подскажите номер вашего договора, я проверю прямо сейчас.", 0.20, 0.15)
            ]
            
            analyzed_segments = []
            full_texts = []
            
            for i, seg in enumerate(segments):
                dialogue_idx = i % len(mock_dialogue)
                speaker_role, text, audio_stress, text_stress = mock_dialogue[dialogue_idx]
                
                # Сопоставляем имена спикеров
                diarizer_speaker = seg.get("speaker", "Спикер A")
                final_speaker = "Оператор (Спикер А)" if diarizer_speaker == "Спикер A" else "Клиент (Спикер Б)"
                
                final_stress = 0.4 * text_stress + 0.6 * audio_stress
                
                analyzed_segments.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "speaker": final_speaker,
                    "text": text,
                    "audio_stress": audio_stress,
                    "text_stress": text_stress,
                    "final_stress": round(final_stress, 2)
                })
                full_texts.append(f"[{final_speaker}]: {text}")
                
            if not analyzed_segments:
                # На случай если тишина или не распозналось ни одного сегмента
                analyzed_segments.append({
                    "start": 0.0,
                    "end": total_features.get("duration", 5.0),
                    "speaker": "Оператор (Спикер А)",
                    "text": "Алло, здравствуйте. Вас приветствует Газпромбанк. Подскажите ваш вопрос.",
                    "audio_stress": 0.15,
                    "text_stress": 0.10,
                    "final_stress": 0.12
                })
                full_texts.append("[Оператор (Спикер А)]: Алло, здравствуйте. Вас приветствует Газпромбанк. Подскажите ваш вопрос.")
                
            avg_final_stress = np.mean([s["final_stress"] for s in analyzed_segments])
            avg_text_stress = np.mean([s["text_stress"] for s in analyzed_segments])
            avg_audio_stress = np.mean([s["audio_stress"] for s in analyzed_segments])
            
            return {
                "transcription": "\n".join(full_texts),
                "text_stress": round(float(avg_text_stress), 2),
                "audio_stress": round(float(avg_audio_stress), 2),
                "final_stress": round(float(avg_final_stress), 2),
                "features": total_features,
                "segments": analyzed_segments
            }
            
        # Реальный режим с инференсом нейросетей
        print("[Pipeline] Запуск диаризации аудио...")
        segments = diarize_audio(audio_path)
        
        import tempfile
        import soundfile as sf
        
        analyzed_segments = []
        full_texts = []
        
        # Определяем роутинг
        asr_route = config.ROUTING.get("asr", "local")
        text_route = config.ROUTING.get("text", "local")
        audio_route = config.ROUTING.get("audio", "local")
        
        for seg in segments:
            seg_y = seg["y"]
            start = seg["start"]
            end = seg["end"]
            diarizer_speaker = seg.get("speaker", "Спикер A")
            
            # Сохраняем сегмент во временный файл
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
                temp_wav_path = temp_wav.name
                
            try:
                sf.write(temp_wav_path, seg_y, 16000)
                
                # 1. Шаг ASR
                if asr_route == "local":
                    chunk_text = self.asr.transcribe(temp_wav_path)
                else:
                    chunk_text = network.remote_asr(asr_route, temp_wav_path)
                    if "[Ошибка связи с удаленным узлом" in chunk_text and config.FAILOVER_TO_LOCAL:
                        chunk_text = self.asr.transcribe(temp_wav_path)
                
                if not chunk_text.strip():
                    continue
                    
                # 2. Шаг Анализа Текста
                if text_route == "local":
                    text_res = self.text_ai.analyze(chunk_text)
                else:
                    text_res = network.remote_text_analysis(text_route, chunk_text)
                    if text_res.get("error") and config.FAILOVER_TO_LOCAL:
                        text_res = self.text_ai.analyze(chunk_text)
                
                text_stress = text_res.get("stress", 0.0)
                
                # 3. Шаг Анализа Звука
                if audio_route == "local":
                    audio_res = self.audio_ai.analyze(temp_wav_path)
                else:
                    audio_res = network.remote_audio_analysis(audio_route, temp_wav_path)
                    if audio_res.get("error") and config.FAILOVER_TO_LOCAL:
                        audio_res = self.audio_ai.analyze(temp_wav_path)
                        
                audio_stress = audio_res.get("stress", 0.0)
                
                # Вычисление итогового стресса сегмента
                seg_final_stress = 0.4 * text_stress + 0.6 * audio_stress
                
                analyzed_segments.append({
                    "start": start,
                    "end": end,
                    "speaker": diarizer_speaker,
                    "text": chunk_text,
                    "audio_stress": round(audio_stress, 2),
                    "text_stress": round(text_stress, 2),
                    "final_stress": round(seg_final_stress, 2)
                })
                full_texts.append(f"[{diarizer_speaker}]: {chunk_text}")
            finally:
                try:
                    os.unlink(temp_wav_path)
                except Exception:
                    pass
                    
        # Если сегментов не было или они пустые, делаем фоллбэк на целый файл
        if not analyzed_segments:
            print("[Pipeline] [WARNING] Реплики не обнаружены. Выполняем анализ целого файла.")
            # 1. Шаг ASR
            if asr_route == "local":
                text = self.asr.transcribe(audio_path)
            else:
                text = network.remote_asr(asr_route, audio_path)
                if "[Ошибка связи с удаленным узлом" in text and config.FAILOVER_TO_LOCAL:
                    text = self.asr.transcribe(audio_path)
                    
            # 2. Шаг Анализа Текста
            if text_route == "local":
                text_res = self.text_ai.analyze(text)
            else:
                text_res = network.remote_text_analysis(text_route, text)
                if text_res.get("error") and config.FAILOVER_TO_LOCAL:
                    text_res = self.text_ai.analyze(text)
            text_stress = text_res.get("stress", 0.0)
            
            # 3. Шаг Анализа Звука
            if audio_route == "local":
                audio_res = self.audio_ai.analyze(audio_path)
            else:
                audio_res = network.remote_audio_analysis(audio_route, audio_path)
                if audio_res.get("error") and config.FAILOVER_TO_LOCAL:
                    audio_res = self.audio_ai.analyze(audio_path)
            audio_stress = audio_res.get("stress", 0.0)
            
            final_stress = 0.4 * text_stress + 0.6 * audio_stress
            
            analyzed_segments.append({
                "start": 0.0,
                "end": total_features.get("duration", 5.0),
                "speaker": "Спикер A",
                "text": text,
                "audio_stress": round(audio_stress, 2),
                "text_stress": round(text_stress, 2),
                "final_stress": round(final_stress, 2)
            })
            full_texts.append(f"[Спикер A]: {text}")
            
        avg_final_stress = np.mean([s["final_stress"] for s in analyzed_segments])
        avg_text_stress = np.mean([s["text_stress"] for s in analyzed_segments])
        avg_audio_stress = np.mean([s["audio_stress"] for s in analyzed_segments])
        
        return {
            "transcription": "\n".join(full_texts),
            "text_stress": round(float(avg_text_stress), 2),
            "audio_stress": round(float(avg_audio_stress), 2),
            "final_stress": round(float(avg_final_stress), 2),
            "features": total_features,
            "segments": analyzed_segments
        }
