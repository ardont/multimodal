import os
import numpy as np
import config

class AudioEmotionAnalyzer:
    def __init__(self):
        if config.MOCK_MODE:
            print("[Audio AI] [MOCK] Загрузка аудио-анализатора...")
            self.model = None
        else:
            import gigaam
            # Безопасно сопоставляем имя устройства для torch/gigaam
            device = "cuda" if "cuda" in getattr(config, "DEVICE_STR", "") else "cpu"
            print(f"[Audio AI] Загрузка модели {config.AUDIO_MODEL_NAME} на device={device}...")
            self.model = gigaam.load_model(config.AUDIO_MODEL_NAME, device=device, fp16_encoder=(device == "cuda"))
            print("[Audio AI] Готово.")

    def extract_acoustic_features(self, audio_path):
        """
        Легковесное извлечение акустических признаков с помощью librosa.
        Работает и в Mock-режиме, так как не требует больших вычислительных ресурсов.
        """
        try:
            import librosa
            # Загружаем аудио (resample до 16кГц для консистентности)
            y, sr = librosa.load(audio_path, sr=16000)
            
            duration = librosa.get_duration(y=y, sr=sr)
            if duration == 0:
                return {
                    "duration": 0.0,
                    "loudness_mean": 0.0,
                    "loudness_max": 0.0,
                    "silence_ratio": 0.0,
                    "tempo_bpm": 0.0
                }
                
            # Энергия (громкость)
            rms = librosa.feature.rms(y=y)
            mean_rms = float(np.mean(rms))
            max_rms = float(np.max(rms))
            
            # Доля тишины (паузы в разговоре)
            # Считаем фреймы ниже -30dB от пикового значения
            db = librosa.amplitude_to_db(rms, ref=np.max)
            silence_ratio = float(np.sum(db < -30) / db.size)
            
            # Оценка темпа речи (BPM - ударов в минуту / примерная частота слогов)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            # Извлекаем значение темпа из массива/числа
            if isinstance(tempo, np.ndarray):
                mean_tempo = float(tempo[0]) if tempo.size > 0 else 120.0
            else:
                mean_tempo = float(tempo)
            
            return {
                "duration": round(duration, 2),
                "loudness_mean": round(mean_rms, 4),
                "loudness_max": round(max_rms, 4),
                "silence_ratio": round(silence_ratio, 2),
                "tempo_bpm": round(mean_tempo, 1)
            }
        except Exception as e:
            print(f"[Audio AI] Ошибка извлечения признаков: {e}")
            # Возвращаем разумные заглушки при ошибке
            return {
                "duration": 4.5,
                "loudness_mean": 0.03,
                "loudness_max": 0.08,
                "silence_ratio": 0.18,
                "tempo_bpm": 115.0
            }

    def analyze(self, audio_path):
        if not audio_path or not os.path.exists(audio_path):
            return {"stress": 0.0, "neutral": 1.0, "features": {}}
            
        # 1. Извлекаем легкие акустические признаки
        features = self.extract_acoustic_features(audio_path)
        
        # 2. Вычисляем эмоциональный скор
        if config.MOCK_MODE:
            # Эвристическая логика оценки стресса по звуку
            # Стресс повышается при:
            # - Высокой громкости (loudness_mean > 0.04)
            # - Быстром темпе (tempo_bpm > 130)
            # - Либо при большом количестве пауз/запинок от волнения (silence_ratio > 0.35)
            stress_score = 0.2  # базовый уровень стресса
            
            if features.get("loudness_mean", 0) > 0.04:
                stress_score += 0.25
            if features.get("tempo_bpm", 120) > 130:
                stress_score += 0.20
            if features.get("silence_ratio", 0) > 0.35:
                stress_score += 0.25
                
            # Ограничиваем [0.05, 0.95] для реалистичности
            stress_score = min(max(stress_score, 0.05), 0.95)
            neutral_score = 1.0 - stress_score
            
            return {
                "stress": round(stress_score, 2),
                "neutral": round(neutral_score, 2),
                "features": features
            }
            
        # Инференс реальной модели GigaAM-Emo
        scores = self.model.get_probs(audio_path)
        
        # Маппинг под категории стресса/аномалии
        # GigaAM-Emo возвращает: angry, sad, neutral, positive
        stress_keys = ['angry', 'sad']
        neutral_keys = ['neutral', 'positive']
        
        stress_score = sum(scores.get(k, 0.0) for k in stress_keys)
        neutral_score = sum(scores.get(k, 0.0) for k in neutral_keys)
        
        # Нормализация
        total = stress_score + neutral_score
        if total > 0:
            stress_score /= total
            neutral_score /= total
            
        return {
            "stress": round(float(stress_score), 2),
            "neutral": round(float(neutral_score), 2),
            "features": features
        }

    def reduce_noise(self, input_path, output_path):
        """
        Спектральное вычитание шума (Spectral Subtraction) для очистки звука.
        Читает input_path, очищает сигнал и записывает в output_path.
        """
        try:
            import soundfile as sf
            import scipy.signal
            
            y, sr = sf.read(input_path)
            # Если стерео, усредняем в моно
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
                
            # Параметры STFT
            nperseg = 1024
            noverlap = 512
            f, t, Zxx = scipy.signal.stft(y, fs=sr, nperseg=nperseg, noverlap=noverlap)
            
            # Шумовой профиль берем из первых 0.25 сек
            noise_frames = int(0.25 * sr / (nperseg - noverlap))
            if Zxx.shape[1] > noise_frames and noise_frames > 0:
                noise_profile = np.mean(np.abs(Zxx[:, :noise_frames]), axis=1, keepdims=True)
            else:
                noise_profile = np.mean(np.abs(Zxx), axis=1, keepdims=True) * 0.1
                
            # Спектральное вычитание
            Zxx_mag = np.abs(Zxx)
            Zxx_phase = np.angle(Zxx)
            
            clean_mag = Zxx_mag - 1.5 * noise_profile
            clean_mag = np.maximum(clean_mag, 0.02 * Zxx_mag)
            
            clean_Zxx = clean_mag * np.exp(1j * Zxx_phase)
            _, clean_y = scipy.signal.istft(clean_Zxx, fs=sr, nperseg=nperseg, noverlap=noverlap)
            
            # Записываем результат
            sf.write(output_path, clean_y, sr)
            return True
        except Exception as e:
            print(f"[Audio AI] Ошибка спектрального шумоподавления: {e}")
            try:
                import shutil
                shutil.copy2(input_path, output_path)
            except Exception:
                pass
            return False

    def analyze_batch(self, audio_paths):
        """Пакетный анализ списка аудиофайлов."""
        if not audio_paths:
            return []
        return [self.analyze(p) for p in audio_paths]

