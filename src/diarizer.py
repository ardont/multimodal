import os
import numpy as np
import librosa
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def diarize_audio(audio_path, num_speakers=2):
    """
    Выполняет диаризацию аудиофайла: разделение аудио на интервалы реплик и
    кластеризацию этих интервалов по спикерам с помощью MFCC и KMeans.
    """
    try:
        # 1. Загружаем аудио (resample до 16кГц для консистентности)
        y, sr = librosa.load(audio_path, sr=16000)
        duration = librosa.get_duration(y=y, sr=sr)
        
        if duration < 1.0:
            return [{"start": 0.0, "end": round(duration, 2), "speaker": "Спикер A", "y": y}]
            
        # 2. Детектируем интервалы голоса (Voice Activity Detection)
        # top_db=25 — порог шума. frame_length и hop_length настроены для сглаживания
        intervals = librosa.effects.split(y, top_db=25, frame_length=2048, hop_length=512)
        
        if len(intervals) == 0:
            return [{"start": 0.0, "end": round(duration, 2), "speaker": "Спикер A", "y": y}]
            
        # Преобразуем интервалы во временные рамки в секундах
        segments = []
        for start_frame, end_frame in intervals:
            start_sec = start_frame / sr
            end_sec = end_frame / sr
            segments.append({
                "start": round(start_sec, 2),
                "end": round(end_sec, 2),
                "y": y[start_frame:end_frame]
            })
            
        # 3. Объединяем реплики, между которыми пауза менее 0.8 секунд
        merged_segments = []
        if segments:
            curr = segments[0]
            for next_seg in segments[1:]:
                if next_seg["start"] - curr["end"] < 0.8:
                    curr["end"] = next_seg["end"]
                    curr_start_idx = int(curr["start"] * sr)
                    curr_end_idx = int(curr["end"] * sr)
                    curr["y"] = y[curr_start_idx:curr_end_idx]
                else:
                    merged_segments.append(curr)
                    curr = next_seg
            merged_segments.append(curr)
        else:
            merged_segments = [{"start": 0.0, "end": round(duration, 2), "speaker": "Спикер A", "y": y}]
            
        if len(merged_segments) < 2:
            for seg in merged_segments:
                seg["speaker"] = "Спикер A"
            return merged_segments
            
        # 4. Извлекаем спектральные фичи (MFCC) для каждого сегмента
        features = []
        valid_segments = []
        for seg in merged_segments:
            seg_y = seg["y"]
            if len(seg_y) < 1600:  # Пропускаем сегменты короче 0.1 секунды
                continue
                
            # Извлекаем 13 коэффициентов MFCC
            mfcc = librosa.feature.mfcc(y=seg_y, sr=sr, n_mfcc=13)
            # Усредняем характеристики по времени, чтобы получить один вектор на реплику
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            feat_vector = np.concatenate([mfcc_mean, mfcc_std])
            
            features.append(feat_vector)
            valid_segments.append(seg)
            
        if len(features) < num_speakers:
            for seg in merged_segments:
                seg["speaker"] = "Спикер A"
            return merged_segments
            
        # 5. Кластеризация KMeans
        features = np.array(features)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        kmeans = KMeans(n_clusters=num_speakers, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_scaled)
        
        # Размечаем сегменты
        for seg, label in zip(valid_segments, labels):
            seg["speaker"] = "Спикер A" if label == 0 else "Спикер B"
            
        # Для пропущенных слишком коротких сегментов ставим Спикер A по умолчанию
        for seg in merged_segments:
            if "speaker" not in seg:
                seg["speaker"] = "Спикер A"
                
        return merged_segments
    except Exception as e:
        print(f"[Diarizer Error] Ошибка диаризации: {e}")
        # При любой ошибке возвращаем аудио как один сегмент
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            duration = librosa.get_duration(y=y, sr=sr)
            return [{"start": 0.0, "end": round(duration, 2), "speaker": "Спикер A", "y": y}]
        except Exception:
            return [{"start": 0.0, "end": 5.0, "speaker": "Спикер A", "y": np.zeros(16000 * 5)}]
