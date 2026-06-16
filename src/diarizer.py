import os
import numpy as np
import librosa
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import config

def diarize_audio_pyannote(audio_path, num_speakers=2):
    """Диаризация с помощью pyannote/speaker-diarization-3.1."""
    token = getattr(config, "HF_TOKEN", "")
    if not token:
        raise ValueError("HF_TOKEN не настроен в config")
        
    from pyannote.audio import Pipeline
    import torch
    
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token
    )
    if pipeline is None:
        raise ValueError("Не удалось загрузить пайплайн PyAnnote (проверьте соглашение на HF)")
        
    # Перенос на GPU при наличии
    device = "cuda" if "cuda" in getattr(config, "DEVICE_STR", "") else "cpu"
    if device == "cuda" and torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        
    diarization = pipeline(audio_path, num_speakers=num_speakers)
    
    # Загружаем аудио для нарезки волновой формы реплик
    y, sr = librosa.load(audio_path, sr=16000)
    duration = len(y) / sr
    
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_sec = max(0.0, turn.start)
        end_sec = min(duration, turn.end)
        start_frame = int(start_sec * sr)
        end_frame = int(end_sec * sr)
        
        # Переводим в стандартные метки спикеров
        speaker_label = "Спикер A" if speaker == "SPEAKER_00" else "Спикер B"
        
        segments.append({
            "start": round(start_sec, 2),
            "end": round(end_sec, 2),
            "speaker": speaker_label,
            "y": y[start_frame:end_frame]
        })
        
    if not segments:
        segments.append({
            "start": 0.0,
            "end": round(duration, 2),
            "speaker": "Спикер A",
            "y": y
        })
        
    return segments

def diarize_audio_kmeans(audio_path, num_speakers=2):
    """Локальная диаризация на основе KMeans (резервный метод)."""
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        duration = librosa.get_duration(y=y, sr=sr)
        
        if duration < 1.0:
            return [{"start": 0.0, "end": round(duration, 2), "speaker": "Спикер A", "y": y}]
            
        intervals = librosa.effects.split(y, top_db=25, frame_length=2048, hop_length=512)
        
        if len(intervals) == 0:
            return [{"start": 0.0, "end": round(duration, 2), "speaker": "Спикер A", "y": y}]
            
        segments = []
        for start_frame, end_frame in intervals:
            start_sec = start_frame / sr
            end_sec = end_frame / sr
            segments.append({
                "start": round(start_sec, 2),
                "end": round(end_sec, 2),
                "y": y[start_frame:end_frame]
            })
            
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
            
        features = []
        valid_segments = []
        for seg in merged_segments:
            seg_y = seg["y"]
            if len(seg_y) < 1600:
                continue
                
            mfcc = librosa.feature.mfcc(y=seg_y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            feat_vector = np.concatenate([mfcc_mean, mfcc_std])
            
            features.append(feat_vector)
            valid_segments.append(seg)
            
        if len(features) < num_speakers:
            for seg in merged_segments:
                seg["speaker"] = "Спикер A"
            return merged_segments
            
        features = np.array(features)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        kmeans = KMeans(n_clusters=num_speakers, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_scaled)
        
        for seg, label in zip(valid_segments, labels):
            seg["speaker"] = "Спикер A" if label == 0 else "Спикер B"
            
        for seg in merged_segments:
            if "speaker" not in seg:
                seg["speaker"] = "Спикер A"
                
        return merged_segments
    except Exception as e:
        print(f"[Diarizer Error] Ошибка KMeans-диаризации: {e}")
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            duration = librosa.get_duration(y=y, sr=sr)
            return [{"start": 0.0, "end": round(duration, 2), "speaker": "Спикер A", "y": y}]
        except Exception:
            return [{"start": 0.0, "end": 5.0, "speaker": "Спикер A", "y": np.zeros(16000 * 5)}]

def diarize_audio(audio_path, num_speakers=2):
    """
    Разделяет аудио по спикерам.
    Пытается применить pyannote/speaker-diarization-3.1, при неудаче
    делает мягкий откат на локальный алгоритм KMeans.
    """
    if not config.MOCK_MODE and getattr(config, "HF_TOKEN", ""):
        try:
            return diarize_audio_pyannote(audio_path, num_speakers)
        except Exception as e:
            print(f"[Diarizer] PyAnnote недоступен ({e}). Откат на локальный KMeans.")
            
    return diarize_audio_kmeans(audio_path, num_speakers)

