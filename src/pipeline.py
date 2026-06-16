from .asr import SpeechToText
from .text_analyzer import TextEmotionAnalyzer
from .audio_analyzer import AudioEmotionAnalyzer
from .diarizer import diarize_audio
from . import network
import config

import time
import os
import hashlib
import numpy as np
import concurrent.futures
import tempfile

class MultimodalPipeline:
    def __init__(self):
        # Проверяем и скачиваем модели GigaAM перед загрузкой, если мы работаем в реальном режиме
        if not config.MOCK_MODE:
            from .downloader import ensure_gigaam_models
            ensure_gigaam_models()
            
        # Инициализируем локальные модели.
        self.asr = SpeechToText()
        self.text_ai = TextEmotionAnalyzer()
        self.audio_ai = AudioEmotionAnalyzer()
        
        # Инициализация LRU-кеша
        self._cache = {}
        self._max_cache_size = 128

    def _get_file_hash(self, path):
        """Вычисляет MD5 хэш файла блоками по 4096 байт для экономии ОЗУ."""
        if not path or not os.path.exists(path):
            return None
        hasher = hashlib.md5()
        try:
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return None

    def _get_fused_stress(self, text, text_stress, audio_stress, silence_ratio, duration, enable_audio_emo=True):
        """
        Динамическое взвешивание модальностей на основе характеристик реплики:
        - Если акустический стресс отключен (enable_audio_emo=False), доверяем тексту на 100%.
        - Если реплика короткая (<= 3 слов), доверяем аудио интонации (вес аудио = 75%).
        - Если реплика длинная (> 3 слов), доверяем текстовому смыслу (вес текста = 65%).
        - Если в аудио много тишины/пауз (> 40%), снижаем вес аудио до 15%.
        """
        if not enable_audio_emo:
            return round(float(text_stress), 2)
            
        words_count = len(text.split())
        
        # Базовые веса по умолчанию
        w_text = 0.4
        w_audio = 0.6
        
        if words_count <= 3:
            w_text = 0.25
            w_audio = 0.75
        else:
            w_text = 0.65
            w_audio = 0.35
            
        if silence_ratio > 0.4:
            w_text = 0.85
            w_audio = 0.15
            
        # Нормализация
        w_text = max(0.1, min(0.9, w_text))
        w_audio = max(0.1, min(0.9, w_audio))
        total = w_text + w_audio
        w_text /= total
        w_audio /= total
        
        final_stress = w_text * text_stress + w_audio * audio_stress
        return round(float(final_stress), 2)

    def _align_words_to_segments(self, words, segments, audio_path):
        """
        Выравнивает распознанные слова по интервалам спикеров из диаризатора.
        Если разметка слов пуста (например, при сбое), делает надежный откат
        на физическую нарезку и транскрипцию по отдельным сегментам.
        """
        if not words:
            print("[Pipeline] Таймкоды слов отсутствуют. Откат на посигментную нарезку.")
            import soundfile as sf
            import tempfile
            aligned = []
            for seg in segments:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_seg:
                    temp_seg_path = temp_seg.name
                try:
                    sf.write(temp_seg_path, seg["y"], 16000)
                    asr_route = config.ROUTING.get("asr", "local")
                    if asr_route == "local":
                        text = self.asr.transcribe(temp_seg_path)
                    else:
                        text = network.remote_asr(asr_route, temp_seg_path)
                        if "[Ошибка" in text and config.FAILOVER_TO_LOCAL:
                            text = self.asr.transcribe(temp_seg_path)
                            
                    if text.strip():
                        spk = seg.get("speaker", "Спикер A")
                        aligned.append({
                            "start": seg["start"],
                            "end": seg["end"],
                            "speaker": "Оператор (Спикер А)" if spk == "Спикер A" else "Клиент (Спикер Б)",
                            "text": text
                        })
                finally:
                    try:
                        os.unlink(temp_seg_path)
                    except Exception:
                        pass
            return aligned

        aligned_segments = []
        for seg in segments:
            seg_words = []
            for w in words:
                w_mid = (w["start"] + w["end"]) / 2.0
                if seg["start"] <= w_mid <= seg["end"]:
                    seg_words.append(w["word"])
            
            if seg_words:
                spk = seg.get("speaker", "Спикер A")
                aligned_segments.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "speaker": "Оператор (Спикер А)" if spk == "Спикер A" else "Клиент (Спикер Б)",
                    "text": " ".join(seg_words)
                })
                
        # Если ничего не отнесли, возвращаем один сегмент со всем текстом
        if not aligned_segments:
            full_text = " ".join([w["word"] for w in words])
            aligned_segments.append({
                "start": 0.0,
                "end": segments[-1]["end"] if segments else 5.0,
                "speaker": "Оператор (Спикер А)",
                "text": full_text
            })
            
        return aligned_segments

    def plot_emotion_timeline(self, segments, duration):
        """Строит сглаженный график стресса во времени через Matplotlib (Agg)."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            steps = np.arange(0, duration, 0.5)
            stress_vals = []
            
            for t in steps:
                val = 0.0
                for seg in segments:
                    if seg["start"] <= t <= seg["end"]:
                        val = seg["final_stress"]
                        break
                else:
                    if segments:
                        closest = min(segments, key=lambda s: min(abs(s["start"] - t), abs(s["end"] - t)))
                        val = closest["final_stress"]
                stress_vals.append(val)
                
            stress_vals = np.array(stress_vals)
            
            # Адаптивное сглаживание и биннинг
            num_points = len(stress_vals)
            if num_points > 50:
                bin_size = 5
                new_steps = []
                new_vals = []
                for i in range(0, num_points, bin_size):
                    new_steps.append(np.mean(steps[i:i+bin_size]))
                    new_vals.append(np.mean(stress_vals[i:i+bin_size]))
                steps = np.array(new_steps)
                stress_vals = np.array(new_vals)
                window = 3
            else:
                window = 5
                
            if len(stress_vals) > window:
                stress_vals = np.convolve(stress_vals, np.ones(window)/window, mode='same')
                # Корректируем края convolve
                stress_vals[0] = stress_vals[1]
                stress_vals[-1] = stress_vals[-2]
                
            fig, ax = plt.subplots(figsize=(7, 2.3), facecolor='#0e121a')
            ax.set_facecolor('#05070a')
            
            # Отрисовка линии и заливки
            ax.plot(steps, stress_vals * 100, color='#60a5fa', linewidth=2.0, label='Индекс стресса')
            ax.fill_between(steps, stress_vals * 100, color='#60a5fa', alpha=0.1)
            
            # Точки пикового напряжения (>= 60%)
            peaks_t = [t for t, v in zip(steps, stress_vals) if v >= 0.6]
            peaks_v = [v * 100 for v in stress_vals if v >= 0.6]
            if peaks_t:
                ax.scatter(peaks_t, peaks_v, color='#ef4444', s=25, zorder=5, label='Пик стресса')
                
            ax.grid(True, color='white', alpha=0.05, linestyle=':')
            ax.tick_params(colors='#9ca3af', labelsize=8)
            ax.spines['bottom'].set_color('#1e293b')
            ax.spines['left'].set_color('#1e293b')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            ax.set_xlabel('Время (секунды)', color='#9ca3af', fontsize=8)
            ax.set_ylabel('Стресс (%)', color='#9ca3af', fontsize=8)
            ax.set_title('Эмоциональная динамика звонка во времени', color='#fff', fontsize=9, fontweight='bold', pad=10)
            ax.set_ylim(-5, 105)
            
            import tempfile
            temp_img = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            plt.tight_layout()
            plt.savefig(temp_img.name, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close()
            return temp_img.name
        except Exception as e:
            print(f"[Chart Error] Не удалось построить график: {e}")
            return None

    def run_analysis(self, audio_path, enable_asr=True, enable_audio_emo=True, enable_coach=True):
        if not audio_path:
            return {
                "transcription": "",
                "text_stress": 0.0,
                "audio_stress": 0.0,
                "final_stress": 0.0,
                "features": {},
                "segments": [],
                "chart_path": None,
                "options": {
                    "enable_asr": enable_asr,
                    "enable_audio_emo": enable_audio_emo,
                    "enable_coach": enable_coach
                }
            }
            
        # 1. Проверка кеша по хэшу файла
        file_hash = self._get_file_hash(audio_path)
        if file_hash and file_hash in self._cache:
            # Проверяем, совпадают ли опции запуска в кэшированном ответе
            cached_res = self._cache[file_hash]
            cached_opts = cached_res.get("options", {})
            if (cached_opts.get("enable_asr", True) == enable_asr and 
                cached_opts.get("enable_audio_emo", True) == enable_audio_emo and 
                cached_opts.get("enable_coach", True) == enable_coach):
                print(f"[Pipeline] [CACHE HIT] Возвращаем результат из кеша для {file_hash}")
                res = self._cache.pop(file_hash)
                self._cache[file_hash] = res
                return res

        # 2. Шумоподавление (вырезаем монотонный фоновый гул)
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix="_clean.wav") as temp_clean:
            clean_wav_path = temp_clean.name
            
        try:
            self.audio_ai.reduce_noise(audio_path, clean_wav_path)
            
            # Акустические характеристики
            total_features = self.audio_ai.extract_acoustic_features(clean_wav_path)
            
            if config.MOCK_MODE:
                # В Mock-режиме эмулируем двух спикеров
                if not enable_asr:
                    avg_audio_stress = 0.07 if enable_audio_emo else 0.0
                    res_dict = {
                        "transcription": "",
                        "text_stress": 0.0,
                        "audio_stress": avg_audio_stress,
                        "final_stress": avg_audio_stress,
                        "features": total_features,
                        "segments": [],
                        "chart_path": None,
                        "options": {
                            "enable_asr": enable_asr,
                            "enable_audio_emo": enable_audio_emo,
                            "enable_coach": enable_coach
                        }
                    }
                    if file_hash:
                        self._cache[file_hash] = res_dict
                    return res_dict

                segments = diarize_audio(clean_wav_path)
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
                    
                    if not enable_audio_emo:
                        audio_stress = 0.0
                    
                    diarizer_speaker = seg.get("speaker", "Спикер A")
                    final_speaker = "Оператор (Спикер А)" if diarizer_speaker == "Спикер A" else "Клиент (Спикер Б)"
                    
                    final_stress = self._get_fused_stress(
                        text, text_stress, audio_stress, 0.1, seg["end"]-seg["start"],
                        enable_audio_emo=enable_audio_emo
                    )
                    
                    analyzed_segments.append({
                        "start": seg["start"],
                        "end": seg["end"],
                        "speaker": final_speaker,
                        "text": text,
                        "audio_stress": audio_stress,
                        "text_stress": text_stress,
                        "final_stress": final_stress
                    })
                    full_texts.append(f"[{final_speaker}]: {text}")
                    
                avg_final_stress = np.mean([s["final_stress"] for s in analyzed_segments]) if analyzed_segments else 0.0
                avg_text_stress = np.mean([s["text_stress"] for s in analyzed_segments]) if analyzed_segments else 0.0
                avg_audio_stress = np.mean([s["audio_stress"] for s in analyzed_segments]) if analyzed_segments else 0.0
                
                chart_path = self.plot_emotion_timeline(analyzed_segments, total_features.get("duration", 5.0))
                
                res_dict = {
                    "transcription": "\n".join(full_texts),
                    "text_stress": round(float(avg_text_stress), 2),
                    "audio_stress": round(float(avg_audio_stress), 2),
                    "final_stress": round(float(avg_final_stress), 2),
                    "features": total_features,
                    "segments": analyzed_segments,
                    "chart_path": chart_path,
                    "options": {
                        "enable_asr": enable_asr,
                        "enable_audio_emo": enable_audio_emo,
                        "enable_coach": enable_coach
                    }
                }
                
                if file_hash:
                    self._cache[file_hash] = res_dict
                return res_dict

            # Реальный инференс
            if not enable_asr:
                avg_audio_stress = 0.0
                if enable_audio_emo:
                    try:
                        audio_route = config.ROUTING.get("audio", "local")
                        if audio_route == "local":
                            score = self.audio_ai.analyze(clean_wav_path)
                        else:
                            score = network.remote_audio_analysis(audio_route, clean_wav_path)
                            if score.get("error") and config.FAILOVER_TO_LOCAL:
                                score = self.audio_ai.analyze(clean_wav_path)
                        avg_audio_stress = score.get("stress", 0.0)
                    except Exception as ex:
                        print(f"[Audio Emo Error] {ex}")

                res_dict = {
                    "transcription": "",
                    "text_stress": 0.0,
                    "audio_stress": round(float(avg_audio_stress), 2),
                    "final_stress": round(float(avg_audio_stress), 2),
                    "features": total_features,
                    "segments": [],
                    "chart_path": None,
                    "options": {
                        "enable_asr": enable_asr,
                        "enable_audio_emo": enable_audio_emo,
                        "enable_coach": enable_coach
                    }
                }
                if file_hash:
                    self._cache[file_hash] = res_dict
                return res_dict

            print("[Pipeline] Запуск диаризации аудио...")
            segments = diarize_audio(clean_wav_path)
            
            asr_route = config.ROUTING.get("asr", "local")
            text_route = config.ROUTING.get("text", "local")
            audio_route = config.ROUTING.get("audio", "local")
            
            def thread_a_task():
                try:
                    if config.HAS_TORCH and torch.cuda.is_available():
                        print(f"[VRAM] Поток А (ASR) старт: {torch.cuda.memory_allocated() / (1024*1024):.1f} MB выделено")
                        
                    if asr_route == "local":
                        asr_res = self.asr.transcribe_with_timestamps(clean_wav_path)
                    else:
                        text_raw = network.remote_asr(asr_route, clean_wav_path)
                        asr_res = {"text": text_raw, "words": []}
                        if "[Ошибка" in text_raw and config.FAILOVER_TO_LOCAL:
                            asr_res = self.asr.transcribe_with_timestamps(clean_wav_path)
                            
                    aligned_segs = self._align_words_to_segments(asr_res.get("words", []), segments, clean_wav_path)
                    
                    texts_to_score = [s["text"] for s in aligned_segs]
                    if text_route == "local":
                        text_scores = self.text_ai.analyze_batch(texts_to_score)
                    else:
                        text_scores = [network.remote_text_analysis(text_route, t) for t in texts_to_score]
                        
                    for s, score in zip(aligned_segs, text_scores):
                        s["text_stress"] = score.get("stress", 0.0)
                        
                    return aligned_segs
                except Exception as ex:
                    print(f"[Thread A Error] Ошибка в потоке распознавания текста: {ex}")
                    return []
                    
            def thread_b_task():
                try:
                    time.sleep(1.5)
                    if config.HAS_TORCH and torch.cuda.is_available():
                        print(f"[VRAM] Поток Б (Audio Emo) старт: {torch.cuda.memory_allocated() / (1024*1024):.1f} MB выделено")
                        
                    audio_scores = []
                    import soundfile as sf
                    for seg in segments:
                        with tempfile.NamedTemporaryFile(delete=False, suffix="_seg.wav") as temp_seg:
                            temp_seg_path = temp_seg.name
                        try:
                            sf.write(temp_seg_path, seg["y"], 16000)
                            if audio_route == "local":
                                score = self.audio_ai.analyze(temp_seg_path)
                            else:
                                score = network.remote_audio_analysis(audio_route, temp_seg_path)
                                if score.get("error") and config.FAILOVER_TO_LOCAL:
                                    score = self.audio_ai.analyze(temp_seg_path)
                                    
                            audio_scores.append({
                                "stress": score.get("stress", 0.0),
                                "features": score.get("features", {})
                            })
                        finally:
                            try:
                                os.unlink(temp_seg_path)
                            except Exception:
                                pass
                    return audio_scores
                except Exception as ex:
                    print(f"[Thread B Error] Ошибка в потоке аудио-эмоций: {ex}")
                    return [{"stress": 0.0, "features": {}} for _ in segments]

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_a = executor.submit(thread_a_task)
                if enable_audio_emo:
                    future_b = executor.submit(thread_b_task)
                    aligned_segs = future_a.result()
                    audio_scores = future_b.result()
                else:
                    aligned_segs = future_a.result()
                    audio_scores = [{"stress": 0.0, "features": {}} for _ in aligned_segs]
                
            full_texts = []
            analyzed_segments = []
            
            for i, seg in enumerate(aligned_segs):
                a_score = audio_scores[i] if i < len(audio_scores) else {"stress": 0.0, "features": {}}
                seg_duration = seg["end"] - seg["start"]
                silence_ratio = a_score.get("features", {}).get("silence_ratio", 0.0)
                
                text_stress = seg.get("text_stress", 0.0)
                audio_stress = a_score.get("stress", 0.0)
                
                final_stress = self._get_fused_stress(
                    seg["text"], text_stress, audio_stress, silence_ratio, seg_duration,
                    enable_audio_emo=enable_audio_emo
                )
                
                seg["audio_stress"] = round(audio_stress, 2)
                seg["final_stress"] = final_stress
                
                analyzed_segments.append(seg)
                full_texts.append(f"[{seg['speaker']}]: {seg['text']}")
                
            avg_final_stress = np.mean([s["final_stress"] for s in analyzed_segments]) if analyzed_segments else 0.0
            avg_text_stress = np.mean([s["text_stress"] for s in analyzed_segments]) if analyzed_segments else 0.0
            avg_audio_stress = np.mean([s["audio_stress"] for s in analyzed_segments]) if analyzed_segments else 0.0
            
            chart_path = self.plot_emotion_timeline(analyzed_segments, total_features.get("duration", 5.0))
            
            res_dict = {
                "transcription": "\n".join(full_texts),
                "text_stress": round(float(avg_text_stress), 2),
                "audio_stress": round(float(avg_audio_stress), 2),
                "final_stress": round(float(avg_final_stress), 2),
                "features": total_features,
                "segments": analyzed_segments,
                "chart_path": chart_path,
                "options": {
                    "enable_asr": enable_asr,
                    "enable_audio_emo": enable_audio_emo,
                    "enable_coach": enable_coach
                }
            }
            
            if file_hash:
                if len(self._cache) >= self._max_cache_size:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[file_hash] = res_dict
                
            return res_dict
            
        finally:
            try:
                os.unlink(clean_wav_path)
            except Exception:
                pass

    def run_analysis_batch(self, audio_paths, enable_asr=True, enable_audio_emo=True, enable_coach=True):
        """
        Пакетная обработка списка файлов. Группирует вызовы моделей,
        использует кеш, слияние без физической нарезки и шахматные потоки.
        """
        if not audio_paths:
            return []
            
        results = [None] * len(audio_paths)
        jobs_indices = []
        jobs_paths = []
        
        # 1. Проверяем кэш
        for idx, path in enumerate(audio_paths):
            file_hash = self._get_file_hash(path)
            if file_hash and file_hash in self._cache:
                cached_res = self._cache[file_hash]
                cached_opts = cached_res.get("options", {})
                if (cached_opts.get("enable_asr", True) == enable_asr and 
                    cached_opts.get("enable_audio_emo", True) == enable_audio_emo and 
                    cached_opts.get("enable_coach", True) == enable_coach):
                    print(f"[Pipeline Batch] [CACHE HIT] Загружаем из кеша {path}")
                    res = self._cache.pop(file_hash)
                    self._cache[file_hash] = res
                    results[idx] = res
                    continue
            
            jobs_indices.append(idx)
            jobs_paths.append(path)
                
        if not jobs_paths:
            return results
            
        # 2. Если ASR отключен, делаем пакетную обработку только по акустике
        if not enable_asr:
            clean_paths = []
            files_features = []
            for path in jobs_paths:
                with tempfile.NamedTemporaryFile(delete=False, suffix="_clean.wav") as temp_clean:
                    clean_path = temp_clean.name
                self.audio_ai.reduce_noise(path, clean_path)
                clean_paths.append(clean_path)
                feats = self.audio_ai.extract_acoustic_features(clean_path)
                files_features.append(feats)
                
            try:
                avg_audio_stresses = [0.0] * len(jobs_paths)
                if enable_audio_emo:
                    if config.MOCK_MODE:
                        avg_audio_stresses = [0.07] * len(jobs_paths)
                    else:
                        flat_audio_res = self.audio_ai.analyze_batch(clean_paths)
                        avg_audio_stresses = [r.get("stress", 0.0) for r in flat_audio_res]
                        
                for k, job_idx in enumerate(jobs_indices):
                    original_path = jobs_paths[k]
                    res_dict = {
                        "transcription": "",
                        "text_stress": 0.0,
                        "audio_stress": round(float(avg_audio_stresses[k]), 2),
                        "final_stress": round(float(avg_audio_stresses[k]), 2),
                        "features": files_features[k],
                        "segments": [],
                        "chart_path": None,
                        "options": {
                            "enable_asr": enable_asr,
                            "enable_audio_emo": enable_audio_emo,
                            "enable_coach": enable_coach
                        }
                    }
                    file_hash = self._get_file_hash(original_path)
                    if file_hash:
                        if len(self._cache) >= self._max_cache_size:
                            self._cache.pop(next(iter(self._cache)))
                        self._cache[file_hash] = res_dict
                    results[job_idx] = res_dict
                return results
            finally:
                for p in clean_paths:
                    try:
                        os.unlink(p)
                    except:
                        pass
            
        # 3. Полный анализ для некэшированных файлов (ASR=True)
        clean_paths = []
        files_segments = []
        files_features = []
        
        for path in jobs_paths:
            with tempfile.NamedTemporaryFile(delete=False, suffix="_clean.wav") as temp_clean:
                clean_path = temp_clean.name
            self.audio_ai.reduce_noise(path, clean_path)
            clean_paths.append(clean_path)
            
            segs = diarize_audio(clean_path)
            files_segments.append(segs)
            
            feats = self.audio_ai.extract_acoustic_features(clean_path)
            files_features.append(feats)
            
        # Двухпоточный пакетный инференс
        def batch_thread_a():
            try:
                if config.HAS_TORCH and torch.cuda.is_available():
                    print(f"[VRAM] Поток А (Batch ASR) старт: {torch.cuda.memory_allocated() / (1024*1024):.1f} MB выделено")
                    
                asr_results = []
                for clean_path in clean_paths:
                    asr_results.append(self.asr.transcribe_with_timestamps(clean_path))
                    
                aligned_files_segs = []
                for asr_res, segs, clean_path in zip(asr_results, files_segments, clean_paths):
                    aligned_files_segs.append(self._align_words_to_segments(asr_res["words"], segs, clean_path))
                    
                all_segs_flat = []
                all_texts_flat = []
                for f_idx, segs in enumerate(aligned_files_segs):
                    for s_idx, seg in enumerate(segs):
                        all_segs_flat.append((f_idx, s_idx))
                        all_texts_flat.append(seg["text"])
                        
                flat_bert_res = self.text_ai.analyze_batch(all_texts_flat)
                
                for (f_idx, s_idx), res in zip(all_segs_flat, flat_bert_res):
                    aligned_files_segs[f_idx][s_idx]["text_stress"] = res.get("stress", 0.0)
                    
                return aligned_files_segs
            except Exception as e:
                print(f"[Batch Thread A Error] {e}")
                return [[] for _ in clean_paths]
                
        def batch_thread_b():
            try:
                time.sleep(1.5)
                if config.HAS_TORCH and torch.cuda.is_available():
                    print(f"[VRAM] Поток Б (Batch Audio Emo) старт: {torch.cuda.memory_allocated() / (1024*1024):.1f} MB выделено")
                    
                all_segs_flat = []
                all_wavs_flat = []
                temp_segment_files = []
                
                import soundfile as sf
                for f_idx, segs in enumerate(files_segments):
                    for s_idx, seg in enumerate(segs):
                        with tempfile.NamedTemporaryFile(delete=False, suffix="_seg.wav") as temp_seg:
                            temp_seg_path = temp_seg.name
                        sf.write(temp_seg_path, seg["y"], 16000)
                        all_segs_flat.append((f_idx, s_idx))
                        all_wavs_flat.append(temp_seg_path)
                        temp_segment_files.append(temp_seg_path)
                        
                flat_audio_res = self.audio_ai.analyze_batch(all_wavs_flat)
                
                for path in temp_segment_files:
                    try:
                        os.unlink(path)
                    except:
                        pass
                        
                f_audio_res = [[{"stress": 0.0, "features": {}} for _ in segs] for segs in files_segments]
                for (f_idx, s_idx), res in zip(all_segs_flat, flat_audio_res):
                    f_audio_res[f_idx][s_idx] = {
                        "stress": res.get("stress", 0.0),
                        "features": res.get("features", {})
                    }
                return f_audio_res
            except Exception as e:
                print(f"[Batch Thread B Error] {e}")
                return [[{"stress": 0.0, "features": {}} for _ in segs] for segs in files_segments]

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(batch_thread_a)
            if enable_audio_emo:
                future_b = executor.submit(batch_thread_b)
                aligned_files_segs = future_a.result()
                files_audio_res = future_b.result()
            else:
                aligned_files_segs = future_a.result()
                files_audio_res = [[{"stress": 0.0, "features": {}} for _ in segs] for segs in files_segments]
            
        for path in clean_paths:
            try:
                os.unlink(path)
            except:
                pass
                
        for k, job_idx in enumerate(jobs_indices):
            original_path = jobs_paths[k]
            aligned_segs = aligned_files_segs[k]
            audio_res = files_audio_res[k]
            total_features = files_features[k]
            
            full_texts = []
            for s_idx, seg in enumerate(aligned_segs):
                a_res = audio_res[s_idx] if s_idx < len(audio_res) else {"stress": 0.0, "features": {}}
                
                seg["audio_stress"] = round(a_res.get("stress", 0.0), 2)
                
                seg_duration = seg["end"] - seg["start"]
                silence_ratio = a_res.get("features", {}).get("silence_ratio", 0.0)
                
                seg["final_stress"] = self._get_fused_stress(
                    seg["text"], seg["text_stress"], seg["audio_stress"],
                    silence_ratio, seg_duration,
                    enable_audio_emo=enable_audio_emo
                )
                full_texts.append(f"[{seg['speaker']}]: {seg['text']}")
                
            avg_final_stress = np.mean([s["final_stress"] for s in aligned_segs]) if aligned_segs else 0.0
            avg_text_stress = np.mean([s["text_stress"] for s in aligned_segs]) if aligned_segs else 0.0
            avg_audio_stress = np.mean([s["audio_stress"] for s in aligned_segs]) if aligned_segs else 0.0
            
            chart_path = self.plot_emotion_timeline(aligned_segs, total_features.get("duration", 5.0))
            
            res_dict = {
                "transcription": "\n".join(full_texts),
                "text_stress": round(float(avg_text_stress), 2),
                "audio_stress": round(float(avg_audio_stress), 2),
                "final_stress": round(float(avg_final_stress), 2),
                "features": total_features,
                "segments": aligned_segs,
                "chart_path": chart_path,
                "options": {
                    "enable_asr": enable_asr,
                    "enable_audio_emo": enable_audio_emo,
                    "enable_coach": enable_coach
                }
            }
            
            file_hash = self._get_file_hash(original_path)
            if file_hash:
                if len(self._cache) >= self._max_cache_size:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[file_hash] = res_dict
                
            results[job_idx] = res_dict
            
        return results

