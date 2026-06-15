import os
import hashlib
import urllib.request
import time

_CACHE_DIR = os.path.expanduser("~/.cache/gigaam")
_URL_DIR = "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM"

_MODEL_HASHES = {
    "v3_e2e_ctc": "367074d6498f426d960b25f49531cf68",
    "emo": "7ce76f9535cb254488985057c0d33006"
}

def get_md5(file_path):
    if not os.path.exists(file_path):
        return None
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception:
        return None

def download_with_retry(url, path, expected_hash, max_retries=5, timeout=15):
    # Проверка хэша
    if expected_hash:
        current_hash = get_md5(path)
        if current_hash == expected_hash:
            print(f"[Downloader] Файл {os.path.basename(path)} уже скачан и проверен.")
            return True
        elif current_hash is not None:
            print(f"[Downloader] Обнаружен поврежденный файл {os.path.basename(path)} (хэш не совпадает). Перезагрузка...")
            try:
                os.remove(path)
            except Exception as e:
                print(f"[Downloader] Не удалось удалить поврежденный файл {path}: {e}")
    else:
        # Если хэш не проверяется (для токенайзера), проверяем просто существование
        if os.path.exists(path) and os.path.getsize(path) > 0:
            print(f"[Downloader] Файл {os.path.basename(path)} уже существует.")
            return True

    os.makedirs(os.path.dirname(path), exist_ok=True)

    for attempt in range(1, max_retries + 1):
        print(f"[Downloader] Скачивание {os.path.basename(path)} (Попытка {attempt}/{max_retries})...")
        try:
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                total_size = int(response.info().get('Content-Length', 0))
                downloaded = 0
                block_size = 1024 * 64  # 64KB блоки для быстрой записи
                
                last_percent = -1
                with open(path, "wb") as f:
                    while True:
                        block = response.read(block_size)
                        if not block:
                            break
                        f.write(block)
                        downloaded += len(block)
                        
                        if total_size > 0:
                            percent = int(downloaded * 100 / total_size)
                            if percent % 10 == 0 and percent != last_percent:
                                print(f"[Downloader] Прогресс: {percent}% ({downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB)")
                                last_percent = percent
                
                # Проверяем хэш после скачивания
                if expected_hash:
                    downloaded_hash = get_md5(path)
                    if downloaded_hash == expected_hash:
                        print(f"[Downloader] Успешно скачано и проверено: {os.path.basename(path)}")
                        return True
                    else:
                        raise ValueError(f"Ошибка проверки хэша для {os.path.basename(path)}")
                else:
                    if os.path.exists(path) and os.path.getsize(path) > 0:
                        print(f"[Downloader] Успешно скачано: {os.path.basename(path)}")
                        return True
                    else:
                        raise ValueError("Скачанный файл пуст")
                        
        except Exception as e:
            print(f"[Downloader] Ошибка скачивания на попытке {attempt}: {e}")
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
            if attempt < max_retries:
                time.sleep(2)  # Пауза перед следующей попыткой
                
    return False

def ensure_gigaam_models():
    """Проверяет и скачивает все необходимые модели GigaAM."""
    print("[Downloader] Проверка наличия моделей GigaAM...")
    
    # 1. Модель ASR v3_e2e_ctc.ckpt
    asr_url = f"{_URL_DIR}/v3_e2e_ctc.ckpt"
    asr_path = os.path.join(_CACHE_DIR, "v3_e2e_ctc.ckpt")
    asr_hash = _MODEL_HASHES["v3_e2e_ctc"]
    
    # 2. Токенайзер ASR v3_e2e_ctc_tokenizer.model
    tok_url = f"{_URL_DIR}/v3_e2e_ctc_tokenizer.model"
    tok_path = os.path.join(_CACHE_DIR, "v3_e2e_ctc_tokenizer.model")
    
    # 3. Модель Emo emo.ckpt
    emo_url = f"{_URL_DIR}/emo.ckpt"
    emo_path = os.path.join(_CACHE_DIR, "emo.ckpt")
    emo_hash = _MODEL_HASHES["emo"]
    
    # Запуск загрузок
    success = True
    success = success and download_with_retry(asr_url, asr_path, asr_hash)
    success = success and download_with_retry(tok_url, tok_path, None)
    success = success and download_with_retry(emo_url, emo_path, emo_hash)
    
    if success:
        print("[Downloader] Все модели GigaAM успешно проверены и готовы к работе.")
    else:
        print("[Downloader] ВНИМАНИЕ: Некоторые модели не удалось скачать автоматически. Возможен сбой при инициализации.")
    return success
