import requests
import time
import os

def ping_node(address):
    """
    Проверяет доступность узла и возвращает системную информацию и пинг.
    address: строка вида "IP:PORT" (например, "127.0.0.1:7860")
    """
    if not address:
        return {"status": "Offline", "ping": None}
        
    url = f"http://{address.strip()}/api/status"
    start_time = time.time()
    try:
        response = requests.get(url, timeout=2.5)
        latency = round((time.time() - start_time) * 1000, 1)
        if response.status_code == 200:
            data = response.json()
            data["ping"] = latency
            data["status"] = "Online"
            return data
    except Exception as e:
        # Узел недоступен
        pass
        
    return {
        "status": "Offline", 
        "ping": None, 
        "cpu": 0, 
        "ram": 0, 
        "gpu_available": False, 
        "gpu_name": "Нет данных"
    }

def remote_asr(address, audio_path):
    """
    Отправляет аудиофайл на удаленный узел для распознавания текста (ASR).
    """
    if not audio_path or not os.path.exists(audio_path):
        return ""
        
    url = f"http://{address.strip()}/api/asr"
    try:
        with open(audio_path, 'rb') as f:
            files = {'file': (os.path.basename(audio_path), f, 'audio/wav')}
            response = requests.post(url, files=files, timeout=60)
            if response.status_code == 200:
                return response.json().get("text", "")
            else:
                print(f"[Network] Ошибка ASR на удаленном узле: {response.text}")
    except Exception as e:
        print(f"[Network] Не удалось связаться с ASR на {address}: {e}")
        
    return "[Ошибка связи с удаленным узлом ASR]"

def remote_text_analysis(address, text):
    """
    Отправляет текст на удаленный узел для анализа эмоций.
    """
    if not text:
        return {"stress": 0.0, "neutral": 1.0}
        
    url = f"http://{address.strip()}/api/text"
    try:
        response = requests.post(url, json={"text": text}, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"[Network] Ошибка анализа текста на удаленном узле: {response.text}")
    except Exception as e:
        print(f"[Network] Не удалось связаться с текстовым анализатором на {address}: {e}")
        
    return {"stress": 0.0, "neutral": 1.0, "error": True}

def remote_audio_analysis(address, audio_path):
    """
    Отправляет аудиофайл на удаленный узел для извлечения акустики и анализа эмоций по звуку.
    """
    if not audio_path or not os.path.exists(audio_path):
        return {"stress": 0.0, "neutral": 1.0, "features": {}}
        
    url = f"http://{address.strip()}/api/audio"
    try:
        with open(audio_path, 'rb') as f:
            files = {'file': (os.path.basename(audio_path), f, 'audio/wav')}
            response = requests.post(url, files=files, timeout=60)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"[Network] Ошибка анализа аудио на удаленном узле: {response.text}")
    except Exception as e:
        print(f"[Network] Не удалось связаться с аудио-анализатором на {address}: {e}")
        
    return {"stress": 0.0, "neutral": 1.0, "features": {}, "error": True}
