import os
import shutil
import tempfile

# Настройка PATH для ffmpeg из imageio-ffmpeg (автоматическое устранение WinError 2 на Windows)
try:
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)
    
    # Создаем копию ffmpeg.exe, так как imageio_ffmpeg поставляет его под другим именем (например, ffmpeg-win64-v4.2.2.exe)
    target_ffmpeg = os.path.join(ffmpeg_dir, "ffmpeg.exe")
    if not os.path.exists(target_ffmpeg):
        try:
            import shutil
            shutil.copy2(ffmpeg_exe, target_ffmpeg)
            print(f"[FFmpeg] Создан файл-псевдоним: {target_ffmpeg}")
        except Exception as ex:
            # Если нет прав на запись в venv, попробуем записать в папку проекта
            target_proj = os.path.join(os.path.dirname(__file__), "ffmpeg.exe")
            if not os.path.exists(target_proj):
                try:
                    shutil.copy2(ffmpeg_exe, target_proj)
                    print(f"[FFmpeg] Создан файл-псевдоним в проекте: {target_proj}")
                except Exception as ex2:
                    print(f"[FFmpeg] Не удалось скопировать бинарный файл: {ex2}")
    
    if ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ffmpeg_dir + os.path.pathsep + os.environ.get("PATH", "")
    
    # Также добавляем папку проекта на случай копирования туда
    proj_dir = os.path.dirname(__file__)
    if proj_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = proj_dir + os.path.pathsep + os.environ.get("PATH", "")
        
    print(f"[FFmpeg] Встроенный FFmpeg успешно настроен в PATH.")
except Exception as e:
    print(f"[FFmpeg] [Warning] Не удалось настроить встроенный FFmpeg: {e}")


import time
import psutil
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import gradio as gr

import sqlite3
import hashlib
import re

import config
from src.pipeline import MultimodalPipeline
import src.network as network

DB_PATH = "gpb_mer.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS calls (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        operator_username TEXT,
        timestamp TEXT NOT NULL,
        duration REAL,
        stress_score REAL,
        compliance_score REAL,
        summary TEXT,
        transcription TEXT
    )
    """)
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                       ("operator", hashlib.sha256("operator".encode()).hexdigest(), "user"))
        cursor.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                       ("admin", hashlib.sha256("admin".encode()).hexdigest(), "admin"))
    conn.commit()
    conn.close()

init_db()

# Инициализируем FastAPI
app = FastAPI(title="GPB MER Distributed Node API", version="1.0.0")

# Инициализируем наш мультимодальный пайплайн
pipeline = MultimodalPipeline()

# Журнал звонков за сессию
call_history = []

# --- FastAPI ЭНДПОИНТЫ ДЛЯ УДАЛЕННЫХ ВЫЧИСЛЕНИЙ ---

class TextRequest(BaseModel):
    text: str

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/api/login")
async def api_login(req: LoginRequest):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    phash = hashlib.sha256(req.password.encode()).hexdigest()
    cursor.execute("SELECT role FROM users WHERE username = ? AND password_hash = ?", (req.username, phash))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"success": True, "role": row[0], "username": req.username}
    return {"success": False, "error": "Неверный логин или пароль"}

@app.get("/api/history")
async def api_history(username: str = None, role: str = None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if role == "admin":
        cursor.execute("SELECT operator_username, timestamp, duration, stress_score, compliance_score, summary, transcription FROM calls ORDER BY id DESC")
    else:
        cursor.execute("SELECT operator_username, timestamp, duration, stress_score, compliance_score, summary, transcription FROM calls WHERE operator_username = ? ORDER BY id DESC", (username,))
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for r in rows:
        history.append({
            "operator": r[0],
            "timestamp": r[1],
            "duration": r[2],
            "stress_score": r[3],
            "compliance_score": r[4],
            "summary": r[5],
            "transcription": r[6]
        })
    return history

@app.get("/download/app")
async def download_app():
    from fastapi.responses import FileResponse
    # Путь к APK-файлу в структуре проекта
    apk_path = os.path.join(os.path.dirname(__file__), "android-app", "app", "build", "outputs", "apk", "debug", "app-debug.apk")
    if os.path.exists(apk_path):
        return FileResponse(
            apk_path, 
            media_type="application/vnd.android.package-archive", 
            filename="gpb_mer_client.apk"
        )
    return {"error": "APK-файл еще не собран. Пожалуйста, соберите Android-проект в Android Studio."}

@app.get("/api/status")
async def get_status():
    """Возвращает системные метрики текущего узла для мониторинга."""
    local_gpu = False
    local_gpu_name = "N/A"
    if config.HAS_TORCH:
        try:
            import torch
            local_gpu = torch.cuda.is_available()
            local_gpu_name = torch.cuda.get_device_name(0) if local_gpu else "N/A"
        except Exception:
            pass
            
    return {
        "cpu": psutil.cpu_percent(),
        "ram": psutil.virtual_memory().percent,
        "gpu_available": local_gpu,
        "gpu_name": local_gpu_name
    }

@app.post("/api/asr")
async def api_asr(file: UploadFile = File(...)):
    """Выполняет ASR (распознавание речи) на этом узле."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp:
        shutil.copyfileobj(file.file, temp)
        temp_path = temp.name
    try:
        text = pipeline.asr.transcribe(temp_path)
        return {"text": text}
    finally:
        try:
            os.unlink(temp_path)
        except Exception:
            pass

@app.post("/api/text")
async def api_text(req: TextRequest):
    """Анализирует текст на эмоции/аномалии на этом узле."""
    res = pipeline.text_ai.analyze(req.text)
    return res

@app.post("/api/audio")
async def api_audio(file: UploadFile = File(...)):
    """Анализирует аудио на акустику и эмоции по звуку на этом узле."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp:
        shutil.copyfileobj(file.file, temp)
        temp_path = temp.name
    try:
        res = pipeline.audio_ai.analyze(temp_path)
        return res
    finally:
        try:
            os.unlink(temp_path)
        except Exception:
            pass

@app.post("/api/analyze")
async def api_analyze(file: UploadFile = File(...), operator: str = Form(None)):
    """Выполняет полный мультимодальный анализ загруженного аудиофайла (ASR, текст, звук + late fusion)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp:
        shutil.copyfileobj(file.file, temp)
        temp_path = temp.name
    try:
        res = pipeline.run_analysis(temp_path)
        # Добавляем расчеты комплаенса для мобильного клиента
        res["compliance"] = check_compliance(res["transcription"])
        # Добавляем суммаризацию
        res["summary"] = generate_summary(res["transcription"], res, res["compliance"])
        
        # Сохраняем в историю, если указан оператор
        if operator:
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                comp = res["compliance"]
                comp_score = (int(comp["greeting"]) + int(comp["goodbye"]) + int(comp["politeness"]) + int(comp["no_stop_words"])) / 4.0
                cursor.execute(
                    "INSERT INTO calls (operator_username, timestamp, duration, stress_score, compliance_score, summary, transcription) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (operator, time.strftime("%Y-%m-%d %H:%M:%S"), res["features"]["duration"], float(res["final_stress"]), comp_score, res["summary"], res["transcription"])
                )
                conn.commit()
                conn.close()
            except Exception as ex:
                print(f"[DB Error] Не удалось сохранить историю звонка: {ex}")
                
        return res
    except Exception as e:
        print(f"[API] Ошибка полного анализа файла: {e}")
        return {"error": str(e)}
    finally:
        try:
            os.unlink(temp_path)
        except Exception:
            pass

# --- GRADIO ИНТЕРФЕЙС ---

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

body, .gradio-container {
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    background: radial-gradient(circle at top left, #0e121a, #05070a) !important;
    color: #f3f4f6 !important;
}

.header-container {
    text-align: center;
    padding: 1.5rem;
    background: linear-gradient(135deg, rgba(37, 99, 235, 0.08) 0%, rgba(124, 58, 237, 0.08) 100%);
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    margin-bottom: 1.5rem;
    backdrop-filter: blur(8px);
}

.header-container h1 {
    font-weight: 800;
    font-size: 2rem;
    background: linear-gradient(to right, #60a5fa, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.25rem 0;
}

.header-container p {
    color: #9ca3af;
    font-size: 0.95rem;
    margin: 0;
}

.gradio-container .input-box {
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    background: rgba(255, 255, 255, 0.02) !important;
    padding: 1.25rem !important;
}

.report-card {
    background: rgba(10, 15, 26, 0.65) !important;
    border-radius: 12px;
    padding: 1.25rem;
    border-left: 5px solid #3b82f6;
    border-top: 1px solid rgba(255,255,255,0.05);
    border-right: 1px solid rgba(255,255,255,0.05);
    border-bottom: 1px solid rgba(255,255,255,0.05);
}

.report-card.stress-high { border-left-color: #ef4444 !important; background: linear-gradient(to right, rgba(239, 68, 68, 0.04), rgba(0,0,0,0)) !important; }
.report-card.stress-med { border-left-color: #f59e0b !important; background: linear-gradient(to right, rgba(245, 158, 11, 0.04), rgba(0,0,0,0)) !important; }
.report-card.stress-low { border-left-color: #10b981 !important; background: linear-gradient(to right, rgba(16, 185, 129, 0.04), rgba(0,0,0,0)) !important; }

.metric-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
}

.badge-stress-high { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.25); }
.badge-stress-med { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.25); }
.badge-stress-low { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.25); }

@keyframes bounce {
    0% { transform: scaleY(0.3); }
    100% { transform: scaleY(1.3); }
}

.bar {
    transform-origin: bottom;
    animation: bounce 0.8s ease-in-out infinite alternate;
}
"""

# Функции управления состоянием узлов и маршрутизацией
def get_node_choices():
    return ["local"] + config.KNOWN_NODES

def add_new_node(address):
    address = address.strip()
    if not address:
        return (
            gr.update(), gr.update(), gr.update(), 
            '<div style="color: #ef4444;">Ошибка: Адрес узла не может быть пустым.</div>'
        )
    # Автоматически добавляем порт, если он пропущен
    if ":" not in address:
        address = f"{address}:{config.PORT}"
        
    if address not in config.KNOWN_NODES:
        config.KNOWN_NODES.append(address)
    
    # Обновляем варианты выбора в выпадающих списках
    choices = get_node_choices()
    success_msg = f'<div style="color: #34d399;">✓ Узел {address} успешно зарегистрирован. Настройте распределение ниже.</div>'
    return (
        gr.update(choices=choices), 
        gr.update(choices=choices), 
        gr.update(choices=choices), 
        success_msg
    )

def save_routing(asr_target, text_target, audio_target, failover_enabled):
    config.ROUTING["asr"] = asr_target
    config.ROUTING["text"] = text_target
    config.ROUTING["audio"] = audio_target
    config.FAILOVER_TO_LOCAL = failover_enabled
    
    # Строим лог роутинга для вывода пользователю
    routes_desc = f"""
    <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 8px; padding: 1rem; margin-top: 1rem;">
        <h4 style="margin: 0 0 0.5rem 0; color: #34d399;">💾 Маршруты вычислений успешно сохранены:</h4>
        <ul style="margin: 0; padding-left: 1.25rem; font-size: 0.95rem; color: #d1d5db;">
            <li><b>Распознавание (ASR):</b> {asr_target}</li>
            <li><b>Анализ текста:</b> {text_target}</li>
            <li><b>Анализ звука:</b> {audio_target}</li>
            <li><b>Резервный локальный откат:</b> {'Включен' if failover_enabled else 'Выключен'}</li>
        </ul>
    </div>
    """
    return routes_desc

def refresh_nodes_status():
    """Опрашивает все узлы и возвращает красивый HTML для визуализации сети."""
    # Получаем локальные характеристики
    local_cpu = psutil.cpu_percent()
    local_ram = psutil.virtual_memory().percent
    local_gpu_available = False
    local_gpu_name = "N/A"
    
    if config.HAS_TORCH:
        try:
            import torch
            local_gpu_available = torch.cuda.is_available()
            local_gpu_name = torch.cuda.get_device_name(0) if local_gpu_available else "N/A"
        except Exception:
            pass
    
    html = '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.25rem;">'
    
    # Рендерим карточку локального ПК
    html += f"""
    <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(96, 165, 250, 0.3); border-radius: 12px; padding: 1.25rem; box-shadow: 0 4px 20px rgba(0,0,0,0.3); backdrop-filter: blur(4px);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.85rem;">
            <h4 style="margin: 0; color: #fff; font-size: 1.15rem; font-weight: 600;">💻 Локальный ПК (Ноутбук)</h4>
            <span class="metric-badge badge-stress-low">Online</span>
        </div>
        <div style="font-size: 0.85rem; color: #9ca3af; margin-bottom: 0.85rem; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 0.5rem;">Роль: Координатор / Клиент</div>
        <div style="margin-bottom: 0.75rem;">
            <span style="color: #9ca3af; font-size: 0.8rem; display: block; margin-bottom: 0.25rem;">Загрузка CPU:</span>
            <div style="font-size: 1.1rem; font-weight: bold; color: #3b82f6;">{local_cpu}%</div>
            <div style="width: 100%; background: rgba(255,255,255,0.08); height: 4px; border-radius: 2px; margin-top: 0.25rem;">
                <div style="width: {local_cpu}%; background: #3b82f6; height: 100%;"></div>
            </div>
        </div>
        <div style="margin-bottom: 0.75rem;">
            <span style="color: #9ca3af; font-size: 0.8rem; display: block; margin-bottom: 0.25rem;">Загрузка RAM:</span>
            <div style="font-size: 1.1rem; font-weight: bold; color: #8b5cf6;">{local_ram}%</div>
            <div style="width: 100%; background: rgba(255,255,255,0.08); height: 4px; border-radius: 2px; margin-top: 0.25rem;">
                <div style="width: {local_ram}%; background: #8b5cf6; height: 100%;"></div>
            </div>
        </div>
        <div>
            <span style="color: #9ca3af; font-size: 0.8rem; display: block; margin-bottom: 0.15rem;">Ускоритель GPU:</span>
            <div style="font-weight: 600; color: {'#34d399' if local_gpu_available else '#9ca3af'}; font-size: 0.9rem;">
                {local_gpu_name if local_gpu_available else 'Нет GPU (MOCK_MODE)'}
            </div>
        </div>
    </div>
    """
    
    # Опрашиваем удаленные узлы
    for node in config.KNOWN_NODES:
        # Исключаем локальный петлевой адрес
        if node in ["127.0.0.1:7860", "localhost:7860"]:
            continue
            
        status = network.ping_node(node)
        
        if status["status"] == "Online":
            badge_html = '<span class="metric-badge badge-stress-low">Online</span>'
            ping_color = "#34d399" if status["ping"] < 40 else "#fbbf24"
            ping_desc = f'<span style="color: {ping_color}; font-weight: 600;">Пинг: {status["ping"]} ms</span>'
            cpu_val = status["cpu"]
            ram_val = status["ram"]
            gpu_desc = f'<span style="color: #34d399; font-weight: 600;">{status["gpu_name"]}</span>' if status["gpu_available"] else '<span style="color: #9ca3af;">Нет GPU</span>'
            border_color = "rgba(16, 185, 129, 0.2)"
        else:
            badge_html = '<span class="metric-badge badge-stress-high">Offline</span>'
            ping_desc = '<span style="color: #ef4444; font-weight: 600;">Связь отсутствует</span>'
            cpu_val = 0
            ram_val = 0
            gpu_desc = '<span style="color: #ef4444;">N/A</span>'
            border_color = "rgba(239, 68, 68, 0.15)"
            
        html += f"""
        <div style="background: rgba(255,255,255,0.02); border: 1px solid {border_color}; border-radius: 12px; padding: 1.25rem; box-shadow: 0 4px 20px rgba(0,0,0,0.3); backdrop-filter: blur(4px);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.85rem;">
                <h4 style="margin: 0; color: #fff; font-size: 1.15rem; font-weight: 600;">🖥 Удаленный ПК</h4>
                {badge_html}
            </div>
            <div style="font-size: 0.85rem; color: #9ca3af; margin-bottom: 0.85rem; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 0.5rem; display: flex; justify-content: space-between;">
                <span>IP: {node}</span>
                {ping_desc}
            </div>
            <div style="margin-bottom: 0.75rem;">
                <span style="color: #9ca3af; font-size: 0.8rem; display: block; margin-bottom: 0.25rem;">Загрузка CPU:</span>
                <div style="font-size: 1.1rem; font-weight: bold; color: { '#3b82f6' if status['status'] == 'Online' else '#6b7280' };">{cpu_val if status['status'] == 'Online' else 'N/A'}%</div>
                <div style="width: 100%; background: rgba(255,255,255,0.08); height: 4px; border-radius: 2px; margin-top: 0.25rem;">
                    <div style="width: {cpu_val}%; background: #3b82f6; height: 100%;"></div>
                </div>
            </div>
            <div style="margin-bottom: 0.75rem;">
                <span style="color: #9ca3af; font-size: 0.8rem; display: block; margin-bottom: 0.25rem;">Загрузка RAM:</span>
                <div style="font-size: 1.1rem; font-weight: bold; color: { '#8b5cf6' if status['status'] == 'Online' else '#6b7280' };">{ram_val if status['status'] == 'Online' else 'N/A'}%</div>
                <div style="width: 100%; background: rgba(255,255,255,0.08); height: 4px; border-radius: 2px; margin-top: 0.25rem;">
                    <div style="width: {ram_val}%; background: #8b5cf6; height: 100%;"></div>
                </div>
            </div>
            <div>
                <span style="color: #9ca3af; font-size: 0.8rem; display: block; margin-bottom: 0.15rem;">Ускоритель GPU:</span>
                <div style="font-weight: 600; font-size: 0.9rem;">{gpu_desc}</div>
            </div>
        </div>
        """
        
    html += '</div>'
    return html

def check_compliance(text):
    text_lower = text.lower()
    greetings = ["здравствуй", "добрый день", "доброе утро", "добрый вечер", "приветствую", "алло", "слушаю"]
    goodbyes = ["до свидания", "всего доброго", "всего хорошего", "до встречи", "хорошего дня", "пока"]
    politeness = ["спасибо", "пожалуйста", "благодарю", "извините", "прошу прощения", "рад помочь"]
    stop_words = ["вы должны", "ваша проблема", "не знаю", "ужас", "бред", "заткнись", "заткнитесь", "глупость"]
    
    has_greeting = any(g in text_lower for g in greetings)
    has_goodbye = any(g in text_lower for g in goodbyes)
    has_politeness = any(p in text_lower for p in politeness)
    has_stop_words = any(s in text_lower for s in stop_words)
    
    found_stops = [s for s in stop_words if s in text_lower]
    
    return {
        "greeting": has_greeting,
        "goodbye": has_goodbye,
        "politeness": has_politeness,
        "no_stop_words": not has_stop_words,
        "found_stops": found_stops
    }

def generate_summary(text, res, compliance):
    if not text.strip():
        return "Диалог пуст или не распознан."
        
    summary_lines = []
    
    # 1. Общий вердикт по стрессу
    final_stress = res.get("final_stress", 0.0)
    stress_status = "Низкий"
    if final_stress > 0.6:
        stress_status = "Критический 🚨"
    elif final_stress > 0.35:
        stress_status = "Умеренный ⚠️"
        
    summary_lines.append(f"📊 **Итог анализа:** Общий уровень стресса: {int(final_stress * 100)}% ({stress_status}).")
    
    # 2. Соблюдение регламента
    passed_rules = []
    failed_rules = []
    
    if compliance.get("greeting"): passed_rules.append("Приветствие")
    else: failed_rules.append("Приветствие")
    
    if compliance.get("goodbye"): passed_rules.append("Прощание")
    else: failed_rules.append("Прощание")
    
    if compliance.get("politeness"): passed_rules.append("Вежливость")
    else: failed_rules.append("Вежливость")
    
    if compliance.get("no_stop_words"): passed_rules.append("Отсутствие стоп-слов")
    else: failed_rules.append(f"Обнаружены стоп-слова ({', '.join(compliance.get('found_stops', []))})")
    
    if passed_rules:
        summary_lines.append(f"✅ **Соблюдено:** {', '.join(passed_rules)}.")
    if failed_rules:
        summary_lines.append(f"❌ **Нарушено:** {', '.join(failed_rules)}.")
        
    # 3. Ключевые моменты (выделение важных предложений)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    key_sentences = []
    
    keywords = ["карта", "счет", "кредит", "ошибка", "проблема", "заблокировано", "деньги", "перевод", "пароль", "договор", "заявление"]
    
    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        s_lower = s_clean.lower()
        if any(kw in s_lower for kw in keywords) or any(st in s_lower for st in ["вы должны", "ваша проблема", "не знаю", "ужас", "бред", "заткнись"]):
            key_sentences.append(f"• {s_clean}")
            
    if key_sentences:
        summary_lines.append("\n📌 **Ключевые моменты разговора:**")
        summary_lines.extend(key_sentences[:4])
    else:
        non_empty = [s.strip() for s in sentences if s.strip()]
        if non_empty:
            summary_lines.append("\n📌 **Ключевые моменты разговора:**")
            for s in non_empty[:2]:
                summary_lines.append(f"• {s}")
                
    # 4. Рекомендация
    if final_stress > 0.4:
        summary_lines.append("\n💡 **Рекомендация:** У оператора зафиксирован повышенный стресс. Рекомендуется сделать перерыв или разобрать диалог с супервизором.")
    elif not compliance.get("greeting") or not compliance.get("goodbye"):
        summary_lines.append("\n💡 **Рекомендация:** Обратить внимание на соблюдение обязательных фраз приветствия и прощания.")
    else:
        summary_lines.append("\n💡 **Рекомендация:** Диалог проведен отлично, регламент полностью соблюден.")
        
    return "\n".join(summary_lines)

def generate_recommendations(res, compliance):
    recs = []
    
    # Рекомендации по комплаенсу
    if not compliance["greeting"]:
        recs.append("👋 <b>Отсутствует приветствие:</b> Менеджер забыл поздороваться. Обязательно используйте стандартные фразы (например: <i>'Добрый день, меня зовут...'</i>).")
    if not compliance["goodbye"]:
        recs.append("🤝 <b>Отсутствует прощание:</b> В конце разговора не зафиксировано вежливого прощания. Рекомендуется завершать звонок фразой <i>'Всего доброго, до свидания'</i>.")
    if not compliance["politeness"]:
        recs.append("✨ <b>Низкий уровень вежливости:</b> Добавьте в диалог больше клиентоориентированных слов (<i>'спасибо', 'пожалуйста', 'буду рад помочь'</i>).")
    if not compliance["no_stop_words"]:
        stops_str = ", ".join([f"'{s}'" for s in compliance["found_stops"]])
        recs.append(f"⚠️ <b>Обнаружены стоп-слова ({stops_str}):</b> Эти фразы вызывают сопротивление клиента. Замените их на конструктивные формулировки.")
        
    # Рекомендации по акустике
    tempo = res['features'].get('tempo_bpm', 0)
    if tempo > 145:
        recs.append("⚡ <b>Слишком быстрый темп речи:</b> Скорость речи превышает 145 BPM. Говорите медленнее, делайте паузы, чтобы клиент успевал усвоить информацию.")
    elif tempo < 70 and tempo > 0:
        recs.append("🐢 <b>Слишком медленный темп речи:</b> Речь звучит пассивно. Постарайтесь говорить более динамично и уверенно.")
        
    # Рекомендации по стрессу
    final_stress = res['final_stress']
    if final_stress >= 0.7:
        recs.append("🔥 <b>Критический стресс:</b> Индекс эмоционального напряжения крайне высок. Менеджеру рекомендуется сделать перерыв и выпить воды перед следующим звонком.")
    elif final_stress >= 0.4:
        recs.append("📈 <b>Повышенное волнение:</b> Зафиксирована умеренная эмоциональная нестабильность. Старайтесь контролировать дыхание и говорить ровным тоном.")
        
    if not recs:
        recs.append("🌟 <b>Идеальный звонок!</b> Все требования комплаенса соблюдены, уровень стресса в норме, темп речи оптимальный. Так держать!")
        
    return recs

def highlight_keywords(text):
    text_highlighted = text
    
    greetings = ["здравствуйте", "добрый день", "доброе утро", "добрый вечер", "приветствую", "алло", "слушаю"]
    goodbyes = ["до свидания", "всего доброго", "всего хорошего", "до встречи", "хорошего дня", "пока"]
    politeness = ["спасибо", "пожалуйста", "благодарю", "извините", "прошу прощения", "рад помочь"]
    stop_words = ["вы должны", "ваша проблема", "не знаю", "ужас", "бред", "заткнись", "заткнитесь", "глупость"]
    
    import re
    
    def repl_green(m): return f'<span style="color: #10b981; font-weight: bold; border-bottom: 1px dashed #10b981; padding: 0 2px;">{m.group(0)}</span>'
    def repl_purple(m): return f'<span style="color: #a78bfa; font-weight: bold; border-bottom: 1px dashed #a78bfa; padding: 0 2px;">{m.group(0)}</span>'
    def repl_blue(m): return f'<span style="color: #60a5fa; font-weight: bold; border-bottom: 1px dashed #60a5fa; padding: 0 2px;">{m.group(0)}</span>'
    def repl_red(m): return f'<span style="color: #ef4444; font-weight: bold; border-bottom: 1px dashed #ef4444; padding: 0 2px;">{m.group(0)}</span>'
    
    for sw in stop_words:
        pattern = re.compile(re.escape(sw), re.IGNORECASE)
        text_highlighted = pattern.sub(repl_red, text_highlighted)
        
    for g in greetings:
        pattern = re.compile(re.escape(g), re.IGNORECASE)
        text_highlighted = pattern.sub(repl_green, text_highlighted)
        
    for p in politeness:
        pattern = re.compile(re.escape(p), re.IGNORECASE)
        text_highlighted = pattern.sub(repl_blue, text_highlighted)
        
    for gb in goodbyes:
        pattern = re.compile(re.escape(gb), re.IGNORECASE)
        text_highlighted = pattern.sub(repl_purple, text_highlighted)
        
    return text_highlighted

def get_local_ips():
    import socket
    ips = []
    try:
        hostname = socket.gethostname()
        for ip in socket.gethostbyname_ex(hostname)[2]:
            if not ip.startswith("127."):
                ips.append(ip)
    except Exception:
        pass
    
    try:
        for interface, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                if addr.family == socket.AF_INET:
                    ip = addr.address
                    if not ip.startswith("127.") and ip not in ips:
                        ips.append(ip)
    except Exception:
        pass
    return ips

def generate_apk_download_html():
    ips = get_local_ips()
    html = """
    <div style="background: rgba(255,255,255,0.02); border: 1px dashed rgba(255,255,255,0.1); border-radius: 12px; padding: 1.25rem; margin-top: 1.5rem; font-family: 'Outfit', sans-serif;">
        <h3 style="margin-top: 0; color: #fff; font-size: 1.15rem; display: flex; align-items: center; gap: 0.5rem;">📱 Мобильное приложение (Android)</h3>
        <p style="color: #9ca3af; font-size: 0.85rem; margin-bottom: 1rem;">
            Вы можете скачать клиентское Android-приложение напрямую на телефон, находясь в той же сети (Wi-Fi или Tailscale VPN). Отсканируйте один из QR-кодов ниже вашей камерой для быстрой загрузки без USB:
        </p>
        <div style="display: flex; gap: 1.25rem; flex-wrap: wrap; justify-content: flex-start;">
    """
    
    for ip in ips:
        url = f"http://{ip}:{config.PORT}/download/app"
        qr_api = f"https://api.qrserver.com/v1/create-qr-code/?size=120x120&data={url}"
        html += f"""
        <div style="text-align: center; background: rgba(0,0,0,0.15); padding: 0.85rem; border-radius: 10px; border: 1px solid rgba(255,255,255,0.05); min-width: 140px;">
            <span style="color: #60a5fa; font-size: 0.75rem; font-weight: bold; display: block; margin-bottom: 0.5rem;">IP: {ip}</span>
            <img src="{qr_api}" alt="QR Code" style="border: 4px solid white; border-radius: 6px; width: 110px; height: 110px; margin: 0 auto 0.75rem auto; display: block;" />
            <a href="{url}" target="_blank" style="display: inline-block; background: #2563eb; color: white; text-decoration: none; padding: 0.35rem 0.65rem; border-radius: 6px; font-size: 0.75rem; font-weight: 600;">Скачать APK</a>
        </div>
        """
        
    # Всегда выводим 127.0.0.1 как резервную
    fallback_url = f"http://127.0.0.1:{config.PORT}/download/app"
    fallback_qr = f"https://api.qrserver.com/v1/create-qr-code/?size=120x120&data={fallback_url}"
    html += f"""
    <div style="text-align: center; background: rgba(0,0,0,0.15); padding: 0.85rem; border-radius: 10px; border: 1px solid rgba(255,255,255,0.05); min-width: 140px;">
        <span style="color: #9ca3af; font-size: 0.75rem; font-weight: bold; display: block; margin-bottom: 0.5rem;">Локально (localhost)</span>
        <img src="{fallback_qr}" alt="QR Code" style="border: 4px solid white; border-radius: 6px; width: 110px; height: 110px; margin: 0 auto 0.75rem auto; display: block;" />
        <a href="{fallback_url}" target="_blank" style="display: inline-block; background: #4b5563; color: white; text-decoration: none; padding: 0.35rem 0.65rem; border-radius: 6px; font-size: 0.75rem; font-weight: 600;">Скачать APK</a>
    </div>
    """
        
    html += """
        </div>
    </div>
    """
    return html

def generate_kpi_html():
    if not call_history:
        return """
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; font-family: 'Outfit', sans-serif;">
            <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 1rem; text-align: center; backdrop-filter: blur(4px);">
                <span style="color: #9ca3af; font-size: 0.75rem; display: block; text-transform: uppercase; letter-spacing: 0.05em;">Всего звонков</span>
                <span style="font-size: 1.8rem; font-weight: 800; color: #fff; margin-top: 0.25rem; display: block;">0</span>
            </div>
            <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 1rem; text-align: center; backdrop-filter: blur(4px);">
                <span style="color: #9ca3af; font-size: 0.75rem; display: block; text-transform: uppercase; letter-spacing: 0.05em;">Средний стресс</span>
                <span style="font-size: 1.8rem; font-weight: 800; color: #10b981; margin-top: 0.25rem; display: block;">0%</span>
            </div>
            <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 1rem; text-align: center; backdrop-filter: blur(4px);">
                <span style="color: #9ca3af; font-size: 0.75rem; display: block; text-transform: uppercase; letter-spacing: 0.05em;">Средний комплаенс</span>
                <span style="font-size: 1.8rem; font-weight: 800; color: #60a5fa; margin-top: 0.25rem; display: block;">100%</span>
            </div>
            <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 1rem; text-align: center; backdrop-filter: blur(4px);">
                <span style="color: #9ca3af; font-size: 0.75rem; display: block; text-transform: uppercase; letter-spacing: 0.05em;">Аномалии (Alerts)</span>
                <span style="font-size: 1.8rem; font-weight: 800; color: #10b981; margin-top: 0.25rem; display: block;">0</span>
            </div>
        </div>
        """
    
    total = len(call_history)
    stresses = []
    compliances = []
    alerts = 0
    
    for call in call_history:
        stress_val = int(call["stress"].replace("%", ""))
        stresses.append(stress_val)
        if stress_val >= 70:
            alerts += 1
            
        comp_val = int(call["compliance"].split("/")[0])
        compliances.append(comp_val)
        
    avg_stress = sum(stresses) / total
    avg_comp_percent = (sum(compliances) / (total * 4)) * 100
    
    if avg_stress >= 70:
        stress_color = "#ef4444"
    elif avg_stress >= 40:
        stress_color = "#f59e0b"
    else:
        stress_color = "#34d399"
        
    if avg_comp_percent >= 80:
        comp_color = "#34d399"
    elif avg_comp_percent >= 50:
        comp_color = "#fbbf24"
    else:
        comp_color = "#ef4444"
        
    alert_color = "#ef4444" if alerts > 0 else "#10b981"
    
    return f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; font-family: 'Outfit', sans-serif;">
        <div style="background: linear-gradient(135deg, rgba(255,255,255,0.01) 0%, rgba(255,255,255,0.03) 100%); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 1rem; text-align: center; backdrop-filter: blur(4px);">
            <span style="color: #9ca3af; font-size: 0.75rem; display: block; text-transform: uppercase; letter-spacing: 0.05em;">Всего звонков</span>
            <span style="font-size: 1.8rem; font-weight: 800; color: #fff; margin-top: 0.25rem; display: block;">{total}</span>
        </div>
        <div style="background: linear-gradient(135deg, rgba(255,255,255,0.01) 0%, rgba(255,255,255,0.03) 100%); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 1rem; text-align: center; backdrop-filter: blur(4px);">
            <span style="color: #9ca3af; font-size: 0.75rem; display: block; text-transform: uppercase; letter-spacing: 0.05em;">Средний стресс</span>
            <span style="font-size: 1.8rem; font-weight: 800; color: {stress_color}; margin-top: 0.25rem; display: block;">{avg_stress:.0f}%</span>
        </div>
        <div style="background: linear-gradient(135deg, rgba(255,255,255,0.01) 0%, rgba(255,255,255,0.03) 100%); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 1rem; text-align: center; backdrop-filter: blur(4px);">
            <span style="color: #9ca3af; font-size: 0.75rem; display: block; text-transform: uppercase; letter-spacing: 0.05em;">Средний комплаенс</span>
            <span style="font-size: 1.8rem; font-weight: 800; color: {comp_color}; margin-top: 0.25rem; display: block;">{avg_comp_percent:.0f}%</span>
        </div>
        <div style="background: linear-gradient(135deg, rgba(255,255,255,0.01) 0%, rgba(255,255,255,0.03) 100%); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 1rem; text-align: center; backdrop-filter: blur(4px);">
            <span style="color: #9ca3af; font-size: 0.75rem; display: block; text-transform: uppercase; letter-spacing: 0.05em;">Аномалии (Alerts)</span>
            <span style="font-size: 1.8rem; font-weight: 800; color: {alert_color}; margin-top: 0.25rem; display: block;">{alerts}</span>
        </div>
    </div>
    """

def generate_history_html():
    if not call_history:
        return """
        <div style="border: 1px dashed rgba(255,255,255,0.05); border-radius: 8px; padding: 1.5rem; text-align: center; color: #9ca3af; font-size: 0.9rem;">
            Журнал пуст. Проведите анализ хотя бы одного звонка или симуляцию.
        </div>
        """
        
    rows = ""
    for call in reversed(call_history):
        stress_val = int(call["stress"].replace("%", ""))
        if stress_val >= 70:
            status_style = "background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.25);"
        elif stress_val >= 40:
            status_style = "background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.25);"
        else:
            status_style = "background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.25);"
            
        comp_val = int(call["compliance"].split("/")[0])
        if comp_val == 4:
            comp_style = "color: #34d399; font-weight: bold;"
        elif comp_val >= 2:
            comp_style = "color: #fbbf24; font-weight: bold;"
        else:
            comp_style = "color: #f87171; font-weight: bold;"
            
        rows += f"""
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.04);">
            <td style="padding: 0.75rem 1rem; color: #9ca3af; font-size: 0.9rem;">{call['time']}</td>
            <td style="padding: 0.75rem 1rem; color: #e5e7eb; font-size: 0.9rem;">{call['duration']}</td>
            <td style="padding: 0.75rem 1rem; {comp_style} font-size: 0.9rem;">{call['compliance']}</td>
            <td style="padding: 0.75rem 1rem; font-weight: bold; color: #f3f4f6; font-size: 0.9rem;">{call['stress']}</td>
            <td style="padding: 0.75rem 1rem; font-size: 0.85rem;"><span class="metric-badge" style="{status_style} padding: 0.15rem 0.5rem; font-size: 0.8rem;">{call['status']}</span></td>
        </tr>
        """
        
    table_html = f"""
    <div style="background: rgba(255,255,255,0.01); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; overflow: hidden; margin-top: 1rem;">
        <table style="width: 100%; border-collapse: collapse; text-align: left;">
            <thead>
                <tr style="background: rgba(255,255,255,0.02); border-bottom: 1px solid rgba(255,255,255,0.08);">
                    <th style="padding: 0.75rem 1rem; color: #9ca3af; font-size: 0.8rem; text-transform: uppercase; font-weight: bold;">Время</th>
                    <th style="padding: 0.75rem 1rem; color: #9ca3af; font-size: 0.8rem; text-transform: uppercase; font-weight: bold;">Длительность</th>
                    <th style="padding: 0.75rem 1rem; color: #9ca3af; font-size: 0.8rem; text-transform: uppercase; font-weight: bold;">Комплаенс</th>
                    <th style="padding: 0.75rem 1rem; color: #9ca3af; font-size: 0.8rem; text-transform: uppercase; font-weight: bold;">Индекс стресса</th>
                    <th style="padding: 0.75rem 1rem; color: #9ca3af; font-size: 0.8rem; text-transform: uppercase; font-weight: bold;">Вердикт</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>
    """
    return table_html

def format_report_html(res, is_simulation=False):
    stress = res['final_stress']
    if stress >= 0.7:
        stress_class = "stress-high"
        badge_class = "badge-stress-high"
        status_text = "Критический стресс / Аномалия"
        gauge_color = "#ef4444"
        timeline_main_color = "#ef4444"
    elif stress >= 0.4:
        stress_class = "stress-med"
        badge_class = "badge-stress-med"
        status_text = "Повышенное волнение"
        gauge_color = "#f59e0b"
        timeline_main_color = "#f59e0b"
    else:
        stress_class = "stress-low"
        badge_class = "badge-stress-low"
        status_text = "Нормальное / Стабильное состояние"
        gauge_color = "#10b981"
        timeline_main_color = "#10b981"
        
    features = res['features']
    compliance = check_compliance(res['transcription'])
    recs = generate_recommendations(res, compliance)
    
    c_greeting_icon = "✓" if compliance["greeting"] else "✗"
    c_greeting_color = "#10b981" if compliance["greeting"] else "#ef4444"
    c_greeting_desc = "Найдено слово приветствия" if compliance["greeting"] else "Приветствие отсутствует"
    timeline_greeting_color = "#10b981" if compliance["greeting"] else "#ef4444"
    
    c_goodbye_icon = "✓" if compliance["goodbye"] else "✗"
    c_goodbye_color = "#10b981" if compliance["goodbye"] else "#ef4444"
    c_goodbye_desc = "Найдено слово прощания" if compliance["goodbye"] else "Прощание отсутствует"
    timeline_goodbye_color = "#10b981" if compliance["goodbye"] else "#ef4444"
    
    c_politeness_icon = "✓" if compliance["politeness"] else "✗"
    c_politeness_color = "#10b981" if compliance["politeness"] else "#ef4444"
    c_politeness_desc = "Вежливые слова найдены" if compliance["politeness"] else "Добавьте больше вежливых фраз"
    
    c_stops_icon = "✓" if compliance["no_stop_words"] else "✗"
    c_stops_color = "#10b981" if compliance["no_stop_words"] else "#ef4444"
    c_stops_desc = "Токсичные стоп-слова не обнаружены" if compliance["no_stop_words"] else f"Обнаружено: {', '.join(compliance['found_stops'])}"
    
    recs_html = "".join([f"<li style='margin-bottom: 0.5rem;'>{r}</li>" for r in recs])
    
    # Сборка HTML реплик диалога
    segments_html = []
    for seg in res.get("segments", []):
        spk = seg.get("speaker", "Спикер A")
        txt = seg.get("text", "")
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        seg_stress = seg.get("final_stress", 0.0)
        
        highlighted_txt = highlight_keywords(txt)
        
        if seg_stress >= 0.7:
            stress_badge_color = "#ef4444"
        elif seg_stress >= 0.4:
            stress_badge_color = "#f59e0b"
        else:
            stress_badge_color = "#10b981"
            
        is_client = "клиент" in spk.lower() or "спикер b" in spk.lower() or "спикер б" in spk.lower()
        
        if is_client:
            align = "flex-end"
            bg = "rgba(139, 92, 246, 0.08)"
            border = "border-right: 3px solid #8b5cf6;"
            margin = "margin-left: 20%;"
            text_align = "right"
        else:
            align = "flex-start"
            bg = "rgba(59, 130, 246, 0.08)"
            border = "border-left: 3px solid #3b82f6;"
            margin = "margin-right: 20%;"
            text_align = "left"
            
        bubble = f"""
        <div style="align-self: {align}; width: 80%; background: {bg}; {border} {margin} padding: 0.65rem 0.85rem; border-radius: 8px; margin-bottom: 0.75rem; text-align: {text_align}; box-sizing: border-box;">
            <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #9ca3af; margin-bottom: 0.25rem;">
                <span style="font-weight: 600;">{spk} ({start:.1f}с - {end:.1f}с)</span>
                <span style="color: {stress_badge_color}; font-weight: bold;">Стресс: {seg_stress * 100:.0f}%</span>
            </div>
            <span style="font-size: 0.95rem; color: #f3f4f6; line-height: 1.5;">"{highlighted_txt}"</span>
        </div>
        """
        segments_html.append(bubble)
        
    dialogue_view_html = f"""
    <div style="display: flex; flex-direction: column; background: rgba(0, 0, 0, 0.2); border: 1px solid rgba(255,255,255,0.05); padding: 1rem; border-radius: 12px; max-height: 250px; overflow-y: auto; margin-bottom: 1.25rem; box-sizing: border-box;">
        {"".join(segments_html)}
    </div>
    """
    
    header_title = "📊 Результат Экспресс-Анализа (Симуляция)" if is_simulation else "📊 Результат Экспресс-Анализа"
    header_subtitle = "Режим быстрой эмуляции сценариев" if is_simulation else "Звонок обработан распределенной нейросетью"
    
    report_html = f"""
    <div class="report-card {stress_class}" style="font-family: 'Outfit', sans-serif; padding: 1.5rem; background: rgba(10, 15, 26, 0.65); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 16px; backdrop-filter: blur(12px); color: #f3f4f6;">
        
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 1.25rem; margin-bottom: 1.5rem; flex-wrap: wrap; gap: 1rem;">
            <div>
                <h3 style="margin: 0; font-size: 1.4rem; font-weight: 700; background: linear-gradient(to right, #60a5fa, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{header_title}</h3>
                <span style="color: #9ca3af; font-size: 0.85rem;">{header_subtitle}</span>
            </div>
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="text-align: right;">
                    <span style="font-size: 0.75rem; color: #9ca3af; display: block; text-transform: uppercase; letter-spacing: 0.05em;">Индекс аномалии</span>
                    <span style="font-size: 1.8rem; font-weight: 800; color: {gauge_color};">{res['final_stress'] * 100:.0f}%</span>
                </div>
                <span class="metric-badge {badge_class}" style="font-size: 0.9rem; padding: 0.4rem 1rem;">{status_text}</span>
            </div>
        </div>
        
        <div style="display: grid; grid-template-columns: 1.25fr 0.75fr; gap: 1.5rem; margin-bottom: 1.5rem; align-items: start;">
            <div>
                <span style="color: #9ca3af; font-size: 0.8rem; display: block; margin-bottom: 0.5rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Диалог (ASR с разделением спикеров):</span>
                {dialogue_view_html}
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.25rem;">
                    <div style="background: rgba(255,255,255,0.02); padding: 0.85rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                        <span style="color: #9ca3af; font-size: 0.8rem; display: block; margin-bottom: 0.25rem;">Текстовый стресс:</span>
                        <div style="font-size: 1.4rem; font-weight: bold; color: #60a5fa;">{res['text_stress'] * 100:.0f}%</div>
                        <div style="width: 100%; background-color: rgba(255,255,255,0.08); height: 6px; border-radius: 3px; overflow: hidden; margin-top: 0.4rem;">
                            <div style="width: {res['text_stress'] * 100}%; background: linear-gradient(to right, #3b82f6, #60a5fa); height: 100%;"></div>
                        </div>
                    </div>
                    <div style="background: rgba(255,255,255,0.02); padding: 0.85rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                        <span style="color: #9ca3af; font-size: 0.8rem; display: block; margin-bottom: 0.25rem;">Акустический стресс:</span>
                        <div style="font-size: 1.4rem; font-weight: bold; color: #a78bfa;">{res['audio_stress'] * 100:.0f}%</div>
                        <div style="width: 100%; background-color: rgba(255,255,255,0.08); height: 6px; border-radius: 3px; overflow: hidden; margin-top: 0.4rem;">
                            <div style="width: {res['audio_stress'] * 100}%; background: linear-gradient(to right, #7c3aed, #a78bfa); height: 100%;"></div>
                        </div>
                    </div>
                </div>

                <span style="color: #9ca3af; font-size: 0.8rem; display: block; margin-bottom: 0.5rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Динамический таймлайн звонка:</span>
                <div style="display: flex; align-items: center; justify-content: space-between; background: rgba(255,255,255,0.01); border: 1px solid rgba(255,255,255,0.04); border-radius: 8px; padding: 0.85rem 1.25rem;">
                    <div style="display: flex; flex-direction: column; align-items: center; gap: 0.25rem; flex: 1;">
                        <div style="width: 12px; height: 12px; border-radius: 50%; background-color: {timeline_greeting_color}; box-shadow: 0 0 8px {timeline_greeting_color};"></div>
                        <span style="font-size: 0.75rem; color: #9ca3af;">Приветствие</span>
                    </div>
                    <div style="height: 2px; background: rgba(255,255,255,0.1); flex: 1.5; margin: 0 0.5rem 12px 0.5rem;"></div>
                    <div style="display: flex; flex-direction: column; align-items: center; gap: 0.25rem; flex: 1;">
                        <div style="width: 12px; height: 12px; border-radius: 50%; background-color: {timeline_main_color}; box-shadow: 0 0 8px {timeline_main_color};"></div>
                        <span style="font-size: 0.75rem; color: #9ca3af;">Диалог</span>
                    </div>
                    <div style="height: 2px; background: rgba(255,255,255,0.1); flex: 1.5; margin: 0 0.5rem 12px 0.5rem;"></div>
                    <div style="display: flex; flex-direction: column; align-items: center; gap: 0.25rem; flex: 1;">
                        <div style="width: 12px; height: 12px; border-radius: 50%; background-color: {timeline_goodbye_color}; box-shadow: 0 0 8px {timeline_goodbye_color};"></div>
                        <span style="font-size: 0.75rem; color: #9ca3af;">Прощание</span>
                    </div>
                </div>
            </div>
            
            <div style="background: rgba(255,255,255,0.02); padding: 1.25rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); height: 100%; box-sizing: border-box;">
                <span style="color: #9ca3af; font-size: 0.8rem; display: block; margin-bottom: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Соблюдение регламента (QA):</span>
                <div style="display: flex; flex-direction: column; gap: 0.75rem;">
                    
                    <div style="display: flex; align-items: center; gap: 0.75rem;">
                        <span style="font-size: 1.2rem; font-weight: bold; color: {c_greeting_color}; width: 20px; text-align: center;">{c_greeting_icon}</span>
                        <div>
                            <span style="font-size: 0.85rem; font-weight: 600; display: block; color: #f3f4f6;">Приветствие</span>
                            <span style="font-size: 0.75rem; color: #9ca3af; display: block; line-height: 1.2;">{c_greeting_desc}</span>
                        </div>
                    </div>
                    
                    <div style="display: flex; align-items: center; gap: 0.75rem; border-top: 1px solid rgba(255,255,255,0.04); padding-top: 0.5rem;">
                        <span style="font-size: 1.2rem; font-weight: bold; color: {c_goodbye_color}; width: 20px; text-align: center;">{c_goodbye_icon}</span>
                        <div>
                            <span style="font-size: 0.85rem; font-weight: 600; display: block; color: #f3f4f6;">Прощание</span>
                            <span style="font-size: 0.75rem; color: #9ca3af; display: block; line-height: 1.2;">{c_goodbye_desc}</span>
                        </div>
                    </div>
                    
                    <div style="display: flex; align-items: center; gap: 0.75rem; border-top: 1px solid rgba(255,255,255,0.04); padding-top: 0.5rem;">
                        <span style="font-size: 1.2rem; font-weight: bold; color: {c_politeness_color}; width: 20px; text-align: center;">{c_politeness_icon}</span>
                        <div>
                            <span style="font-size: 0.85rem; font-weight: 600; display: block; color: #f3f4f6;">Вежливость</span>
                            <span style="font-size: 0.75rem; color: #9ca3af; display: block; line-height: 1.2;">{c_politeness_desc}</span>
                        </div>
                    </div>
                    
                    <div style="display: flex; align-items: center; gap: 0.75rem; border-top: 1px solid rgba(255,255,255,0.04); padding-top: 0.5rem;">
                        <span style="font-size: 1.2rem; font-weight: bold; color: {c_stops_color}; width: 20px; text-align: center;">{c_stops_icon}</span>
                        <div>
                            <span style="font-size: 0.85rem; font-weight: 600; display: block; color: #f3f4f6;">Отсутствие стоп-слов</span>
                            <span style="font-size: 0.75rem; color: #9ca3af; display: block; line-height: 1.2;">{c_stops_desc}</span>
                        </div>
                    </div>
                    
                </div>
            </div>
        </div>
        
        <div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.08) 0%, rgba(37, 99, 235, 0.08) 100%); border: 1px solid rgba(139, 92, 246, 0.25); border-radius: 12px; padding: 1.25rem; margin-bottom: 1.5rem; box-shadow: 0 4px 15px rgba(139, 92, 246, 0.15);">
            <h4 style="margin: 0 0 0.75rem 0; color: #c084fc; font-size: 1rem; font-weight: 700; display: flex; align-items: center; gap: 0.5rem;">
                💡 Интеллектуальные рекомендации (AI Coach):
            </h4>
            <ul style="margin: 0; padding-left: 1.25rem; font-size: 0.9rem; line-height: 1.6; color: #d1d5db; display: flex; flex-direction: column; gap: 0.5rem;">
                {recs_html}
            </ul>
        </div>
        
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
            <h4 style="margin: 0; color: #e5e7eb; font-size: 1rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">🎙 Акустические характеристики звука</h4>
            <div style="display: flex; align-items: flex-end; gap: 3px; height: 20px;">
                <div class="bar" style="width: 3px; height: 12px; background-color: {gauge_color}; border-radius: 2px; animation: bounce 0.8s ease-in-out infinite alternate;"></div>
                <div class="bar" style="width: 3px; height: 18px; background-color: {gauge_color}; border-radius: 2px; animation: bounce 0.5s ease-in-out infinite alternate; animation-delay: 0.15s;"></div>
                <div class="bar" style="width: 3px; height: 8px; background-color: {gauge_color}; border-radius: 2px; animation: bounce 1.1s ease-in-out infinite alternate; animation-delay: 0.3s;"></div>
                <div class="bar" style="width: 3px; height: 15px; background-color: {gauge_color}; border-radius: 2px; animation: bounce 0.7s ease-in-out infinite alternate; animation-delay: 0.1s;"></div>
                <div class="bar" style="width: 3px; height: 10px; background-color: {gauge_color}; border-radius: 2px; animation: bounce 0.9s ease-in-out infinite alternate; animation-delay: 0.2s;"></div>
            </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 0.75rem;">
            <div style="background: rgba(255,255,255,0.01); padding: 0.75rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.04); text-align: center;">
                <span style="color: #9ca3af; font-size: 0.75rem; display: block; margin-bottom: 0.25rem;">Длительность</span>
                <span style="font-size: 1.1rem; font-weight: 600; color: #f3f4f6;">{features.get('duration', 0)} сек</span>
            </div>
            <div style="background: rgba(255,255,255,0.01); padding: 0.75rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.04); text-align: center;">
                <span style="color: #9ca3af; font-size: 0.75rem; display: block; margin-bottom: 0.25rem;">Громкость (RMS)</span>
                <span style="font-size: 1.1rem; font-weight: 600; color: #f3f4f6;">{features.get('loudness_mean', 0)}</span>
            </div>
            <div style="background: rgba(255,255,255,0.01); padding: 0.75rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.04); text-align: center;">
                <span style="color: #9ca3af; font-size: 0.75rem; display: block; margin-bottom: 0.25rem;">Доля тишины</span>
                <span style="font-size: 1.1rem; font-weight: 600; color: #f3f4f6;">{features.get('silence_ratio', 0) * 100:.0f}%</span>
            </div>
            <div style="background: rgba(255,255,255,0.01); padding: 0.75rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.04); text-align: center;">
                <span style="color: #9ca3af; font-size: 0.75rem; display: block; margin-bottom: 0.25rem;">Темп речи</span>
                <span style="font-size: 1.1rem; font-weight: 600; color: #f3f4f6;">{features.get('tempo_bpm', 0)} BPM</span>
            </div>
        </div>
    </div>
    """
    
    comp_score = 0
    if compliance["greeting"]: comp_score += 1
    if compliance["goodbye"]: comp_score += 1
    if compliance["politeness"]: comp_score += 1
    if compliance["no_stop_words"]: comp_score += 1
    
    now = time.strftime("%H:%M:%S")
    call_history.append({
        "time": now,
        "duration": f"{features.get('duration', 0)} с",
        "compliance": f"{comp_score}/4",
        "stress": f"{stress * 100:.0f}%",
        "status": status_text
    })
    
    return report_html

def run_simulation(example_type):
    # Симуляция разных сценариев
    if example_type == 1:
        text = "Добрый день! Спасибо большое за ожидание. Подскажите, пожалуйста, номер вашего договора, я с радостью вам помогу всего хорошего."
        res = {
            "transcription": text,
            "text_stress": 0.05,
            "audio_stress": 0.12,
            "final_stress": 0.09,
            "features": {
                "duration": 15,
                "loudness_mean": -22,
                "silence_ratio": 0.18,
                "tempo_bpm": 115
            },
            "segments": [
                {
                    "start": 0.0,
                    "end": 15.0,
                    "speaker": "Оператор (Спикер А)",
                    "text": text,
                    "audio_stress": 0.12,
                    "text_stress": 0.05,
                    "final_stress": 0.09
                }
            ]
        }
    elif example_type == 2:
        text = "Да заткнитесь вы уже! Это ваша проблема, что вы не прочитали договор. Вы должны были внести платеж вчера! Это бред какой-то."
        res = {
            "transcription": text,
            "text_stress": 0.88,
            "audio_stress": 0.95,
            "final_stress": 0.92,
            "features": {
                "duration": 18,
                "loudness_mean": -12,
                "silence_ratio": 0.05,
                "tempo_bpm": 138
            },
            "segments": [
                {
                    "start": 0.0,
                    "end": 18.0,
                    "speaker": "Клиент (Спикер Б)",
                    "text": text,
                    "audio_stress": 0.95,
                    "text_stress": 0.88,
                    "final_stress": 0.92
                }
            ]
        }
    else:
        text = "Здравствуйте... Ой, извините, я не знаю, наверное... Да-да, сейчас я посмотрю информацию, подождите секундочку, пожалуйста, я постараюсь быстрее..."
        res = {
            "transcription": text,
            "text_stress": 0.35,
            "audio_stress": 0.55,
            "final_stress": 0.47,
            "features": {
                "duration": 22,
                "loudness_mean": -18,
                "silence_ratio": 0.08,
                "tempo_bpm": 156
            },
            "segments": [
                {
                    "start": 0.0,
                    "end": 22.0,
                    "speaker": "Клиент (Спикер Б)",
                    "text": text,
                    "audio_stress": 0.55,
                    "text_stress": 0.35,
                    "final_stress": 0.47
                }
            ]
        }
        
    report_html = format_report_html(res, is_simulation=True)
    return report_html, generate_kpi_html(), generate_history_html()

def predict(audio):
    if audio is None:
        return """
        <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 8px; padding: 1rem; color: #f87171; text-align: center;">
            ⚠️ Пожалуйста, запишите или загрузите аудиофайл.
        </div>
        """, generate_kpi_html(), generate_history_html()
        
    res = pipeline.run_analysis(audio)
    report_html = format_report_html(res, is_simulation=False)
    return report_html, generate_kpi_html(), generate_history_html()

# Создаем интерфейс Gradio
with gr.Blocks(title="GPB MER Distributed MVP") as demo:
    # Инжектируем CSS напрямую через HTML для совместимости с Gradio 6
    gr.HTML(f"<style>{custom_css}</style>")
    # Заголовок
    with gr.Row(elem_classes="header-container"):
        with gr.Column():
            gr.HTML(
                """
                <div style="text-align: center;">
                    <h1 style="margin: 0; font-weight: 800;">🎙 Распределенная система анализа речи</h1>
                    <p style="color: #9ca3af; margin: 0.25rem 0 0 0;">Мультимодальный скоринг стресса и мониторинг узлов вычисления (Tailscale VPN)</p>
                </div>
                """
            )
            
    # Вкладки интерфейса
    with gr.Tabs() as tabs:
        # Вкладка 1: Анализ речи
        with gr.Tab("🎙 Анализатор"):
            with gr.Row():
                with gr.Column(scale=5):
                    gr.Markdown("### 📥 Входной аудиопоток")
                    audio_in = gr.Audio(
                        sources=["microphone", "upload"], 
                        type="filepath", 
                        label="Запишите голос или загрузите аудиофайл (.wav / .mp3)",
                        elem_classes="input-box"
                    )
                    btn = gr.Button("🚀 Запустить распределенный анализ", variant="primary")
                    
                    gr.Markdown("### 🎭 Быстрый старт (Тестовые сценарии)")
                    with gr.Row():
                        sim_btn_1 = gr.Button("🟢 Пример 1 (Вежливый)", variant="secondary")
                        sim_btn_2 = gr.Button("🔴 Пример 2 (Конфликт)", variant="secondary")
                        sim_btn_3 = gr.Button("🟡 Пример 3 (Волнение)", variant="secondary")
                    
                with gr.Column(scale=6):
                    gr.Markdown("### 📊 Отчет анализатора")
                    output_html = gr.HTML(
                        value="""
                        <div style="border: 1px dashed rgba(255,255,255,0.1); border-radius: 12px; padding: 3rem; text-align: center; color: #9ca3af;">
                            Ожидание входных данных. Запишите аудио и нажмите кнопку «Запустить анализ» или выберите тестовый сценарий для демонстрации.
                        </div>
                        """
                    )
            
            gr.Markdown("### 📈 Сводный дашборд KPI (Сессия)")
            kpi_dashboard = gr.HTML(value=generate_kpi_html())
            
            gr.Markdown("### 📜 Журнал звонков за сессию")
            history_table = gr.HTML(value=generate_history_html())
            
            # Блок скачивания мобильного приложения
            gr.HTML(value=generate_apk_download_html())
            
            # Логика событий
            btn.click(fn=predict, inputs=[audio_in], outputs=[output_html, kpi_dashboard, history_table])
            
            # Функции-обертки для демо-кнопок
            def load_sim1(): return run_simulation(1)
            def load_sim2(): return run_simulation(2)
            def load_sim3(): return run_simulation(3)
            
            sim_btn_1.click(fn=load_sim1, inputs=[], outputs=[output_html, kpi_dashboard, history_table])
            sim_btn_2.click(fn=load_sim2, inputs=[], outputs=[output_html, kpi_dashboard, history_table])
            sim_btn_3.click(fn=load_sim3, inputs=[], outputs=[output_html, kpi_dashboard, history_table])
            
        # Вкладка 2: Настройка распределения
        with gr.Tab("⚙️ Распределение вычислений"):
            gr.Markdown("### 📡 Маршрутизация моделей по узлам сети")
            
            with gr.Row():
                with gr.Column(scale=5):
                    gr.Markdown("#### Добавить вычислительный узел (Воркер)")
                    node_ip_input = gr.Textbox(
                        placeholder="Например: 100.115.20.12:7860", 
                        label="Адрес ПК в сети Tailscale (IP:Port)"
                    )
                    add_node_btn = gr.Button("➕ Зарегистрировать узел", variant="secondary")
                    add_node_status = gr.HTML(value="")
                    
                with gr.Column(scale=6):
                    gr.Markdown("#### Маршруты выполнения задач")
                    asr_select = gr.Dropdown(
                        choices=get_node_choices(), 
                        value=config.ROUTING["asr"], 
                        label="Распознавание речи (ASR / GigaAM)"
                    )
                    text_select = gr.Dropdown(
                        choices=get_node_choices(), 
                        value=config.ROUTING["text"], 
                        label="Анализ семантики текста (RuBERT)"
                    )
                    audio_select = gr.Dropdown(
                        choices=get_node_choices(), 
                        value=config.ROUTING["audio"], 
                        label="Анализ акустики звука (GigaAM Emo)"
                    )
                    failover_check = gr.Checkbox(
                        label="Авто-переключение на локальный инференс при сбое сети (Smart Network Failover)",
                        value=config.FAILOVER_TO_LOCAL
                    )
                    save_routes_btn = gr.Button("💾 Сохранить маршруты", variant="primary")
                    save_routes_status = gr.HTML(value="")
            
            # Логика добавления узла
            add_node_btn.click(
                fn=add_new_node, 
                inputs=[node_ip_input], 
                outputs=[asr_select, text_select, audio_select, add_node_status]
            )
            # Логика сохранения роутинга
            save_routes_btn.click(
                fn=save_routing,
                inputs=[asr_select, text_select, audio_select, failover_check],
                outputs=[save_routes_status]
            )
            
        # Вкладка 3: Мониторинг узлов сети
        with gr.Tab("🖥 Сетевой мониторинг") as monitor_tab:
            gr.Markdown("### 📊 Статус подключенных устройств")
            refresh_btn = gr.Button("🔄 Обновить статусы узлов", variant="secondary")
            nodes_status_html = gr.HTML(
                value="<p style='color: #9ca3af;'>Нажмите «Обновить статусы узлов» для сканирования сети...</p>"
            )
            
            # Логика обновления статусов
            refresh_btn.click(fn=refresh_nodes_status, inputs=[], outputs=[nodes_status_html])
            
            # Автоматически обновлять статус при переходе на вкладку
            monitor_tab.select(fn=refresh_nodes_status, inputs=[], outputs=[nodes_status_html])

# Монтируем Gradio в FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import threading
    import webbrowser
    import subprocess
    
    # Автоматически открываем браузер и выводим кликабельные ссылки через 2.0 секунды
    def auto_open():
        time.sleep(2.0)
        ips = get_local_ips()
        url = f"http://127.0.0.1:{config.PORT}"
        print("\n" + "=" * 70)
        print("🚀 СЕРВЕР ЗАПУЩЕН И ГОТОВ К РАБОТЕ!")
        print("👉 Ссылки для открытия в браузере (зажмите Ctrl и кликните):")
        print(f"   Локально: {url}")
        for ip in ips:
            print(f"   В сети:   http://{ip}:{config.PORT}")
        print("=" * 70 + "\n")
        
        try:
            # На Windows webbrowser.open часто работает нестабильно в venv,
            # поэтому сначала пытаемся открыть через системный 'start'
            if os.name == 'nt':
                subprocess.Popen(f"start {url}", shell=True)
            else:
                webbrowser.open(url)
        except Exception as e:
            try:
                webbrowser.open(url)
            except Exception as ex:
                print(f"[Server] Не удалось автоматически открыть браузер: {ex}")
            
    threading.Thread(target=auto_open, daemon=True).start()
    
    uvicorn.run(app, host="0.0.0.0", port=config.PORT)
