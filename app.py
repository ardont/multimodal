import os
import shutil
import tempfile
import time
import psutil
import uvicorn
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import gradio as gr

import config
from src.pipeline import MultimodalPipeline
import src.network as network

# Инициализируем FastAPI
app = FastAPI(title="GPB MER Distributed Node API", version="1.0.0")

# Инициализируем наш мультимодальный пайплайн
pipeline = MultimodalPipeline()

# --- FastAPI ЭНДПОИНТЫ ДЛЯ УДАЛЕННЫХ ВЫЧИСЛЕНИЙ ---

class TextRequest(BaseModel):
    text: str

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
    background: rgba(255, 255, 255, 0.02) !important;
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

def save_routing(asr_target, text_target, audio_target):
    config.ROUTING["asr"] = asr_target
    config.ROUTING["text"] = text_target
    config.ROUTING["audio"] = audio_target
    
    # Строим лог роутинга для вывода пользователю
    routes_desc = f"""
    <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 8px; padding: 1rem; margin-top: 1rem;">
        <h4 style="margin: 0 0 0.5rem 0; color: #34d399;">💾 Маршруты вычислений успешно сохранены:</h4>
        <ul style="margin: 0; padding-left: 1.25rem; font-size: 0.95rem; color: #d1d5db;">
            <li><b>Распознавание (ASR):</b> {asr_target}</li>
            <li><b>Анализ текста:</b> {text_target}</li>
            <li><b>Анализ звука:</b> {audio_target}</li>
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

def predict(audio):
    if audio is None:
        return """
        <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 8px; padding: 1rem; color: #f87171; text-align: center;">
            ⚠️ Пожалуйста, запишите или загрузите аудиофайл.
        </div>
        """
        
    res = pipeline.run_analysis(audio)
    
    # Определение уровня стресса
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
    
    # Проверка скриптового соответствия
    compliance = check_compliance(res['transcription'])
    recs = generate_recommendations(res, compliance)
    
    # Стилизация чек-листа комплаенса
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
    
    # Генерация HTML-рекомендаций
    recs_html = "".join([f"<li style='margin-bottom: 0.5rem;'>{r}</li>" for r in recs])
    
    # Генерация HTML отчета
    report_html = f"""
    <div class="report-card {stress_class}" style="font-family: 'Outfit', sans-serif; padding: 1.5rem; background: rgba(10, 15, 26, 0.65); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 16px; backdrop-filter: blur(12px); color: #f3f4f6;">
        
        <!-- Верхняя панель: Заголовок и итоговый индекс -->
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 1.25rem; margin-bottom: 1.5rem; flex-wrap: wrap; gap: 1rem;">
            <div>
                <h3 style="margin: 0; font-size: 1.4rem; font-weight: 700; background: linear-gradient(to right, #60a5fa, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">📊 Результат Экспресс-Анализа</h3>
                <span style="color: #9ca3af; font-size: 0.85rem;">Звонок обработан распределенной нейросетью</span>
            </div>
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="text-align: right;">
                    <span style="font-size: 0.75rem; color: #9ca3af; display: block; text-transform: uppercase; letter-spacing: 0.05em;">Индекс аномалии</span>
                    <span style="font-size: 1.8rem; font-weight: 800; color: {gauge_color};">{res['final_stress'] * 100:.0f}%</span>
                </div>
                <span class="metric-badge {badge_class}" style="font-size: 0.9rem; padding: 0.4rem 1rem;">{status_text}</span>
            </div>
        </div>
        
        <!-- Средняя часть: Текст и Чек-лист регламента -->
        <div style="display: grid; grid-template-columns: 1.25fr 0.75fr; gap: 1.5rem; margin-bottom: 1.5rem; align-items: start;">
            
            <!-- Левая колонка: ASR и таймлайн -->
            <div>
                <span style="color: #9ca3af; font-size: 0.8rem; display: block; margin-bottom: 0.5rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Распознанный текст (ASR):</span>
                <div style="background: rgba(255,255,255,0.02); padding: 1rem; border-radius: 10px; border: 1px solid rgba(255,255,255,0.05); line-height: 1.6; font-style: italic; color: #e5e7eb; font-size: 1rem; margin-bottom: 1.25rem; max-height: 120px; overflow-y: auto;">
                    "{res['transcription']}"
                </div>
                
                <!-- Стресс-индексы -->
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

                <!-- Динамический таймлайн звонка -->
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
            
            <!-- Правая колонка: Чек-лист регламента -->
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
        
        <!-- Советник ИИ -->
        <div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.08) 0%, rgba(37, 99, 235, 0.08) 100%); border: 1px solid rgba(139, 92, 246, 0.25); border-radius: 12px; padding: 1.25rem; margin-bottom: 1.5rem; box-shadow: 0 4px 15px rgba(139, 92, 246, 0.15);">
            <h4 style="margin: 0 0 0.75rem 0; color: #c084fc; font-size: 1rem; font-weight: 700; display: flex; align-items: center; gap: 0.5rem;">
                💡 Интеллектуальные рекомендации (AI Coach):
            </h4>
            <ul style="margin: 0; padding-left: 1.25rem; font-size: 0.9rem; line-height: 1.6; color: #d1d5db; display: flex; flex-direction: column; gap: 0.5rem;">
                {recs_html}
            </ul>
        </div>
        
        <!-- Акустические метрики -->
        <h4 style="margin: 0 0 0.75rem 0; color: #e5e7eb; font-size: 1rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">🎙 Акустические характеристики звука</h4>
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
    return report_html

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
                    
                with gr.Column(scale=6):
                    gr.Markdown("### 📊 Отчет анализатора")
                    output_html = gr.HTML(
                        value="""
                        <div style="border: 1px dashed rgba(255,255,255,0.1); border-radius: 12px; padding: 3rem; text-align: center; color: #9ca3af;">
                            Ожидание входных данных. Запишите аудио и нажмите кнопку «Запустить анализ».
                        </div>
                        """
                    )
            btn.click(fn=predict, inputs=[audio_in], outputs=[output_html])
            
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
                        label="Распознавание речи (ASR / Whisper)"
                    )
                    text_select = gr.Dropdown(
                        choices=get_node_choices(), 
                        value=config.ROUTING["text"], 
                        label="Анализ семантики текста (RuBERT)"
                    )
                    audio_select = gr.Dropdown(
                        choices=get_node_choices(), 
                        value=config.ROUTING["audio"], 
                        label="Анализ акустики звука (Wav2Vec2)"
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
                inputs=[asr_select, text_select, audio_select],
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
    print(f"[Server] Запуск распределенного узла на http://0.0.0.0:{config.PORT}")
    uvicorn.run(app, host="0.0.0.0", port=config.PORT)
