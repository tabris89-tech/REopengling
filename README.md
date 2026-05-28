# REopengling

![CI](https://github.com/tabris89-tech/REopengling/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/github/license/tabris89-tech/REopengling)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)

**AI Video Editor** — форк [OpenGling](https://github.com/poopstain112/opengling) с улучшениями.

Автоматически удаляет тишину, слова-паразиты и неудачные дубли из видео.

---

## ✨ Что нового (наши изменения)

- **URL download** — yt-dlp (1700+ сайтов: YouTube, VK, RuTube, Yandex Disk, Google Drive), прямые ссылки `.mp4`/`.webm` и HTTP streaming
- **Bootstrap** — авто-проверка FFmpeg, CUDA, spaCy, yt-dlp при запуске. Portable: ничего руками устанавливать не нужно
- **Time range (HH:MM:SS)** — обработка только нужного отрезка видео (обрезается ДО анализа, экономит время)
- **Whisper large-v3 по умолчанию** — максимальная точность распознавания
- **Streaming upload 4GB** — загрузка больших файлов чанками по 8MB без OOM
- **Веб-интерфейс: URL input** — вставь ссылку прямо в браузере (прогресс скачивания + авто-обработка)
- **Прогресс-бар в реальном времени** — больше не висит на 0% при анализе
- **Звук в экспортированном видео** — исправлена потеря аудиодорожки
- **Экспорт в 10 раз быстрее** — без повторного запуска Whisper
- **Всплывающие уведомления** — о завершении загрузки, анализа и экспорта
- **Устойчивость к ошибкам** — больше KeyError при перезапуске сервера

---

## 🚀 Быстрый старт

### 1. Установи FFmpeg

**Windows:**
```bash
winget install FFmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

### 2. Скачай и установи REopengling

```bash
git clone https://github.com/tabris89-tech/REopengling.git
cd REopengling

# Создай виртуальное окружение (рекомендуется)
python -m venv venv

# Активируй
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Установи пакет
pip install -e .

# Дополнительные возможности (по желанию):
pip install -e ".[all]"     # Всё включено
pip install -e ".[zoom]"    # Авто-зум (mediapipe, opencv)
pip install -e ".[mcp]"     # MCP сервер для Claude
pip install -e ".[youtube]" # Генерация метаданных YouTube

# Скачай модель для поиска слов-паразитов
python -m spacy download en_core_web_sm
```

### 3. Запусти веб-интерфейс

```bash
opengling serve
```

Открой браузер → http://localhost:8000

Готово. Загружай видео и смотри результат.

---

## 📖 Использование

### Веб-интерфейс (рекомендуется)

```bash
opengling serve
```

Возможности:
- Загрузить видео
- Посмотреть расшифровку
- Увидеть, что будет вырезано
- Переключить отдельные правки (оставить/вырезать)
- Экспорт в MP4, Final Cut Pro, Premiere Pro, DaVinci Resolve

### Командная строка

```bash
# Полный цикл: тишина + паразиты + дубли
opengling process video.mp4

# Скачать из URL и сразу обработать
opengling process --url https://youtube.com/watch?v=... --start 00:01:00 --end 00:05:00

# С шумоподавлением и авто-зумом
opengling process video.mp4 --noise --zoom

# С экспортом субтитров
opengling process video.mp4 --captions srt

# Скачать видео из URL
opengling download https://youtube.com/watch?v=...

# Информация о видео по ссылке
opengling inspect https://youtube.com/watch?v=...

# Только расшифровка
opengling transcribe video.mp4

# Анализ без редактирования (предпросмотр)
opengling analyze video.mp4 --detailed
```

### MCP для AI-ассистентов

REopengling можно подключить к Claude через MCP.

Добавь в свой `mcp_config.json`:
```json
{
  "mcpServers": {
    "opengling": {
      "command": "python",
      "args": ["-m", "opengling.mcp_server"]
    }
  }
}
```

> **Примечание:** В `mcp_config.json` в этом репозитории уже настроены серверы `filesystem`, `git` и `github`. Пути в нём (`C:\Users\kantairon\opengling`) замени на свои.

**Доступные инструменты:**
- `process_video` — полный пайплайн обработки
- `transcribe` — расшифровка аудио/видео
- `analyze_video` — анализ без редактирования
- `generate_captions` — генерация SRT/VTT субтитров
- `generate_youtube_metadata` — создание заголовков, описаний, глав
- `export_timeline` — экспорт в монтажные программы

### Python API

```python
from opengling import VideoProcessor, ProcessingConfig

config = ProcessingConfig(
    remove_fillers=True,
    detect_bad_takes=True,
    remove_noise=True,
    whisper_model="large-v3",
)

processor = VideoProcessor(config)
result = processor.process("input.mp4", "output.mp4")

print(f"Сэкономлено: {result.time_saved:.1f}с ({result.time_saved_percentage:.1f}%)")
print(f"Тишина удалена: {result.silences_removed}")
print(f"Паразиты удалены: {result.fillers_removed}")
print(f"Дубли удалены: {result.bad_takes_removed}")
```

---

## ⚙️ Настройки

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `silence_threshold` | 0.5 | Мин. длительность тишины для удаления (сек) |
| `silence_padding` | 0.1 | Отступ вокруг речи (сек) |
| `remove_fillers` | True | Удалять слова-паразиты |
| `detect_bad_takes` | True | Удалять неудачные дубли |
| `remove_noise` | False | Шумоподавление |
| `noise_reduction_strength` | 0.5 | Интенсивность шумоподавления |
| `auto_zoom` | False | Авто-зум по лицу |
| `max_zoom` | 1.5 | Макс. увеличение |
| `whisper_model` | "large-v3" | Модель Whisper |
| `language` | None | Язык (авто если None) |

### Модели Whisper

| Модель | Скорость | Точность | VRAM |
|--------|----------|----------|------|
| tiny | 🚀 | Низкая | ~1 GB |
| base | ⚡ | Хорошая | ~1 GB |
| small | 🚶 | Лучше | ~2 GB |
| medium | 🐢 | Отличная | ~5 GB |
| **large-v3** ⭐ | 🐌 | **Наилучшая** | ~10 GB |

---

## 🛠 Технологии

- **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** — распознавание речи
- **[yt-dlp](https://github.com/yt-dlp/yt-dlp)** — скачивание видео из 1700+ сайтов
- **[FFmpeg](https://ffmpeg.org/)** — обработка видео/аудио
- **[MoviePy](https://github.com/Zulko/moviepy)** — видеомонтаж
- **[WebRTC VAD](https://github.com/wiseman/py-webrtcvad)** — детекция речи
- **[spaCy](https://spacy.io/)** — NLP
- **[MediaPipe](https://github.com/google/mediapipe)** — детекция лица
- **[noisereduce](https://github.com/timsainb/noisereduce)** — шумоподавление
- **[Ollama](https://ollama.ai/)** — локальный LLM
- **[MCP](https://modelcontextprotocol.io/)** — Model Context Protocol

---

## 🗺️ Roadmap

- ✅ URL download (yt-dlp, 1700+ сайтов)
- ✅ Bootstrap: авто-установка FFmpeg, проверка CUDA/spaCy
- ✅ Time range (HH:MM:SS — обрезание до обработки)
- ✅ Whisper large-v3 по умолчанию
- ✅ 4GB streaming upload
- ✅ Веб-интерфейс: URL input + прогресс
- ⬜ Пакетный режим (batch processing нескольких файлов)
- ⬜ Темы оформления (тёмная/светлая)
- ⬜ drag & drop в веб-интерфейсе
- ⬜ Интеграция с Telegram ботом
- ⬜ Экспорт в Google Docs / Notion
- ⬜ Нативная Windows сборка (pyinstaller)

---

## 📝 Оригинальный проект

REopengling — форк [poopstain112/opengling](https://github.com/poopstain112/opengling).  
Распространяется под лицензией **MIT** — см. [LICENSE](LICENSE).

---

## 🤝 Вклад

Пулл-реквесты и issue приветствуются.
