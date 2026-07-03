# vLLM-Omni Audio Classifier

Клиентский pipeline для асинхронной классификации аудио через vLLM-Omni HTTP `/v1/chat/completions` в режиме:

```text
input:  audio | text | audio + text
output: text only
```

Модель должна возвращать класс строго внутри XML-подобного тега:

```xml
<answer>class_name</answer>
```

Pipeline извлекает значение из `<answer>...</answer>`, нормализует его, проверяет по списку разрешённых классов из YAML-конфига и пишет результат в JSONL.

---

## 1. Для чего это нужно

Этот сервис закрывает задачу массовой разметки аудио, например:

```text
- human
- answering_machine
- unknown
```

или любую другую задачу классификации, где модель получает аудио, опциональную текстовую инструкцию и должна вернуть один допустимый класс.

Pipeline не запускает vLLM-Omni backend. Он является только клиентом поверх уже поднятого HTTP endpoint.

Ожидаемый backend-контракт:

```text
vLLM-Omni endpoint: http://host:port/v1
model name:         omni-model или другой SERVED_MODEL_NAME
request endpoint:   /chat/completions
output modalities:  ["text"]
```

---

## 2. Структура проекта

```text
vllm_omni_audio_classifier/
├── audio_classifier_service.py
├── requirements.txt
├── configs/
│   └── audio_classifier.example.yaml
├── prompts/
│   ├── audio_classifier_system.txt
│   └── few_shots.example.yaml
├── examples/
│   └── audio_samples.csv
└── outputs/
```

Назначение файлов:

```text
audio_classifier_service.py
  Основной async pipeline классификации.

requirements.txt
  Python-зависимости: httpx, PyYAML, tqdm.

configs/audio_classifier.example.yaml
  Пример полного YAML-конфига.

prompts/audio_classifier_system.txt
  Базовый системный промпт с общими правилами разметки.

prompts/few_shots.example.yaml
  Примеры few-shot разметки.

examples/audio_samples.csv
  Пример входного manifest-файла.

outputs/
  Директория для JSONL-результатов.
```

---

## 3. Установка

Из директории проекта:

```bash
pip install -r requirements.txt
```

Минимальные зависимости:

```text
httpx
PyYAML
tqdm
```

`tqdm` опционален. Если он недоступен, pipeline всё равно работает и печатает progress в stderr.

---

## 4. Быстрый запуск

Dry-run первого запроса без отправки в модель:

```bash
python audio_classifier_service.py --config configs/audio_classifier.example.yaml --dry-run
```

Реальная разметка:

```bash
python audio_classifier_service.py --config configs/audio_classifier.example.yaml
```

Ограничить обработку первыми 10 строками manifest:

```bash
python audio_classifier_service.py --config configs/audio_classifier.example.yaml --limit 10
```

---

## 5. Главная идея pipeline

Pipeline выполняет следующие шаги:

```text
1. Читает YAML-конфиг.
2. Загружает список классов.
3. Загружает system prompt из файла.
4. Компилирует итоговый system prompt: базовые правила + список классов + контракт ответа.
5. Загружает few-shot examples, если они указаны.
6. Загружает CSV/JSONL manifest с аудиосэмплами.
7. Для каждого сэмпла собирает OpenAI-compatible chat payload.
8. Отправляет запросы асинхронно с concurrency=N.
9. Получает текстовый ответ модели.
10. Извлекает класс из <answer>...</answer>.
11. Проверяет класс по YAML classes.
12. Пишет результат построчно в JSONL.
```

Итоговый chat-контекст имеет такую структуру:

```text
system

user few-shot-1
assistant few-shot-1

user few-shot-2
assistant few-shot-2

...

user few-shot-N
assistant few-shot-N

user REAL-QUERY
```

Few-shot не вставляются в system prompt. Это полноценные демонстрационные chat-turn пары перед реальным запросом.

---

## 6. YAML-конфиг

Пример полного конфига:

```yaml
endpoint:
  base_url: "http://127.0.0.1:48005/v1"
  model: "omni-model"
  api_key: ""
  api_key_env: "OMNI_API_KEY"

sampling:
  max_tokens: 64
  temperature: 0.0
  top_p: 1.0
  repetition_penalty: 1.0
  presence_penalty: 0.0
  frequency_penalty: 0.0
  seed: null
  stop: []

io:
  input_path: "examples/audio_samples.csv"
  input_format: "auto"
  output_path: "outputs/audio_labels.jsonl"
  output_format: "jsonl"
  id_field: "id"
  audio_field: "audio_path"
  text_field: "text"
  resume: true

runtime:
  concurrency: 8
  timeout_seconds: 300
  retries: 2
  retry_backoff_seconds: 2.0
  request_jitter_seconds: 0.0

prompt:
  system_prompt_path: "prompts/audio_classifier_system.txt"
  user_template: |
    Разметь текущий аудиосэмпл.

    Дополнительная информация/инструкция:
    {text}

    Верни только один класс в формате <answer>class_name</answer>.
  few_shots_path: "prompts/few_shots.example.yaml"

classes:
  - name: "human"
    description: "Живой человек отвечает, говорит, реагирует на обращение или ведёт диалог."
  - name: "answering_machine"
    description: "Автоответчик, голосовая почта, роботизированное сообщение, недоступность абонента, сигнал после сообщения."
  - name: "unknown"
    description: "Невозможно уверенно определить класс из-за шума, тишины, обрыва или недостатка информации."

parsing:
  answer_tag: "answer"
  case_sensitive: false
  allow_bare_label: false
  strip_quotes: true
```

---

## 7. Секция `endpoint`

```yaml
endpoint:
  base_url: "http://127.0.0.1:48005/v1"
  model: "omni-model"
  api_key: ""
  api_key_env: "OMNI_API_KEY"
```

Поля:

```text
base_url
  OpenAI-compatible base URL. Pipeline сам добавляет /chat/completions.

model
  Served model name. Должен совпадать с SERVED_MODEL_NAME на стороне vLLM-Omni.

api_key
  Явный API key. Можно оставить пустым.

api_key_env
  Имя env-переменной, из которой брать API key, если api_key пустой.
```

Если backend без авторизации, оставь:

```yaml
api_key: ""
api_key_env: "OMNI_API_KEY"
```

Если нужен ключ:

```bash
export OMNI_API_KEY=secret-token
```

---

## 8. Секция `sampling`

```yaml
sampling:
  max_tokens: 64
  temperature: 0.0
  top_p: 1.0
  repetition_penalty: 1.0
  presence_penalty: 0.0
  frequency_penalty: 0.0
  seed: null
  stop: []
```

Поля напрямую попадают в JSON body `/chat/completions`.

Рекомендованный режим для классификации:

```yaml
max_tokens: 32
or
max_tokens: 64

temperature: 0.0
top_p: 1.0
repetition_penalty: 1.0
presence_penalty: 0.0
frequency_penalty: 0.0
```

Почему так:

```text
temperature: 0.0
  Максимальная стабильность. Для разметки это обычно лучше вариативности.

top_p: 1.0
  Не ограничиваем nucleus sampling, потому что temperature=0 уже делает decoding детерминированным.

max_tokens: 32/64
  Ответ должен быть коротким: <answer>class_name</answer>.

repetition_penalty: 1.0
  Для короткого класса penalty обычно не нужен.
```

Для спорных аудио можно попробовать:

```yaml
temperature: 0.2
top_p: 0.9
repetition_penalty: 1.05
```

Но для production-разметки лучше начинать с greedy.

---

## 9. Секция `io`

```yaml
io:
  input_path: "examples/audio_samples.csv"
  input_format: "auto"
  output_path: "outputs/audio_labels.jsonl"
  output_format: "jsonl"
  id_field: "id"
  audio_field: "audio_path"
  text_field: "text"
  resume: true
```

Поля:

```text
input_path
  Путь до входного manifest-файла.

input_format
  auto | csv | jsonl.
  auto определяет формат по расширению .csv, .jsonl или .ndjson.

output_path
  Путь до выходного JSONL-файла.

output_format
  Сейчас поддерживается jsonl.

id_field
  Имя поля с ID сэмпла.
  Если ID нет, используется индекс строки.

audio_field
  Имя поля с путём до аудио.

text_field
  Имя поля с дополнительным текстом/инструкцией.

resume
  Если true, pipeline читает output JSONL и пропускает уже обработанные id.
```

---

## 10. Формат входного CSV

Пример:

```csv
id,audio_path,text
sample_001,/data/audio/call_001.mp3,"Определи тип ответа в звонке."
sample_002,/data/audio/call_002.wav,"Определи тип ответа в звонке."
sample_003,,"Текстовый тест без аудио."
```

Минимальные допустимые варианты строки:

```text
audio_path only
  Модель получает только аудио и user_template.

text only
  Модель получает только текст.

audio_path + text
  Модель получает аудио и дополнительный текст.
```

Если в строке нет ни audio_path, ни text, сэмпл завершится ошибкой.

---

## 11. Формат входного JSONL

Пример:

```jsonl
{"id":"sample_001","audio_path":"/data/audio/call_001.mp3","text":"Определи тип ответа в звонке."}
{"id":"sample_002","audio_path":"/data/audio/call_002.wav","text":"Определи тип ответа в звонке."}
```

JSONL удобен, если у сэмплов есть дополнительные поля:

```jsonl
{"id":"sample_003","audio_path":"/data/audio/call_003.mp3","text":"Определи тип ответа.","source":"batch_17","duration_sec":12.3}
```

Все поля, кроме `id_field`, `audio_field`, `text_field`, попадают в `extra`. Их можно использовать в `user_template` как placeholders.

Например, если строка содержит `duration_sec`, можно использовать:

```yaml
user_template: |
  Разметь аудио.
  Длительность: {duration_sec} секунд.
  Дополнительная инструкция: {text}
  Верни <answer>class_name</answer>.
```

---

## 12. Пути до аудио

`audio_path` может быть:

```text
- локальный путь
- http:// URL
- https:// URL
- data: URL
```

Локальные файлы автоматически читаются и кодируются в `data:<mime>;base64,...`.

MIME определяется через расширение файла. Например:

```text
.wav -> audio/x-wav или audio/wav, зависит от mimetypes
.mp3 -> audio/mpeg
.flac -> audio/flac
```

Если MIME определить не удалось, используется fallback `audio/wav`.

Важно: локальные относительные пути в manifest считаются относительно директории manifest-файла.

Пример:

```text
configs/audio_classifier.yaml
examples/audio_samples.csv
examples/audio/call_001.mp3
```

Если в CSV указано:

```csv
audio/call_001.mp3
```

то путь будет резолвиться как:

```text
examples/audio/call_001.mp3
```

---

## 13. Секция `runtime`

```yaml
runtime:
  concurrency: 8
  timeout_seconds: 300
  retries: 2
  retry_backoff_seconds: 2.0
  request_jitter_seconds: 0.0
```

Поля:

```text
concurrency
  Сколько сэмплов размечать одновременно.

timeout_seconds
  HTTP timeout на один запрос.

retries
  Сколько повторных попыток делать после ошибки.
  Фактическое число попыток = retries + 1.

retry_backoff_seconds
  Базовая задержка между retry.
  Задержка растёт как backoff * attempt.

request_jitter_seconds
  Случайная задержка перед запросом от 0 до указанного значения.
  Может снизить синхронные всплески нагрузки.
```

Рекомендации для одной H100:

```yaml
runtime:
  concurrency: 1
  timeout_seconds: 300
  retries: 2
  retry_backoff_seconds: 2.0
  request_jitter_seconds: 0.0
```

После стабильного запуска можно поднять:

```yaml
concurrency: 2
```

Если backend показывает хороший запас KV cache и стабильную latency, можно пробовать выше. Для тяжёлых аудио и длинного `MAX_MODEL_LEN=32768` высокий concurrency может быстро привести к очередям, OOM или таймаутам.

---

## 14. Секция `classes`

Классы задаются списком строк или mapping-объектов.

Короткий вариант:

```yaml
classes:
  - human
  - answering_machine
  - unknown
```

Расширенный вариант:

```yaml
classes:
  - name: "human"
    description: "Живой человек отвечает, говорит, реагирует на обращение или ведёт диалог."
  - name: "answering_machine"
    description: "Автоответчик, голосовая почта, роботизированное сообщение, недоступность абонента, сигнал после сообщения."
  - name: "unknown"
    description: "Невозможно уверенно определить класс из-за шума, тишины, обрыва или недостатка информации."
```

Descriptions автоматически добавляются в скомпилированный system prompt.

Ограничения:

```text
- class name не должен быть пустым
- class names должны быть уникальными
- модель обязана вернуть один из этих классов
```

---

## 15. Секция `prompt`

```yaml
prompt:
  system_prompt_path: "prompts/audio_classifier_system.txt"
  user_template: |
    Разметь текущий аудиосэмпл.

    Дополнительная информация/инструкция:
    {text}

    Верни только один класс в формате <answer>class_name</answer>.
  few_shots_path: "prompts/few_shots.example.yaml"
```

Поля:

```text
system_prompt_path
  Файл с базовыми правилами классификации.

user_template
  Шаблон user prompt для каждого реального сэмпла.

few_shots_path
  Optional YAML-файл с few-shot examples.
```

---

## 16. Как компилируется system prompt

Файл `system_prompt_path` не используется как есть. Pipeline добавляет к нему два блока:

```text
1. Список разрешённых классов.
2. Жёсткий контракт формата ответа.
```

Например, если `audio_classifier_system.txt` содержит:

```text
Ты классификатор аудио. Твоя задача — определить ровно один класс из разрешённого списка.

Правила:
- Используй только информацию из аудио и дополнительной текстовой инструкции пользователя.
- Не придумывай факты, которых нет в аудио.
- Не добавляй объяснения вне тега ответа.
```

то итоговый system prompt будет примерно таким:

```text
Ты классификатор аудио. Твоя задача — определить ровно один класс из разрешённого списка.

Правила:
- Используй только информацию из аудио и дополнительной текстовой инструкции пользователя.
- Не придумывай факты, которых нет в аудио.
- Не добавляй объяснения вне тега ответа.

Разрешённые классы:
- human: Живой человек отвечает, говорит, реагирует на обращение или ведёт диалог.
- answering_machine: Автоответчик, голосовая почта, роботизированное сообщение, недоступность абонента, сигнал после сообщения.
- unknown: Невозможно уверенно определить класс из-за шума, тишины, обрыва или недостатка информации.

Формат ответа строго обязателен:
- Верни ровно один тег <answer>...</answer>.
- Внутри тега должен быть ровно один класс из списка: human, answering_machine, unknown.
- Никаких пояснений, JSON, markdown, дополнительных тегов или текста вне answer не добавляй.
```

Так проще менять классы в YAML, не переписывая руками prompt-файл.

---

## 17. `user_template` и placeholders

`user_template` форматируется через Python `.format(...)`.

Доступные стандартные placeholders:

```text
{id}
  ID сэмпла.

{index}
  Индекс строки во входном manifest.

{text}
  Значение из text_field. Если текста нет, пустая строка.

{classes}
  Список имён классов через запятую.
```

Также доступны поля из `extra`, то есть любые поля manifest, кроме id/audio/text.

Пример:

```yaml
user_template: |
  Это реальный аудиосэмпл для разметки.
  Используй приложенное аудио как основной источник истины.

  ID сэмпла: {id}
  Источник: {source}
  Дополнительная инструкция: {text}

  Выбери один класс из списка: {classes}.
  Верни только <answer>class_name</answer>.
```

Если placeholder отсутствует в данных, pipeline завершится `ConfigError`.

---

## 18. Few-shot: порядок сообщений

Few-shot examples вставляются перед реальным запросом как пары user/assistant.

Итоговый порядок сообщений:

```text
system

user few-shot-1
assistant few-shot-1

user few-shot-2
assistant few-shot-2

...

user few-shot-N
assistant few-shot-N

user REAL-QUERY
```

Few-shot assistant answer всегда компилируется так:

```xml
<answer>label</answer>
```

Это закрепляет формат ответа для модели.

---

## 19. Few-shot без аудио

Пример `few_shots.example.yaml`:

```yaml
- text: "В аудио слышен живой человек: он здоровается и отвечает на вопрос оператора."
  label: "human"

- text: "В аудио слышно автоматическое сообщение: абонент недоступен, оставьте сообщение после сигнала."
  label: "answering_machine"
```

В таком режиме user few-shot содержит только text part. Аудио не прикладывается.

Это нормально, если такие examples явно воспринимаются как текстовые описания аудио. Они помогают:

```text
- закрепить классы
- показать формат ответа
- показать граничные случаи
- объяснить правила разметки
```

Но они не обучают модель акустическим паттернам напрямую:

```text
- шум
- бип
- пауза
- синтетический голос
- интонация
- автоответчик по звучанию
```

Чтобы не спутать модель, text-only few-shot лучше писать явно:

```yaml
- text: |
    Это демонстрационный пример. В этом примере аудио заменено текстовым описанием.

    Описание аудио:
    Слышно автоматическое сообщение: абонент недоступен, затем короткий сигнал.
  label: "answering_machine"
```

А реальный запрос в `user_template` лучше формулировать иначе:

```yaml
user_template: |
  Это реальный аудиосэмпл для разметки.
  Используй приложенное аудио как основной источник истины.

  Дополнительная инструкция:
  {text}

  Верни только один класс в формате <answer>class_name</answer>.
```

Тогда модель видит разницу:

```text
few-shot:
  демонстрационный текстовый пример

real query:
  реальный аудиосэмпл
```

---

## 20. Few-shot с аудио

Пример:

```yaml
- audio_path: "/data/few_shots/human_001.wav"
  text: "Это демонстрационный пример. Классифицируй приложенное аудио."
  label: "human"

- audio_path: "/data/few_shots/am_001.wav"
  text: "Это демонстрационный пример. Классифицируй приложенное аудио."
  label: "answering_machine"
```

Тогда few-shot user turn будет multimodal:

```text
user:
  audio_url
  text

assistant:
  <answer>label</answer>
```

Плюс audio few-shot:

```text
- ближе к реальной задаче
- может помочь при акустических граничных случаях
- показывает модели реальные примеры звучания классов
```

Минусы:

```text
- каждый request тащит few-shot audio заново
- base64 payload становится больше
- растёт network overhead
- растёт preprocessing cost
- растёт latency
- уменьшается полезный контекст под target sample
```

Практический режим:

```text
1. Сначала 3–8 text-described few-shot.
2. Если качество недостаточно — добавить 2–4 коротких audio few-shot.
3. Не добавлять много длинных audio few-shot на одной H100.
```

---

## 21. Секция `parsing`

```yaml
parsing:
  answer_tag: "answer"
  case_sensitive: false
  allow_bare_label: false
  strip_quotes: true
```

Поля:

```text
answer_tag
  Имя тега для извлечения класса.
  По умолчанию answer.

case_sensitive
  Если false, HUMAN, Human и human будут сопоставлены с human.

allow_bare_label
  Если true, pipeline принимает ответ без тега, если весь текст равен допустимому классу.
  Для production лучше false.

strip_quotes
  Если true, убирает кавычки вокруг извлечённого label.
```

Рекомендуемый строгий режим:

```yaml
parsing:
  answer_tag: "answer"
  case_sensitive: false
  allow_bare_label: false
  strip_quotes: true
```

---

## 22. Как собирается HTTP payload

Для каждого target sample формируется payload:

```json
{
  "model": "omni-model",
  "messages": [
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "...compiled system prompt..."
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "audio_url",
          "audio_url": {
            "url": "data:audio/mpeg;base64,..."
          }
        },
        {
          "type": "text",
          "text": "...compiled user prompt..."
        }
      ]
    }
  ],
  "modalities": ["text"],
  "stream": false,
  "max_tokens": 64,
  "temperature": 0.0,
  "top_p": 1.0,
  "repetition_penalty": 1.0
}
```

Если few-shot включены, они вставляются между system и target user.

Ключевой момент:

```json
"modalities": ["text"]
```

Именно это требует text-only output. Модель не должна генерировать audio output.

---

## 23. Dry-run

Dry-run нужен, чтобы проверить итоговую компиляцию запроса без отправки в модель.

```bash
python audio_classifier_service.py --config configs/audio_classifier.example.yaml --dry-run
```

Он печатает первый pending request в JSON.

Проверяй в dry-run:

```text
- правильный base_url не нужен, запрос не отправляется
- system prompt содержит правила, классы и output contract
- few-shot идут turn-парами
- target user содержит audio_url и text
- modalities равно ["text"]
- sampling параметры корректные
```

---

## 24. Resume

Если `io.resume: true`, pipeline читает уже существующий `output_path` и собирает множество обработанных `id`.

```yaml
io:
  resume: true
```

Поведение:

```text
- если id уже есть в output JSONL, сэмпл пропускается
- статус не учитывается: ok, invalid_answer и failed считаются завершёнными
- чтобы переобработать всё, удали output JSONL или поставь resume: false
```

Если нужно переобработать только failed/invalid, проще сделать отдельный manifest из плохих строк output.

---

## 25. Output JSONL

Пример успешной строки:

```json
{"id":"sample_001","index":0,"status":"ok","class":"answering_machine","extracted_answer":"answering_machine","valid":true,"error":null,"raw_text":"<answer>answering_machine</answer>","attempts":1,"latency_seconds":2.431,"audio_path":"/data/audio/call_001.mp3","text":"Определи тип ответа в звонке.","extra":{},"usage":{"prompt_tokens":123,"completion_tokens":8,"total_tokens":131}}
```

Поля:

```text
id
  ID сэмпла.

index
  Индекс строки во входном manifest.

status
  ok | invalid_answer | failed.

class
  Нормализованный валидный класс или null.

extracted_answer
  То, что было внутри <answer>...</answer>, до нормализации.

valid
  true, если класс валиден.

error
  Ошибка парсинга или запроса.

raw_text
  Сырой текст ответа модели.

attempts
  Количество попыток.

latency_seconds
  Время обработки сэмпла.

audio_path
  Исходный audio_path из manifest.

text
  Исходный text из manifest.

extra
  Остальные поля manifest.

usage
  Usage из ответа vLLM, если он пришёл.
```

---

## 26. Статусы результата

```text
ok
  Модель вернула ровно один тег <answer>...</answer>, а класс валиден.

invalid_answer
  HTTP-запрос прошёл, но ответ не соответствует контракту.
  Например: нет answer tag, несколько answer tags, класс не из списка.

failed
  Запрос не удалось выполнить после всех retry.
  Например: timeout, HTTP 500, connection error, ошибка чтения аудио.
```

`invalid_answer` не ретраится, потому что технически запрос выполнен. Если нужно ретраить invalid answers, это можно добавить отдельным режимом, но по умолчанию лучше не скрывать проблемы prompt-контракта.

---

## 27. Типичные конфиги

### 27.1 Строгая production-разметка

```yaml
sampling:
  max_tokens: 32
  temperature: 0.0
  top_p: 1.0
  repetition_penalty: 1.0
  presence_penalty: 0.0
  frequency_penalty: 0.0
  seed: null
  stop: []

runtime:
  concurrency: 1
  timeout_seconds: 300
  retries: 2
  retry_backoff_seconds: 2.0
  request_jitter_seconds: 0.0
```

### 27.2 Более мягкий режим для спорных аудио

```yaml
sampling:
  max_tokens: 64
  temperature: 0.2
  top_p: 0.9
  repetition_penalty: 1.05
  presence_penalty: 0.0
  frequency_penalty: 0.0
```

### 27.3 Высокая асинхронность для коротких файлов

```yaml
runtime:
  concurrency: 4
  timeout_seconds: 300
  retries: 2
  retry_backoff_seconds: 2.0
  request_jitter_seconds: 0.05
```

Использовать только после проверки, что backend выдерживает нагрузку.

---

## 28. Рекомендации под Instruct / Thinking

Pipeline одинаково работает с Instruct и Thinking, если backend поднят в режиме:

```text
audio/text -> text
```

Практическая разница:

```text
Instruct
  Обычно лучше следует пользовательским инструкциям и формату.
  Может быть предпочтительнее для строгой классификации с классами.

Thinking
  Может быть сильнее для сложных рассуждений по аудио/тексту.
  Но для production-классификации нужно жёстко запрещать вывод reasoning и требовать только <answer>...</answer>.
```

Для Thinking в system prompt особенно полезно добавить:

```text
Не выводи ход рассуждений. Верни только итоговый класс внутри <answer>...</answer>.
```

---

## 29. Пример system prompt для автоответчика

```text
Ты классификатор аудио телефонных звонков. Нужно определить, кто или что отвечает в начале звонка.

Главное правило:
- Используй аудио как основной источник истины.
- Если есть дополнительная текстовая инструкция пользователя, используй её только как уточнение задачи.
- Не придумывай детали, которых нет в аудио.
- Не выводи рассуждения.
- Не объясняй выбор.
- Верни только один XML-подобный тег answer.

Критерии:
- human: живой человек говорит, отвечает на обращение, реагирует на ситуацию или ведёт диалог.
- answering_machine: автоответчик, голосовая почта, роботизированное сообщение, операторская фраза недоступности, сообщение после которого ожидается сигнал.
- unknown: аудио слишком плохое, слишком короткое, пустое, обрывочное или не позволяет уверенно выбрать human/answering_machine.

Если слышен автоматический текст вроде "абонент недоступен", "оставьте сообщение", "после сигнала" — выбирай answering_machine.
Если слышен живой человек, даже короткое "алло" — выбирай human.
Если есть только шум/тишина/обрыв — выбирай unknown.
```

Классы всё равно лучше держать в YAML `classes`, а не только в prompt-файле, потому что именно YAML используется для валидации.

---

## 30. Пример few-shot для автоответчика

```yaml
- text: |
    Это демонстрационный пример. В этом примере аудио заменено текстовым описанием.

    Описание аудио:
    Слышен живой человек. Он говорит "алло" и затем ждёт ответа.
  label: "human"

- text: |
    Это демонстрационный пример. В этом примере аудио заменено текстовым описанием.

    Описание аудио:
    Слышно автоматическое сообщение: "Абонент временно недоступен".
  label: "answering_machine"

- text: |
    Это демонстрационный пример. В этом примере аудио заменено текстовым описанием.

    Описание аудио:
    В записи почти тишина, речь неразборчива, слышен только шум.
  label: "unknown"
```

---

## 31. Пример manifest для автоответчика

```csv
id,audio_path,text
call_001,/data/calls/call_001.mp3,"Определи, живой человек это или автоответчик."
call_002,/data/calls/call_002.wav,"Определи, живой человек это или автоответчик."
call_003,/data/calls/call_003.flac,"Определи, живой человек это или автоответчик."
```

---

## 32. Команды запуска для реального проекта

Dry-run:

```bash
python /home/user5/vllm_omni_audio_classifier/audio_classifier_service.py --config /home/user5/vllm_omni_audio_classifier/configs/audio_classifier.yaml --dry-run
```

Первые 10 сэмплов:

```bash
python /home/user5/vllm_omni_audio_classifier/audio_classifier_service.py --config /home/user5/vllm_omni_audio_classifier/configs/audio_classifier.yaml --limit 10
```

Полная разметка:

```bash
python /home/user5/vllm_omni_audio_classifier/audio_classifier_service.py --config /home/user5/vllm_omni_audio_classifier/configs/audio_classifier.yaml
```

Посмотреть плохие ответы:

```bash
python - <<'PY'
import json
from pathlib import Path
path = Path('/home/user5/vllm_omni_audio_classifier/outputs/audio_labels.jsonl')
for line in path.read_text(encoding='utf-8').splitlines():
    item = json.loads(line)
    if item.get('status') != 'ok':
        print(json.dumps(item, ensure_ascii=False, indent=2))
PY
```

Если нужен строго однострочный shell-вариант:

```bash
python -c "import json; from pathlib import Path; path=Path('/home/user5/vllm_omni_audio_classifier/outputs/audio_labels.jsonl'); [print(json.dumps(item, ensure_ascii=False)) for item in map(json.loads, path.read_text(encoding='utf-8').splitlines()) if item.get('status') != 'ok']"
```

---

## 33. Производительность и concurrency

На стороне backend для одной H100 часто реальный bottleneck — не Python-клиент, а vLLM-Omni Stage 0: модель, KV cache, audio encoder preprocessing и prefill.

Стартовая рекомендация:

```yaml
runtime:
  concurrency: 1
```

Дальше повышать постепенно:

```text
1 -> 2 -> 4
```

После каждого шага смотреть:

```text
- latency_seconds в output JSONL
- timeout/failure rate
- GPU memory
- vLLM logs по KV cache / preemption / OOM
- HTTP 500/timeout
```

Если много timeout:

```text
- уменьшить concurrency
- увеличить timeout_seconds
- уменьшить max_tokens
- сократить few-shot
- убрать audio few-shot
- уменьшить MAX_MODEL_LEN на backend
```

---

## 34. Частые ошибки

### 34.1 `Missing <answer>...</answer> tag`

Модель не соблюла формат.

Что сделать:

```text
- temperature: 0.0
- max_tokens: 32 или 64
- усилить system prompt
- добавить few-shot с правильным <answer>label</answer>
- оставить allow_bare_label: false, чтобы видеть проблему явно
```

### 34.2 `Extracted label is not in configured classes`

Модель вернула класс, которого нет в YAML.

Что сделать:

```text
- проверить class names
- не использовать русские/английские варианты вперемешку без правил
- добавить descriptions
- добавить few-shot по спорному классу
```

### 34.3 `Audio file not found`

Локальный путь из manifest не найден.

Что проверить:

```text
- абсолютный путь
- относительный путь от директории manifest
- mount директории, если запускаешь внутри контейнера
```

### 34.4 HTTP 404

Обычно неправильный `endpoint.base_url`.

Должно быть:

```yaml
endpoint:
  base_url: "http://127.0.0.1:48005/v1"
```

Pipeline сам добавляет `/chat/completions`. Не надо писать:

```text
http://127.0.0.1:48005/v1/chat/completions
```

### 34.5 HTTP timeout

Что сделать:

```text
- уменьшить runtime.concurrency
- увеличить runtime.timeout_seconds
- проверить, что backend не упал
- уменьшить audio few-shot
- уменьшить max_tokens
```

### 34.6 Слишком большой JSON body

Причина обычно в длинных audio few-shot или длинных аудиофайлах.

Что сделать:

```text
- использовать text-described few-shot вместо audio few-shot
- обрезать аудио до нужного фрагмента
- использовать remote URL, если backend имеет доступ к нему
- уменьшить число few-shot
```

---

## 35. Практическая схема разработки разметчика

Рекомендуемый порядок:

```text
1. Запустить backend vLLM-Omni в режиме audio/text -> text.
2. Проверить один audio/text запрос простым smoke-клиентом.
3. Создать YAML config.
4. Написать system prompt без few-shot.
5. Сделать dry-run и проверить payload.
6. Запустить --limit 10.
7. Посмотреть output JSONL.
8. Исправить prompt/classes.
9. Добавить text-described few-shot.
10. Снова --limit 10/50.
11. Если качество стабильно — запустить полный manifest.
12. Если есть акустические ошибки — добавить 2–4 audio few-shot.
13. Поднять concurrency только после стабильной одиночной разметки.
```

---

## 36. Минимальный production config для автоответчика

```yaml
endpoint:
  base_url: "http://127.0.0.1:48005/v1"
  model: "omni-model"
  api_key: ""
  api_key_env: "OMNI_API_KEY"

sampling:
  max_tokens: 32
  temperature: 0.0
  top_p: 1.0
  repetition_penalty: 1.0
  presence_penalty: 0.0
  frequency_penalty: 0.0
  seed: null
  stop: []

io:
  input_path: "/data/manifests/calls.csv"
  input_format: "csv"
  output_path: "/data/outputs/call_labels.jsonl"
  output_format: "jsonl"
  id_field: "id"
  audio_field: "audio_path"
  text_field: "text"
  resume: true

runtime:
  concurrency: 1
  timeout_seconds: 300
  retries: 2
  retry_backoff_seconds: 2.0
  request_jitter_seconds: 0.0

prompt:
  system_prompt_path: "/data/prompts/audio_classifier_system.txt"
  user_template: |
    Это реальный аудиосэмпл для разметки.
    Используй приложенное аудио как основной источник истины.

    Дополнительная инструкция:
    {text}

    Верни только один класс в формате <answer>class_name</answer>.
  few_shots_path: "/data/prompts/few_shots.yaml"

classes:
  - name: "human"
    description: "Живой человек отвечает, говорит, реагирует на обращение или ведёт диалог."
  - name: "answering_machine"
    description: "Автоответчик, голосовая почта, роботизированное сообщение, недоступность абонента, сигнал после сообщения."
  - name: "unknown"
    description: "Невозможно уверенно определить класс из-за шума, тишины, обрыва или недостатка информации."

parsing:
  answer_tag: "answer"
  case_sensitive: false
  allow_bare_label: false
  strip_quotes: true
```

---

## 37. Контракт качества

Pipeline сам по себе не гарантирует правильность класса. Он гарантирует технический контракт:

```text
- запросы уходят в endpoint
- output запрошен как text-only
- ответ извлекается только из <answer>...</answer>
- класс валидируется по YAML
- ошибки не теряются
- output пишется инкрементально
- можно продолжить разметку через resume
```

Качество классификации зависит от:

```text
- checkpoint: Instruct/Thinking
- system prompt
- descriptions классов
- few-shot examples
- sampling
- качества аудио
- длины и содержательности аудиофрагмента
- backend limits: MAX_MODEL_LEN, concurrency, available KV cache
```

---

## 38. Что важно не смешивать

```text
Backend config
  Как поднята модель: H100, vLLM-Omni, stage config, max model len, served model name.

Classifier YAML
  Как клиент размечает: endpoint, sampling, classes, prompt, few-shot, concurrency.

Manifest
  Что размечается: id, audio_path, text, дополнительные поля.

Output JSONL
  Что получилось: class, raw_text, status, error, usage, latency.
```

Не стоит зашивать классы в backend. Классы — это задача клиентского classifier pipeline.

---

## 39. Короткий чеклист перед полной разметкой

```text
[ ] Backend отвечает на /v1/models.
[ ] Один audio/text запрос проходит через простой smoke-клиент.
[ ] YAML endpoint.base_url заканчивается на /v1.
[ ] endpoint.model совпадает с SERVED_MODEL_NAME.
[ ] classes уникальны.
[ ] system_prompt_path существует.
[ ] few_shots_path либо существует, либо пустой/null.
[ ] input_path существует.
[ ] output_path ведёт в доступную директорию.
[ ] dry-run показывает modalities: ["text"].
[ ] few-shot порядок: system -> user/assistant pairs -> real user.
[ ] sampling.temperature = 0.0 для первой production-разметки.
[ ] runtime.concurrency начинается с 1 или 2.
[ ] resume=true, если разметка большая.
```
