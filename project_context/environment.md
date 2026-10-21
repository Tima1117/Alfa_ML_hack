# Облачное окружение Claude Code

## Подключение к LLM (настроено пользователем в настройках окружения)
- **Сетевой секрет** (Network secrets): тип Bearer, сайт `changellenge.up.railway.app`, путь `/v1/`,
  заголовок `Authorization: Bearer <ключ команды>`. Прокси окружения сам добавляет ключ к запросам —
  в контейнере ключа нет и быть не должно.
- **Переменные окружения**:
  ```
  OPENAI_BASE_URL=https://changellenge.up.railway.app/v1
  OPENAI_MODEL=qwen/qwen3.8-27b
  OPENAI_API_KEY=injected-by-proxy
  ```
  `OPENAI_API_KEY` — заглушка: без неё `LLMClient.from_env()` считает LLM ненастроенной.
- Новые настройки подхватываются только в новой сессии.
- **Расход по ключу команды**: `GET https://changellenge.up.railway.app/key/info` → `info.spend` (₽ за сутки),
  `max_budget` (600), `budget_reset_at` (00:00 UTC = 03:00 МСК). Общий на всю команду. Печатать только эти поля.
- Прокси: непотоковый запрос обрывается на ~30 с (`upstream request failed`) → только `stream=True`;
  поле `reasoning` (`{"max_tokens": N}` или `{"effort": "none"}`) управляет рассуждением Qwen, `usage.cost` — ₽.

## Сеть
- pypi доступен; `huggingface.co`, `ollama.com` заблокированы (403) — локальную модель для отладки не скачать
  без разрешения пользователя в Allowed domains.

## Python
- Системный Python 3.13, нужен 3.12 → `uv venv -p 3.12 .venv` и пакеты из `CLAUDE.md`
  (подмножество `Base/requirements.txt` без torch/autogluon/lightautoml — агенту и тестам они не нужны).
- Машина: 4 CPU, ~15 GB RAM (на платформе 8 CPU / 32 GB) — замеры времени здесь консервативные.

## Секреты
Ключ команды нигде не хранить в репо/коде/логах. Организаторы просят при проблемах передавать
название ключа и текст ошибки, но не сам ключ.
