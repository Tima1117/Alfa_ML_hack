# Alfa Hack AutoML — fork for the LLM-agent part

Хакатон «Альфа Хак AutoML»: AI-агент, который сам решает табличную ML-задачу от `task.md` до `submission.csv`
(цикл AutoResearch). Команда из трёх человек; **в этом форке моя роль — LLM-агент (`agent/`)**,
ML-ядро (`ml_core/`) пишет Тима, инфраструктуру (`solution.py`, ZIP) — третий участник.

**Перед работой прочитай `project_context/README.md`, `repo_map.md`, `pitfalls.md` и `insights.md`** — там кейс, роли, архитектура,
решения, статус, план и места, где раньше терялись время и контекст. PDF кейса не перечитывай — выжимка в `case.md`.

## Правила
- Для других LLM-ассистентов те же правила лежат в `AGENTS.md` — меняешь правила здесь, меняй и там.
- Ветка: `claude/fervent-lamport-31kwal`. Коммиты и push только в неё, PR — только по просьбе пользователя.
- Ключ к LLM никогда не печатать, не писать в код, конфиги, коммиты и чат. Код берёт подключение только
  из `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL` (на платформе их подставляют организаторы).
- Пользователь общается по-русски; объяснять просто, без лишнего жаргона.
- `ml_core/` — зона Тимы: правки только минимальные и отдельными коммитами, с описанием для него.
- Документацию в `project_context/` обновлять после значимых изменений (особенно `status.md`, `next_steps.md`).

## Команды
```bash
uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python numpy==1.26.4 scipy==1.12.0 pandas==2.2.3 \
  pyarrow==19.0.1 joblib==1.4.2 threadpoolctl==3.6.0 scikit-learn==1.5.2 catboost==1.2.8 lightgbm==4.6.0 \
  xgboost==2.1.4 pydantic==2.10.6 openai==1.68.2 httpx==0.28.1 tenacity==9.0.0 python-dotenv==1.0.1 pytest
.venv/bin/python -m pytest tests/ -q -p no:warnings --deselect tests/test_agent_research.py  # ~25 с, в цикле правок
.venv/bin/python -m pytest tests/ -q -p no:warnings            # ~2.5 мин, 57 тестов, перед коммитом
.venv/bin/python scripts/run_agent_local.py "Base/Datasets/Регрессия/task.md" --budget-s 600 [--no-llm]
```
Не запускай тесты параллельно с прогонами агента (замедление в 2–3 раза, искажённые замеры). Быстрые проверки —
на регрессии (`--budget-s 300`, ~2 мин); длинные прогоны — через `run_in_background`, не ждать через `pgrep -f`.
