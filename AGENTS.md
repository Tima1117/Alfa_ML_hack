# AGENTS.md — вход для любого LLM-ассистента

Проект: решение кейса «Альфа Хак AutoML» — AI-агент, который сам решает табличную ML-задачу от `task.md`
до `submission.csv` (цикл AutoResearch: гипотеза → эксперимент → оценка → следующая гипотеза).
Claude Code читает `CLAUDE.md`; правила там те же.

## Прочитать перед работой (по порядку, не дольше пары минут)
1. `project_context/repo_map.md` — что где лежит, кто владелец, что можно трогать.
2. `project_context/pitfalls.md` — где раньше терялись время, деньги и контекст.
3. `project_context/insights.md` — главные находки: что работает, особенности прокси Qwen, во что обходится LLM.
4. `project_context/status.md` и `next_steps.md` — где мы сейчас и что дальше.
5. По задаче: `docs/AGENT_CONTRACT.md` (агент), `docs/CONTRACTS.md` (ml_core), `project_context/case.md` (кейс).
   PDF кейса не читать — выжимка в `case.md`.

## Правила
- Ключ к LLM нигде не печатать и не сохранять (код, конфиги, логи, коммиты, чат). Подключение — только
  через `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL`.
- LLM-лимит команды общий: 600 ₽ в сутки. Перед дорогими прогонами смотреть расход
  (`GET https://changellenge.up.railway.app/key/info` → `info.spend`), согласовывать бюджет с человеком.
- `Base/` не менять. `ml_core/` — зона Тимы: только минимальные правки отдельными коммитами с описанием.
- Только библиотеки из `Base/requirements.txt`; Python 3.12; ставить пакеты из кода нельзя.
- После значимых изменений обновлять `project_context/status.md` и `next_steps.md`.
- Общение с пользователем — по-русски, просто, без лишнего жаргона.

## Команды
```bash
uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python numpy==1.26.4 scipy==1.12.0 pandas==2.2.3 \
  pyarrow==19.0.1 joblib==1.4.2 threadpoolctl==3.6.0 scikit-learn==1.5.2 catboost==1.2.8 lightgbm==4.6.0 \
  xgboost==2.1.4 pydantic==2.10.6 openai==1.68.2 httpx==0.28.1 tenacity==9.0.0 python-dotenv==1.0.1 pytest
.venv/bin/python -m pytest tests/ -q -p no:warnings --deselect tests/test_agent_research.py  # ~20 с, в цикле правок
.venv/bin/python -m pytest tests/ -q -p no:warnings                                          # ~2 мин, перед коммитом
.venv/bin/python scripts/run_agent_local.py "Base/Datasets/Регрессия/task.md" --budget-s 300 [--no-llm]
```
Тесты и прогоны агента одновременно не запускать. Долгие прогоны — в фоне и из отдельной копии
(`git worktree add --detach <папка> HEAD`), если параллельно правите код.
