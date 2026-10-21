# Карта репозитория: что где лежит и как с этим работать

Для человека и для LLM-ассистента. Сначала эта страница, потом — только нужный модуль
(`grep -n "^def \|^class " <файл>`, затем читать конкретную функцию). Целиком большие файлы не открывать.

## Корень
| путь | что это | чьё | трогать? |
|---|---|---|---|
| `CLAUDE.md` | правила для Claude Code: ветка, ключ, команды | LLM-агент | да, при смене правил |
| `AGENTS.md` | то же для любых других LLM-ассистентов (Codex, Cursor, …) | LLM-агент | вместе с CLAUDE.md |
| `CLEANUP_BEFORE_MAIN.md` | что можно удалить перед переносом в основной репозиторий команды | LLM-агент | да |
| `README.md` | обзор ml_core (англ., от Тимы) + раздел про агента | Тима | минимально |
| `Base/` | материалы организаторов: `Альфа Хак_AutoML.pdf`, `baseline.py`, `requirements.txt`, `Datasets/<задача>/` | организаторы | **нет** (только читать) |
| `ml_core/` | ML-ядро: профиль данных, валидация, обучение, предсказания | Тима | только минимальные правки отдельными коммитами |
| `agent/` | LLM-агент (цикл AutoResearch) | LLM-агент | да |
| `tests/` | pytest: ml_core и агент (55 тестов) | оба | да |
| `scripts/run_agent_local.py` | локальный запуск агента на одной задаче | LLM-агент | да |
| `examples/` | прогоны ml_core без LLM (Тима) и их журналы в `examples/runs/` | Тима | нет |
| `docs/CONTRACTS.md` | контракт ml_core: объекты, операции, гарантии | Тима | нет |
| `docs/AGENT_CONTRACT.md` | контракт агента: `run_research`, Executor, env-переменные | LLM-агент | при изменении интерфейса |
| `docs/acceptance/` | отчёты прогонов приёмки `agent-v1` (копии `report.md`) | LLM-агент | при новой приёмке |
| `docs/presentation/agent_deck.html` | презентация решения со схемами (открыть в браузере) | LLM-агент | при значимых изменениях |
| `project_context/` | память проекта для новых сессий (см. ниже) | все | после значимых изменений |
| `skills/` | справочники для ИИ-ассистентов (EDA, polars, sklearn, shap, lightning); кодом **не** используются | Тима | нет |
| `runs/`, `.venv/` | локальные результаты прогонов и окружение | — | в git не попадают (`.gitignore`) |

## `project_context/` — с чего начать новую сессию
| файл | когда читать |
|---|---|
| `README.md` | всегда первым: оглавление |
| `pitfalls.md` | всегда: где раньше терялись время, деньги и контекст |
| `case.md` | суть кейса, ограничения, score, правила ZIP (PDF не перечитывать) |
| `team.md` | кто что делает, что кому передать |
| `architecture.md` | как устроен один прогон агента |
| `decisions.md` | почему сделано именно так |
| `status.md` | что сделано, измеренные результаты, расход LLM |
| `next_steps.md` | что делать дальше по порядку |
| `handoff.md` | что считаем готовым результатом нашей части и когда передаём команде |
| `acceptance.md` | итог приёмки `agent-v1`: критерии, прогоны, расход, риски |
| `insights.md` | хайлайты: что узнали о качестве, LLM, прокси, деньгах и процессе |
| `environment.md` | облачное окружение: ключ, прокси, расход по ключу |
| `repo_map.md` | эта страница |

## `agent/` — модули (подробнее: `architecture.md`)
| файл | ответственность | ключевые функции |
|---|---|---|
| `research.py` | главный цикл, отчёт, точка входа | `run_research`, `_Researcher.prepare/research/finish` |
| `llm.py` | клиент Qwen: поток, JSON + починка, бюджет, ошибки прокси | `LLMClient.complete_json`, `openai_transport`, `is_budget_exceeded` |
| `prompts.py` | тексты промптов (англ.) | `UNDERSTAND_SYSTEM`, `RESEARCH_SYSTEM`, `research_user` |
| `schemas.py` | что LLM обязана вернуть | `TaskUnderstanding`, `HypothesisBatch`, `Hypothesis` |
| `context.py` | сжатые сводки данных и истории для промпта | `describe_task`, `describe_history` |
| `understand.py` | метрика и валидация (правила важнее LLM) | `decide_metric`, `decide_validation` |
| `hypotheses.py` | гипотеза → `ExperimentConfig`; seeds, playbook, local search | `materialize`, `sanitize_hyperparameters` |
| `features.py`, `_feature_sandbox_runner.py` | песочница для кода признаков от LLM | `FeatureSandbox` |
| `executor.py`, `_worker.py` | запуск ml_core в подпроцессе с жёстким таймаутом | `SubprocessExecutor`, `InProcessExecutor` |
| `task_files.py` | поиск csv рядом с `task.md` | `locate_task_files` |

## Что получается после прогона (`runs/agent/<задача>/` или `--out-dir`)
| файл | содержимое |
|---|---|
| `submission.csv` | ответ (пишется атомарно: fallback → после seeds → чекпоинты → финальный) |
| `report.md` / `report.json` | итог: метрика, валидация, журнал всех гипотез, расход LLM (`llm_usage`: вызовы, токены, `cost` в ₽) |
| `agent.log` | лог по времени; фильтр: `grep -E "(->|run «|refit|LLM|done)" agent.log` |
| `experiments/experiments.jsonl` | сырой журнал ml_core по каждому эксперименту |
| `validation/`, `features/` | сохранённое разбиение и принятый песочницей код признаков |

## Как работать
```bash
# окружение и тесты — команды в CLAUDE.md / AGENTS.md
.venv/bin/python scripts/run_agent_local.py "Base/Datasets/Регрессия/task.md" --budget-s 300            # быстрая проверка, ~5 мин
.venv/bin/python scripts/run_agent_local.py "Base/Datasets/Многоклассовая классификация/task.md" --budget-s 1800  # долго, в фоне
.venv/bin/python scripts/run_agent_local.py <task.md> --no-llm                                         # без LLM, бесплатно
```
- LLM подключается только через `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL`; ключ не печатать и не коммитить.
- Расход по ключу команды за сутки: `GET https://changellenge.up.railway.app/key/info` → `info.spend` (₽ из 600).
- Тесты и прогоны агента не запускать одновременно (4 CPU — замеры искажаются в 2–3 раза).
- Код агента не править во время прогона из этой же папки: воркер импортирует модули заново на каждый
  эксперимент. Для долгих прогонов — отдельная копия: `git worktree add --detach <папка> HEAD`.
