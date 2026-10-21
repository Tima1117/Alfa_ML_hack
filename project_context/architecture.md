# Архитектура

```
task.md ──▶ agent.research.run_research ──▶ Executor ──▶ ml_core (profiling / validation / training / prediction)
                 │        ▲                                   │
                 ▼        │ JSON-гипотезы                     ▼
            LLM (Qwen) ───┘                     experiments.jsonl, submission.csv (атомарно)
```

## ml_core/ (Тима) — см. docs/CONTRACTS.md
`profiling.infer_task_spec / profile_data`, `validation.create_validation`, `training.run_experiment`
(никогда не падает, результат со статусом), `prediction.refit_and_predict / write_fallback_submission`,
`experiment_log` (append-only jsonl), `model_registry` (baseline, ridge, logistic_regression, random_forest,
hist_gradient_boosting, catboost, lightgbm, xgboost), `feature_steps` (добавлено нами).

## agent/ (наша часть) — см. docs/AGENT_CONTRACT.md
| модуль | что делает |
|---|---|
| `research.py` | главный цикл и точка входа `run_research(task_md, deadline_ts, executor, llm, ...)`, отчёт |
| `llm.py` | клиент OpenAI-совместимого API: поток, ограничение рассуждения, JSON + pydantic, один раунд «починки» (без рассуждения, если ответ пуст), бюджет вызовов/токенов, учёт ₽, отключение при сбоях и `Budget has been exceeded` |
| `schemas.py` | что LLM должна вернуть: `TaskUnderstanding`, `HypothesisBatch`, `Hypothesis` |
| `prompts.py` | промпты (англ.), списки моделей/полей берутся из ml_core |
| `context.py` | компактные сводки данных и истории экспериментов для промпта |
| `understand.py` | метрика (текст task.md важнее LLM), выбор валидации (time_split только при строго более позднем периоде) |
| `hypotheses.py` | гипотеза → строгий `ExperimentConfig` (чистка/ограничение параметров, выброс параметров, которых модель не принимает, стабильный id), seeds, playbook, local search |
| `features.py` + `_feature_sandbox_runner.py` | песочница для кода `make_features(df)` от LLM |
| `executor.py` + `_worker.py` | `SubprocessExecutor` (по умолчанию, жёсткий таймаут) и `InProcessExecutor` |
| `task_files.py` | поиск csv-файлов задачи |

## Один прогон
1. Найти файлы → профиль данных → **сразу fallback-сабмит**.
2. Понимание (1 вызов LLM): метрика, time/group-колонки, что можно выбросить.
3. Seeds: LightGBM и CatBoost → refit → настоящий сабмит.
4. Раунды: LLM предлагает до 3 гипотез (изменение относительно лучшего конфига: модель, параметры,
   предобработка, встроенные шаги признаков, свой код признаков) → каждая проверяется на одной и той же
   валидации → история и ошибки уходят в следующий промпт. Без LLM / после неё — playbook, затем local search.
   После 4 раундов без рекорда LLM «отдыхает» 6 раундов и возвращается (терпение 2); раунд LLM не начинается,
   если на ответ и один эксперимент уже не хватает времени.
5. Refit лучшего на всём train (чекпоинт раз в 30 мин + финальный) → `report.md/json` в work_dir.

Время: агент знает только `deadline_ts`, держит резерв на финальный refit (самокалибруется по факту).
