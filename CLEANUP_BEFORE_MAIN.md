# Что можно удалить перед переносом в основной репозиторий

Основной репозиторий команды — `Tima1117/Alfa_ML_hack`. Этот форк добавляет к нему агента, тесты и
документацию. Ниже — что из добавленного нужно только форку, а что удалять нельзя.

## Уже не попадает в git — ничего делать не нужно
`.venv/`, `runs/`, `__pycache__/`, `.pytest_cache/`, `catboost_info/`, `agent_work/`, `submission.csv`
в папках задач — всё это в `.gitignore`. Если переносите папку целиком (копированием, а не git'ом), удалите их руками.

## Можно спокойно удалить (код и тесты от них не зависят)
| что | зачем было в форке | после удаления |
|---|---|---|
| `CLEANUP_BEFORE_MAIN.md` | этот список | — |
| `CLAUDE.md` | правила для Claude Code в форке; в нём указана ветка форка | если команда работает с Claude Code — оставьте, но замените строку про ветку |
| `project_context/environment.md` | настройки облачного окружения Claude Code этого форка (секрет, прокси) | убрать строку из `project_context/README.md` |
| `project_context/pitfalls.md` | где ИИ-ассистент терял время в прошлых сессиях | убрать упоминания в `AGENTS.md`, `CLAUDE.md`, `project_context/README.md` |
| `docs/presentation/` | HTML-презентация решения | убрать ссылки в `README.md`, `project_context/README.md`, `project_context/repo_map.md` |

```bash
git rm -r CLEANUP_BEFORE_MAIN.md project_context/environment.md project_context/pitfalls.md
# по желанию: git rm CLAUDE.md;  git rm -r docs/presentation
grep -rn "pitfalls.md\|environment.md\|presentation/\|CLEANUP_BEFORE_MAIN" --include=*.md .   # найти оставшиеся ссылки
```

## Лучше оставить (полезно команде и её LLM-ассистентам)
- `AGENTS.md` — вход для любого LLM-ассистента: что читать, правила, команды.
- `project_context/` — кейс, архитектура, решения, статус, план, карта репозитория (`repo_map.md`),
  план передачи (`handoff.md`). Если команда такой контекст не ведёт — можно удалить целиком,
  но тогда поправьте ссылки в `README.md` и `AGENTS.md`.

## Удалять нельзя
| что | почему |
|---|---|
| `agent/` | сам агент |
| `ml_core/` | ML-ядро; в форке в нём есть наш коммит `a680722` (`feature_steps`) — согласовать с Тимой |
| `tests/` | тесты ml_core и агента (56) |
| `scripts/run_agent_local.py` | локальный запуск агента на задаче |
| `docs/AGENT_CONTRACT.md`, `docs/CONTRACTS.md` | контракты для инфраструктуры и ml_core |
| `Base/`, `README.md`, `.gitignore`, `examples/`, `skills/` | были в основном репозитории до форка |

Для ZIP-сабмита этот список не важен: в архив попадают только корневые `.py` и `requirements.txt`,
сборку делает инфраструктура.
