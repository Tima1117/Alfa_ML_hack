# Контекст проекта — с чего начать

Папка для того, чтобы новая сессия (или новый человек) за пару минут поняла, где мы находимся.
Подробные технические контракты здесь не дублируются — на них ссылки.

| файл | о чём |
|---|---|
| [case.md](case.md) | суть кейса, ограничения, формула score, инструкция организаторов про ключ и ZIP |
| [team.md](team.md) | кто что делает, репозитории, что кому передать |
| [architecture.md](architecture.md) | карта кода: `ml_core/` и `agent/`, как идёт один прогон |
| [decisions.md](decisions.md) | принятые решения и почему |
| [insights.md](insights.md) | **хайлайты**: что узнали о качестве, LLM, прокси, деньгах и процессе |
| [status.md](status.md) | что сделано, измеренные результаты, известные ограничения |
| [next_steps.md](next_steps.md) | что делать дальше, по порядку |
| [repo_map.md](repo_map.md) | что где лежит в репозитории, кто владелец, что получается после прогона |
| [acceptance.md](acceptance.md) | итог приёмки `agent-v1`: критерии, прогоны с LLM и без, расход, риски |
| [handoff.md](handoff.md) | что считаем готовым результатом нашей части и когда передаём команде |
| [environment.md](environment.md) | облачное окружение: ключ, переменные, установка зависимостей |
| [pitfalls.md](pitfalls.md) | **где теряли время и контекст** — прочитать до первых команд |

Презентация решения со схемами: [docs/presentation/agent_deck.html](../docs/presentation/agent_deck.html).

Технические контракты:
- [docs/CONTRACTS.md](../docs/CONTRACTS.md) — ml_core (Тима): объекты, операции, гарантии.
- [docs/AGENT_CONTRACT.md](../docs/AGENT_CONTRACT.md) — агент: точка входа `run_research`, Executor, env-переменные.

Исходники кейса: `Base/Альфа Хак_AutoML.pdf`, `Base/baseline.py`, `Base/requirements.txt`, `Base/Datasets/*/task.md`.
