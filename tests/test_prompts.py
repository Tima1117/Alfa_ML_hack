"""research_user()'s first-round diversity nudge: present only once, on the very first LLM round."""

from agent import prompts


def _user(**overrides):
    kwargs = dict(
        data_brief="brief", validation_brief="validation", understanding="x", best_config_json="none yet",
        history="no experiments yet", notes=[], minutes_left=100, n_hypotheses=3,
    )
    kwargs.update(overrides)
    return prompts.research_user(**kwargs)


def test_first_round_asks_for_two_different_strategies():
    text = _user(first_round=True)
    assert "FIRST round" in text and "genuinely different strategies" in text


def test_later_rounds_get_no_first_round_nudge():
    text = _user(first_round=False)
    assert "FIRST round" not in text

    text_default = _user()  # first_round defaults to False
    assert "FIRST round" not in text_default
