# Agent report: Регрессия

- status: **ok**
- task: regression, metric **rmse** (lower is better, from task.md)
- validation: time_split × 1 — all test rows have Year_Factor later than any train row; validate on the latest train period
- best: `dbbe2f138d75` = 41.855030941277725 ± 0.0; submitted: `dbbe2f138d75`
- experiments: 82, rounds: 28, elapsed: 12.2 min
- LLM: on (qwen/qwen3.8-27b); usage: {"calls": 6, "failed_calls": 0, "prompt_tokens": 27088, "completion_tokens": 18937, "cost": 4.6813, "seconds": 221.9}

## Initial assessment

The target is likely driven mainly by building use and class, location, and annual weather severity, especially heating/cooling degree days and monthly temperatures. The main risk is temporal distribution shift because test Year_Factor values are later than all train values, plus substantial missingness in energy_star_rating and wind/fog features. A gradient boosting model with careful categorical encoding for facility_type, State_Factor, and building_class, and time-based validation by Year_Factor, is a strong first approach.

## Journal

| round | source | hypothesis | status | metric | ± | s | new best | note |
|---|---|---|---|---|---|---|---|---|
| 0 | seed | LightGBM default | success | 50.554 | 0 | 3 | ✓ |  |
| 0 | seed | CatBoost default | success | 51.147 | 0 | 9 |  |  |
| 1 | llm | LightGBM + log1p target | success | 47.53 | 0 | 4 | ✓ |  |
| 1 | llm | LightGBM + domain features | success | 42.968 | 0 | 4 | ✓ | feature code OK: +['floor_area_log', 'building_age', 'annual_temp_range', 'energy_star_missing', 'wind_max_sin', 'wind_max_cos', 'wind_peak_sin', 'wind_peak_cos |
| 1 | llm | XGBoost + log1p target | success | 189.27 | 0 | 3 |  |  |
| 2 | llm | LightGBM deeper + lower LR | success | 45.193 | 0 | 8 |  |  |
| 2 | llm | CatBoost + log1p + domain features | success | 45.898 | 0 | 14 |  | ignored reserved hyperparameter 'allow_writing_files' |
| 2 | llm | LightGBM + extra building/weather features | success | 45.187 | 0 | 4 |  | feature code OK: +['log_floor_area', 'building_age', 'annual_temp_range', 'annual_min_range', 'annual_max_range', 'hdd_cdd_ratio', 'hdd_plus_cdd', 'area_sq', 'e |
| 3 | llm | LightGBM num_leaves=31 | success | 44.673 | 0 | 3 |  |  |
| 3 | llm | LightGBM min_child_samples=50 | success | 43.95 | 0 | 4 |  |  |
| 3 | llm | CatBoost depth=8 lr=0.03 iters=2000 | success | 46.899 | 0 | 32 |  | ignored reserved hyperparameter 'allow_writing_files' |
| 4 | llm | LightGBM 400 trees | success | 43.108 | 0 | 3 |  |  |
| 4 | llm | LightGBM stronger reg + fewer leaves | success | 43.919 | 0 | 3 |  |  |
| 4 | llm | HistGradientBoosting + log1p + domain features | success | 47.62 | 0 | 6 |  |  |
| 5 | llm | LightGBM + missing_indicators | success | 46.2 | 0 | 4 |  |  |
| 5 | llm | XGBoost + log1p + domain features (retry) | success | 150.54 | 0 | 3 |  |  |
| 5 | llm | LightGBM + count_encode for facility_type | success | 46.812 | 0 | 3 |  |  |
| 6 | playbook | log1p target | duplicate |  |  | 0 |  | same config as an earlier experiment, skipped |
| 6 | playbook | LightGBM more trees, lower LR | success | 46.375 | 0 | 8 |  |  |
| 6 | playbook | LightGBM regularized | success | 42.874 | 0 | 2 | ✓ |  |
| 7 | playbook | CatBoost shorter, stronger L2 | success | 44.295 | 0 | 4 |  |  |
| 7 | playbook | missing-value indicators | success | 42.788 | 0 | 2 | ✓ |  |
| 7 | playbook | count encoding of categoricals | success | 45.71 | 0 | 3 |  |  |
| 8 | playbook | XGBoost | success | 43.754 | 0 | 7 |  |  |
| 8 | playbook | CatBoost deeper | success | 45.917 | 0 | 20 |  |  |
| 8 | playbook | LightGBM bigger leaves | success | 43.281 | 0 | 9 |  |  |
| 9 | playbook | drop Year_Factor | success | 44.721 | 0 | 2 |  |  |
| 10 | local-search | tune: learning_rate 0.05→0.0674 | success | 43.17 | 0 | 2 |  |  |
| 10 | local-search | tune: subsample 0.8→0.6823 | success | 44.071 | 0 | 2 |  |  |
| 10 | local-search | tune: colsample_bytree 0.6→0.6744 | success | 44.073 | 0 | 3 |  |  |
| 11 | local-search | tune: n_estimators 400→268 | success | 43.043 | 0 | 2 |  |  |
| 11 | local-search | tune: colsample_bytree 0.6→0.4298 | success | 43.355 | 0 | 2 |  |  |
| 11 | local-search | tune: n_estimators 400→675, min_child_samples 50→30 | success | 43.622 | 0 | 3 |  |  |
| 12 | local-search | tune: subsample 0.8→0.884 | success | 44.18 | 0 | 3 |  |  |
| 12 | local-search | tune: n_estimators 400→509 | success | 42.692 | 0 | 3 | ✓ |  |
| 12 | local-search | tune: learning_rate 0.05→0.0778, num_leaves 31→19 | success | 42.767 | 0 | 2 |  |  |
| 13 | local-search | tune: min_child_samples 50→33 | success | 44.738 | 0 | 2 |  |  |
| 13 | local-search | tune: colsample_bytree 0.6→0.5236, n_estimators 509→306 | success | 43.355 | 0 | 2 |  |  |
| 13 | local-search | tune: reg_lambda 5.0→0.0 | success | 43.186 | 0 | 2 |  |  |
| 14 | local-search | tune: num_leaves 31→40, min_child_samples 50→66 | success | 43.537 | 0 | 3 |  |  |
| 14 | local-search | tune: subsample 0.8→0.6891, reg_lambda 5.0→0.8634 | success | 44.469 | 0 | 3 |  |  |
| 14 | local-search | tune: n_estimators 509→724 | success | 42.573 | 0 | 4 | ✓ |  |
| 15 | local-search | tune: subsample 0.8→0.9343 | success | 43.432 | 0 | 3 |  |  |
| 15 | local-search | tune: n_estimators 724→430 | success | 42.65 | 0 | 2 |  |  |
| 15 | local-search | tune: num_leaves 31→18, learning_rate 0.05→0.0665 | success | 43.273 | 0 | 3 |  |  |
| 16 | local-search | tune: min_child_samples 50→65, subsample 0.8→0.7253 | success | 44.775 | 0 | 3 |  |  |
| 16 | local-search | tune: subsample 0.8→0.9398, colsample_bytree 0.6→0.7207 | success | 46.763 | 0 | 3 |  |  |
| 16 | local-search | tune: colsample_bytree 0.6→0.6827 | success | 42.818 | 0 | 3 |  |  |
| 17 | local-search | tune: reg_lambda 5.0→2.3886 | success | 42.553 | 0 | 3 | ✓ |  |
| 17 | local-search | tune: reg_lambda 5.0→9.3856 | success | 43.733 | 0 | 3 |  |  |
| 17 | local-search | tune: min_child_samples 50→75, colsample_bytree 0.6→0.4348 | success | 44.07 | 0 | 3 |  |  |
| 18 | local-search | tune: subsample 0.8→0.9269 | success | 44.323 | 0 | 3 |  |  |
| 18 | local-search | tune: learning_rate 0.05→0.0651, num_leaves 31→17 | success | 43.876 | 0 | 3 |  |  |
| 18 | local-search | tune: colsample_bytree 0.6→0.4081, n_estimators 724→1238 | success | 42.581 | 0 | 5 |  |  |
| 19 | local-search | tune: n_estimators 724→1200 | success | 42.418 | 0 | 5 | ✓ |  |
| 19 | local-search | tune: colsample_bytree 0.6→0.7384 | success | 43.332 | 0 | 6 |  |  |
| 19 | local-search | tune: num_leaves 31→42 | success | 43.375 | 0 | 6 |  |  |
| 20 | local-search | tune: colsample_bytree 0.6→0.518 | success | 42.721 | 0 | 5 |  |  |
| 20 | local-search | tune: reg_lambda 2.3886→0.0, learning_rate 0.05→0.0892 | success | 44.776 | 0 | 5 |  |  |
| 20 | local-search | tune: num_leaves 31→56 | success | 44.285 | 0 | 7 |  |  |
| 21 | local-search | tune: subsample 0.8→0.8899 | success | 45.157 | 0 | 5 |  |  |
| 21 | local-search | tune: min_child_samples 50→64, colsample_bytree 0.6→0.521 | success | 43.578 | 0 | 5 |  |  |
| 21 | local-search | tune: colsample_bytree 0.6→0.4476 | success | 43.315 | 0 | 5 |  |  |
| 22 | local-search | tune: learning_rate 0.05→0.0374 | success | 42.827 | 0 | 5 |  |  |
| 22 | local-search | tune: reg_lambda 2.3886→4.672 | success | 43.205 | 0 | 5 |  |  |
| 22 | local-search | tune: num_leaves 31→41 | success | 44.034 | 0 | 6 |  |  |
| 23 | local-search | tune: subsample 0.8→0.8743 | success | 43.376 | 0 | 6 |  |  |
| 23 | local-search | tune: reg_lambda 2.3886→0.0113, min_child_samples 50→70 | success | 42.85 | 0 | 5 |  |  |
| 23 | local-search | tune: min_child_samples 50→31, subsample 0.8→0.7439 | success | 45.956 | 0 | 5 |  |  |
| 24 | local-search | tune: reg_lambda 2.3886→0.0, learning_rate 0.05→0.0646 | success | 44.555 | 0 | 5 |  |  |
| 24 | local-search | tune: num_leaves 31→42 | duplicate |  |  | 0 |  | same config as an earlier experiment, skipped |
| 24 | local-search | tune: min_child_samples 50→40 | success | 44.204 | 0 | 4 |  |  |
| 25 | local-search | tune: learning_rate 0.05→0.0383 | success | 41.855 | 0 | 5 | ✓ |  |
| 25 | local-search | tune: min_child_samples 50→88, n_estimators 1200→844 | success | 42.589 | 0 | 4 |  |  |
| 25 | local-search | tune: num_leaves 31→45, min_child_samples 50→65 | success | 43.256 | 0 | 6 |  |  |
| 26 | local-search | tune: n_estimators 1200→728, learning_rate 0.0383→0.0307 | success | 43.147 | 0 | 3 |  |  |
| 26 | local-search | tune: colsample_bytree 0.6→0.522, reg_lambda 2.3886→0.0 | success | 43.309 | 0 | 4 |  |  |
| 26 | local-search | tune: colsample_bytree 0.6→0.4355 | success | 42.544 | 0 | 5 |  |  |
| 27 | local-search | tune: min_child_samples 50→28 | success | 45.292 | 0 | 5 |  |  |
| 27 | local-search | tune: colsample_bytree 0.6→0.4433 | success | 42.754 | 0 | 5 |  |  |
| 27 | local-search | tune: subsample 0.8→0.9435, colsample_bytree 0.6→0.4362 | success | 44.236 | 0 | 5 |  |  |
| 28 | local-search | tune: n_estimators 1200→1544, num_leaves 31→45 | success | 42.987 | 0 | 7 |  |  |
| 28 | local-search | tune: learning_rate 0.0383→0.06 | success | 43.799 | 0 | 5 |  |  |
| 28 | local-search | tune: learning_rate 0.0383→0.0256 | success | 43.138 | 0 | 5 |  |  |

## Best config

```json
{
  "model_name": "lightgbm",
  "hyperparameters": {
    "n_estimators": 1200,
    "learning_rate": 0.0383,
    "num_leaves": 31,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.6,
    "min_child_samples": 50,
    "reg_lambda": 2.3886
  },
  "preprocessing": {
    "target_transform": "log1p"
  },
  "feature_steps": [
    "py:feat_3aed1b0012c6.py",
    "missing_indicators"
  ]
}
```

## Notes

- round 1 analysis: LightGBM (50.55) edges out CatBoost (51.15) as the best model. The target site_eui is right-skewed (mean=84, median=77, max=993), suggesting a log1p transform could significantly help the model learn the distribution, especially the tail. The 36 monthly temperature features are redundant and could be distilled into derived features (annual range, heating/cooling balance). Several features have high missingness (energy_star_rating 41.7%, wind/fog ~50-62%) where missing indicators carry signal. Fl
- round 2 analysis: LightGBM with log1p target + domain features (42.97) is clearly the best, with log1p contributing ~3 RMSE and features ~4.5 RMSE. XGBoost with log1p was catastrophically bad (189), likely a bug. CatBoost default (51.15) underperformed LightGBM but hasn't been tested with log1p + features. The single-fold time-split validation means we need robust improvements, not noise-chasing. With ~11 min left, I'll focus on: (1) deeper LightGBM with lower LR for better convergence, (2) CatBoost with log1p + 
- round 3 analysis: LightGBM with log1p + domain features (42.97) is clearly best. The 1200-tree/lower-LR variant (45.19) was worse, suggesting the model overfits beyond ~600 trees on this time-split validation. CatBoost with the same setup (45.90) trails LightGBM. Extra features hurt (45.19), so the current feature set is well-tuned. With time-split (future year held out), simpler/regularized models should generalize better to the temporal shift.
- round 4 analysis: LightGBM + log1p + domain features (42.97) is the clear best. The 1200-tree variant (45.19) and CatBoost (45.90) both underperform, suggesting the current 600-tree LightGBM is near the sweet spot for this time-split validation. Extra features hurt (45.19), so the feature set is well-tuned. With ~9 min left, I'll test: (1) fewer trees to check if we're slightly overfitting, (2) stronger regularization at 600 trees, and (3) HistGradientBoosting as a different model family that may handle temporal 
- round 5 analysis: LightGBM with log1p + domain features (42.97) is the clear best. All parameter tweaks (more trees, stronger regularization, fewer leaves, higher min_child_samples) have been negative, confirming the model is near its sweet spot for this time-split validation. The main remaining opportunities are: (1) explicit missing indicators for the high-missingness features (energy_star_rating 41.7%, wind/fog ~50-62%), (2) a different model family (XGBoost) that may handle the temporal shift differently, and
