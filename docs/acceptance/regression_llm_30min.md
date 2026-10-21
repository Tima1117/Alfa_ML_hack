# Agent report: Регрессия

- status: **ok**
- task: regression, metric **rmse** (lower is better, from task.md)
- validation: time_split × 1 — all test rows have Year_Factor later than any train row; validate on the latest train period
- best: `42a95433b16b` = 42.14861802480531 ± 0.0; submitted: `42a95433b16b`
- experiments: 141, rounds: 49, elapsed: 27.1 min
- LLM: on (qwen/qwen3.8-27b); usage: {"calls": 19, "failed_calls": 1, "prompt_tokens": 108854, "completion_tokens": 25939, "cost": 5.2119, "seconds": 543.3}

## Initial assessment

Target is driven primarily by building type/size/age and heating/cooling degree days; the distribution is right-skewed (median 77, max 993), so a log-transform of the target is advisable. Key risks: temporal shift (test Year_Factor values are all later than train), ~42% missing energy_star_rating, and ~50-62% missing wind/fog features. Start with LightGBM or XGBoost using temporal (last-period) cross-validation, log-transform the target, and let the booster handle missing values natively.

## Journal

| round | source | hypothesis | status | metric | ± | s | new best | note |
|---|---|---|---|---|---|---|---|---|
| 0 | seed | LightGBM default | success | 50.554 | 0 | 3 | ✓ |  |
| 0 | seed | CatBoost default | success | 51.147 | 0 | 9 |  |  |
| 1 | llm | Log1p target transform | success | 47.53 | 0 | 4 | ✓ |  |
| 1 | llm | Domain features: temp range, HDD/CDD ratio, log area | success | 45.637 | 0 | 4 | ✓ | feature code OK: +['annual_temp_range', 'hdd_cdd_ratio', 'log_floor_area', 'building_age', 'elev_hdd'] -[] (0.5s) |
| 1 | llm | XGBoost with log1p target | success | 51.193 | 0 | 4 |  |  |
| 2 | llm | LightGBM more trees lower LR | success | 46.107 | 0 | 11 |  |  |
| 2 | llm | Extra features: temp variability, area interactions, age proxy | invalid |  |  | 0 |  | unknown generated step 'py:feat_extra.py': submit new code via feature_code |
| 2 | llm | CatBoost with log1p target and deeper trees | failed |  |  | 0 |  | ignored reserved hyperparameter 'allow_writing_files'; model construction failed: TypeError: CatBoostRegressor.__init__() got an unexpected keyword argument 'au |
| 3 | llm | CatBoost with log1p and existing features | success | 44.229 | 0 | 19 | ✓ | ignored reserved hyperparameter 'allow_writing_files' |
| 3 | llm | LightGBM more regularized (fewer leaves, higher min_child) | failed |  |  | 0 |  | ignored reserved hyperparameter 'allow_writing_files'; model construction failed: TypeError: CatBoostRegressor.__init__() got an unexpected keyword argument 'su |
| 3 | llm | Add area-degree-day and elevation interaction features | success | 48.461 | 0 | 22 |  | feature code OK: +['log_area_x_hdd', 'log_area_x_cdd', 'hdd_per_area', 'cdd_per_area', 'elev_x_avg_temp', 'elev_x_hdd', 'elev_x_cdd', 'esr_x_log_area', 'esr_x_h |
| 4 | llm | CatBoost depth=7 | success | 44.615 | 0 | 25 |  | ignored reserved hyperparameter 'allow_writing_files' |
| 4 | llm | CatBoost with bagging_temperature=0.5 | success | 44.229 | 0 | 18 |  | ignored reserved hyperparameter 'allow_writing_files' |
| 4 | llm | CatBoost 3000 iterations lr=0.02 | success | 45.006 | 0 | 28 |  | ignored reserved hyperparameter 'allow_writing_files' |
| 5 | llm | CatBoost with rsm=0.7 feature subsampling | success | 44.967 | 0 | 17 |  | ignored reserved hyperparameter 'allow_writing_files' |
| 5 | llm | CatBoost with l2_leaf_reg=10 | success | 44.632 | 0 | 18 |  | ignored reserved hyperparameter 'allow_writing_files' |
| 5 | llm | Drop noisy wind/fog features | success | 44.349 | 0 | 17 |  | ignored reserved hyperparameter 'allow_writing_files' |
| 6 | llm | CatBoost with random_strength=2 | success | 44.586 | 0 | 19 |  | ignored reserved hyperparameter 'allow_writing_files' |
| 6 | llm | CatBoost with border_count=64 | success | 48.28 | 0 | 16 |  | ignored reserved hyperparameter 'allow_writing_files' |
| 6 | llm | Add missing indicators for sparse features | success | 44.454 | 0 | 17 |  | ignored reserved hyperparameter 'allow_writing_files' |
| 7 | llm | Drop Year_Factor (temporal leakage risk) | success | 44.825 | 0 | 18 |  | ignored reserved hyperparameter 'allow_writing_files' |
| 7 | llm | CatBoost lower LR (0.01) with 4000 iterations | success | 44.682 | 0 | 34 |  | ignored reserved hyperparameter 'allow_writing_files' |
| 7 | llm | CatBoost Lossguide growth with leaf_count=31 | failed |  |  | 0 |  | depth clamped 0 -> 1; ignored reserved hyperparameter 'allow_writing_files'; model construction failed: TypeError: CatBoostRegressor.__init__() got an unexpecte |
| 8 | playbook | log1p target | duplicate |  |  | 0 |  | same config as an earlier experiment, skipped |
| 8 | playbook | LightGBM more trees, lower LR | success | 48.357 | 0 | 9 |  |  |
| 8 | playbook | LightGBM regularized | success | 44.154 | 0 | 3 | ✓ |  |
| 9 | playbook | CatBoost shorter, stronger L2 | success | 45.06 | 0 | 4 |  |  |
| 9 | playbook | missing-value indicators | success | 43.394 | 0 | 2 | ✓ |  |
| 9 | playbook | count encoding of categoricals | success | 43.523 | 0 | 2 |  |  |
| 10 | playbook | XGBoost | success | 47.349 | 0 | 7 |  |  |
| 10 | playbook | CatBoost deeper | success | 46.454 | 0 | 21 |  |  |
| 10 | playbook | LightGBM bigger leaves | success | 43.077 | 0 | 8 | ✓ |  |
| 11 | playbook | drop Year_Factor | success | 44.971 | 0 | 8 |  |  |
| 12 | local-search | tune: learning_rate 0.04→0.0539 | success | 45.412 | 0 | 8 |  |  |
| 12 | local-search | tune: subsample 0.8→0.6823 | success | 43.841 | 0 | 8 |  |  |
| 12 | local-search | tune: colsample_bytree 0.7→0.7744 | success | 45.382 | 0 | 8 |  |  |
| 13 | local-search | tune: n_estimators 1000→669 | success | 43.036 | 0 | 5 | ✓ |  |
| 13 | local-search | tune: colsample_bytree 0.7→0.5298 | success | 44.618 | 0 | 5 |  |  |
| 13 | local-search | tune: n_estimators 1000→1688, min_child_samples 10→6 | success | 44.635 | 0 | 13 |  |  |
| 14 | llm | Drop temporal-leakage features (Year_Factor, days_with_fog) | success | 44.309 | 0 | 6 |  |  |
| 14 | llm | XGBoost with log1p and categorical handling | success | 43.442 | 0 | 6 |  |  |
| 14 | llm | LightGBM lower leaves + higher reg_lambda | success | 43.136 | 0 | 5 |  |  |
| 15 | llm | Drop only days_with_fog | success | 43.234 | 0 | 6 |  |  |
| 15 | llm | Add L1 regularization (reg_alpha=1.0) | success | 43.953 | 0 | 6 |  |  |
| 15 | llm | min_child_samples=20 | success | 45.303 | 0 | 6 |  |  |
| 16 | local-search | tune: subsample 0.8→0.884 | success | 45.437 | 0 | 6 |  |  |
| 16 | local-search | tune: n_estimators 669→851 | success | 43.102 | 0 | 7 |  |  |
| 16 | local-search | tune: learning_rate 0.04→0.0622, num_leaves 127→78 | success | 45.906 | 0 | 4 |  |  |
| 17 | local-search | tune: min_child_samples 10→7 | success | 44.65 | 0 | 5 |  |  |
| 17 | local-search | tune: colsample_bytree 0.7→0.6236, n_estimators 669→402 | success | 43.604 | 0 | 4 |  |  |
| 17 | local-search | tune: reg_lambda 5.0→0.0 | success | 45.227 | 0 | 5 |  |  |
| 18 | local-search | tune: num_leaves 127→166, min_child_samples 10→13 | success | 44.774 | 0 | 7 |  |  |
| 18 | local-search | tune: subsample 0.8→0.6891, reg_lambda 5.0→0.8634 | success | 44.31 | 0 | 6 |  |  |
| 18 | local-search | tune: n_estimators 669→951 | success | 43.043 | 0 | 8 |  |  |
| 19 | local-search | tune: subsample 0.8→0.9343 | success | 48.429 | 0 | 5 |  |  |
| 19 | local-search | tune: n_estimators 669→397 | success | 42.873 | 0 | 3 | ✓ |  |
| 19 | local-search | tune: num_leaves 127→72, learning_rate 0.04→0.0532 | success | 44.983 | 0 | 3 |  |  |
| 20 | local-search | tune: min_child_samples 10→13, subsample 0.8→0.7253 | success | 45.29 | 0 | 3 |  |  |
| 20 | local-search | tune: subsample 0.8→0.9398, colsample_bytree 0.7→0.8207 | success | 45.936 | 0 | 4 |  |  |
| 20 | local-search | tune: colsample_bytree 0.7→0.7827 | success | 44.202 | 0 | 3 |  |  |
| 21 | local-search | tune: reg_lambda 5.0→2.3886 | success | 44.489 | 0 | 4 |  |  |
| 21 | local-search | tune: reg_lambda 5.0→9.3856 | success | 43.635 | 0 | 4 |  |  |
| 21 | local-search | tune: min_child_samples 10→15, colsample_bytree 0.7→0.5348 | success | 43.399 | 0 | 4 |  |  |
| 22 | llm | Add L1 regularization (reg_alpha=2.0) | success | 43.822 | 0 | 4 |  |  |
| 22 | llm | Lower min_child_samples to 5 | success | 43.517 | 0 | 4 |  |  |
| 22 | llm | Lower learning rate to 0.03 with 500 trees | success | 44.786 | 0 | 5 |  |  |
| 24 | llm | num_leaves=100 | success | 44.5 | 0 | 4 |  |  |
| 24 | llm | subsample=0.85 | success | 45.562 | 0 | 3 |  |  |
| 24 | llm | Add floor_area x total_degree_days interaction | success | 44.359 | 0 | 4 |  | feature code OK: +['floor_area_x_total_dd', 'log_floor_area_x_total_dd'] -[] (0.2s) |
| 25 | local-search | tune: subsample 0.8→0.9269 | success | 46.507 | 0 | 3 |  |  |
| 25 | local-search | tune: learning_rate 0.04→0.052, num_leaves 127→71 | success | 45.145 | 0 | 3 |  |  |
| 25 | local-search | tune: colsample_bytree 0.7→0.5081, n_estimators 397→679 | success | 45.22 | 0 | 5 |  |  |
| 26 | local-search | tune: n_estimators 397→658 | success | 42.944 | 0 | 5 |  |  |
| 26 | local-search | tune: colsample_bytree 0.7→0.8384 | success | 44.388 | 0 | 4 |  |  |
| 26 | local-search | tune: num_leaves 127→171 | success | 44.655 | 0 | 4 |  |  |
| 27 | local-search | tune: colsample_bytree 0.7→0.618 | success | 43.617 | 0 | 4 |  |  |
| 27 | local-search | tune: reg_lambda 5.0→2.4864, learning_rate 0.04→0.0714 | success | 46.278 | 0 | 3 |  |  |
| 27 | local-search | tune: num_leaves 127→228 | success | 44.686 | 0 | 5 |  |  |
| 28 | local-search | tune: subsample 0.8→0.8899 | success | 46.593 | 0 | 3 |  |  |
| 28 | local-search | tune: min_child_samples 10→13, colsample_bytree 0.7→0.621 | success | 43.155 | 0 | 4 |  |  |
| 28 | local-search | tune: colsample_bytree 0.7→0.5476 | success | 43.294 | 0 | 3 |  |  |
| 29 | local-search | tune: learning_rate 0.04→0.0299 | success | 44.683 | 0 | 4 |  |  |
| 29 | local-search | tune: reg_lambda 5.0→7.2834 | success | 42.307 | 0 | 4 | ✓ |  |
| 29 | local-search | tune: num_leaves 127→169 | success | 44.12 | 0 | 4 |  |  |
| 30 | local-search | tune: subsample 0.8→0.8743 | success | 45.258 | 0 | 4 |  |  |
| 30 | local-search | tune: reg_lambda 7.2834→4.9061, min_child_samples 10→14 | success | 43.684 | 0 | 4 |  |  |
| 30 | local-search | tune: min_child_samples 10→6, subsample 0.8→0.7439 | success | 44.834 | 0 | 3 |  |  |
| 31 | llm | XGBoost with same features | success | 43.23 | 0 | 3 |  |  |
| 31 | llm | reg_lambda 7.2834→9.5 | success | 44.142 | 0 | 4 |  |  |
| 31 | llm | Drop days_with_fog | success | 44.164 | 0 | 4 |  |  |
| 32 | llm | reg_lambda 7.2834→8.5 | success | 43.509 | 0 | 4 |  |  |
| 32 | llm | Add reg_alpha=1.0 | success | 42.83 | 0 | 4 |  |  |
| 32 | llm | Add reg_alpha=0.5 + reg_lambda=8.0 | success | 44.089 | 0 | 4 |  |  |
| 33 | local-search | tune: reg_lambda 7.2834→4.517, learning_rate 0.04→0.0517 | success | 44.345 | 0 | 4 |  |  |
| 33 | local-search | tune: num_leaves 127→171 | success | 44.382 | 0 | 4 |  |  |
| 33 | local-search | tune: min_child_samples 10→8 | success | 43.543 | 0 | 4 |  |  |
| 34 | local-search | tune: learning_rate 0.04→0.0306 | success | 44.111 | 0 | 4 |  |  |
| 34 | local-search | tune: min_child_samples 10→18, n_estimators 397→279 | success | 44.034 | 0 | 3 |  |  |
| 34 | local-search | tune: num_leaves 127→184, min_child_samples 10→13 | success | 43.835 | 0 | 4 |  |  |
| 35 | local-search | tune: n_estimators 397→241, learning_rate 0.04→0.0321 | success | 44.295 | 0 | 3 |  |  |
| 35 | local-search | tune: colsample_bytree 0.7→0.622, reg_lambda 7.2834→1.878 | success | 43.831 | 0 | 4 |  |  |
| 35 | local-search | tune: colsample_bytree 0.7→0.5355 | success | 43.869 | 0 | 4 |  |  |
| 36 | local-search | tune: min_child_samples 10→6 | success | 43.767 | 0 | 4 |  |  |
| 36 | local-search | tune: colsample_bytree 0.7→0.5433 | success | 43.869 | 0 | 4 |  |  |
| 36 | local-search | tune: subsample 0.8→0.9435, colsample_bytree 0.7→0.5362 | success | 45.862 | 0 | 3 |  |  |
| 37 | local-search | tune: n_estimators 397→511, num_leaves 127→186 | success | 42.916 | 0 | 5 |  |  |
| 37 | local-search | tune: learning_rate 0.04→0.0626 | success | 44.221 | 0 | 3 |  |  |
| 37 | local-search | tune: learning_rate 0.04→0.0267 | success | 43.923 | 0 | 4 |  |  |
| 38 | local-search | tune: num_leaves 127→73 | success | 43.308 | 0 | 3 |  |  |
| 38 | local-search | tune: num_leaves 127→217, n_estimators 397→619 | success | 44.285 | 0 | 7 |  |  |
| 38 | local-search | tune: subsample 0.8→0.9432 | success | 47.236 | 0 | 4 |  |  |
| 39 | llm | reg_lambda 7.28→9.5 | duplicate |  |  | 0 |  | same config as an earlier experiment, skipped |
| 39 | llm | feature_fraction_bynode=0.8 | success | 43.38 | 0 | 4 |  |  |
| 39 | llm | min_child_samples=20 | success | 43.645 | 0 | 4 |  |  |
| 40 | llm | reg_lambda=9.0 | success | 43.231 | 0 | 4 |  |  |
| 40 | llm | reg_alpha=0.5 | success | 44.011 | 0 | 4 |  |  |
| 40 | llm | reg_lambda=10.0 + reg_alpha=0.3 | success | 43.871 | 0 | 4 |  |  |
| 41 | local-search | tune: n_estimators 397→508 | success | 42.296 | 0 | 5 | ✓ |  |
| 41 | local-search | tune: subsample 0.8→0.7228 | success | 45.023 | 0 | 4 |  |  |
| 41 | local-search | tune: subsample 0.8→0.8704 | success | 44.956 | 0 | 5 |  |  |
| 42 | local-search | tune: colsample_bytree 0.7→0.8967, min_child_samples 10→8 | success | 43.614 | 0 | 5 |  |  |
| 42 | local-search | tune: reg_lambda 7.2834→5.2692, num_leaves 127→81 | success | 43.812 | 0 | 4 |  |  |
| 42 | local-search | tune: n_estimators 508→774, subsample 0.8→0.9061 | success | 47.073 | 0 | 6 |  |  |
| 43 | local-search | tune: learning_rate 0.04→0.0263 | success | 43.612 | 0 | 5 |  |  |
| 43 | local-search | tune: colsample_bytree 0.7→0.5698, reg_lambda 7.2834→4.0504 | success | 43.52 | 0 | 4 |  |  |
| 43 | local-search | tune: num_leaves 127→79 | success | 44.647 | 0 | 3 |  |  |
| 44 | local-search | tune: num_leaves 127→229 | success | 43.898 | 0 | 6 |  |  |
| 44 | local-search | tune: n_estimators 508→281 | success | 42.902 | 0 | 3 |  |  |
| 44 | local-search | tune: min_child_samples 10→8 | success | 43.686 | 0 | 4 |  |  |
| 45 | local-search | tune: reg_lambda 7.2834→4.2162, min_child_samples 10→7 | success | 44.405 | 0 | 5 |  |  |
| 45 | local-search | tune: reg_lambda 7.2834→10.1731 | success | 43.423 | 0 | 4 |  |  |
| 45 | local-search | tune: num_leaves 127→95, reg_lambda 7.2834→3.6895 | success | 44.003 | 0 | 4 |  |  |
| 46 | local-search | tune: reg_lambda 7.2834→10.688, learning_rate 0.04→0.0641 | success | 44.031 | 0 | 4 |  |  |
| 46 | local-search | tune: learning_rate 0.04→0.0525, reg_lambda 7.2834→9.4364 | success | 44.475 | 0 | 4 |  |  |
| 46 | local-search | tune: num_leaves 127→185, subsample 0.8→0.7115 | success | 44.498 | 0 | 5 |  |  |
| 47 | llm | reg_lambda 7.2834→8.5 | success | 43.417 | 0 | 5 |  |  |
| 47 | llm | Add degree_days x floor_area features | success | 42.839 | 0 | 5 |  | feature code OK: +['hdd_per_area', 'cdd_per_area', 'hdd_log_area', 'cdd_log_area', 'total_dd', 'dd_range'] -[] (0.3s) |
| 47 | llm | n_estimators 508→560 | success | 42.229 | 0 | 5 | ✓ |  |
| 48 | llm | n_estimators 560→650 | success | 42.149 | 0 | 6 | ✓ |  |
| 48 | llm | n_estimators 620, lr 0.035 | success | 43.309 | 0 | 5 |  |  |
| 48 | llm | reg_lambda 7.2834→6.5 | success | 43.43 | 0 | 5 |  |  |
| 49 | llm | n_estimators 650→750 | success | 42.336 | 0 | 6 |  |  |
| 49 | llm | n_estimators 650→800 | success | 42.367 | 0 | 7 |  |  |
| 49 | llm | n_estimators 700, lr 0.038 | success | 43.499 | 0 | 6 |  |  |

## Best config

```json
{
  "model_name": "lightgbm",
  "hyperparameters": {
    "n_estimators": 650,
    "learning_rate": 0.04,
    "num_leaves": 127,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "min_child_samples": 10,
    "reg_lambda": 7.2834
  },
  "preprocessing": {
    "target_transform": "log1p"
  },
  "feature_steps": [
    "py:feat_15fb8d1fa5e4.py",
    "missing_indicators"
  ]
}
```

## Notes

- round 1 analysis: LightGBM (50.554) slightly beats CatBoost (51.147). The target is heavily right-skewed (median 77, max 993), which means the model is likely struggling with the long tail. A log1p target transform should significantly improve RMSE by making the target distribution more symmetric. Additionally, we can extract useful domain features from the 36 monthly temperature columns (annual range, heating/cooling balance) and building characteristics (age, area interactions).
- round 2 analysis: LightGBM with log1p target + domain features (45.637) is clearly the best. The ~2 RMSE improvement from features confirms that building/climate interactions matter. The temporal split (latest Year_Factor held out) is a key risk, so features that capture stable physical relationships (not time-specific patterns) are most valuable. We have 26 min left, so I'll focus on: (1) increasing model capacity, (2) adding more physically-motivated features, and (3) testing CatBoost with log1p to see if it ge
- «Extra features: temp variability, area interactions, age proxy» invalid: unknown generated step 'py:feat_extra.py': submit new code via feature_code
- round 3 analysis: LightGBM with log1p + domain features (45.637) is the clear best. The 'more trees lower LR' variant (46.107) was slightly worse, suggesting the model isn't underfitting but may be slightly overfitting to temporal patterns. CatBoost failed due to an invalid parameter (auto_class_weights is for classification). XGBoost (51.193) is much worse. The temporal split (test years later than train) means we should favor models/params that generalize well rather than memorize. Key opportunities: (1) fix Ca
- round 4 analysis: CatBoost with log1p target + existing domain features (feat_15fb8d1fa5e4.py) is the clear best at 44.229. Adding more interaction features made it worse (48.461), suggesting the model already captures the key relationships and extra features add noise. The temporal split (test years later than train) means regularization and generalization are critical. LightGBM with the same setup gets 45.637, confirming CatBoost handles this data better. With 23 min left, I'll test small CatBoost hyperparamete
- round 5 analysis: CatBoost with log1p (44.229) is the clear best. Depth=7 and more iterations both made things worse, suggesting the model is already well-tuned in terms of capacity. The temporal split (test years later than train) means generalization is critical. The wind/fog features have 50-62% missing and 'train/test values barely overlap', making them likely noise sources. Feature subsampling (rsm) and stronger L2 regularization are the most promising untried directions for improving temporal generalization
- round 6 analysis: CatBoost with log1p target (44.229) remains the clear best after 16 experiments. Most parameter changes (deeper trees, more iterations, stronger L2, feature subsampling) made things worse, suggesting the model is near its optimum for this configuration. The temporal split (test years later than train) is the key challenge, so regularization strategies that improve generalization are most promising. The 41.7% missing energy_star_rating and 50-62% missing wind/fog features represent structured mis
- round 7 analysis: CatBoost with log1p (44.229) remains the best after 19 experiments. Most parameter tweaks (deeper trees, stronger regularization, feature subsampling, border_count) made things worse, suggesting the model is near its optimum for symmetric tree growth. The key unexplored risk is Year_Factor, which has 'train/test values barely overlap' - the model may be using it as a temporal proxy that won't generalize. A lower learning rate with more iterations is also untried and could improve generalization 
- round 14 analysis: LightGBM with log1p target + domain features + missing_indicators (43.036) is the clear best after 37 experiments. Most parameter tweaks made things worse, suggesting the model is near its optimum. The key remaining risk is temporal leakage: Year_Factor and days_with_fog have 'train/test values barely overlap', meaning the model may be memorizing temporal patterns that won't transfer. Dropping these features could improve generalization to the held-out test period.
- round 15 analysis: LightGBM at 43.036 is the clear best after 37 experiments. Most parameter tweaks (lower leaves, lower subsample, lower colsample, more trees) made things worse, suggesting the model is near its optimum. The drop-both-temporal-features experiment (44.309) was worse, but that conflated two effects. days_with_fog has 61.8% missing and barely overlaps train/test, making it a likely noise source, while Year_Factor (0% missing, 5 values) may carry useful signal. L1 regularization (reg_alpha) is untrie
- round 22 analysis: LightGBM at 42.873 is well-tuned after 61 experiments. Most parameter tweaks (subsample, colsample, reg_lambda, min_child_samples) made things worse, confirming we're near the optimum. The key untried direction is L1 regularization (reg_alpha), which could improve temporal generalization by inducing sparsity and reducing reliance on noisy/temporal features. The temporal split (test years later than train) means the model needs to generalize rather than memorize, and L1's feature selection effect
- round 24 analysis: LightGBM at 42.873 is well-tuned after 61 experiments. Most parameter tweaks (reg_lambda, reg_alpha, min_child_samples, learning rate, subsample, colsample) made things worse, confirming we're near the optimum. The temporal split makes generalization critical, but dropping temporal features also hurt. With 11 minutes left, I'll test a few untried small variations: num_leaves=100 (between the tried 95 and 127), a slightly higher subsample (0.85), and a domain-specific feature interaction (floor_a
- round 31 analysis: LightGBM at 42.307 is well-tuned after 85 experiments. reg_lambda=7.2834 was the key improvement over 5.0 (42.873→42.307). All other parameter directions (more trees, more leaves, different subsample, lower LR, different min_child_samples) made things worse. With 8 minutes left, I'll try: (1) XGBoost as a different model family with the same features, (2) slightly higher reg_lambda since 7.2834 > 5.0 helped, and (3) dropping days_with_fog which has 61.8% missing and barely overlaps train/test.
- round 32 analysis: LightGBM at 42.307 is well-tuned after 85 experiments. The key improvement was reg_lambda 5.0→7.2834. XGBoost (43.23) and dropping days_with_fog (44.164) both hurt. With 8 minutes left, I'll try small steps in the direction that helped (slightly higher reg_lambda) and the untried L1 regularization (reg_alpha) which could improve temporal generalization by inducing feature sparsity.
- round 39 analysis: LightGBM at 42.307 is well-tuned after 109 experiments. The key improvement was reg_lambda 5.0→7.2834. All other parameter directions (more trees, more/fewer leaves, different subsample, different LR, reg_alpha=1.0, XGBoost, dropping features) made things worse. With 6 minutes left, I'll try small steps in the one direction that helped (higher reg_lambda) and one untried regularization mechanism (feature_fraction_bynode).
- round 40 analysis: LightGBM at 42.307 is well-tuned after 111 experiments. The only meaningful improvement was reg_lambda 5.0→7.2834. All other parameter directions (more/fewer leaves, different subsample, lower LR, reg_alpha=1.0, feature_fraction_bynode, min_child_samples=20, XGBoost, CatBoost, dropping features) made things worse. With 5 minutes left, I'll try small steps in the one direction that helped (higher reg_lambda) and a gentler L1 (reg_alpha=0.5 vs the 1.0 that hurt).
- round 47 analysis: LightGBM at 42.296 is extremely well-tuned after 132 experiments. The only meaningful improvement was reg_lambda 5.0→7.2834. All other parameter directions (more/fewer trees, different leaves, subsample, LR, reg_alpha, different models, dropping features) made things worse. With 3 minutes left, I'll try a small step in the one direction that helped (slightly higher reg_lambda) and one untried feature interaction that captures domain physics (degree_days × floor_area as total energy demand).
- round 48 analysis: LightGBM at 42.229 is extremely well-tuned after 135 experiments. The only consistent improvement direction was increasing n_estimators (397→508→560 gave 42.873→42.296→42.229). All other parameter changes (reg_lambda higher/lower, reg_alpha, different models, feature interactions, dropping features) made things worse. With 2 minutes left, the safest bet is continuing the n_estimators trend slightly upward.
- round 49 analysis: The consistent trend is that increasing n_estimators has been the only reliable improvement direction: 397→508→560→650 gave 42.873→42.296→42.229→42.149. The gains are diminishing but still positive. With 1 minute left, the safest strategy is to continue pushing n_estimators upward in small steps.
