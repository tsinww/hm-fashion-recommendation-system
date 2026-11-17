# Public Kaggle implicit-ALS baseline

This folder preserves the downloaded source of Julián Peller's Apache-2.0 Kaggle notebook, **H&M - Implicit ALS model [0.014]**, plus a local adapter that evaluates ALS on this project's four-week history and final validation week.

The downloaded notebook used transactions after 2020-08-21 and described its validation as three weeks of training plus one validation week. It only reported MAP@12 through `implicit` during validation and generated competition submissions with `filter_already_liked_items=False`. It did not report candidate Recall@K or Recall@12.

Run a small smoke test:

```bash
.venv/bin/python external_baselines/kaggle_implicit_als/run_four_week_als.py \
  --factors 10 --iterations 2 --customers 200 --candidate-k 100 --tag smoke
```

Run the comparable experiment using the notebook's default `train` settings:

```bash
.venv/bin/python external_baselines/kaggle_implicit_als/run_four_week_als.py \
  --factors 200 --iterations 15 --customers 2000 --candidate-k 1000 --tag full
```

The original notebook contains a large hyperparameter grid, but its downloaded source does not preserve the output value of `best_params`. The full command therefore uses the notebook's explicit training defaults rather than claiming to reproduce an unavailable tuned configuration.
