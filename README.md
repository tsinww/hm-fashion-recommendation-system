# H&M Personalized Recommendation: Leakage-Safe Offline Experiment

This repository turns the original exploratory notebook into a reproducible offline recommendation experiment. It now contains both the original serial ablation and a parallel retrieval architecture: BPR MF, implicit ALS, co-purchase, popularity, age-segment popularity, and two-tower candidates are unioned; exact history articles are removed from discovery sources; a bounded repeat source is added; and DLRM and LightGBM LambdaRank are compared on identical candidates. Every method is evaluated on the same customers, temporal holdout, and eligible item catalog.

The measured result is useful precisely because it does **not** assume that a deeper pipeline is better: the serial two-tower/DLRM pipeline does not beat retrieval-only models. A separately reproduced public implicit-ALS baseline is substantially better at Top-12 ranking than the project's BPR MF, even though its Candidate Recall@1,000 is slightly lower.

## What was found in the original project

`Rec_sys.ipynb` contains:

- recent transaction loading;
- distinct customer–article positives;
- uniform negative sampling;
- user/item embeddings with dot-product scores and binary cross-entropy;
- a duplicated three-epoch MF training loop.

Only the imports and file discovery cells have saved execution output. Candidate generation, the reported recall experiment, two-tower training, DLRM, Top-12 ranking, and end-to-end evaluation are not present as executed notebook artifacts. Therefore the previously quoted recall values (overall `~0.8243`, history-customer `~0.7108`, fallback `~63.20%`) cannot be reproduced from the checked-in notebook.

The original filter starts on 2020-08-26 and validates from 2020-09-16. That is **three** context weeks, not four, and it produces exactly 26,503 context items. The implemented experiment corrects this.

## Leakage-safe temporal design

All intervals are half-open: the start is included and the end is excluded.

| Purpose | Context available to model | Subsequent purchase labels |
|---|---:|---:|
| Train retrieval and ranking models | 2020-08-12–2020-09-08 | 2020-09-09–2020-09-15 |
| Final evaluation | 2020-08-19–2020-09-15 | 2020-09-16–2020-09-22 |

The final validation week is never used by `fit_mf`, two-tower fitting, DLRM or LightGBM fitting, feature encoders, or negative sampling. It is read only after all models are trained to construct relevance sets and metrics. There is no hyperparameter search on the final week.

At each snapshot, the eligible catalog is the set of articles purchased during that snapshot's four-week context. This gives 29,278 training-snapshot items and 29,009 final-snapshot items. Validation-only articles are treated as unknown/unavailable and correctly count as misses; 95.64% of distinct sampled validation customer–article pairs are in the eligible final catalog.

## Stage objectives and inference distributions

1. **Recent popularity** ranks eligible articles by context purchase count. It is also the fallback for customers with no context purchases.
2. **MF retrieval** learns 32-dimensional user/item embeddings from distinct context pairs using Bayesian Personalized Ranking (BPR): each observed pair is contrasted with a uniformly sampled unobserved article. BPR directly optimizes relative positive-versus-negative ordering, which is better aligned with retrieval than the notebook's pointwise BCE. Candidate scoring is batched; the code never creates a full user-by-item matrix.
3. **Two-tower narrowing** uses customer demographics, context-derived preferences/activity, and article metadata. It is trained with weighted BCE on retrieved positives and hard negatives sampled **from MF's candidates**, matching the distribution it sees at inference. It narrows 1,000 candidates to 100.
4. **DLRM ranking** embeds the same sparse customer/article fields, constructs pairwise embedding interactions, and combines them with MF score, two-tower score, popularity, price, recency, activity, candidate rank, and repeat-purchase indicators. Its weighted-BCE examples are sampled **after two-tower narrowing**, matching its inference distribution.

Missing categorical values map to an explicit missing category, missing numeric values are imputed, customers absent from MF use popularity, customers absent from metadata receive zero/OOV features, and validation-only articles remain explicit catalog misses.

### Architecture assessment

MF and a conventional two-tower are both dot-product retrieval models. Running them serially is often redundant unless the second tower adds enough side-information signal to preserve or improve candidate recall. Here it does not: narrowing from 1,000 to 100 reduces macro candidate Recall from `0.42375` to `0.09626`. Once a relevant item is discarded, DLRM cannot restore it.

The concrete next architecture should be **MF plus popularity/feature-retrieval candidate union → one ranker**, with no mandatory two-tower bottleneck. The two-tower can be an alternative source for cold-start candidates; take the union before ranking. Train the ranker directly on that union using candidate-aware negatives and a pairwise/listwise ranking loss. Until that design beats the fixed MF baseline on an earlier development window, MF-only should remain the selected model.

## Reproduce the experiment

Expected layout:

```text
Data/
  articles.csv
  customers.csv
  sample_submission.csv
  transactions_train.csv
```

Create an environment and install the three runtime dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Run metric tests and the end-to-end smoke test first:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python run_experiment.py --profile smoke
```

### Reproduce the best observed routed result

The complete selected architecture can be regenerated and evaluated with one
entry point:

```bash
.venv/bin/python run_best_system.py --rebuild
```

This command regenerates the four-week 1,012-candidate pool, trains the
LightGBM--CatBoost active-user rankers, builds the inactive-user KNN and
time-decayed ItemCF additions, and finally routes each customer to the matching
Top-12 output. It writes the combined predictions and metrics under
`artifacts/best_routed_system/`.

If the upstream artifacts already exist, recompute only routing and evaluation:

```bash
.venv/bin/python run_best_system.py
```

The observed routed result is `0.02887 MAP@12` and `0.06113 Recall@12` over the
fixed 2,000-customer holdout (`0.05497 MAP@12` for 885 active customers and
`0.00816` for 1,115 inactive customers). The underlying component training is
temporally leakage-safe, but this combined number remains a **post-hoc
diagnostic**: the routing rule was selected after examining final-week cohort
performance. The uniform LightGBM--CatBoost result, `0.02765 MAP@12`, is the
safer current headline until the frozen route is tested on another temporal
fold.

Run the measured practical experiment:

```bash
.venv/bin/python run_experiment.py --profile experiment
```

Run the four-week BPR MF seen-item ablation:

```bash
.venv/bin/python run_mf_seen_ablation.py
```

Run the parallel retrieval architecture (smoke test first):

```bash
.venv/bin/python run_hybrid_experiment.py --profile hybrid_smoke
.venv/bin/python run_hybrid_experiment.py --profile hybrid_experiment
```

Run the no-new/repeat-split ablation and retain its candidate arrays:

```bash
.venv/bin/python run_hybrid_experiment.py \
  --profile hybrid_unsplit_smoke --save-candidates
.venv/bin/python run_hybrid_experiment.py \
  --profile hybrid_unsplit_experiment --save-candidates
```

Isolate history deletion while retaining the controlled repeat source:

```bash
.venv/bin/python run_hybrid_experiment.py \
  --profile hybrid_allow_seen_repeat_smoke --save-candidates
.venv/bin/python run_hybrid_experiment.py \
  --profile hybrid_allow_seen_repeat_experiment --save-candidates
```

Preserve all 1,000 unseen candidates and expand final capacity for repeats:

```bash
.venv/bin/python run_hybrid_experiment.py \
  --profile hybrid_expanded_repeat_smoke --save-candidates
.venv/bin/python run_hybrid_experiment.py \
  --profile hybrid_expanded_repeat_experiment --save-candidates
```

Run the matched eight-week-history experiment and fixed-cohort comparison:

```bash
.venv/bin/python run_hybrid_experiment.py \
  --profile hybrid_eight_week_smoke --save-candidates
.venv/bin/python run_hybrid_experiment.py \
  --profile hybrid_eight_week_experiment --save-candidates
.venv/bin/python compare_history_horizons.py
```

Run the separately persisted six-week experiment and new 4/6/8 comparison:

```bash
.venv/bin/python run_hybrid_experiment.py \
  --profile hybrid_six_week_smoke --save-candidates
.venv/bin/python run_hybrid_experiment.py \
  --profile hybrid_six_week_experiment --save-candidates
.venv/bin/python compare_history_horizons_4_6_8.py
```

Audit how longer history changes the cold-start cohort:

```bash
.venv/bin/python run_cold_start_audit.py
```

Run the isolated inactive-user architecture, then compare it with the preserved
four-week candidate pool and ranker (the old artifacts are read-only inputs):

```bash
.venv/bin/python run_inactive_experiment.py --profile inactive_knn_smoke
.venv/bin/python run_inactive_experiment.py --profile inactive_knn_experiment
.venv/bin/python compare_inactive_experiment.py
.venv/bin/python run_inactive_dlrm.py
```

On macOS, `run_inactive_experiment.py` locates a bundled OpenMP library and
restarts itself with the required library path. If LightGBM is imported from a
different entry point and reports a missing `libomp.dylib`, run it as:

```bash
DYLD_LIBRARY_PATH=.venv/lib/python3.12/site-packages/sklearn/.dylibs \
  .venv/bin/python <script.py>
```

Run the adapted public Kaggle implicit-ALS baseline (install its two extra dependencies first):

```bash
.venv/bin/pip install -r external_baselines/kaggle_implicit_als/requirements.txt
.venv/bin/python external_baselines/kaggle_implicit_als/run_four_week_als.py \
  --factors 200 --iterations 15 --customers 2000 --candidate-k 1000 --tag full
```

Run the documented top-solution-inspired reproductions. These are adaptations
of the published write-ups, not the winners' unreleased private code:

```bash
.venv/bin/python run_top_solution_reproductions.py --solution first
.venv/bin/python run_top_solution_reproductions.py --solution first-ensemble
.venv/bin/python run_top_solution_reproductions.py --solution second
```

Apply CatBoost and a fixed LightGBM–CatBoost rank blend to the project's own
saved four-week 1,012-candidate pool:

```bash
.venv/bin/python run_own_ranker_ensemble.py
```

The first run scans the 3.2 GB transaction CSV in chunks and stores a 64 MB recent-transaction cache under `artifacts/cache/`. Outputs:

- `artifacts/<profile>/results.json`: configuration, split sizes, losses, metrics, environment, and stage runtimes;
- `artifacts/predictions/<profile>_top12.csv`: exactly 12 ranked article IDs for each sampled evaluation customer.

CLI overrides are available for `--seed`, `--max-train-users`, `--max-eval-users`, `--candidate-k`, and `--rerank-k`.

## Measured results

Configuration: seed 42; 3 MF epochs; 2 downstream epochs; 6,000 downstream-training customers; 2,000 randomly sampled final-week customers; 1,000 MF candidates; 100 candidates after two-tower narrowing; CPU execution. The sample contains 885 customers with context history and 1,115 cold-start customers, preserving the final week's approximate cohort mix.

Definitions:

- **Candidate Recall@K:** for each customer, the fraction of distinct validation articles present in that method's retained K candidates, macro-averaged across customers.
- **Recall@12:** the same fraction using only the final 12 recommendations.
- **MAP@12:** mean average precision through rank 12, with binary relevance and distinct validation articles. Earlier relevant hits receive more credit.

### Overall

| Method | Candidate K | Candidate Recall@K | MAP@12 | Recall@12 |
|---|---:|---:|---:|---:|
| Recent popularity | 1,000 | 0.41631 | 0.00648 | 0.02119 |
| MF | 1,000 | **0.42375** | **0.00873** | **0.02522** |
| MF → two-tower | 100 | 0.09626 | 0.00429 | 0.01722 |
| MF → two-tower → DLRM | 100 | 0.09626 | 0.00419 | 0.01527 |

MF improves MAP@12 by 34.8% and Recall@12 by 19.1% relative to recent popularity on this fixed sample. These are offline sample results, not production or Kaggle leaderboard claims.

### Reproduced public implicit-ALS baseline

The Apache-2.0 Kaggle notebook **H&M - Implicit ALS model [0.014]** was downloaded and preserved under `external_baselines/kaggle_implicit_als/`. Its original code used three history weeks for validation and did not report Candidate Recall or Recall@12. The local adapter keeps its implicit-ALS interaction-count method but uses the same four-week history, final validation week, 2,000 customers, eligible catalog, cold-start fallback, and metric implementation as this project. The run used the notebook's explicit training defaults (`200` factors, `15` iterations, regularization `0.01`) because its downloaded source did not preserve the output of its large tuning grid.

| Method (history items allowed) | Candidate Recall@1,000 | MAP@12 | Recall@12 |
|---|---:|---:|---:|
| Project BPR MF | **0.42375** | 0.00873 | 0.02522 |
| Public implicit ALS adaptation | 0.39825 | **0.01734** | **0.03655** |

For the 885 customers with four-week purchase history, ALS obtained `0.36769` Candidate Recall@1,000, `0.02959` MAP@12, and `0.05621` Recall@12. Thus ALS retrieved fewer total validation items within 1,000 than BPR MF, but put relevant articles much nearer the top. This is why Candidate Recall alone is not enough to select the model. The complete run took 80.1 seconds locally (77.3 seconds training), and the full record is in `artifacts/external_als/results_full.json`.

Strictly excluding history items reduced ALS overall results to `0.36745` Candidate Recall@1,000, `0.00962` MAP@12, and `0.02505` Recall@12. As with the BPR ablation, exact repeats should therefore be handled as a controlled candidate source rather than removed from every recommendation path.

### First- and second-place-inspired pipelines

The competition winners published architecture descriptions but not complete
runnable implementations. The following runs reproduce the disclosed structures
on the same fixed 2,000 final-week customers and preserve their outputs separately.
They must not be described as the teams' exact models or leaderboard scores.

The first-place-inspired run uses the saved six-week multi-source pool, truncates
it to the published Top-100 retrieval width, adds user–item, customer–section,
7/30/42-day popularity, recency, repeat, and buyer-age interaction features,
downsamples candidate negatives, and trains one LightGBM classifier. It excludes
the winners' unreleased retrieval details and their final five-LightGBM plus
seven-CatBoost ensemble.

| First-place-inspired single model | Candidate Recall@100 | MAP@12 | Recall@12 |
|---|---:|---:|---:|
| Overall | 0.13521 | **0.02801** | 0.06229 |
| Six-week history customers (1,084) | 0.17878 | 0.04595 | 0.09480 |
| Six-week cold customers (916) | 0.08366 | 0.00679 | 0.02382 |

This approximately matches the project's best overall MAP but falls far below
the winners' reported local single-model score. The important missing component
is their exact high-precision Top-100 retrieval: the write-up names repeat,
ItemCF, similar-item, popularity, image-embedding, and learned retrieval families
but does not disclose enough implementation detail to reproduce them exactly.

A separately preserved experiment trains CatBoost on exactly the same sampled
examples and Top-100 candidates, then combines LightGBM and CatBoost using an
equal reciprocal-rank fusion fixed before evaluation. No blend weight is fitted
to the final labels.

| Top-100 ranker | Candidate Recall@100 | MAP@12 | Recall@12 |
|---|---:|---:|---:|
| LightGBM | 0.13521 | 0.02798 | 0.06273 |
| CatBoost | 0.13521 | 0.02732 | 0.06204 |
| Equal LightGBM–CatBoost rank fusion | 0.13521 | **0.02833** | **0.06345** |

CatBoost does not win alone, but its different ordering improves the fixed
ensemble by `0.00036` MAP over the matched LightGBM run. Relative to the
four-week `0.02733` system, the ensemble improves MAP by 3.7% and Recall@12 by
5.1%. This is the strongest implemented, pre-specified ranking experiment; its
outputs are stored under `artifacts/first_place_ensemble/`.

The second-place-inspired implementation follows the more detailed published
pipeline: two-month recent/global and customer-section popularity produces 600
candidates; a first LightGBM selects 130; every eligible article purchased in
the two-month context is added; and a richer LightGBM performs final ranking.
The stages use three consecutive snapshots so the first ranker cannot see the
labels used to train the second ranker.

| Second-place-inspired stage | Candidate Recall | MAP@12 | Recall@12 |
|---|---:|---:|---:|
| Initial 600 after first-ranker ordering | 0.34594 | 0.02150 | 0.04779 |
| First LightGBM Top 130 | 0.17051 | 0.02150 | 0.04779 |
| Top 130 + all two-month history (K=330) | 0.24456 | 0.02150 | 0.04779 |
| Final LightGBM | 0.24456 | **0.02571** | **0.05230** |

The final ranker improves MAP over the incoming order by 19.6%, but this
adaptation does not beat the project's `0.02733` four-week system or the
`0.02801` first-place-inspired single model. It also excludes the second-place
teammate's co-occurrence/random-graph-walk candidates, custom LambdaRank-MAP
objective, Rust feature framework, and several private higher-order features.
Complete records are in `artifacts/first_place_inspired/` and
`artifacts/second_place_inspired/`; neither overwrites an existing pool.

### Parallel candidate union → DLRM/LightGBM

The implemented architecture is:

```text
BPR MF ────────────┐
Implicit ALS ──────┤
Same-day co-purchase├─ remove exact history articles ─┐
Recent popularity ─┤                                  │
Age popularity ────┤                                  ├─ union (K=1,000) ─ DLRM/LightGBM ─ Top 12
Two-tower ─────────┘                                  │
Controlled repeat source ─────────────────────────────┘
```

Every discovery source is fitted independently. Two-tower retrieves directly from the eligible catalog and is not a serial filter. The union deduplicates articles and orders them by summed reciprocal source rank. DLRM and LightGBM receive identical customer, article, reciprocal source-rank, support-count, repeat, and union-rank features.

Ranker training uses the earlier snapshot only: August 12–September 8 context generates candidates, and September 9–15 supplies labels. Positives are next-week purchases present in that exact candidate distribution; negatives are sampled from the same union. DLRM uses weighted BCE; LightGBM uses customer-grouped LambdaRank with truncation level 12. Both are applied without refitting or tuning on September 16–22 labels.

Practical configuration: seed 42; 6,000 ranker-training customers; 2,000 final evaluation customers; BPR/ALS K=500 each; co-purchase/global popularity/age popularity K=100 each; two-tower K=200; repeat K=12; union K=1,000; 64-factor/8-iteration ALS; 3-epoch BPR; 2-epoch two-tower and DLRM; and 250 LightGBM trees. Discovery-source rows below have different K values and therefore their Candidate Recall numbers should not be compared without considering K.

| Method | Candidate K | Candidate Recall@K | MAP@12 | Recall@12 |
|---|---:|---:|---:|---:|
| BPR MF, history removed | 500 | 0.27407 | 0.00729 | 0.02318 |
| Implicit ALS, history removed | 500 | 0.26908 | 0.01083 | 0.02597 |
| Co-purchase, history removed | 100 | 0.10949 | **0.01210** | 0.02980 |
| Global popularity, history removed | 100 | 0.09706 | 0.00632 | 0.02006 |
| Age-segment popularity, history removed | 100 | 0.10885 | 0.00763 | 0.02493 |
| Two-tower, history removed | 200 | 0.11115 | 0.00415 | 0.01309 |
| Repeat source only | 12 | 0.03285 | 0.02064 | 0.03285 |
| New-item source union | 1,000 | 0.40848 | 0.00950 | 0.02803 |
| New-item union + repeats | 1,000 | **0.44101** | **0.01174** | **0.04187** |
| DLRM ranking of the same union | 1,000 | **0.44101** | 0.01089 | 0.03128 |
| LightGBM LambdaRank of the same union | 1,000 | **0.44101** | **0.02732** | **0.05956** |

Candidate Recall is identical for the final three rows because both rankers only reorder the same 1,000 articles. The reciprocal-rank union beat DLRM, while LightGBM raised MAP@12 from `0.01174` to `0.02732` and Recall@12 from `0.04187` to `0.05956`. Relative to the earlier BPR-only K=1,000 result, the complete candidate-union/LightGBM system improved Candidate Recall from `0.42375` to `0.44101`, MAP@12 from `0.00873` to `0.02732`, and Recall@12 from `0.02522` to `0.05956`. This comparison changes both retrieval and ranking, so it is an architecture result, not an isolated ALS or LightGBM effect.

For the 885 history customers, LightGBM reached `0.46733` Candidate Recall@1,000, `0.05346` MAP@12, and `0.11203` Recall@12. For the 1,115 cold-start customers, it reached `0.42012`, `0.00658`, and `0.01791`, respectively. The simple union remains better for cold-start customers (`0.00787` MAP@12 and `0.02028` Recall@12), so a cohort-aware fallback remains an open improvement. The repeat source has zero cold-start coverage by construction.

The ranker-training union retrieved 7,201 of 18,922 distinct next-week positive pairs (`38.06%`) for the sampled 6,000 customers, producing 151,221 candidate-aware training rows. LightGBM training took 0.74 seconds and scoring the two-million final candidate pairs took 4.26 seconds; the complete pipeline took 226.7 seconds. Its highest-gain features were union rank, article product type/color, repeat-source rank, co-purchase rank, price, section, user recency, and age. Detailed metrics and feature importance are recorded in `artifacts/hybrid_experiment/results.json`.

#### CatBoost on the project's own four-week candidates

The complete expanded four-week train and final pools were reused without
rerunning retrieval. LightGBM and CatBoost were trained on the same 152,292
candidate-aware examples, and their rankings were combined with a pre-specified
equal reciprocal-rank fusion. Candidate Recall is identical because all methods
only reorder the same 1,012 candidates.

| Ranker on own four-week pool | Candidate Recall@1,012 | MAP@12 | Recall@12 |
|---|---:|---:|---:|
| LightGBM | 0.44256 | 0.02733 | 0.06038 |
| CatBoost | 0.44256 | 0.02674 | 0.05835 |
| Equal LightGBM–CatBoost rank fusion | 0.44256 | **0.02765** | **0.06111** |

The fusion improves MAP by `0.00032` absolute (`1.17%` relative) and Recall@12
by `0.00073` over the matched LightGBM. The gain comes from history customers:
their MAP rises from `0.05424` to `0.05497`; inactive MAP is effectively flat
and slightly lower (`0.005980` to `0.005977`). Thus CatBoost adds useful model
diversity but does not by itself move the original four-week system above
`0.028`. Full results and predictions are stored separately in
`artifacts/own_four_week_ranker_ensemble/`.

### New/repeat split ablation

The direct-union ablation uses the same seed, temporal windows, 6,000 ranker-training customers, 2,000 evaluation customers, generator cutoffs, 1,000-candidate width, and ranker configurations. Its only architectural change is that BPR, ALS, co-purchase, popularity, segment popularity, and two-tower may each return previously purchased or new articles directly. No history filtering or dedicated repeat generator is applied. Both rankers are retrained on this matching candidate distribution.

| Policy and ranking | Candidate Recall@1,000 | MAP@12 | Recall@12 |
|---|---:|---:|---:|
| Explicit new/repeat split, reciprocal-rank union | **0.44101** | 0.01174 | **0.04187** |
| Direct unsplit union, reciprocal-rank union | 0.43543 | **0.01229** | 0.03615 |
| Explicit new/repeat split, DLRM | **0.44101** | 0.01089 | 0.03128 |
| Direct unsplit union, DLRM | 0.43543 | **0.01376** | **0.03747** |
| Explicit new/repeat split, LightGBM | **0.44101** | **0.02732** | **0.05956** |
| Direct unsplit union, LightGBM | 0.43543 | 0.02383 | 0.05449 |

For the strongest ranker, LightGBM, the explicit split raises MAP@12 from `0.02383` to `0.02732` and Recall@12 from `0.05449` to `0.05956`. The benefit is concentrated among the 885 history customers: LightGBM MAP@12 rises from `0.04521` to `0.05346`, while Candidate Recall@1,000 rises from `0.45473` to `0.46733`. Cold-start candidate recall is identical (`0.42012`) because those customers have no four-week purchases to filter or restore. Their LightGBM scores are close: unsplit `0.00686` MAP@12 versus split `0.00658`.

The split is therefore supported for the final LightGBM system, but not universally: direct union is better for DLRM and slightly better for the raw union's MAP@12. This is why the project retains both policies as a measured ablation rather than claiming that manual repeat handling always wins. Full outputs are in `artifacts/hybrid_unsplit_experiment/results.json`; its separately saved train and final candidate pools are in the same directory and do not overwrite the original experiment.

To isolate whether deleting seen items from discovery is necessary, a third run retains the exact same bounded repeat source while allowing all six discovery generators to return both seen and unseen articles. This changes only history filtering; it does not remove the explicit repeat path.

| Policy with repeat source retained | Candidate Recall@1,000 | MAP@12 | Recall@12 |
|---|---:|---:|---:|
| Filter seen discovery items, union ordering | 0.44101 | 0.01174 | 0.04187 |
| Allow seen discovery items, union ordering | **0.44114** | **0.01581** | **0.04668** |
| Filter seen discovery items, LightGBM | 0.44101 | **0.02732** | **0.05956** |
| Allow seen discovery items, LightGBM | **0.44114** | 0.02664 | 0.05886 |

Among history customers, filtering raises LightGBM MAP@12 only from `0.05294` to `0.05346` and Recall@12 from `0.11116` to `0.11203`; meanwhile, allowing seen discovery items slightly raises Candidate Recall@1,000 from `0.46733` to `0.46763`. Thus the controlled repeat source is the important architectural choice, while mandatory deletion has only a marginal one-seed LightGBM advantage and is not established as statistically significant. Filtering is retained in the current best measured configuration, but repeated-seed validation is required before claiming that deletion itself is necessary. Full results and separately persisted pools are in `artifacts/hybrid_allow_seen_repeat_experiment/`.

The expanded-capacity experiment tests whether repeats should displace new candidates. It retains the original 1,000-item filtered discovery union, adds the bounded repeat source, and ranks 1,012 candidates. Customers with fewer than 12 repeats receive additional unseen candidates in the unused slots, so every customer has a fixed-width ranker input while the original Top-1,000 unseen set is preserved.

| Final capacity | Candidate Recall | LightGBM MAP@12 | LightGBM Recall@12 |
|---|---:|---:|---:|
| 1,000 total candidates | 0.44101 | 0.02732 | 0.05956 |
| 1,012 expanded candidates | **0.44256** | **0.02733** | **0.06038** |

The extra capacity retrieved 51 additional next-week positive training pairs and raised overall Candidate Recall by `0.00155`. Overall MAP@12 was effectively unchanged (`+0.00001`), while Recall@12 rose by `0.00082`. For history customers, MAP@12 increased from `0.05346` to `0.05424`; for cold customers it fell from `0.00658` to `0.00598`, despite higher candidate recall. The expanded design cleanly prevents repeat candidates from consuming discovery capacity, but its ranking improvement is modest and requires repeated-seed validation. Full results and saved pools are in `artifacts/hybrid_expanded_repeat_experiment/`.

A final matched 1,012-capacity run isolates deletion after preserving the extra slots:

| 1,012-capacity policy | Overall Candidate Recall | Overall MAP@12 | History MAP@12 | History Recall@12 |
|---|---:|---:|---:|---:|
| Allow seen discovery items + repeats | **0.44269** | **0.02763** | 0.05339 | 0.11140 |
| Delete seen discovery items + repeats | 0.44256 | 0.02733 | **0.05424** | **0.11230** |

At equal capacity, deletion improves history MAP@12 by `1.59%` relative and history Recall@12 by `0.80%`, but reduces overall MAP@12 because the allow-seen model performs better on cold customers. Restricting relevance to genuinely new validation articles for the 847 history customers with at least one such purchase gives Candidate Recall `0.42994` versus `0.43005`, MAP@12 `0.00811` versus `0.00801`, and Recall@12 `0.04436` versus `0.04542` for allow-seen versus delete-seen, respectively. These mixed, extremely small differences show that deletion does not materially improve new-item retrieval at K≈1,000. Full matched outputs are in `artifacts/hybrid_allow_seen_repeat_expanded_experiment/` and `artifacts/hybrid_expanded_repeat_experiment/`.

### Four-week versus eight-week history

The first longer-history experiment changes only the snapshot horizon from 28 to 56 days while retaining seed 42, the same 6,000 training-label customers, the same 2,000 evaluation customers, source cutoffs, 1,000 filtered discovery slots, 12 extra repeat slots, and ranker settings. Because the eligible catalog is defined from each snapshot context, it grows from 29,009 to 36,054 articles; this is an end-to-end horizon comparison, not a catalog-fixed causal ablation. All candidate generators in this first test use the full context without recency weighting.

| Context | History customers | Cold customers | Candidate Recall@1,012 | LightGBM MAP@12 | LightGBM Recall@12 | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| 4 weeks | 885 | 1,115 | **0.44256** | **0.02733** | 0.06038 | 248.8 s |
| 8 weeks | 1,225 | 775 | 0.35406 | 0.02604 | **0.06060** | 380.2 s |

Extending every component uniformly to eight weeks recovers 340 apparent cold customers but reduces overall Candidate Recall by `0.08849` and MAP@12 by `0.00130`; Recall@12 is effectively flat (`+0.00022`). Every individual discovery source loses Candidate Recall, while the repeat source improves from `0.03285` to `0.03609`. The longer unweighted context expands the catalog and dilutes current-fashion signals, so simply feeding all generators twice as much history is not supported.

The same customer IDs were also evaluated under fixed behavioral cohorts:

| Fixed cohort | Customers | 4-week Candidate Recall | 8-week Candidate Recall | 4-week MAP@12 | 8-week MAP@12 | 4-week Recall@12 | 8-week Recall@12 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Active in latest 4 weeks | 885 | **0.46889** | 0.41855 | **0.05424** | 0.05019 | **0.11230** | 0.11052 |
| Recovered in weeks 5–8 | 340 | **0.38454** | 0.28954 | 0.00711 | **0.00858** | 0.01751 | **0.02333** |
| Still cold after 8 weeks | 775 | **0.43793** | 0.30873 | 0.00549 | **0.00611** | 0.01990 | **0.01993** |

Longer history helps final ranking for the 340 recovered dormant customers despite lower candidate recall, but it harms the already-active cohort enough to reduce overall MAP. The next justified design is therefore selective: retain short recent windows for catalog/trend retrieval, and use the older four weeks only for customer profiles, repeat evidence, and explicitly recency-weighted features. Full run artifacts are in `artifacts/hybrid_eight_week_experiment/`; the reproducible fixed-cohort comparison is in `artifacts/history_horizon_comparison/results.json`.

### Six-week follow-up

The six-week follow-up is a new isolated run; it does not overwrite the four- or eight-week pools, predictions, results, or original comparison. Its ranker-training context is July 29–September 8 and its final context is August 5–September 15. All other practical settings remain matched.

| Context | History customers | Cold customers | Eligible catalog | Candidate Recall@1,012 | LightGBM MAP@12 | LightGBM Recall@12 | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 weeks | 885 | 1,115 | 29,009 | **0.44256** | 0.02733 | 0.06038 | 248.8 s |
| 6 weeks | 1,084 | 916 | 32,792 | 0.38502 | **0.02826** | 0.05953 | 290.3 s |
| 8 weeks | 1,225 | 775 | 36,054 | 0.35406 | 0.02604 | **0.06060** | 380.2 s |

Six weeks gives the best one-seed MAP@12, improving over four weeks by `0.00092` absolute (`3.37%` relative), but it does not improve Recall@12 and its Candidate Recall is substantially lower. Candidate Recall falls monotonically as the unweighted context and catalog expand, while LightGBM ranking quality peaks at six weeks. This means six weeks is a promising configuration, not yet a proven replacement: the result needs rolling-week and repeated-seed validation.

The fixed cohorts explain where the six-week MAP gain comes from:

| Fixed cohort | Customers | 4-week MAP / Recall@12 | 6-week MAP / Recall@12 | 8-week MAP / Recall@12 |
|---|---:|---:|---:|---:|
| Active in latest 4 weeks | 885 | **0.05424 / 0.11230** | 0.05366 / 0.10711 | 0.05019 / 0.11052 |
| Recovered in weeks 5–6 | 199 | 0.00805 / 0.02212 | 0.01003 / **0.03169** | **0.01241** / 0.03029 |
| Recovered in weeks 7–8 | 141 | **0.00577** / 0.01099 | 0.00575 / **0.01683** | 0.00316 / 0.01351 |
| Still cold after 8 weeks | 775 | 0.00549 / 0.01990 | **0.00802 / 0.02010** | 0.00611 / 0.01993 |

The six-week model slightly weakens the already-active cohort but improves ranking for the newly recovered 5–6-week cohort and for the customers still interaction-cold. The latter improvement comes from changes to global/segment retrieval and the ranker, not personalization from purchase history, because those 775 users remain without an interaction profile. The new reproducible comparison is stored in `artifacts/history_horizon_comparison_4w_6w_8w/results.json`; older comparison artifacts remain unchanged.

### Cold-start lookback audit

The four-week definition labels 1,115 of the 2,000 evaluation customers as cold, but a full-history scan shows that most are dormant rather than truly new. A 90-day context recovers 678 (`60.8%`) of them; 180 days recovers 820 (`73.5%`); 365 days recovers 911 (`81.7%`); and any prior history recovers 957 (`85.8%`). Only 158 customers have no transaction before the four-week context, equal to `14.2%` of the apparent-cold cohort and `7.9%` of all evaluation customers. Detailed counts are in `artifacts/cold_start_audit/results.json` and can be reproduced with `run_cold_start_audit.py`.

This motivates three serving/evaluation cohorts: active customers use the verified 1,000-unseen-plus-repeat LightGBM path; dormant customers use a longer, time-decayed history plus user/item-neighborhood candidates and a dormant-matched ranker; truly transaction-free customers use metadata segments and recent/local popularity. Similarity-based collaborative retrieval should target dormant customers, because true cold customers have no interaction profile from which collaborative similarity can be calculated.

### Inactive-user similarity experiment

The separate inactive route keeps the verified four-week catalog and trend
window. An evaluation customer enters this route only when they made no purchase
during those four weeks. It then distinguishes three cases using transactions
strictly before the context:

- 820 customers have a transaction in the preceding 180 days;
- 137 have older history, but none in those 180 days;
- 158 have no prior transaction in the available data.

Candidate retrieval combines 300 behavioral-KNN, 400 metadata-KNN, 200
preference-matched-popularity, 200 segment-popularity, 300 global-popularity,
and up to 12 controlled-repeat candidates. The deduplicated input has 1,000
discovery slots plus 12 repeat slots. Behavioral profiles use article attributes
with a 90-day half-life and 50 nearest active users from a bounded 50,000-user
reference pool. Metadata KNN uses customer fields and an exact-postal-code
similarity boost. The inactive-only LightGBM ranker includes explicit
`has_history_180d`, `has_any_history`, historical activity, and
`days_since_last_purchase` features; missing purchase age is represented by a
separate indicator rather than silently imputed as recent.

The ranker is trained on 3,211 inactive customers from the earlier snapshot,
using 76,566 candidate-distribution-matched examples (3,646 positives). The
final week remains evaluation-only. The complete practical run took 135.7
seconds on the recorded local environment.

| Final-week inactive method | Candidate Recall | MAP@12 | Recall@12 |
|---|---:|---:|---:|
| Preserved four-week union (1,012) | **0.42165** | **0.00787** | 0.02028 |
| Preserved general LightGBM | 0.42165 | 0.00598 | 0.01917 |
| New inactive union (1,012) | 0.38646 | 0.00736 | 0.02016 |
| New inactive-only LightGBM | 0.38646 | 0.00686 | **0.02276** |
| Global popularity alone (300) | 0.20194 | 0.00762 | 0.02095 |

Relative to the old general LightGBM on the same 1,115 customers, the
inactive-only ranker improves MAP@12 by 14.7% and Recall@12 by 18.7%. It does
**not**, however, beat the preserved raw union or global popularity on MAP@12,
and its candidate recall is lower. The experiment therefore validates cohort
features and improves the learned ranker, but does not justify replacing the
current inactive fallback. The correct selected system remains the preserved
raw union for inactive customers while similarity retrieval is treated as an
ablation requiring better source blending and additional earlier training
weeks.

| Inactive cohort | Customers | New LGBM Candidate Recall | New LGBM MAP@12 | New LGBM Recall@12 |
|---|---:|---:|---:|---:|
| History within 180 days | 820 | 0.37560 | **0.00735** | 0.02127 |
| Only stale history | 137 | **0.42374** | 0.00587 | 0.02277 |
| No prior transaction | 158 | 0.41050 | 0.00513 | **0.03048** |

Behavioral KNN applies only to the first cohort and by itself reaches 0.11960
Candidate Recall@300. Metadata KNN is available to all cohorts but is weak by
itself (0.00347 MAP@12 overall). These source-level measurements are important:
they show that “more personalization” is not automatically stronger than
current-fashion popularity for sparse users. Complete matched results are in
`artifacts/inactive_knn_comparison/results.json`; the separately saved new pool,
predictions, user-state features, and run record are under
`artifacts/inactive_knn_experiment/`.

The KNN source weights were subsequently tuned using only the earlier
September 9–15 labels and then frozen for September 16–22. A grid over
`0, 0.25, 0.5, 1, 2, 4, 8` selected `0.25` for both behavioral and metadata
KNN, with the remaining preference/popularity/repeat sources held at `1.0`.

| Inactive KNN-pool ordering | MAP@12 | Recall@12 |
|---|---:|---:|
| Equal source weights | 0.00736 | 0.02016 |
| Earlier-selected KNN weights | **0.00746** | **0.02390** |
| Preserved older raw candidate pool | **0.00787** | 0.02028 |

Thus KNN reweighting improves the new similarity pool, especially Recall@12,
but does not beat the older raw pool on MAP. Giving KNN weights of `2`–`8`
substantially reduced earlier-week MAP, so user similarity is retained as a
supporting signal rather than made dominant. The complete grid is stored in
`artifacts/inactive_knn_weight_tuning/results.json`.

The stronger preserved pool was then retained in full and augmented with a
bounded KNN list. A second earlier-week grid selected the Top 25 KNN candidates
with a `0.5` reciprocal-rank contribution. Overlap was high, so this added only
4.8 unique candidates per customer on average instead of forcing 25 extra
items.

| Final inactive pool | Candidate Recall | MAP@12 | Recall@12 |
|---|---:|---:|---:|
| Preserved 1,012-candidate raw pool | 0.42165 | 0.00787 | 0.02028 |
| Preserved pool + bounded KNN | **0.42467** | **0.00808** | **0.02029** |

This is the first KNN experiment that improves the stronger inactive MAP
baseline. The gain is largest for the 137 stale-history customers, whose
MAP rises from `0.01117` to `0.01300`. Recent-dormant and transaction-free
cohorts do not improve consistently, so further routing should be validated on
additional earlier weeks rather than chosen from the final subgroup table.
The selected pool, predictions, complete grid, and cohort results are in
`artifacts/inactive_old_plus_knn/`.

Recent-dormant customers were then given a dedicated time-decayed ItemCF source.
The relationship graph uses 180 days of same-customer, same-day co-purchases:
the historical query article may be outside the current catalog, but every
neighbor must be currently eligible. Each customer's older purchases receive a
60-day exponential decay. An earlier-week grid selected 50 ItemCF candidates
with a small `0.1` reciprocal-rank contribution, and that configuration was
frozen for final evaluation.

| Recent-dormant method (820 users) | Candidate Recall | MAP@12 | Recall@12 |
|---|---:|---:|---:|
| Preserved pool + bounded KNN | 0.42353 | 0.00757 | 0.01791 |
| Base + time-decayed ItemCF | **0.42936** | **0.00769** | **0.01933** |

Across all 1,115 inactive customers, MAP rises from `0.00808` to `0.00816`
and Recall@12 from `0.02029` to `0.02133`; stale and transaction-free users are
unchanged because ItemCF is routed only to the recent-dormant cohort. The
complete grid, 180-day cache configuration, and predictions are stored under
`artifacts/recent_dormant_itemcf/`.

#### Inactive ranker ablation on identical saved candidates

The saved 1,012-wide inactive training and final pools were also ranked with a
compact DLRM. Retrieval was not rerun, so raw reciprocal-rank ordering,
LightGBM, and DLRM have identical `0.38646` Candidate Recall. The DLRM was
trained on 72,920 within-customer positive/negative pairs from the earlier
snapshot using the pairwise objective
`softplus(score_negative - score_positive)`. The final week was used only for
evaluation.

| Ranker on the same inactive pool | MAP@12 | Recall@12 |
|---|---:|---:|
| Raw reciprocal-rank ordering | **0.00736** | 0.02016 |
| Inactive LightGBM LambdaRank | 0.00686 | **0.02276** |
| Pairwise DLRM | 0.00601 | 0.02055 |

DLRM does not improve the inactive group in this controlled comparison. It
loses `12.4%` MAP relative to LightGBM and `18.3%` relative to the raw ordering,
while its Recall@12 is also below LightGBM. The result supports keeping DLRM as
an ablation rather than adding it to the selected architecture. The deterministic
seed-42 run, model weights, and Top-12 predictions are stored under
`artifacts/inactive_dlrm_experiment/`.

### Seen-item filtering ablation (four-week BPR MF)

This experiment uses the exact final MF configuration above: August 19–September 15 history, September 16–22 validation, pairwise BPR, one uniform unobserved negative per positive per epoch, three epochs, 1,000 candidates, and the same 2,000 evaluation customers. It also uses the original final-MF training seed (46) and evaluation seed (42), so the unfiltered side exactly matches the main result. History interactions remain in MF training; only recommendation-time output filtering changes.

| Policy | Candidate Recall@1,000 | MAP@12 | Recall@12 |
|---|---:|---:|---:|
| Allow exact history articles | **0.42375** | **0.00873** | **0.02522** |
| Exclude exact history articles | 0.40058 | 0.00766 | 0.02287 |

For the 885 sampled history customers, strict filtering reduced overall MAP@12 from `0.01014` to `0.00772` and Recall@12 from `0.03061` to `0.02528`. However, when evaluation is restricted to genuinely new validation articles, filtering improved history-customer MAP@12 from `0.00863` to `0.00907` and Recall@12 from `0.02769` to `0.02828`. New-item Candidate Recall@1,000 was essentially unchanged (`0.40469` versus `0.40508`); filtering fills freed positions with additional unseen articles.

Only 2.49% of sampled validation pairs were exact repeats, but those repeats were relatively predictable: the unfiltered MF retrieved 65.98% of them within 1,000 candidates for the 116 affected customers. The result supports a hybrid policy—filter exact history articles from normal discovery sources, then restore a small, explicit repeat-candidate source—rather than either allowing repeats everywhere or banning them completely. Full measurements are in `artifacts/mf_seen_ablation/results.json`.

### Customers with context purchase history

| Method | Candidate Recall@K | MAP@12 | Recall@12 |
|---|---:|---:|---:|
| Recent popularity | 0.40852 | 0.00504 | 0.02149 |
| MF | **0.42534** | **0.01014** | **0.03061** |
| MF → two-tower | 0.10274 | 0.00449 | 0.01884 |
| MF → two-tower → DLRM | 0.10274 | 0.00491 | 0.01776 |

### Cold-start customers (no context purchase history)

| Method | Candidate Recall@K | MAP@12 | Recall@12 |
|---|---:|---:|---:|
| Recent popularity | **0.42249** | **0.00762** | **0.02095** |
| MF (popularity fallback) | **0.42249** | **0.00762** | **0.02095** |
| MF → two-tower | 0.09112 | 0.00413 | 0.01593 |
| MF → two-tower → DLRM | 0.09112 | 0.00362 | 0.01330 |

The final practical run took 118.0 seconds on the recorded CPU environment; 80.9 seconds were feature construction. Runtime is saved for reproducibility, not presented as a serving-latency result.

### Interpreting the earlier 10,000-candidate recall

The earlier setup's 10,000 candidates cover `10,000 / 26,503 = 37.7%` of its three-week catalog. A recall near 0.82 is therefore much less selective than it first appears and should not be read as evidence that the final Top-12 ranking is strong. The current 1,000-candidate run covers only 3.45% of the corrected 29,009-item catalog. Its Recall@1,000 is not directly comparable to the earlier result because both the temporal context and K changed.

## Resume bullets (verified wording)

- Built a leakage-safe H&M fashion recommender with parallel BPR-MF, implicit-ALS, co-purchase, popularity, segment-popularity, two-tower, and controlled-repeat candidate sources; deduplicated source candidates and ranked a bounded union without materializing a full user–item score matrix.
- Identified that exact repeats were only `2.49%` of sampled validation pairs but were highly predictable; a bounded frequency/recency-weighted repeat source improved active-customer MAP@12 by `18.2%` and overall MAP@12 by `14.7%` against a matched unsplit candidate union.
- Reproduced key ideas from public first- and second-place solution write-ups under the same local temporal protocol; the uniform LightGBM--CatBoost system reached `0.02765 MAP@12`, while the best observed cohort-routed diagnostic reached `0.02887 MAP@12` and `0.06113 Recall@12`.

## Verified versus still pending

**Implemented and executed:** configurable 28-, 42-, and 56-day temporal snapshots; chunked data loading/cache; missing/OOV handling; BPR MF and implicit ALS; global and segment popularity; vectorized same-day co-purchase; full-catalog two-tower retrieval; seen-item filtering; controlled repeats; split-versus-direct, deletion-only, expanded-capacity, and history-horizon ablations with separately persisted pools; fixed-cohort 4/6/8-week comparison; candidate-aware DLRM and LightGBM LambdaRank; CatBoost and fixed equal-rank LightGBM–CatBoost fusion on both the first-place-inspired Top-100 pool and the project's own four-week 1,012-candidate pool; the separate inactive-user behavioral-KNN, metadata-KNN, preference, segment, popularity, repeat, and inactive-ranker experiment; the identical-pool inactive raw/LightGBM/pairwise-DLRM ablation; leakage-safe first- and second-place-inspired pipeline reproductions; batched candidates; Top-12 CSV output; overall/history/inactive-subgroup metrics; smoke and practical profiles; seventeen unit tests; deterministic seed/config/runtime logging; and a downloaded public implicit-ALS notebook adapted and executed on the identical four-week evaluation.

**Preserved:** the original `Rec_sys.ipynb` and raw CSV files were not modified.

**Still needs validation before stronger claims:** repeated runs across several seeds; multiple earlier rolling weeks for ranker training/tuning; inactive-source removal and calibrated source-blending ablations; item-to-item retrieval from older preferences; full-population scoring; and Kaggle test/leaderboard evaluation. The inactive similarity route improved the old learned ranker but did not beat the raw-union fallback on MAP@12, so it is not presented as the selected model. No deployment, online lift, or serving-latency claim has been made.
