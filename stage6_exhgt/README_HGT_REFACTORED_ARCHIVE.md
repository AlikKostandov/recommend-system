# ExHGT

## Архитектура

```text
hgt_common.py
    общая логика HGT: config, загрузка таблиц, сборка HeteroData, модель,
    обучение, temporal 80/20, synthetic cold-start, evaluation, explainability.

hgt_exp_*.ipynb
    только сценарии экспериментов: какие stage запускать, какие параметры
    threshold/min_pos/k/metrics использовать, как сохранять результаты и строить графики.
```

`hgt_common.py` не содержит специальных сценариев.
Он получает строку `stage` и включает признаки по наличию подстрок:

```python
"genres"
"directors"
"actors"
"countries"
"tags"
"age_group"
"occupation"
```

Например:

```python
stage = "genres_actors_age_group"
```

означает, что граф будет содержать связи по жанрам, актёрам и возрастной группе.

## Файлы

- `hgt_common.py` — общая библиотека HGT.
- `hgt_exp_single_context.ipynb` — каждый признак отдельно: `raw`, `genres`, `directors`, `actors`, `countries`, `tags`, `age_group`, `occupation`.
- `hgt_exp_all_vs_without_one_context.ipynb` — полная модель против полной модели без одного признака.
- `hgt_exp_temporal_80_20.ipynb` — обычные temporal 80/20 эксперименты.
- `hgt_exp_cold_start.ipynb` — искусственный cold-start.
- `hgt_exp_random_context_order.ipynb` — случайные порядки добавления контекста.
- `hgt_exp_explainability.ipynb` — шаблон для post-hoc объяснений.

## Как задавать параметры

В каждом notebook параметры задаются через `HGTExperimentConfig`:

```python
config = HGTExperimentConfig(
    threshold_values=[5.0],
    min_pos_values=[5],
    k_values=[20],
    target_metrics=["recall", "ndcg"],
    epochs=100,
)
```

Запуск temporal-сценария:

```python
result_df, artifacts = runner.run_temporal_80_20(
    stage="genres_directors_actors_countries_tags_age_group_occupation",
    threshold=5.0,
    min_pos=5,
    k_values=[20],
)
```

Запуск cold-start:

```python
cold_df = runner.run_cold_start_grid(
    stage="genres_directors_actors_countries_tags_age_group_occupation",
    threshold=5.0,
    min_pos=5,
    cold_n_values=[1, 3, 5],
    k_values=[20],
)
```

## Метрики

По умолчанию поддерживаются:

```python
Recall@K
NDCG@K
HitRate@K
```

## Данные

`load_hgt_tables(DATA_DIR)` ожидает CSV-файлы:

```text
user_movie_rates.csv
users.csv
movies.csv
movie_genres.csv
movie_actors.csv
movie_directors.csv
movie_countries.csv
movie_tag.csv
```
