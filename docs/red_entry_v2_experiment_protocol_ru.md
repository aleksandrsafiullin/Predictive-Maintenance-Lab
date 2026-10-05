# Протокол ограниченной матрицы RED-entry v2

Runner: `scripts/run_red_entry_v2_matrix.py`. Он обучает отдельную задачу первого зарегистрированного входа в RED. Старые signal/RUL-проекты и их результаты не изменяются. Единственный разрешённый store — `research-projects` рядом с заданным `research_data_freeze.json`.

## Команды

Выполнять из корня репозитория. `PDM_PROJECTS_ROOT` обязателен; runner отклоняет оригинальный store.

```sh
PDM_PROJECTS_ROOT=output/red-entry-v2-20261002/research-projects \
.venv/bin/python scripts/run_red_entry_v2_matrix.py \
  --projects legacy-bearings,legacy-filters \
  --engines all --input-modes age_context,sensor_only,hybrid \
  --seeds 42,73 --development-folds 2 \
  --research-freeze output/red-entry-v2-20261002/strict_research_data_freeze.json \
  --output-dir output/red-entry-v2-20261002/matrix-smoke-strict --smoke
```

Полный ограниченный запуск использует ту же команду без `--smoke` и отдельный `--output-dir output/red-entry-v2-20261002/matrix-bounded`. Для продолжения добавить `--resume` к неизменённой команде. Smoke и полный запуск имеют разные contracts; заменить один другим при resume запрещено. `--engines` принимает `all` или список поддержанных движков; `--development-folds 0` разрешает явный запуск только исходного исследовательского split. Такая сокращённая матрица не заменяет заранее предусмотренные grouped-проверки.

`--research-freeze PATH` задаёт другую предварительную фиксацию; соответствующий изолированный store должен лежать в `PATH.parent/research-projects`. Перечень проектов должен существовать в этой фиксации.

## Бюджет и выбор

Полный бюджет: не более 20 эпох, early stopping patience 5, максимум 64 origins на физическую единицу, история 16 фактических отсчётов, hidden size 32; recurrent learning rate 0.001. Smoke: 1 эпоха и максимум 4 origins. Фактическое число эпох и время сохраняются для каждой задачи. Baseline и boosting/Full CNS используют свои фиксированные процедуры адаптеров, замороженные хешами исходного кода; предел recurrent-эпох не превращается в обещание одинаковой вычислительной стоимости разных движков.

Две базы × исходный split и два development folds × семь движков × три режима × два seeds = 252 задачи. Это ограничение объёма, а не оценка времени в секундах. Full CNS использует все 166700 нейронов официального графа; размер графа и число окон могут доминировать во времени и памяти даже при коротком обучении. Нет синтетического или уменьшенного заменителя при отсутствии источника. Недоступность сохраняется отдельно от ошибки fit.

Фактический объём приёмки 2 октября заранее задан в `output/red-entry-v2-20261002/execution_plan.json`: **64 задачи**. Первая серия сравнивает все семь движков в `hybrid` на двух базах и двух seeds (28 задач). Вторая проверяет три набора входов у заранее выбранного GRU на исходном split и двух фиксированных физических development folds, двух базах и двух seeds (36 задач). Это полное обучение в зафиксированном ограниченном бюджете, без `--smoke`; exhaustive-поиск всех архитектур во всех абляциях не выполняется. Выбор GRU как представителя абляций сделан до просмотра новых Test-результатов.

```sh
PDM_PROJECTS_ROOT=output/red-entry-v2-20261002/research-projects \
.venv/bin/python scripts/run_red_entry_v2_matrix.py \
  --projects legacy-bearings,legacy-filters --engines all \
  --input-modes hybrid --seeds 42,73 --development-folds 0 \
  --output-dir output/red-entry-v2-20261002/architecture-hybrid

PDM_PROJECTS_ROOT=output/red-entry-v2-20261002/research-projects \
.venv/bin/python scripts/run_red_entry_v2_matrix.py \
  --projects legacy-bearings,legacy-filters --engines gru \
  --input-modes age_context,sensor_only,hybrid --seeds 42,73 \
  --development-folds 2 \
  --output-dir output/red-entry-v2-20261002/gru-context-ablation
```

Фактический allowlist после исправления: `age_context` = known_age + operating_context; `sensor_only` = только sensor, без наработки, часов, номера строки, age-прокси и operating_context/regime; `hybrid` = все три роли. Quality служит масками/границами достоверности. Первоначальные 36 абляций сохранялись с sensor + operating_context и проверяли исключение возраста при общем режиме; они сохранены отдельно, но для строгой проверки раздела 8 заменены новой 36-task серией `gru-context-ablation-strict`. Это исправление контракта, а не выбор входов по Test.

Преобразования обучаются на Train. Checkpoint и калибровка используют Validation; повторное использование Validation остаётся exploratory. Test не участвует в выборе архитектуры, режима, seed, эпохи или порога. Обе заранее объявленные development-разбивки берутся из `research_data_freeze.json`, охватывают только исходные Train + Validation и сохраняют исторические Test/holdout. Проверка физической идентичности запрещает разнести циклы одного оборудования между частями.

Каждый fold публикуется как новый immutable snapshot: копируются исходные канонические файлы, создаются новый split, fingerprint и ID. Активный snapshot не переключается. Исходные snapshots не редактируются. Эти folds не создают независимый holdout; уже просмотренный Test сохраняет статус `explored`.

## Коррекция строгого sensor-only

`strict_sensor_only_correction_plan.json` фиксирует причину, неизменные данные/физические folds, seeds и бюджет до повторного fit. `strict_research_data_freeze.json` меняет только protocol.input_modes.sensor_only и добавляет происхождение исправления; исходная фиксация сохранена. Новая серия использует точный архив `frozen-source-strict/source_archive.json`. Старые 28 hybrid-архитектур сохраняют вычислительную семантику: реальные state/transform/windows age_context/hybrid и старый Full CNS replay повторены побитово.

```sh
PDM_PROJECTS_ROOT=output/red-entry-v2-20261002/research-projects \
.venv/bin/python scripts/run_red_entry_v2_matrix.py \
  --projects legacy-bearings,legacy-filters --engines gru \
  --input-modes age_context,sensor_only,hybrid --seeds 42,73 \
  --development-folds 2 \
  --research-freeze output/red-entry-v2-20261002/strict_research_data_freeze.json \
  --output-dir output/red-entry-v2-20261002/gru-context-ablation-strict
```

Итоговый соответствующий контракту объём: 28 архитектур + 36 строгих абляций = 64. Всего сохранено 100 ограниченных запусков, включая первые 36 до исправления; они не увеличивают независимую доказательную базу. Исторические команды выше описывают исходную фиксацию; текущий код требует новую strict freeze. Для проверки старого resume используется архивный preflight helper; исходные bytes не переписываются.

## Freeze, восстановление и файлы

До fit сохраняется `frozen_contract.json`: протокол, полный бюджет, проекты и source fingerprints, hashes всех файлов исходных snapshots, effective RED rule, горизонты, назначения folds, движки, режимы, seeds, хеши исходного кода и официальных источников Full CNS. Изменение этих данных требует нового output directory.

Freeze охватывает все Python-файлы `src/pdm`, сам runner, YAML-протокол, `pyproject.toml`, lockfile и локальные версии Python/установленных пакетов. Так изменение вычисления порога, recurrent encoder или проверки split также запрещает resume прежней серии.

`matrix_manifest.json` содержит hash freeze, immutable fold snapshots и их hashes, задачи, статусы, фактическое время, selection и ссылки на сохранённые runs. Состояния: `running`, `completed`, `failed`, `unavailable`. Каждая запись состояния сохраняется атомарно. Manifest имеет checksum; завершённая задача пропускается только после проверки всех её artifacts. Ошибочные, недоступные и прерванные задачи повторяются при resume. Причина ошибки сохраняется; ненулевой exit code означает наличие `failed`, а наличие `unavailable` нужно отдельно читать в manifest.

Для каждого завершённого run хешируются training contract, feature state, targets и target identity, predictions, weights/model, calibration, metrics и собственный manifest. Итоговые `comparison.json` и `comparison.csv` также хешируются. Resume проверяет их до перезаписи. Checksum защищает от случайного изменения; это не цифровая подпись против атакующего, способного переписать файлы и checksums.

## Сравнение и границы доказательств

`comparison.json` содержит строки по методу, режиму, seed, split/fold и горизонту. `native` показывает собственное доступное покрытие; `common` оценивает только пересечение одинаковых `(unit_id, episode_id, timestamp_s)` с известным исходом и конечной поддержанной вероятностью у всех завершённых методов/режимов/seeds в данном snapshot, части и горизонте. Незавершённые методы перечисляются в `status_counts`: общая таблица завершённых методов не является полной запланированной матрицей.

Brier сначала усредняется по origins одной физической единицы, затем по физическим единицам. Это known-outcome Brier, не IPCW survival Brier. Неизвестный follow-up, плохой контекст, отсутствие horizon support и NaN-хвосты baseline исключаются; они не превращаются в отрицательные события. Для оценки приведены количество всех sampled origins, число поддержанных известных origins, число физических единиц и per-unit scores. Все sampled origins включают подтверждённые event/post-event строки; их отношение к оценённым строкам не является долей доступности до события. Qualified policy coverage оценивается отдельно по потенциальным origins: подтверждённые inactive строки исключены, неопределённые риск/качество/контекст остаются обязательствами знаменателя. `seed_unit_spread` сохраняет значения каждого seed и каждой единицы без bootstrap по пересекающимся строкам.

Сравнение является описательным: общей базы может не хватать после пересечения age-context с sensor-only. Доступность метода нельзя скрывать одним усреднённым score. Отдельный `metrics.json` каждого run сохраняет event/alert evaluation, где она поддерживается; per-horizon comparison остаётся пригодным, когда целая survival distribution недоступна.

`pairwise_common` содержит все пары завершённых методов на одном snapshot, части и горизонте. Это позволяет оценить доступную общую часть двух методов, когда глобальное пересечение пусто из-за ограниченной поддержки третьего. Пары не выбираются по значениям метрик. `diagnostic_reference` отдельно сохраняет одну треть средней непрерывной Train-длительности и достижимость такого упреждения; подтверждённое событие остаётся в знаменателе даже без пригодных ранних origins. События с неизвестной историей указаны отдельно.

Модели этой матрицы выдают hazard. Отдельно вычисляется MAE причинного `last_value` baseline: прогноз равен последнему измерению в origin, оценка использует только реально записанное измерение в `origin + horizon` с допуском 1e-6 с, без интерполяции. Разрыв, плохое качество, смена цикла и неполное наблюдение исключаются; сохраняются физически равные веса, единицы сигнала и покрытие. Это вспомогательный численный baseline, а не оценка срока RED и не MAE обученного signal-head.

Прогнозы для отчёта ограничены теми же 64 origins на физическую единицу. Оценка тревог сохраняет полную хронологию оборудования; в пропущенных точках вероятность остаётся неизвестной. Поэтому нагрузка и исходы эпизодов при неполном покрытии имеют статус частичной оценки, который исключает положительный допуск. Ограниченные метрики нельзя выдавать за полностью оценённую непрерывную политику предупреждений.

Smoke подтверждает контракт сохранения/восстановления, но не качество. Исторический Test является exploratory/regression. Независимый holdout отсутствует, эксплуатационные требования не заполнены; quality gate не может получить `passed`. Большему числу эпох не приписывается гарантированный выигрыш.

## Проверка runner

```sh
.venv/bin/python -m pytest tests/test_red_entry_runner.py
.venv/bin/ruff check scripts/run_red_entry_v2_matrix.py tests/test_red_entry_runner.py
```

Тесты используют tiny fixtures: freeze/resume, целостность completed artifacts и reports, раздельные unavailable/failed, общий finite support без ложных отрицательных исходов, физические folds, реальный baseline fit/save/compare/resume. Полная матрица, Full CNS fit и UI в этих тестах не выполняются.
