"""User-facing help and label copy for the Streamlit UI."""
from __future__ import annotations

APPEARANCE_HELP = (
    "Switch between a light and a dark look. Your choice is kept while you move between screens "
    "and after a page reload."
)

# Projects and Import
PROJECT_NAME_HELP = "A name to find this project later. It does not affect the data or the model."
PROJECT_SOURCE_FORMAT_HELP = (
    "The kind of files you will import. It decides which columns are read and which signal is used. "
    "It is fixed when the project is created."
)
CREATE_PROJECT_HELP = "Creates an empty project with this name and format. You import data on the next step."
IMPORT_SOURCE_MODE_HELP = (
    "Upload a folder from this browser, or type a folder path on the computer running the app. "
    "Use a path for very large data."
)
IMPORT_FOLDER_HELP = (
    "Pick the folder that holds this set's files. Each file or subfolder should belong to one physical unit "
    "(one machine, bearing, or filter)."
)
IMPORT_VALIDATION_FROM_HELP = (
    "Validation data picks the best saved model during training. Split from training sets aside whole units "
    "from the Training folder; Separate folder uses units you choose."
)
IMPORT_TEST_FROM_HELP = (
    "Test data is kept away from training and model selection, so its score shows how the model may do on "
    "new units. Split from training or use your own folder."
)
IMPORT_SPLIT_SETTINGS_HELP = (
    "Weights and seed for units split automatically from the Training folder. Separate folders are not split."
)
IMPORT_VIEW_HELP = "Open this set on Data Quality. Viewing does not change the data."
IMPORT_CARD_EMPTY = "Import to see units, rows, and gaps."
IMPORT_SPLIT_WEIGHTS_CAPTION = (
    "A separate folder stays separate. These weights split the remaining automatic units; actual unit counts "
    "appear after import."
)
IMPORT_TRAIN_WEIGHT_HELP = (
    "Share of automatically split units used to train. More training units usually help the model learn; "
    "fewer validation or test units make their scores less reliable."
)
IMPORT_VALIDATION_WEIGHT_HELP = (
    "Share of automatically split units used to choose the best saved model. Weights must add up to 100%."
)
IMPORT_TEST_WEIGHT_HELP = (
    "Share of automatically split units kept for the final, untouched score. Weights must add up to 100%."
)
IMPORT_SPLIT_SEED_HELP = (
    "A number that fixes which units land in each set. Same files and seed give the same split; "
    "a different seed gives a different random split."
)
IMPORT_SIGNAL_COLUMN_HELP = (
    "Exact CSV column holding the measurement to forecast, e.g. vibration or pressure. Must start with a "
    "letter or underscore; then letters, digits or underscores only."
)
IMPORT_SIGNAL_NAME_HELP = "Friendly name shown on charts and results. It does not affect the model."
IMPORT_SIGNAL_UNIT_HELP = (
    "Unit of the signal (e.g. g, Pa, °C). Shown on charts and used for your limits. It does not convert values."
)
IMPORT_RED_CONDITION_HELP = (
    "Whether trouble means the signal going up (for example vibration or filter pressure) or going down. "
    "It sets how yellow and red limits are compared."
)
IMPORT_YELLOW_LIMIT_HELP = (
    "Early-warning level in the signal's unit. Results mark a forecast yellow when it reaches this level. "
    "It does not change training; it only marks Results."
)
IMPORT_RED_LIMIT_HELP = (
    "Action level in the signal's unit. Results mark a forecast red when it reaches this level. "
    "For a rising signal it must be above yellow. It does not change training; it only marks Results."
)
IMPORT_SUBMIT_HELP = (
    "Copies the files into this project, checks every row, and splits units into Train, Validation and Test. "
    "Runs in the background."
)

# Data Quality
QUALITY_TABS_CAPTION = (
    "Each tab shows one set. Training learns from Training Data, Validation Data picks the best model, "
    "Testing Data is scored once at the end."
)
QUALITY_INSPECT_UNIT_HELP = (
    "Choose one physical unit to see its measurements over time. Viewing does not change the data."
)
QUALITY_CONTINUE_HELP = "Go to Training. Available when all three sets have usable measurements."
QUALITY_UNITS_HELP = (
    "Physical units (machines, bearings, filters) in this set. A unit's rows never appear in two sets."
)
QUALITY_ADMITTED_ROWS_HELP = (
    "Rows kept after import checks. Rows the import rejected are not counted; any remaining missing values "
    "are listed below."
)
QUALITY_ZONE_MODEL_CAPTION = (
    "Signal models in this project learn to forecast future {label} values, not zone classes. Zones are derived "
    "from those values with the same limits, and Results uses them to report the expected red entry."
)
QUALITY_ZONE_SUMMARY_HELP = (
    "Every admitted measurement in this set, colored by the yellow/red limits saved with the data. "
    "Not zoned rows have no valid limit yet, such as the start of a baseline."
)
QUALITY_NO_ZONES = "No valid yellow/red limits are saved with this data, so measurements are not zoned."
QUALITY_GAPS_HELP = (
    "Breaks in a unit's timeline. The model never learns or forecasts across a gap, so many gaps mean fewer "
    "usable windows."
)
QUALITY_MOVE_UNITS_HELP = (
    "Choose whole physical units to move. All rows of a unit move together; rows and windows cannot be moved "
    "on their own."
)
QUALITY_MOVE_TO_HELP = "The set the chosen units join. Each set must keep at least one unit."
QUALITY_MOVE_SUBMIT_HELP = (
    "Saves a new data snapshot with the changed sets. Earlier model runs keep the previous snapshot, so train "
    "again on the new data."
)
QUALITY_MOVE_LEGACY = (
    "This project uses the published split of its source dataset. Create a new project to change the split."
)
QUALITY_MOVE_DONE = (
    "Moved {n} unit(s) to {name}. A new data snapshot is active; earlier model runs stay with the previous "
    "data snapshot."
)
QUALITY_MOVE_TEST_OPTIMISM = "Changing Testing units after reviewing results makes later Test scores optimistic."
QUALITY_MOVE_FIXED_HSE = "{n} official HSE test unit(s) are fixed in Testing Data."
QUALITY_MOVE_JOB_ACTIVE = "Wait for the current job to finish before changing sets."
QUALITY_MOVE_PREVIEW = "After the move: Training {train} · Validation {validation} · Testing {test} units."
LEGACY_PREPARE_HELP = (
    "Reads the imported files, checks them, and builds the prepared snapshot used for training. "
    "Runs in the background."
)
LEGACY_SENSOR_UNIT_HELP = "Choose one unit to plot its raw sensor signal. Viewing does not change the data."
LEGACY_SOURCE_FORMAT_HELP = (
    "The public dataset these files come from. It decides how the files are read and which signal is used."
)
LEGACY_SOURCE_FILES_HELP = (
    "Upload the XJTU-SY ZIP, or the HSE ZIP or both HSE CSV files. For very large archives, use the local "
    "path instead."
)
LEGACY_OPEN_PROJECT_HELP = "Pick an imported dataset to continue with. Opening it does not change the data."

# Training
TRAIN_MODEL_HELP = (
    "The forecasting method. GRU and LSTM are neural networks that learn patterns over time; Quantile boosting "
    "is a fast tree method that also gives a rough 5–95% range (not calibrated). Try GRU first."
)
TRAIN_MODEL_CAPTION = (
    "The model forecasts the signal itself in its native unit, not the remaining life of the unit."
)
TRAIN_HISTORY_HELP = (
    "How many past measurements the model sees for each forecast. More history can capture slower trends but "
    "needs longer unbroken records; too long and training finds no usable windows."
)
TRAIN_HORIZONS_HELP = (
    "How far ahead to forecast, in seconds, separated by commas. Use multiples of your sampling interval "
    "(default: 1×, 2×, 3×); a horizon with no matching measurement cannot be trained."
)
TRAIN_SEED_HELP = (
    "Fixes the random start and shuffling. Same data and seed give repeatable results; change it to check that "
    "a result is not luck."
)
TRAIN_EPOCHS_HELP = (
    "Epoch = one full pass over the training data. More epochs give the model more chances to improve but take "
    "longer; the app keeps the epoch that scored best on Validation."
)
TRAIN_HIDDEN_HELP = (
    "Size of the model's memory. Larger can learn more complex patterns but trains slower and may memorize "
    "small datasets. 32 is a good start."
)
TRAIN_BATCH_HELP = (
    "How many examples the model looks at before each update. Smaller is noisier but updates more often; "
    "larger is smoother and faster on big data."
)
TRAIN_BOOSTING_ITER_HELP = (
    "Number of trees added one after another. More can fit finer detail but may overfit; the app also tries "
    "half this number and keeps whichever scores better on Validation."
)
TRAIN_BOOSTING_NO_EPOCHS_CAPTION = (
    "Quantile boosting has no epoch count; it tries two tree counts and keeps the better one on Validation."
)
TRAIN_SUBMIT_HELP = (
    "Starts training in the background with these settings. Training uses Train units, picks the best model "
    "on Validation, then scores Test once."
)
TRAIN_STOP_HELP = (
    "Asks the job to stop after its current safe step. No model is saved from a stopped run; earlier saved runs "
    "stay available."
)
TRAIN_VALIDATION_MAE_HELP = (
    "Average size of the forecast error on Validation units, in the signal's unit; each unit counts equally. "
    "Lower is better. This set was used to pick the model."
)
TRAIN_TEST_MAE_HELP = (
    "Average forecast error on Test units, which the model never saw during training or selection; each unit "
    "counts equally. The fairest estimate for new units."
)
TRAIN_STALE_RUNS_CAPTION = (
    "{n} earlier model run(s) were trained on a previous data snapshot and are not shown. Train again on this data."
)
LEGACY_MODEL_FAMILIES_HELP = (
    "Which model types to train on the same data and target. Each one is trained and scored separately so you "
    "can compare them."
)
LEGACY_START_TRAINING_HELP = (
    "Starts one background run that trains each selected model, picks checkpoints on Validation and scores "
    "Test after freezing."
)
LEGACY_COMPARE_MODELS_HELP = "Pick which trained models appear in the Test comparison. Hiding one does not delete it."
