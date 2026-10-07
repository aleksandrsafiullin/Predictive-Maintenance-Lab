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
QUALITY_GAPS_HELP = (
    "Breaks in a unit's timeline. The model never learns or forecasts across a gap, so many gaps mean fewer "
    "usable windows."
)
QUALITY_MOVE_TO_HELP = (
    "The set this unit will join. The unit leaves its current set. Training, Validation, and Test each keep "
    "at least one unit."
)
QUALITY_MOVE_SUBMIT_HELP = (
    "Publishes a new data snapshot with this unit in the chosen set. Earlier model runs stay on the previous "
    "snapshot."
)
QUALITY_REPLACE_WITH_HELP = (
    "A unit from another set that trades places with the unit you are inspecting. Set counts stay the same."
)
QUALITY_REPLACE_SUBMIT_HELP = (
    "Publishes one new data snapshot in which the two units trade sets. Earlier model runs stay on the "
    "previous snapshot."
)
QUALITY_MOVE_TEST_OPTIMISM = (
    "Changing Testing units after reviewing results makes later Test scores optimistic."
)
QUALITY_MOVE_DONE = (
    "Moved {unit} to {destination}. A new data snapshot is active; earlier model runs stay with the previous "
    "data snapshot."
)
QUALITY_REPLACE_DONE = (
    "Swapped {unit_a} with {unit_b}. A new data snapshot is active; earlier model runs stay with the previous "
    "data snapshot."
)
QUALITY_REPLACE_NONE = "No unit in another set can take this place."
QUALITY_SUGGEST_LIMITS_HELP = (
    "Fills yellow and red from Training Data for the selected direction. Zones only label the chart. "
    "Nothing is saved until you press Save."
)
QUALITY_SUGGEST_DONE = (
    "Suggested yellow {yellow:g} and red {red:g} from Training Data. Press Save to keep them."
)
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
    "GRU/LSTM learn sequences; boosting uses trees; Full MaleCNS uses the public connectome. "
    "All learn a trend corridor at ±10%, expanding up to ±15%. RED entry follows from its boundaries."
)
TRAIN_FULL_CNS_CAPTION = (
    "Uses all classified MaleCNS v1.0 neurons and original directed connections. Synapse counts are scaled "
    "for a fixed reservoir; its dynamics are mathematical, not measured fly activity. Training may be slow."
)
TRAIN_HISTORY_HELP = (
    "How many past measurements the model sees for each forecast. More history can capture slower trends but "
    "needs longer unbroken records; too long and training finds no usable windows."
)
TRAIN_HORIZONS_HELP = (
    "Comma-separated seconds, preferably multiples of the sampling interval. Each time is a direct model "
    "output and needs matching Training measurements. Missing calibration targets remain unknown."
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
    "small datasets. The current starting value is 64."
)
TRAIN_BATCH_HELP = (
    "How many examples the model looks at before each update. Smaller is noisier but updates more often; "
    "larger is smoother and faster on big data."
)
TRAIN_BOOSTING_ITER_HELP = (
    "Fixed number of tree iterations on Training units. A learned corridor readout is then selected on Validation."
)
TRAIN_BOOSTING_NO_EPOCHS_CAPTION = (
    "Quantile boosting uses a fixed iteration count."
)
TRAIN_SUBMIT_HELP = (
    "Fits a bounded trend corridor on Training, selects its weights on Validation, "
    "then measures containment and width on Test. Half-width stays within ±10–15%; misses count as errors."
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
