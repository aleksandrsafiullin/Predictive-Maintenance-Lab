# Create ML inspired frontend implementation plan

## Product flow

The workspace opens at **Import data**. A project appears only when source files have been supplied or a prepared local snapshot exists. The user then proceeds through **Data Quality → Training → Results → Compare**. The navigation communicates completion, current step, and the next available action; it does not present datasets or screens as unrelated radio options.

## Architecture

1. Keep the existing Streamlit and Python model pipeline. Reuse the download, prepare, train, evaluation, and report functions rather than rewriting their scientific logic.
2. Make import the owner of source selection. Show discovered local projects and accept a local path or browser upload; ask for the supported source format (XJTU-SY bearings or HSE filters) as part of import. Preserve the large-file path workflow because the bearing archive is several gigabytes. Do not imply arbitrary CSV schemas are supported.
3. Derive workflow availability from real artifacts: raw files, a prepared snapshot, and saved runs/evaluations. Keep previously completed steps accessible so users can revisit evidence. Never make a disabled-looking label that still launches training.
4. Replace the sidebar radios with a compact project area and step navigation. Keep worker status available but visually secondary. Provide clear next-step actions after import, quality review, and training.
5. Add a shared light/dark design system with restrained surfaces, readable typography, quiet borders, semantic status colors, and responsive widths. Apply it to the existing report and neural explorer without obscuring their technical labels.
6. Verify no-data, existing-data, and active-worker states; test dataset switching, light/dark theme, narrow viewport, and that model reports remain tied to their dataset and run.

## Acceptance criteria

- The sidebar contains no `Dataset` or `Screen` radio groups.
- The first step provides an actual import path and browser file upload for supported files, with honest size guidance and source identification.
- A fresh workspace cannot appear to have a ready project; a prepared local snapshot is discoverable without being treated as a preset demo button.
- Quality, training, and results form one navigable sequence with the current and completed states visible.
- Light and dark themes can be selected in the UI and remain legible after reruns.
- Existing preparation, training, reporting, and comparison calculations continue to run unchanged.

## Reference

Apple describes Create ML's workflow as gathering data, training, and evaluation, and its app as input, training, and output with live progress and model preview: [Create ML](https://developer.apple.com/documentation/createml) and [Introducing the Create ML App](https://developer.apple.com/videos/play/wwdc2019/430/). This plan uses that workflow as a design reference, without copying Apple's UI assets.

## Implemented and verified

- Import is the first step. It discovers locally prepared projects, accepts supported browser files or a local path, and hands source acquisition to the existing worker. The quality page owns inspection and preparation.
- Artifact-aware navigation leads through quality, in-app future-red model training, test results, and comparison on the same frozen matrix target. Historical RUL comparisons remain available as a separate comparison target.
- The interface has dark and light palettes, including report charts, tables, inputs, and the neural explorer. A selected theme remains active while navigating within the session.
- The affected Streamlit UI suites passed with `PYTHONPATH=src .venv/bin/python -m pytest` (165 tests), and Ruff passed for `src tests scripts`. Desktop and narrow browser layouts were inspected; the browser console reported no errors.
- A full model training run was not started as part of frontend verification. Browser uploads retain Streamlit's 200 MB per-file limit, so the multi-gigabyte bearing archive uses the local path input.
