"""Exercise preserved research UI helpers without exposing them in product navigation."""

import runpy

from pdm.paths import project_root

namespace = runpy.run_path(str(project_root() / "src" / "pdm" / "app.py"), run_name="pdm_legacy_ui_test")
namespace["legacy_main"]()
