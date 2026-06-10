"""Regression guard for the MockTool schema-shadow UserWarning.

MockTool stores its schema on the ``schema_`` field with a ``schema`` alias so
the field name does not shadow the deprecated ``BaseModel.schema`` method. That
shadow used to emit a pydantic UserWarning on every ``import korrel``. The
warning fires once, at class-definition time, so this runs a fresh interpreter
with the specific warning escalated to an error and asserts a clean import plus
an unchanged public surface (``schema=`` construction and ``tool.schema`` read).
"""

import subprocess
import sys


def test_import_emits_no_schema_shadow_warning() -> None:
    code = (
        "import warnings\n"
        "warnings.filterwarnings('error', message=r'Field name .schema.')\n"
        "import korrel\n"
        "from korrel import MockTool\n"
        "t = MockTool(name='t', schema={'type': 'object'}, respond=lambda a, s: None)\n"
        "assert t.schema == {'type': 'object'}\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
