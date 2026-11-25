from pathlib import Path

import pytest

from chatbot.rag_adapter import validate_task2_root
from core.exceptions import Task2IntegrationError


def test_validate_task2_root_accepts_expected_package_layout(tmp_path: Path):
    required = [
        tmp_path / "app" / "__init__.py",
        tmp_path / "app" / "config.py",
        tmp_path / "app" / "services" / "rag_service.py",
        tmp_path / "app" / "services" / "vector_store.py",
    ]
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# test fixture\n", encoding="utf-8")

    assert validate_task2_root(tmp_path) == tmp_path.resolve()


def test_validate_task2_root_reports_missing_services(tmp_path: Path):
    with pytest.raises(Task2IntegrationError, match="Task 2 could not be located"):
        validate_task2_root(tmp_path)
