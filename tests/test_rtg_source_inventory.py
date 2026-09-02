"""Filesystem and import-closure checks for the ATTR-RTG source seal."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from aletheion_state_models.benchmarks.transition_risk.rtg_pipeline_seal import (
    EVALUATOR_RELATIVE_PATH,
    GENERATOR_RELATIVE_PATH,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_source_inventory import (
    SOURCE_RELATIVE_PATHS,
)

ROOT = Path(__file__).resolve().parents[1]
_PYTHON_ROOTS = (
    "src/aletheion_state_models",
    "src/drm_language_emitter",
    "world_model",
    "transformer",
)
_SEPARATE_GROUPS = {GENERATOR_RELATIVE_PATH, EVALUATOR_RELATIVE_PATH}


def _permitted_filesystem_set() -> set[str]:
    paths = {
        path.relative_to(ROOT).as_posix()
        for relative in _PYTHON_ROOTS
        for path in (ROOT / relative).rglob("*.py")
        if "__pycache__" not in path.parts
    }
    paths.difference_update(_SEPARATE_GROUPS)
    paths.update(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "scripts").glob("run_attr_rtg_*.py")
    )
    paths.update({"pyproject.toml", "docs/ATTR_RTG_PREREGISTRATION.md"})
    for directory in (ROOT / "configs", ROOT / "transformer"):
        paths.update(
            path.relative_to(ROOT).as_posix()
            for path in directory.iterdir()
            if path.is_file() and path.suffix in {".json", ".yaml", ".yml"}
        )
    return paths


def _module_index() -> dict[str, Path]:
    result = {}
    for relative in _PYTHON_ROOTS:
        for path in (ROOT / relative).rglob("*.py"):
            parts = list(path.relative_to(ROOT).parts)
            if parts[0] == "src":
                parts.pop(0)
            if parts[-1] == "__init__.py":
                module = ".".join(parts[:-1])
            else:
                module = ".".join(parts)[:-3]
            result[module] = path
    return result


def _local_imports(path: Path, modules: dict[str, Path]) -> set[Path]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    current = next((name for name, item in modules.items() if item == path), "")
    package = current if path.name == "__init__.py" else current.rpartition(".")[0]
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parent = package.split(".")
                parent = parent[: len(parent) - (node.level - 1)]
                base = ".".join((*parent, *((node.module,) if node.module else ())))
            else:
                base = node.module or ""
            if base:
                imported_modules.add(base)
            imported_modules.update(
                f"{base}.{alias.name}" if base else alias.name
                for alias in node.names
            )
    result: set[Path] = set()
    for module in imported_modules:
        if module in modules:
            result.add(modules[module])
            parts = module.split(".")
            for index in range(1, len(parts)):
                package_path = modules.get(".".join(parts[:index]))
                if package_path is not None and package_path.name == "__init__.py":
                    result.add(package_path)
    return result


def _execution_closure() -> set[str]:
    modules = _module_index()
    pending = [
        *(ROOT / "scripts").glob("run_attr_rtg_*.py"),
        ROOT / GENERATOR_RELATIVE_PATH,
        ROOT / EVALUATOR_RELATIVE_PATH,
    ]
    closure = set(pending)
    while pending:
        for imported in _local_imports(pending.pop(), modules):
            if imported not in closure:
                closure.add(imported)
                pending.append(imported)
    return {path.relative_to(ROOT).as_posix() for path in closure}


def test_literal_inventory_exactly_matches_permitted_filesystem_set():
    assert len(SOURCE_RELATIVE_PATHS) == len(set(SOURCE_RELATIVE_PATHS))
    assert tuple(sorted(SOURCE_RELATIVE_PATHS)) == SOURCE_RELATIVE_PATHS
    assert set(SOURCE_RELATIVE_PATHS) == _permitted_filesystem_set()


def test_static_execution_closure_is_fully_sealed():
    sealed = set(SOURCE_RELATIVE_PATHS) | _SEPARATE_GROUPS
    assert _execution_closure() <= sealed


def test_finalized_pipeline_paths_use_the_exact_literal_inventory():
    document = json.loads(
        (ROOT / "runs/attr_rtg/pipeline_paths.json").read_text(encoding="utf-8")
    )
    assert tuple(document["sources"]) == SOURCE_RELATIVE_PATHS
    assert document["sources"] == {
        relative: relative for relative in SOURCE_RELATIVE_PATHS
    }
