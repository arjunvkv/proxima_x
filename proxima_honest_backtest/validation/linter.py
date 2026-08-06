from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from glob import glob
from typing import Dict, List, Optional


@dataclass
class LintResult:
    file_path: str
    passed: bool
    violations: List[Dict[str, object]] = field(default_factory=list)
    error_count: int = 0
    warning_count: int = 0

    def __post_init__(self) -> None:
        self.error_count = sum(1 for v in self.violations if v.get("severity") == "error")
        self.warning_count = sum(1 for v in self.violations if v.get("severity") == "warning")


_LOOKAHEAD_PATTERNS: List[Dict[str, object]] = [
    {
        "pattern": re.compile(r"\.shift\(\s*-\s*\d+"),
        "message": "shift with negative lag looks into future data",
        "severity": "error",
    },
    {
        "pattern": re.compile(r"\.bfill\s*\("),
        "message": "backward fill uses future data",
        "severity": "error",
    },
    {
        "pattern": re.compile(r"center\s*=\s*True"),
        "message": "centered window leaks future information",
        "severity": "warning",
    },
    {
        "pattern": re.compile(r"\.iloc\s*\[\s*-\d"),
        "message": "negative iloc indexing may access future data",
        "severity": "warning",
    },
    {
        "pattern": re.compile(r"\.iat\s*\[\s*-\d"),
        "message": "negative iat indexing may access future data",
        "severity": "warning",
    },
    {
        "pattern": re.compile(r"\.rolling\s*\(.*center\s*=\s*True"),
        "message": "centered rolling window uses future data",
        "severity": "error",
    },
    {
        "pattern": re.compile(r"\.get\(\s*[\"'](?:close|high|low)[\"']\s*\)"),
        "message": "reads forming-bar close/high/low via .get() — same-bar lookahead; use history + open only",
        "severity": "error",
    },
]


class LookAheadLinter:
    def __init__(self, rules: Optional[List[str]] = None) -> None:
        self.rules = rules
        self._qtype_available = False
        self._qtype_linter = None
        self._init_qtype()

    def _init_qtype(self) -> None:
        try:
            import qtype  # type: ignore[import-untyped]
            self._qtype_available = True
            self._qtype_linter = qtype.Linter()
        except ImportError:
            self._qtype_available = False

    def lint_file(self, file_path: str) -> LintResult:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()
        except Exception as exc:
            return LintResult(
                file_path=file_path,
                passed=False,
                violations=[{"line": 0, "col": 0, "message": f"cannot read file: {exc}", "severity": "error"}],
            )

        return self.lint_strategy_code(code, file_path)

    def lint_directory(self, dir_path: str, pattern: str = "*.py") -> List[LintResult]:
        results: List[LintResult] = []
        search_pattern = os.path.join(dir_path, "**", pattern)
        for file_path in glob(search_pattern, recursive=True):
            if os.path.isfile(file_path):
                results.append(self.lint_file(file_path))
        return results

    def lint_strategy_code(self, code: str, file_path: str = "<string>") -> LintResult:
        if self._qtype_available and self._qtype_linter is not None:
            return self._qtype_lint(code, file_path)
        return self._regex_lint(code, file_path)

    def _qtype_lint(self, code: str, file_path: str) -> LintResult:
        violations: List[Dict[str, object]] = []
        try:
            qtype_result = self._qtype_linter.lint(code)
            for issue in qtype_result.get("issues", []):
                violations.append({
                    "line": issue.get("line", 0),
                    "col": issue.get("col", 0),
                    "message": issue.get("message", "unknown qtype violation"),
                    "severity": issue.get("severity", "warning"),
                })
        except Exception as exc:
            violations.append({
                "line": 0,
                "col": 0,
                "message": f"qtype linting error: {exc}",
                "severity": "warning",
            })

        passed = all(v.get("severity") != "error" for v in violations)
        return LintResult(file_path=file_path, passed=passed, violations=violations)

    def _regex_lint(self, code: str, file_path: str) -> LintResult:
        violations: List[Dict[str, object]] = []

        try:
            tree = ast.parse(code)
            self._check_ast(tree, violations)
        except SyntaxError:
            pass

        for rule in _LOOKAHEAD_PATTERNS:
            pattern: re.Pattern = rule["pattern"]
            for match in pattern.finditer(code):
                line_number = code[: match.start()].count("\n") + 1
                violation: Dict[str, object] = {
                    "line": line_number,
                    "col": match.start() - code.rfind("\n", 0, match.start()) - 1,
                    "message": str(rule["message"]),
                    "severity": str(rule["severity"]),
                }
                violations.append(violation)

        passed = all(v.get("severity") != "error" for v in violations)
        return LintResult(file_path=file_path, passed=passed, violations=violations)

    def _check_ast(self, tree: ast.AST, violations: List[Dict[str, object]]) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "shift":
                    for arg in node.args:
                        if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                            violations.append({
                                "line": node.lineno,
                                "col": node.col_offset,
                                "message": "shift with negative argument detected",
                                "severity": "error",
                            })
                if isinstance(func, ast.Attribute) and func.attr == "bfill":
                    violations.append({
                        "line": node.lineno,
                        "col": node.col_offset,
                        "message": "backward fill detected",
                        "severity": "error",
                    })
                if isinstance(func, ast.Attribute) and func.attr == "rolling":
                    for kw in node.keywords:
                        if kw.arg == "center" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                            violations.append({
                                "line": node.lineno,
                                "col": node.col_offset,
                                "message": "centered rolling window detected",
                                "severity": "error",
                            })
            if isinstance(node, ast.Subscript):
                if isinstance(node.slice, ast.UnaryOp) and isinstance(node.slice.op, ast.USub):
                    violations.append({
                        "line": node.lineno,
                        "col": node.col_offset,
                        "message": "negative subscript indexing detected",
                        "severity": "warning",
                    })

            # STRICT-HONEST-CONTRACT RULE: a strategy may act ONLY on `history`
            # (closes strictly before the current bar) + the current bar's `open`.
            # Reading the forming bar's `close`/`high`/`low` is SAME-BAR LOOKAHEAD
            # (the Ultra Monster bug). `bars[p]["close"]` / `bar["close"]` / etc.
            # inside a strategy body is flagged as an error.
            if isinstance(node, ast.Subscript):
                key = node.slice
                is_ohlc_key = (
                    isinstance(key, ast.Constant) and isinstance(key.value, str)
                    and key.value in ("close", "high", "low")
                )
                if is_ohlc_key:
                    root = node.value
                    while isinstance(root, ast.Subscript):
                        root = root.value
                    if isinstance(root, ast.Name) and root.id in ("bars", "bar"):
                        violations.append({
                            "line": node.lineno,
                            "col": node.col_offset,
                            "message": (
                                f"reads forming-bar '{key.value}' — same-bar "
                                f"lookahead; use `history` (completed closes) + `open` only"
                            ),
                            "severity": "error",
                        })
