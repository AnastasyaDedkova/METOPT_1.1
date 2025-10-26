import io
import os
import runpy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contextlib import redirect_stdout
import numpy as np
import pytest

import lp_solver as ss

def run_cli_with_file(tmp_path: Path, text: str):
    """Создаем временный файл с задачей и выполняем main(), вернув stdout."""
    p = tmp_path / "problem.txt"
    p.write_text(text, encoding="utf-8")
    fake_argv = ["simplex_solver.py", str(p)]
    buf = io.StringIO()
    # подменяем sys.argv и stdout
    old_argv = sys.argv
    try:
        sys.argv = fake_argv
        with redirect_stdout(buf):
            runpy.run_module("simplex_solver", run_name="__main__")
    finally:
        sys.argv = old_argv
    return buf.getvalue().strip()


# Парсеры форматов
def test_parse_math_basic():
    text = """
    max 3x1 + 2x2
    x1 + x2 <= 4
    x1 <= 2
    x2 <= 3
    x >= 0
    """
    sense, c, A, b, rels = ss._parse_math(text)
    assert sense == "MAX"
    np.testing.assert_allclose(c, [3, 2])
    np.testing.assert_allclose(A, [[1, 1], [1, 0], [0, 1]])
    np.testing.assert_allclose(b, [4, 2, 3])
    assert rels == ["<=", "<=", "<="]


def test_parse_math_ignores_x_ge_0_declaration():
    text = """
    min x1 + x2
    x1 + 2x2 >= 4
    3x1 + x2  >= 3
    # это комментарий
    x >= 0
    """
    sense, c, A, b, rels = ss._parse_math(text)
    assert sense == "MIN"
    np.testing.assert_allclose(c, [1, 1])
    np.testing.assert_allclose(A, [[1, 2], [3, 1]])
    np.testing.assert_allclose(b, [4, 3])
    assert rels == [">=", ">="]

def test_parse_table_basic():
    text = """\
MAX
2 2
3 2
1 1 <= 4
1 0 <= 2
"""
    sense, c, A, b, rels = ss._parse_table(text)
    assert sense == "MAX"
    np.testing.assert_allclose(c, [3, 2])
    np.testing.assert_allclose(A, [[1, 1], [1, 0]])
    np.testing.assert_allclose(b, [4, 2])
    assert rels == ["<=", "<="]


# Приведение к стандартной форме (to_standard)
def test_to_standard_normalizes_negative_rhs_and_sets_basis():
    A = np.array([[-1.0]])
    b = np.array([-2.0])
    rels = ["<="]
    sense = "MAX"
    c = np.array([1.0])

    std = ss.to_standard(sense, c, A.copy(), b.copy(), rels.copy())

    assert std.A.shape[0] == 1
    assert std.A.shape[1] == 3
    row = std.A[0]
    assert row[0] == pytest.approx(1.0, abs=1e-12)
    assert row[1] == pytest.approx(-1.0, abs=1e-12)
    assert row[2] == pytest.approx(1.0, abs=1e-12)
    assert std.vtypes == ["orig", "slack", "art"]
    assert std.basis == [2]
    assert std.b[0] == pytest.approx(2.0, abs=1e-12)

def test_to_standard_bases_for_each_rel():
    A = np.array([[1.0, 2.0]])
    b = np.array([5.0])
    rels = ["<="]
    c = np.array([1.0, 1.0])
    std = ss.to_standard("MAX", c, A.copy(), b.copy(), rels.copy())
    assert std.vtypes[-1] == "slack"
    assert std.basis == [2]

    A = np.array([[1.0]])
    b = np.array([3.0])
    rels = ["="]
    c = np.array([0.0])
    std = ss.to_standard("MAX", c, A.copy(), b.copy(), rels.copy())
    assert std.vtypes[-1] == "art"
    assert std.basis == [1]

    A = np.array([[1.0]])
    b = np.array([2.0])
    rels = [">="]
    c = np.array([0.0])
    std = ss.to_standard("MAX", c, A.copy(), b.copy(), rels.copy())
    assert std.vtypes == ["orig", "slack", "art"]
    assert std.basis == [2]  # индекс art


# Симплекс: статусы и корректные решения
def test_simplex_optimal_unique():
    text = """
    max 3x1 + 2x2
    x1 + x2 <= 4
    x1 <= 2
    x2 <= 3
    """
    sense, c, A, b, rels = ss._parse_math(text)
    status, z, x = ss.simplex(ss.to_standard(sense, c, A, b, rels))
    assert status == ss.Status.OPTIMAL
    assert z == pytest.approx(10.0, rel=1e-9, abs=1e-9)
    np.testing.assert_allclose(x, [2.0, 2.0], atol=1e-9)


def test_simplex_requires_phase1_and_finds_min():
    text = """
    min x1 + x2
    x1 + 2x2 >= 4
    3x1 + x2  >= 3
    """
    sense, c, A, b, rels = ss._parse_math(text)
    status, z, x = ss.simplex(ss.to_standard(sense, c, A, b, rels))
    assert status == ss.Status.OPTIMAL
    assert z == pytest.approx(11/5, rel=1e-9, abs=1e-9)
    np.testing.assert_allclose(x, [2/5, 9/5], atol=1e-8)


def test_simplex_infeasible():
    text = """
    max x1
    x1 >= 1
    x1 <= 0
    """
    sense, c, A, b, rels = ss._parse_math(text)
    status, z, x = ss.simplex(ss.to_standard(sense, c, A, b, rels))
    assert status == ss.Status.INFEASIBLE


def test_simplex_unbounded():
    text = """
    max x1 + x2
    x1 - x2 >= 0
    x1 + x2 >= 1
    """
    sense, c, A, b, rels = ss._parse_math(text)
    status, z, x = ss.simplex(ss.to_standard(sense, c, A, b, rels))
    assert status == ss.Status.UNBOUNDED


def test_simplex_equality_altopt():
    text = """
    max x1 + x2
    x1 + x2 = 5
    """
    sense, c, A, b, rels = ss._parse_math(text)
    status, z, x = ss.simplex(ss.to_standard(sense, c, A, b, rels))
    assert status == ss.Status.OPTIMAL
    assert z == pytest.approx(5.0, abs=1e-9)
    assert x.sum() == pytest.approx(5.0, abs=1e-8)
    assert np.all(x >= -1e-8)


def test_simplex_degenerate_bland():
    text = """
    max x1
    x1 + x2 <= 1
    x1 <= 1
    x2 <= 1
    """
    sense, c, A, b, rels = ss._parse_math(text)
    status, z, x = ss.simplex(ss.to_standard(sense, c, A, b, rels))
    assert status == ss.Status.OPTIMAL
    assert z == pytest.approx(1.0, abs=1e-9)
    assert x[0] == pytest.approx(1.0, abs=1e-8)
    assert x[1] == pytest.approx(0.0, abs=1e-8)


def test_table_optimal():
    text = """\
MAX
2 2
3 2
1 1 <= 4
1 0 <= 2
"""
    sense, c, A, b, rels = ss._parse_table(text)
    status, z, x = ss.simplex(ss.to_standard(sense, c, A, b, rels))
    assert status == ss.Status.OPTIMAL
    assert z == pytest.approx(10.0, abs=1e-9)
    np.testing.assert_allclose(x, [2.0, 2.0], atol=1e-8)


def test_table_unbounded():
    text = """\
MAX
2 2
1 1
1 -1 >= 0
1  1 >= 1
"""
    sense, c, A, b, rels = ss._parse_table(text)
    status, z, x = ss.simplex(ss.to_standard(sense, c, A, b, rels))
    assert status == ss.Status.UNBOUNDED


def test_cli_outputs_optimal(tmp_path):
    text = """
    max 3x1 + 2x2
    x1 + x2 <= 4
    x1 <= 2
    x2 <= 3
    """
    out = run_cli_with_file(tmp_path, text)
    assert "F = 10" in out
    assert "x1 = 2" in out and "x2 = 2" in out


def test_cli_outputs_unbounded(tmp_path):
    text = """
    max x1 + x2
    x1 - x2 >= 0
    x1 + x2 >= 1
    """
    out = run_cli_with_file(tmp_path, text)
    assert out.strip().upper().endswith("UNBOUNDED")


def test_cli_outputs_infeasible(tmp_path):
    text = """
    max x1
    x1 >= 1
    x1 <= 0
    """
    out = run_cli_with_file(tmp_path, text)
    assert out.strip().upper().endswith("INFEASIBLE")

def test_max_variables_scaling():
    n = 200        # можно поднять до 300-400, если хватает времени
    k = n // 2
    # Составим задание в math-формате одной строкой - нагрузим парсер.
    obj = " + ".join(f"x{i}" for i in range(1, n + 1))
    lines = [f"max {obj}"]

    lines.append(f"{obj} <= {k}")
    for i in range(1, n + 1):
        lines.append(f"x{i} <= 1")
    lines.append("x >= 0")

    text = "\n".join(lines)

    sense, c, A, b, rels = ss._parse_math(text)
    status, z, x = ss.simplex(ss.to_standard(sense, c, A, b, rels))

    assert status == ss.Status.OPTIMAL
    assert z == pytest.approx(float(k), rel=1e-9, abs=1e-9)

    # Проверяем структуру решения: 0 <= xi <= 1, сумма k
    assert np.all(x >= -1e-9)
    assert np.all(x <= 1 + 1e-9)
    assert x.sum() == pytest.approx(float(k), abs=1e-6)
