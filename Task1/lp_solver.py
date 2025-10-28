"""
Приводит к каноническому виду, запускает двухфазный симплекс
с правилом Бланда (защита от циклов).
Определяет три статуса:
- OPTIMAL - выводит  F = <значение>  и  x1 = … x2 = … …
- UNBOUNDED - целевая функция неограничена
- INFEASIBLE - область допустимых решений пуста

Запуск:
    python lp_solver.py problem.txt
"""
from __future__ import annotations

import re
import sys
from typing import List, Tuple
import numpy as np

# Парсеры входа

_math_term = re.compile(r'([+\-]?\s*\d*(?:\.\d*)?)\s*x(\d+)', re.IGNORECASE)

def _parse_expr(expr: str) -> dict[int, float]:
    coeffs: dict[int, float] = {}
    for m in _math_term.finditer(expr):
        raw, idx = m.groups()
        raw = raw.replace(' ', '')
        coef = 1.0 if raw in ('', '+') else -1.0 if raw == '-' else float(raw)
        j = int(idx) - 1
        coeffs[j] = coeffs.get(j, 0.0) + coef
    if not coeffs:
        raise ValueError('empty expression')
    return coeffs


def _parse_math(text: str):
    lines = [
        l.split('#', 1)[0].strip()
        for l in text.splitlines()
        if l.split('#', 1)[0].strip()
    ]
    if not lines:
        raise ValueError('empty file')

    head = lines[0].split(None, 1)
    if head[0].lower() not in ('max', 'min'):
        raise ValueError('not math format')
    sense = head[0].upper()
    if len(head) == 1:
        raise ValueError('objective missing')
    obj = _parse_expr(head[1])

    constr: List[Tuple[dict[int, float], str, float]] = []
    free_set: Set[int] = set()
    all_free_flag = False  # встретилась строка "x < 0"

    # шаблоны для свободных переменных
    rx_free_single = re.compile(r'^x(\d+)\s+(free|свободная|unrestricted|без_ограничений)$', re.IGNORECASE)
    rx_free_list   = re.compile(r'^free:\s*(.*)$', re.IGNORECASE)
    rx_all_ge0     = re.compile(r'^x\s*>?=\s*0$', re.IGNORECASE)
    rx_all_lt0     = re.compile(r'^x\s*<\s*0$', re.IGNORECASE)
    rx_one_lt0     = re.compile(r'^x(\d+)\s*<\s*0$', re.IGNORECASE)

    for ln in lines[1:]:
        low = ln.lower()

        if rx_all_ge0.fullmatch(ln):
            continue

        if rx_all_lt0.fullmatch(ln):
            all_free_flag = True
            continue

        m_one_lt0 = rx_one_lt0.fullmatch(ln)
        if m_one_lt0:
            j = int(m_one_lt0.group(1)) - 1
            free_set.add(j)
            continue

        m_free_single = rx_free_single.fullmatch(ln)
        if m_free_single:
            j = int(m_free_single.group(1)) - 1
            free_set.add(j)
            continue

        m_free_list = rx_free_list.fullmatch(ln)
        if m_free_list:
            for tok in m_free_list.group(1).split():
                if tok.isdigit():
                    free_set.add(int(tok) - 1)
            continue

        m_rel = re.search(r'(<=|>=|=)', ln)
        if not m_rel:
            raise ValueError(f'cannot parse line {ln!r}')
        rel = m_rel.group(1)
        left, right = ln.split(rel, 1)
        coeffs = _parse_expr(left)
        rhs = float(right.strip())
        constr.append((coeffs, rel, rhs))

    if not constr:
        raise ValueError('no constraints')

    n = max(max(d) for d, _, _ in constr + [(obj, '', 0)]) + 1
    m = len(constr)

    if all_free_flag:
        free_set = set(range(n))
    else:
        free_set = {j for j in free_set if 0 <= j < n}

    c = np.zeros(n)
    for j, v in obj.items():
        c[j] = v
    A = np.zeros((m, n))
    b = np.zeros(m)
    rels: List[str] = []
    for i, (d, rel, rhs) in enumerate(constr):
        for j, v in d.items():
            A[i, j] = v
        b[i] = rhs
        rels.append(rel)

    return sense, c, A, b, rels, free_set


def _parse_table(text: str):
    lines = [
        l.split('#', 1)[0].strip()
        for l in text.splitlines()
        if l.split('#', 1)[0].strip()
    ]
    if not lines:
        raise ValueError('empty file')
    sense = lines[0].upper()
    m, n = map(int, lines[1].split())
    c = np.fromstring(lines[2], sep=' ', dtype=float)
    if c.size != n:
        raise ValueError('objective length mismatch')

    A = np.empty((m, n))
    b = np.empty(m)
    rels: List[str] = []
    for i in range(m):
        parts = lines[3 + i].split()
        A[i] = list(map(float, parts[:n]))
        rels.append(parts[n])
        b[i] = float(parts[n + 1])

    free_set: Set[int] = set()
    all_free_flag = False

    rx_free_list   = re.compile(r'^FREE:\s*(.*)$', re.IGNORECASE)
    rx_all_lt0     = re.compile(r'^x\s*<\s*0$', re.IGNORECASE)
    rx_one_lt0     = re.compile(r'^x(\d+)\s*<\s*0$', re.IGNORECASE)

    for ln in lines[3 + m:]:
        if rx_all_lt0.fullmatch(ln):
            all_free_flag = True
            continue
        m_one = rx_one_lt0.fullmatch(ln)
        if m_one:
            free_set.add(int(m_one.group(1)) - 1)
            continue
        m_free = rx_free_list.fullmatch(ln)
        if m_free:
            for tok in m_free.group(1).split():
                if tok.isdigit():
                    free_set.add(int(tok) - 1)

    if all_free_flag:
        free_set = set(range(n))
    else:
        free_set = {j for j in free_set if 0 <= j < n}

    return sense, c, A, b, rels, free_set


def parse_lp(path: str):
    txt = open(path, 'r', encoding='utf-8').read()
    try:
        return _parse_math(txt)
    except Exception:
        return _parse_table(txt)


# Приведение к стандартной форме

EPS = 1e-9

class StandardLP:
    def __init__(
        self,
        c: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        basis: List[int],
        vtypes: List[str],
        orig_min: bool,
        n_orig: int,
        recon: List[List[tuple[int, float]]],  # для каждого исходного x_j: [(col, weight), ...]
    ):
        self.c = c
        self.A = A
        self.b = b
        self.basis = basis
        self.vtypes = vtypes
        self.orig_min = orig_min
        self.n_orig = n_orig
        self.recon = recon


def _apply_free_splitting(
    c: np.ndarray, A: np.ndarray, free_set: Set[int]
) -> tuple[np.ndarray, np.ndarray, List[str], List[List[tuple[int, float]]]]:
    m, n = A.shape
    vtypes: List[str] = []
    recon: List[List[tuple[int, float]]] = [[] for _ in range(n)]

    cols: List[np.ndarray] = []
    c_new: List[float] = []

    next_col = 0
    for j in range(n):
        col = A[:, j]
        if j in free_set:
            # xj = xj+ - xj-
            cols.append(col.copy())
            c_new.append(c[j])
            vtypes.append('orig')
            recon[j].append((next_col, +1.0))
            next_col += 1

            cols.append((-col).copy())
            c_new.append(-c[j])
            vtypes.append('orig')
            recon[j].append((next_col, -1.0))
            next_col += 1
        else:
            cols.append(col.copy())
            c_new.append(c[j])
            vtypes.append('orig')
            recon[j].append((next_col, +1.0))
            next_col += 1

    A2 = np.column_stack(cols) if cols else A.copy()
    c2 = np.array(c_new, dtype=float)
    return A2, c2, vtypes, recon


def to_standard(sense, c, A, b, rels, free_set: Set[int]):
    m, n = A.shape
    orig_min = (sense == 'MIN')
    if orig_min:
        c = -c

    A1, c1, vtypes, recon = _apply_free_splitting(c, A, free_set)
    m, n1 = A1.shape

    rels = list(rels)
    A_work = A1.copy()
    b_work = b.copy()
    for i in range(m):
        if b_work[i] < -EPS:
            A_work[i, :] *= -1
            b_work[i] *= -1
            if rels[i] in ('<=', '<'):
                rels[i] = '>='
            elif rels[i] in ('>=', '>'):
                rels[i] = '<='
            # = оставляем как есть

    extra_cols: list[np.ndarray] = []
    basis: list[int] = [-1] * m
    next_col = n1

    for i, rel in enumerate(rels):
        if rel in ('<=', '<'):
            s = np.zeros(m); s[i] = 1.0
            extra_cols.append(s)
            vtypes.append('slack')
            basis[i] = next_col
            next_col += 1

        elif rel in ('>=', '>'):
            s = np.zeros(m); s[i] = -1.0
            extra_cols.append(s)
            vtypes.append('slack')
            next_col += 1

            a = np.zeros(m); a[i] = 1.0
            extra_cols.append(a)
            vtypes.append('art')
            basis[i] = next_col
            next_col += 1

        else:  # =
            a = np.zeros(m); a[i] = 1.0
            extra_cols.append(a)
            vtypes.append('art')
            basis[i] = next_col
            next_col += 1

    Afull = np.hstack([A_work] + [col.reshape(-1, 1) for col in extra_cols]) if extra_cols else A_work
    cfull = np.concatenate([c1, np.zeros(Afull.shape[1] - len(c1))])

    return StandardLP(
        c=cfull,
        A=Afull,
        b=b_work.copy(),
        basis=basis,
        vtypes=vtypes,
        orig_min=orig_min,
        n_orig=A.shape[1],
        recon=recon,
    )


# Симплекс

def pivot(T, basis, row, col):
    piv = T[row, col]
    T[row] /= piv
    for r in range(T.shape[0]):
        if r != row:
            T[r] -= T[r, col] * T[row]
    basis[row] = col


def argmin_ratio(col, rhs):
    idx = [i for i, a in enumerate(col) if a > EPS]
    if not idx:
        return None
    ratios = rhs[idx] / col[idx]
    min_val = ratios.min()
    for i in idx:
        if abs(rhs[i] / col[i] - min_val) <= 1e-12:
            return i
    return None


class Status:
    OPTIMAL = 'OPTIMAL'
    UNBOUNDED = 'UNBOUNDED'
    INFEASIBLE = 'INFEASIBLE'


def _simplex_core(T, basis):
    m = T.shape[0] - 1
    N = T.shape[1] - 1
    while True:
        rc = T[-1, :N]
        cand = np.where(rc > EPS)[0]
        if cand.size == 0:
            return Status.OPTIMAL
        col = int(cand[0])
        row = argmin_ratio(T[:m, col], T[:m, -1])
        if row is None:
            return Status.UNBOUNDED
        pivot(T, basis, row, col)


def simplex(std: StandardLP):
    m, N = std.A.shape

    # Фаза 1
    T = np.zeros((m + 1, N + 1))
    T[:m, :N] = std.A
    T[:m, -1] = std.b
    T[-1, :N] = [-1 if t == 'art' else 0 for t in std.vtypes]

    basis = std.basis.copy()
    for i, j in enumerate(basis):
        if std.vtypes[j] == 'art':
            T[-1] += T[i]

    st = _simplex_core(T, basis)
    if st == Status.UNBOUNDED:
        return st, np.nan, np.empty(0)
    if abs(T[-1, -1]) > 1e-7:
        return Status.INFEASIBLE, np.nan, np.empty(0)

    for r, var in enumerate(basis):
        if std.vtypes[var] == 'art':
            for c in range(N):
                if std.vtypes[c] != 'art' and abs(T[r, c]) > 1e-9:
                    pivot(T, basis, r, c)
                    break

    keep = [j for j, t in enumerate(std.vtypes) if t != 'art']
    A2 = T[:m, keep]
    rhs = T[:m, -1]
    c2 = std.c[keep]
    N2 = len(keep)

    # Фаза 2
    T2 = np.zeros((m + 1, N2 + 1))
    T2[:m, :N2] = A2
    T2[:m, -1] = rhs
    T2[-1, :N2] = c2

    basis2 = [keep.index(j) for j in basis if j in keep]
    for i, j in enumerate(basis2):
        T2[-1] -= T2[-1, j] * T2[i]

    st = _simplex_core(T2, basis2)
    if st != Status.OPTIMAL:
        return st, np.nan, np.empty(0)

    x_keep = np.zeros(N2)
    for i, j in enumerate(basis2):
        x_keep[j] = T2[i, -1]

    keep_pos: Dict[int, int] = {orig_col: pos for pos, orig_col in enumerate(keep)}

    x_orig = np.zeros(std.n_orig)
    for j in range(std.n_orig):
        val = 0.0
        for col_idx, weight in std.recon[j]:
            pos = keep_pos.get(col_idx)
            if pos is not None:
                val += weight * x_keep[pos]
        x_orig[j] = val

    z = -T2[-1, -1]
    if std.orig_min:
        z = -z

    return Status.OPTIMAL, z, x_orig


# 4

def main(argv: List[str]):
    if len(argv) != 2:
        print('Usage: python lp_solver.py <problem.txt>')
        sys.exit(2)

    try:
        sense, c, A, b, rels, free_set = parse_lp(argv[1])
    except Exception as e:
        print('Parse error:', e)
        sys.exit(1)

    status, z, x = simplex(to_standard(sense, c, A, b, rels, free_set))

    if status == Status.OPTIMAL:
        print(f'F = {z:g}')
        print(' '.join(f'x{i} = {xi:g}' for i, xi in enumerate(x, 1)))
    elif status == Status.UNBOUNDED:
        print('UNBOUNDED')
    else:
        print('INFEASIBLE')


if __name__ == '__main__':
    main(sys.argv)

