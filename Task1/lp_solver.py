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


# 1. Парсеры входа
_math_term = re.compile(r'([+\-]?\s*\d*(?:\.\d*)?)\s*x(\d+)')

def _parse_expr(expr: str) -> dict[int, float]:
    """Разобрать линейное выражение (x1, x2, …) -> dict{index: coeff}."""
    coeffs: dict[int, float] = {}
    for m in _math_term.finditer(expr):
        raw, idx = m.groups()
        raw = raw.replace(' ', '')
        coef = 1.0 if raw in ('', '+') else -1.0 if raw == '-' else float(raw)
        coeffs[int(idx) - 1] = coeffs.get(int(idx) - 1, 0.0) + coef
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
    for ln in lines[1:]:
        if re.fullmatch(r'x\s*>?=\s*0', ln.lower()):
            continue                       # x >= 0
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
    return sense, c, A, b, rels


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
    return sense, c, A, b, rels

def parse_lp(path: str):
    txt = open(path, 'r', encoding='utf-8').read()
    try:
        return _parse_math(txt)
    except Exception:
        return _parse_table(txt)

# 2. Перевод к каноническому виду
EPS = 1e-9

class StandardLP:
    def __init__(self, c, A, b, basis, vtypes, orig_min):
        self.c = c
        self.A = A
        self.b = b
        self.basis = basis
        self.vtypes = vtypes
        self.orig_min = orig_min


def to_standard(sense, c, A, b, rels):
    """
    Приводит задачу к равенствам A·z = b, z ≥ 0
    и формирует начальный базис для двухфазного симплекса.

    <= - +slack s (базис именно в своей строке)
    >= - –surplus s  +artificial a (в базис только a в своей строке)
    = - +artificial a (базис)
    """
    m, n = A.shape
    orig_min = sense == 'MIN'
    if orig_min:
        c = -c

    # Будем добавлять дополнительные столбцы и заполнять basis[i] для каждой строки i.
    vtypes: list[str] = ['orig'] * n
    extra_cols: list[np.ndarray] = []
    basis: list[int] = [-1] * m

    # Индекс следующего добавляемого столбца
    next_col = n
    for i in range(A.shape[0]):
        if b[i] < -EPS:
            A[i, :] *= -1
            b[i] *= -1
            if rels[i] == '<=' or rels[i] == '<':
                rels[i] = '>='
            elif rels[i] == '>=' or rels[i] == '>':
                rels[i] = '<='
            # для '=' знак не меняем

    for i, rel in enumerate(rels):
        if rel in ('<=', '<'):  # a·x ≤ b -- +s_i, s_i базис в строке i
            s = np.zeros(m); s[i] = 1.0
            extra_cols.append(s)
            vtypes.append('slack')
            basis[i] = next_col
            next_col += 1

        elif rel in ('>=', '>'):  # a·x ≥ b -- -s_i + a_i, базис a_i в строке i
            s = np.zeros(m); s[i] = -1.0
            extra_cols.append(s)
            vtypes.append('slack')
            next_col += 1

            a = np.zeros(m); a[i] = 1.0
            extra_cols.append(a)
            vtypes.append('art')
            basis[i] = next_col
            next_col += 1

        else:  # a·x = b -- +a_i, базис a_i
            a = np.zeros(m); a[i] = 1.0
            extra_cols.append(a)
            vtypes.append('art')
            basis[i] = next_col
            next_col += 1

    Afull = np.hstack([A] + [col.reshape(-1, 1) for col in extra_cols]) if extra_cols else A
    cfull = np.concatenate([c, np.zeros(Afull.shape[1] - len(c))])

    return StandardLP(cfull, Afull, b.copy(), basis, vtypes, orig_min)

# 3. Симплекс
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

    # -- Phase I --
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

    # удаляем добавленные x из базиса
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

    # -- Phase II --
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

    x = np.zeros(N2)
    for i, j in enumerate(basis2):
        x[j] = T2[i, -1]

    z = -T2[-1, -1]
    if std.orig_min:
        z = -z

    n_orig = std.vtypes.count('orig')
    return Status.OPTIMAL, z, x[:n_orig]


# 4. CLI
def main(argv: List[str]):
    if len(argv) != 2:
        print('Usage: python simplex_solver.py <problem.txt>')
        sys.exit(2)

    try:
        sense, c, A, b, rels = parse_lp(argv[1])
    except Exception as e:
        print('Parse error:', e)
        sys.exit(1)

    status, z, x = simplex(to_standard(sense, c, A, b, rels))

    if status == Status.OPTIMAL:
        print(f'F = {z:g}')
        print(' '.join(f'x{i} = {xi:g}' for i, xi in enumerate(x, 1)))
    elif status == Status.UNBOUNDED:
        print('UNBOUNDED')
    else:
        print('INFEASIBLE')


if __name__ == '__main__':
    main(sys.argv)
