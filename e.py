from typing import Any, Dict, Generator, List, Set, Tuple

import numpy as np


def _compute_dict_size(term: str, dim_dict: Dict[str, int]):
    size = 1
    for c in term:
        size *= dim_dict[c]
    return size


def _find_contraction(
    positions: Tuple[int], in_sets: List[Set[int]], out_set: Set[int]
):
    idx_contract: Set[int] = set()
    idx_remain = out_set.copy()
    remaining: List[Set[int]] = []
    for i, v in enumerate(in_sets):
        if i in positions:
            idx_contract |= v
        else:
            remaining.append(v)
            idx_remain |= v
    n_res = idx_remain & idx_contract
    idx_removed = idx_contract - n_res
    remaining.append(n_res)
    return (n_res, remaining, idx_removed, idx_contract)


def _flop_count(
    idx_contraction: str, inner: bool, num_terms: int, size_dict: Dict[str, int]
):
    overall = _compute_dict_size(idx_contraction, size_dict)
    op_factor = max(1, num_terms - 1)
    if inner:
        op_factor += 1
    return overall * op_factor


def _combinations(list: List[int]) -> Generator[Tuple[int], Any, None]:
    if len(list) == 0:
        return
    for i in range(len(list)):
        t = []
        t.append(list[i])
        for i in range(i + 1, len(list)):
            t.append(list[i])
            yield tuple(t)
            t.pop()


def _optimal(
    in_sets: List[Set[int]], out_set: Set[int], idx_dict: Dict[str, int], max_size: int
):
    f_results: List[Tuple[int, List[Tuple[int]], List[Set[int]]]] = [(0, [], in_sets)]
    for it in range(len(in_sets) - 1):
        it_results: List[Tuple[int, List[Tuple[int]], List[Set[int]]]] = []
        for curr in f_results:
            cost, positions, remaining = curr
            for con in _combinations(range(len(in_sets) - it)):
                cont = _find_contraction(con, remaining, out_set)
                new_res, new_in_set, idx_removed, idx_contract = cont
                new_size = _compute_dict_size(new_res, idx_dict)
                if new_size > max_size:
                    continue
                total_cost = cost + _flop_count(
                    idx_contract, idx_removed, len(con), idx_dict
                )
                new_pos = positions + [con]
                it_results.append((total_cost, new_pos, new_in_set))
        if it_results:
            f_results = it_results
        else:
            path = min(f_results, key=lambda x: x[0])[1]
            path += [tuple(range(len(in_sets) - it))]
            return path
    if len(f_results) == 0:
        return [tuple(range(len(in_sets)))]
    path = min(f_results, key=lambda x: x[0])[1]
    return path


def _greedy(
    in_sets: List[Set[int]], out_set: Set[int], idx_dict: Dict[str, int], max_size: int
):
    raise NotImplementedError


def _can_dot(inputs: List[int], result: Set[int], idx_removed: Set[int]) -> bool:
    if len(idx_removed) == 0:
        return False
    if len(inputs) != 2:
        return False
    in_left, in_right = inputs
    for c in set(in_left + in_right):
        nl, nr = in_left.count(c), in_right.count(c)
        if (nl > 1) or (nr > 1) or (nl + nr > 2):
            return False
        if nl + nr - 1 == int(c in result):
            return False
    s_left = set(in_left)
    s_right = set(in_right)
    keep_left = s_left - idx_removed
    keep_right = s_right - idx_removed
    rs = len(idx_removed)
    if in_left == in_right:
        return True
    if s_left == s_right:
        return False
    # GEMM & GEMV cases

    if in_left[-rs:] == in_right[:rs]:
        return True
    if in_left[:rs] == in_right[-rs:]:
        return True
    if in_left[-rs:] == in_right[-rs:]:
        return True
    if in_left[:rs] == in_right[:rs]:
        return True

    if not keep_left or not keep_right:
        return False

    return True


def parse_path(path: str) -> Tuple[str, str]:
    in_scripts, out_scripts = "", ""
    if "->" in path:
        in_scripts, out_scripts = path.split("->")
    else:
        in_scripts = path
        tmp_scripts: str = in_scripts.replace(",", "")
        for s in sorted(set(tmp_scripts)):
            if tmp_scripts.count(s) == 1:
                out_scripts += s
    return in_scripts, out_scripts


def _einsum_path(path: str, *operands, ptype="greedy"):
    in_script, out_script = parse_path(path)
    in_list = in_script.split(",")
    in_sets = [set(x) for x in in_list]
    out_set = set(out_script)
    indicies = set(in_script.replace(",", ""))

    dim_dict = {}
    broadcast_indicies = [[] for x in range(len(in_list))]
    for i, chars in enumerate(in_list):
        # TODO: check against operand shape
        assert len(chars) == operands[i].ndim, "operand shape mismatch"
        for ci, c in enumerate(chars):
            dim = operands[i].shape[ci]
            if dim == 1:
                broadcast_indicies[i].append(c)
            if c in dim_dict:
                if dim_dict[c] == 1:
                    dim_dict[c] = dim
                elif dim not in (1, dim_dict[c]):
                    raise ValueError(f"dimension mismatch {c}")
            else:
                dim_dict[c] = dim
    # print(dim_dict)
    broadcast_indicies = [set(x) for x in broadcast_indicies]
    size_list = [_compute_dict_size(term, dim_dict) for term in in_list + [out_script]]
    max_size = max(size_list)
    inner_product = (sum(len(x) for x in in_sets) - len(indicies)) > 0
    naive_cost = _flop_count(indicies, inner_product, len(in_list), dim_dict)
    path = []
    if len(in_list) in [1, 2] or indicies == out_set:
        path = [tuple(range(len(in_list)))]
    elif ptype == "optimal":
        path = _optimal(in_sets, out_set, dim_dict, max_size)
    elif ptype == "greedy":
        path = _greedy(in_sets, out_set, dim_dict, max_size)

    contraction_list = []
    scale_list = []
    for cn, ci in enumerate(path):
        ci = tuple(sorted(list(ci), reverse=True))
        contract = _find_contraction(ci, in_sets, out_set)
        out_inds, in_sets, idx_removed, idx_contract = contract
        do_blas = False
        scale_list.append(len(idx_contract))
        bcast = set()
        tmp_inputs: List[str] = []
        for x in ci:
            tmp_inputs.append(in_list.pop(x))
            bcast |= broadcast_indicies.pop(x)
        new_bcast = bcast - idx_removed
        if not len(idx_removed & bcast):
            do_blas = _can_dot(tmp_inputs, out_inds, idx_removed)
        idx_result = out_script

        if (cn - len(path)) != -1:
            sort_result = [(dim_dict[ind], ind) for ind in out_inds]
            idx_result = "".join([x[1] for x in sorted(sort_result)])
        in_list.append(idx_result)
        broadcast_indicies.append(new_bcast)
        einsum_str = ",".join(tmp_inputs) + "->" + idx_result
        contraction_list.append((ci, idx_removed, einsum_str, in_list.copy(), do_blas))

    overall_contraction = in_script + "->" + out_script
    header = ("scaling", "current", "remaining")

    max_i = max(size_list)

    path_print = "  Complete contraction:  %s\n" % overall_contraction
    path_print += "  Largest intermediate:  %.3e elements\n" % max_i
    path_print += "-" * 74 + "\n"
    path_print += "%6s %24s %40s\n" % header
    path_print += "-" * 74

    for n, contraction in enumerate(contraction_list):
        inds, idx_rm, einsum_str, remaining, blas = contraction
        remaining_str = ",".join(remaining) + "->" + out_script
        path_run = (scale_list[n], einsum_str, remaining_str)
        path_print += "\n%4d    %24s %40s" % path_run
    return (path, path_print)


# print([x for x in _combinations("ABCD")])
# print(_optimal([set('abd'), set('ac'), set('bdc')], set(), {'a': 1, 'b': 2, 'c': 3, 'd': 4}, 5000))
# from numpy.core.einsumfunc import _optimal_path
# print(_optimal_path([set('abd'), set('ac'), set('bdc')], set(), {'a': 1, 'b': 2, 'c': 3, 'd': 4}, 5000))
# print(parse_path("ij,jk,kl"))
# from numpy.core.einsumfunc import _parse_einsum_input
# print(_parse_einsum_input(["ij,jk,kl", np.ones((2,2)), np.ones((2,2)), np.ones((2,2))]))
# exit()
xmx = _einsum_path(
    "ij,jk,kl", np.ones((2, 2)), np.ones((2, 5)), np.ones((5, 2)), ptype="optimal"
)
# xnp = np.einsum_path("ij,jk,kl", np.ones((2,2)), np.ones((2,5)), np.ones((5,2)), optimize="optimal")
print(xmx[0])
# print(xnp[0])
print("-----------------")
print(xmx[1])
# print("-----------------")
# print("numpy")
# print(xnp[1])
