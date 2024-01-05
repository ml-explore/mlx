from collections import Counter, OrderedDict
from typing import Any, Dict, Generator, List, Set, Tuple

import numpy as np


def _compute_dict_size(term: str, dim_dict: Dict[str, int]):
    size = 1
    for c in term:
        size *= dim_dict[c]
    return size


def _find_contraction(
    positions: Tuple[int], in_sets: List[Set[str]], out_set: Set[str]
):
    idx_contract: Set[str] = set()
    idx_remain = out_set.copy()
    remaining: List[Set[str]] = []
    for i, v in enumerate(in_sets):
        if i in positions:
            idx_contract.update(v)
        else:
            remaining.append(v)
            idx_remain.update(v)
    n_res = idx_remain.intersection(idx_contract)
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
    in_sets: List[Set[str]], out_set: Set[str], idx_dict: Dict[str, int], max_size: int
):
    f_results: List[Tuple[int, List[Tuple[int]], List[Set[int]]]] = [(0, [], in_sets)]
    for it in range(len(in_sets) - 1):
        it_results: List[Tuple[int, List[Tuple[int]], List[Set[int]]]] = []
        for curr in f_results:
            for con in _combinations(range(len(in_sets) - it)):
                cont = _find_contraction(con, curr[2], out_set)
                new_size = _compute_dict_size(cont[0], idx_dict)
                if new_size > max_size:
                    continue
                total_cost = curr[0] + _flop_count(
                    cont[3], len(cont[2]) > 0, 2, idx_dict
                )
                new_pos = curr[1] + [con]
                it_results.append((total_cost, new_pos, cont[1]))
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


# def _parse_possible_contraction(positions: Tuple[int], in_sets: List[Set[int]], out_set: Set[int], idx_dict: Dict[str, int], max_size: int, path_cost: int, naive_cost: int):
#     contract = _find_contraction(positions, in_sets, out_set)
#     idx_result, new_in_set, idx_removed, idx_contract = contract
#     new_size = _compute_dict_size(idx_result, idx_dict)
#     if new_size > max_size:
#         return None
#     old_sizes = (_compute_dict_size(in_sets[p], idx_dict) for p in positions)
#     removed_size = sum(old_sizes) - new_size
#     cost = _flop_count(idx_contract, idx_removed, len(positions), idx_dict)
#     sort = (-removed_size, cost)
#     if (path_cost + cost) > naive_cost:
#         return None
#     return (sort, positions, new_in_set)

# def _update_other_results(results: List[Tuple[tuple[int, int], Tuple[int], List[Set[int]]]], best: Tuple[tuple[int, int], Tuple[int], List[Set[int]]]):
#     best_con = best[1]
#     bx, by = best_con
#     mod_results = []
#     for cost, (x, y), con_sets in results:
#         if x in best_con or y in best_con:
#             continue
#         del con_sets[by-int(by > x) - int(by > y)]
#         del con_sets[bx-int(bx > x) - int(bx > y)]
#         con_sets.insert(-1, best[2][-1])
#         mod_con = int(x > bx) - int(x > by), y - int(y > bx) - int(y > by)
#         mod_results.append((cost, mod_con, con_sets))
#     return mod_results

# def _greedy(
#     in_sets: List[Set[int]], out_set: Set[int], idx_dict: Dict[str, int], max_size: int
# ):
#     if len(in_sets) == 1:
#         return [(0,)]
#     elif len(in_sets) == 2:
#         return [(0, 1,)]

#     contract = _find_contraction(range(len(in_sets)), in_sets, out_set)
#     idx_result, new_in_set, idx_removed, idx_contract = contract
#     naive_cost = _flop_count(idx_contract, idx_removed, len(in_sets), idx_dict)
#     loop = [x for x in _combinations(range(len(in_sets)))]
#     path_cost = 0
#     path = []
#     known_contractions: List[Tuple[tuple[int, int], Tuple[int], List[Set[int]]]] = []
#     for it in range(len(in_sets) - 1):
#         for ci in loop:
#             if in_sets[ci[0]].isdisjoint(in_sets[ci[1]]):
#                 continue
#             res = _parse_possible_contraction(ci, in_sets, out_set, idx_dict, max_size, path_cost, naive_cost)
#             if res is not None:
#                 known_contractions.append(res)
#         if len(known_contractions) == 0:
#             for pos in _combinations(range(len(in_sets))):
#                 res = _parse_possible_contraction(pos, in_sets, out_set, idx_dict, max_size, path_cost, naive_cost)
#                 if res is not None:
#                     known_contractions.append(res)
#             if len(known_contractions) == 0:
#                 path.append(tuple(range(len(in_sets))))
#                 break
#         best = min(known_contractions, key=lambda x: x[0])
#         known_contractions = _update_other_results(known_contractions, best)
#         in_sets = best[2]
#         new_tensor_pos = len(in_sets) - 1
#         loop = [(i, new_tensor_pos) for i in range(new_tensor_pos)]
#         path.append(best[1])
#         path_cost += best[0][1]
#     return path


def _can_dot(inputs: List[str], result: Set[int], idx_removed: Set[str]) -> bool:
    print(f"inputs {inputs} result {result} idx_removed {idx_removed}")
    if len(idx_removed) == 0:
        return False
    if len(inputs) != 2:
        return False
    in_left = Counter(inputs[0])
    in_right = Counter(inputs[1])
    if in_left == in_right:
        return True
    for c in in_left.keys():
        nl, nr = in_left.get(c, 0), in_right.get(c, 0)
        print(f"got {c} {nl} {nr}")
        if (nl > 1) or (nr > 1) or (nl + nr > 2):
            return False
        if nl + nr - 1 == int(c in result):
            return False
    for c in in_right.keys():
        nl, nr = in_left.get(c, 0), in_right.get(c, 0)
        if (nl > 1) or (nr > 1) or (nl + nr > 2):
            return False
        if nl + nr - 1 == int(c in result):
            return False
    if in_left.keys() == in_right.keys():
        return False

    rs = len(idx_removed)
    # GEMM & GEMV cases
    if inputs[0][-rs:] == inputs[1][:rs]:
        return True
    if inputs[0][:rs] == inputs[1][-rs:]:
        return True
    if inputs[0][-rs:] == inputs[1][-rs:]:
        return True
    if inputs[0][:rs] == inputs[1][:rs]:
        return True
    print("TESTING")

    keep_left = in_left.keys() - idx_removed
    keep_right = in_right.keys() - idx_removed
    if not keep_left or not keep_right:
        return False

    return True


def parse_path(path: str) -> Tuple[str, str]:
    in_scripts, out_scripts = "", ""
    if "->" in path:
        in_scripts, out_scripts = path.split("->")
    else:
        in_scripts = path
        tmp_check = OrderedDict()
        for c in sorted(in_scripts.replace(",", "")):
            tmp_check[c] = None
        for s in tmp_check.keys():
            if in_scripts.count(s) == 1:
                out_scripts += s
    return in_scripts, out_scripts


def _einsum_path(path: str, *operands, optimize="greedy"):
    in_script, out_script = parse_path(path)
    in_list = in_script.split(",")
    in_sets = [set(x) for x in in_list]
    out_set = set(out_script)
    indicies = set(in_script.replace(",", ""))

    dim_dict = {}
    broadcast_indicies = [set() for x in range(len(in_list))]
    for i, chars in enumerate(in_list):
        # TODO: check against operand shape
        assert len(chars) == operands[i].ndim, "operand shape mismatch"
        for ci, c in enumerate(chars):
            dim = operands[i].shape[ci]
            if dim == 1:
                broadcast_indicies[i].add(c)
            if c in dim_dict:
                if dim_dict[c] == 1:
                    dim_dict[c] = dim
                elif dim not in (1, dim_dict[c]):
                    raise ValueError(f"dimension mismatch {c}")
            else:
                dim_dict[c] = dim
    # print(dim_dict)
    # broadcast_indicies = [set(x) for x in broadcast_indicies]
    max_size = 0
    for term in in_list + [out_script]:
        max_size = max(max_size, _compute_dict_size(term, dim_dict))
    path: List = []
    if len(in_list) in [1, 2] or indicies == out_set:
        path = [tuple(range(len(in_list)))]
    elif optimize == "optimal":
        path = _optimal(in_sets, out_set, dim_dict, max_size)
    # print(f"path_size {len(path)}")
    # elif optimize == "greedy":
    #     path = _greedy(in_sets, out_set, dim_dict, max_size)
    contraction_list: List[Tuple[int], Set[str], str, List[str], bool] = []
    # scale_list = []
    for cn, ci in enumerate(path):
        ci = tuple(sorted(list(ci), reverse=True))
        cont = _find_contraction(ci, in_sets, out_set)
        in_sets = cont[1]
        do_blas = False
        # scale_list.append(len(idx_contract))
        bcast = set()
        tmp_inputs: List[str] = []
        for x in ci:
            tmp_inputs.append(in_list.pop(x))
            bcast.update(broadcast_indicies.pop(x))

        if not len(cont[2] & bcast):
            do_blas = _can_dot(tmp_inputs, cont[0], cont[2])
        idx_result = out_script
        if (cn - len(path)) != -1:
            sort_result = [(dim_dict[ind], ind) for ind in cont[0]]
            idx_result = "".join([x[1] for x in sorted(sort_result)])
        in_list.append(idx_result)
        new_bcast = bcast - cont[2]
        broadcast_indicies.append(new_bcast)
        einsum_str = ",".join(tmp_inputs) + "->" + idx_result
        contraction_list.append((ci, cont[2], einsum_str, in_list.copy(), do_blas))
    return (operands, contraction_list)
    # overall_contraction = in_script + "->" + out_script
    # header = ("scaling", "current", "remaining")

    # max_i = max(size_list)

    # path_print = "  Complete contraction:  %s\n" % overall_contraction
    # path_print += "  Largest intermediate:  %.3e elements\n" % max_i
    # path_print += "-" * 74 + "\n"
    # path_print += "%6s %24s %40s\n" % header
    # path_print += "-" * 74

    # for n, contraction in enumerate(contraction_list):
    #     inds, idx_rm, einsum_str, remaining, blas = contraction
    #     remaining_str = ",".join(remaining) + "->" + out_script
    #     path_run = (scale_list[n], einsum_str, remaining_str)
    #     path_print += "\n%4d    %24s %40s" % path_run
    return (path, path_print)


tests = [
    ("ij,jk,kl", np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2))),
    # ("i,i", np.ones(1), np.ones(1)),
    # ("ij,jk", np.ones((2,2)), np.ones((2,2))),
    # ("ijk,jil->kl", np.arange(60.).reshape(3,4,5), np.arange(24.).reshape(4,3,2)),
    # ("ijk,ilm,njm,nlk,abc->", np.ones(64).reshape(2,4,8), np.ones(64).reshape(2,4,8), np.ones(64).reshape(2,4,8), np.ones(64).reshape(2,4,8), np.ones(64).reshape(2,4,8)),
    # ("ea,fb,abcd,gc,hd->efgh",  np.random.rand(10, 10),  np.random.rand(10, 10), np.random.rand(10, 10, 10, 10), np.random.rand(10, 10),  np.random.rand(10, 10))
]


def log(xmx, xnp):
    print(f"\tsteps: {len(xmx[1])}")
    for o in zip(xmx[0], xnp[0]):
        np.testing.assert_allclose(*o)
    for i, o in enumerate(zip(xmx[1], xnp[1])):
        log = o[0]
        print(
            f"\t{log[0]} einsum: {log[2]:20s} removing: {log[1]} {log[3]} dot: {log[-1]}"
        )
        assert o[0] == o[1], f"contraction {i} mismatch\n${o[0]}\n${o[1]}"


def test_optimal(test):
    xmx = _einsum_path(*test, optimize="optimal")
    xnp = np.einsum_path(*test, optimize="optimal", einsum_call=True)
    log(xmx, xnp)


def test_greedy(test):
    xmx = _einsum_path(*test, optimize="greedy")
    xnp = np.einsum_path(*test, optimize="greedy", einsum_call=True)
    log(xmx, xnp)


for test in tests:
    print(f"testing optimal {test[0]}")
    test_optimal(test)
    # print(f"testing greedy {test[0]}")
    # test_greedy(test)
