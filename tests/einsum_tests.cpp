// Copyright © 2023 Apple Inc.

#include <set>

#include "doctest/doctest.h"
#include "mlx/einsum.h"
#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("einsum_path") {
  std::vector<EinsumPath> expected;
  expected.push_back({{1, 0}, {'j'}, "jk,ij->ik", {"kl", "ik"}, true});
  expected.push_back({{1, 0}, {'k'}, "ik,kl->il", {"il"}, true});
  auto x = einsum_path("ij,jk,kl", {ones({2, 2}), ones({2, 2}), ones({2, 2})});
  CHECK_EQ(x.size(), expected.size());
  for (int i = 0; i < x.size(); i++) {
    CHECK_EQ(x.at(i).args, expected.at(i).args);
    CHECK_EQ(x.at(i).remaining, expected.at(i).remaining);
    CHECK_EQ(x.at(i).einsum_str, expected.at(i).einsum_str);
    CHECK_EQ(x.at(i).removing, expected.at(i).removing);
    CHECK_EQ(x.at(i).can_dot, expected.at(i).can_dot);
  }
  expected.clear();
  expected.push_back(
      {{
           0,
       },
       {},
       "jki->ijk",
       {"ijk"},
       false});
  x = einsum_path("jki", {ones({2, 3, 4})});
  CHECK_EQ(x.size(), expected.size());
  for (int i = 0; i < x.size(); i++) {
    CHECK_EQ(x.at(i).args, expected.at(i).args);
    CHECK_EQ(x.at(i).remaining, expected.at(i).remaining);
    CHECK_EQ(x.at(i).einsum_str, expected.at(i).einsum_str);
    CHECK_EQ(x.at(i).removing, expected.at(i).removing);
    CHECK_EQ(x.at(i).can_dot, expected.at(i).can_dot);
  }
  expected.clear();
  expected.push_back({{1, 0}, {'i'}, "i,i->", {""}, false});
  x = einsum_path("i,i", {ones({2}), ones({1})});
  CHECK_EQ(x.size(), expected.size());
  for (int i = 0; i < x.size(); i++) {
    CHECK_EQ(x.at(i).args, expected.at(i).args);
    CHECK_EQ(x.at(i).remaining, expected.at(i).remaining);
    CHECK_EQ(x.at(i).einsum_str, expected.at(i).einsum_str);
    CHECK_EQ(x.at(i).removing, expected.at(i).removing);
    CHECK_EQ(x.at(i).can_dot, expected.at(i).can_dot);
  }
  expected.clear();
  expected.push_back({{1, 0}, {'j'}, "jk,ij->ik", {"ik"}, true});
  x = einsum_path("ij,jk", {ones({2, 2}), ones({2, 2})});
  CHECK_EQ(x.size(), expected.size());
  for (int i = 0; i < x.size(); i++) {
    CHECK_EQ(x.at(i).args, expected.at(i).args);
    CHECK_EQ(x.at(i).remaining, expected.at(i).remaining);
    CHECK_EQ(x.at(i).einsum_str, expected.at(i).einsum_str);
    CHECK_EQ(x.at(i).removing, expected.at(i).removing);
    CHECK_EQ(x.at(i).can_dot, expected.at(i).can_dot);
  }
  expected.clear();
  expected.push_back({{1, 0}, {'i', 'j'}, "jil,ijk->kl", {"kl"}, true});

  x = einsum_path("ijk,jil->kl", {ones({3, 4, 5}), ones({4, 3, 2})});
  CHECK_EQ(x.size(), expected.size());
  for (int i = 0; i < x.size(); i++) {
    CHECK_EQ(x.at(i).args, expected.at(i).args);
    CHECK_EQ(x.at(i).remaining, expected.at(i).remaining);
    CHECK_EQ(x.at(i).einsum_str, expected.at(i).einsum_str);
    CHECK_EQ(x.at(i).removing, expected.at(i).removing);
    CHECK_EQ(x.at(i).can_dot, expected.at(i).can_dot);
  }
  expected.clear();
  expected.push_back(
      {{3, 0}, {'k'}, "nlk,ijk->injl", {"ilm", "njm", "abc", "injl"}, true});
  expected.push_back(
      {{1, 0}, {'m'}, "njm,ilm->injl", {"abc", "injl", "injl"}, true});
  expected.push_back(
      {{2, 1}, {'n', 'i', 'l', 'j'}, "injl,injl->", {"abc", ""}, true});
  expected.push_back({{1, 0}, {'c', 'a', 'b'}, ",abc->", {""}, false});

  x = einsum_path(
      "ijk,ilm,njm,nlk,abc->",
      {ones({2, 4, 8}),
       ones({2, 4, 8}),
       ones({2, 4, 8}),
       ones({2, 4, 8}),
       ones({2, 4, 8})});
  CHECK_EQ(x.size(), expected.size());
  for (int i = 0; i < x.size(); i++) {
    CHECK_EQ(x.at(i).args, expected.at(i).args);
    CHECK_EQ(x.at(i).remaining, expected.at(i).remaining);
    CHECK_EQ(x.at(i).einsum_str, expected.at(i).einsum_str);
    CHECK_EQ(x.at(i).removing, expected.at(i).removing);
    CHECK_EQ(x.at(i).can_dot, expected.at(i).can_dot);
  }
  expected.clear();
  expected.push_back(
      {{2, 0}, {'a'}, "abcd,ea->bcde", {"fb", "gc", "hd", "bcde"}, true});
  expected.push_back(
      {{3, 0}, {'b'}, "bcde,fb->cdef", {"gc", "hd", "cdef"}, true});
  expected.push_back({{2, 0}, {'c'}, "cdef,gc->defg", {"hd", "defg"}, true});
  expected.push_back({{1, 0}, {'d'}, "defg,hd->efgh", {"efgh"}, true});
  x = einsum_path(
      "ea,fb,abcd,gc,hd->efgh",
      {
          ones({10, 10}),
          ones({10, 10}),
          ones({10, 10, 10, 10}),
          ones({10, 10}),
          ones({10, 10}),
      });
  CHECK_EQ(x.size(), expected.size());
  for (int i = 0; i < x.size(); i++) {
    CHECK_EQ(x.at(i).args, expected.at(i).args);
    CHECK_EQ(x.at(i).remaining, expected.at(i).remaining);
    CHECK_EQ(x.at(i).einsum_str, expected.at(i).einsum_str);
    CHECK_EQ(x.at(i).removing, expected.at(i).removing);
    CHECK_EQ(x.at(i).can_dot, expected.at(i).can_dot);
  }
};