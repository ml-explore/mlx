// Copyright © 2024 Apple Inc.

#include "doctest/doctest.h"
#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("test einsum path") {
  std::vector<std::vector<int>> expected = {{1, 2}, {0, 1}};
  auto path =
      einsum_path("ij,jk,kl", {ones({2, 2}), ones({2, 4}), ones({4, 2})}).first;
  CHECK_EQ(path, expected);

  expected = {{0}};
  path = einsum_path("jki", {ones({2, 3, 4})}).first;
  CHECK_EQ(path, expected);

  expected = {{0, 1}};
  path = einsum_path("i,i", {ones({2}), ones({1})}).first;
  CHECK_EQ(path, expected);

  expected = {{0, 1}};
  path = einsum_path("ij,jk", {ones({2, 2}), ones({2, 2})}).first;
  CHECK_EQ(path, expected);

  expected = {{0, 1}};
  path = einsum_path("ijk,jil->kl", {ones({3, 4, 5}), ones({4, 3, 2})}).first;
  CHECK_EQ(path, expected);

  expected = {{0, 3}, {1, 3}, {0, 2}, {0, 1}};
  path = einsum_path(
             "ijk,ilm,njm,nlk,abc->",
             {ones({2, 6, 8}),
              ones({2, 4, 5}),
              ones({3, 6, 5}),
              ones({3, 4, 8}),
              ones({9, 4, 7})})
             .first;
  CHECK_EQ(path, expected);

  expected = {{0, 2}, {0, 3}, {0, 2}, {0, 1}};
  path = einsum_path(
             "ea,fb,abcd,gc,hd->efgh",
             {ones({10, 10}),
              ones({10, 10}),
              ones({10, 10, 10, 10}),
              ones({10, 10}),
              ones({10, 10})})
             .first;
  CHECK_EQ(path, expected);
}

TEST_CASE("test einsum") {
  CHECK_THROWS(einsum("i,j", {array({1.0})}));
  CHECK_THROWS(einsum("ijk", {full({2, 2}, 2.0f)}));
  CHECK_THROWS(einsum("", {}));
  CHECK_THROWS(einsum("ij", {array({1, 2})}));
  CHECK_THROWS(einsum("", {array({1, 2})}));
  CHECK_THROWS(einsum("i,ij", {array({1, 2}), array({2, 3})}));
  CHECK_THROWS(einsum("i,i", {array({1, 2}), array({2, 3, 4})}));
  CHECK_THROWS(einsum("i->ii", {array({1, 2})}));
  CHECK_THROWS(einsum("12", {zeros({4, 4})}));
  CHECK_THROWS(einsum("ii->i", {zeros({3, 2})}));

  auto x = einsum("jki", {full({2, 3, 4}, 3.0f)});
  auto expected = full({4, 2, 3}, 3.0f);
  CHECK_EQ(allclose(x, expected).item<bool>(), true);

  x = einsum("ij,jk->ik", {full({2, 2}, 2.0f), full({2, 2}, 3.0f)});
  expected = array({12.0f, 12.0f, 12.0f, 12.0f}, {2, 2});
  CHECK_EQ(allclose(x, expected).item<bool>(), true);

  x = einsum("i,j->ij", {full({2}, 15.0f), full({4}, 20.0f)});
  expected = full({2, 4}, 300.0f);
  CHECK_EQ(allclose(x, expected).item<bool>(), true);
}
