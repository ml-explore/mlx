# Copyright © 2024 Apple Inc.

import unittest

import mlx.core as mx
import mlx_tests
import numpy as np


class TestEinsum(mlx_tests.MLXTestCase):

    def test_simple_path(self):
        a = mx.zeros((5, 5))
        path = mx.einsum_path("ii", a)
        self.assertEqual(path[0], [(0,)])

        path = mx.einsum_path("ij->i", a)
        self.assertEqual(path[0], [(0,)])

        path = mx.einsum_path("ii->i", a)
        self.assertEqual(path[0], [(0,)])

        a = mx.zeros((5, 8))
        b = mx.zeros((8, 3))
        path = mx.einsum_path("ij,jk", a, b)
        self.assertEqual(path[0], [(0, 1)])
        path = mx.einsum_path("ij,jk -> ijk", a, b)
        self.assertEqual(path[0], [(0, 1)])

        a = mx.zeros((5, 8))
        b = mx.zeros((8, 3))
        c = mx.zeros((3, 7))
        path = mx.einsum_path("ij,jk,kl", a, b, c)

        self.assertEqual(path[0], [(0, 1), (0, 1)])

        a = mx.zeros((5, 8))
        b = mx.zeros((8, 10))
        c = mx.zeros((10, 7))
        path = mx.einsum_path("ij,jk,kl", a, b, c)
        self.assertEqual(path[0], [(1, 2), (0, 1)])

    def test_longer_paths(self):
        chars = "abcdefghijklmopqABC"
        sizes = [2, 3, 4, 5, 4, 3, 2, 6, 5, 4, 3, 2, 5, 7, 4, 3, 2, 3, 4]
        dim_dict = {c: s for c, s in zip(chars, sizes)}
        cases = [
            "eb,cb,fb->cef",
            "dd,fb,be,cdb->cef",
            "dd,fb,be,cdb->cef",
            "bca,cdb,dbf,afc->",
            "dcc,fce,ea,dbf->ab",
            "dcc,fce,ea,dbf->ab",
        ]

        for case in cases:
            subscripts = case[: case.find("->")].split(",")
            inputs = []
            for s in subscripts:
                shape = [dim_dict[c] for c in s]
                inputs.append(np.ones(shape))
            np_path = np.einsum_path(case, *inputs)

            inputs = [mx.array(i) for i in inputs]
            mx_path = mx.einsum_path(case, *inputs)
            self.assertEqual(np_path[0][1:], mx_path[0])

    def test_simple_einsum(self):
        a = mx.arange(4 * 4).reshape(4, 4)
        a_mx = mx.einsum("ii->i", a)
        a_np = np.einsum("ii->i", a)
        self.assertTrue(np.array_equal(a_mx, a_np))

        a = mx.arange(2 * 2 * 2).reshape(2, 2, 2)
        a_mx = mx.einsum("iii->i", a)
        a_np = np.einsum("iii->i", a)
        self.assertTrue(np.array_equal(a_mx, a_np))

        a = mx.arange(2 * 2 * 3 * 3).reshape(2, 2, 3, 3)
        a_mx = mx.einsum("iijj->ij", a)
        a_np = np.einsum("iijj->ij", a)
        self.assertTrue(np.array_equal(a_mx, a_np))

        a = mx.arange(2 * 2 * 3 * 3).reshape(2, 3, 2, 3)
        a_mx = mx.einsum("ijij->ij", a)
        a_np = np.einsum("ijij->ij", a)
        self.assertTrue(np.array_equal(a_mx, a_np))

        # Test some simple reductions
        a = mx.arange(2 * 2).reshape(2, 2)
        a_mx = mx.einsum("ii", a)
        a_np = np.einsum("ii", a)
        self.assertTrue(np.array_equal(a_mx, a_np))

        a = mx.arange(2 * 4).reshape(2, 4)
        a_mx = mx.einsum("ij->", a)
        a_np = np.einsum("ij->", a)
        self.assertTrue(np.array_equal(a_mx, a_np))

        a = mx.arange(2 * 4).reshape(2, 4)
        a_mx = mx.einsum("ij->i", a)
        a_np = np.einsum("ij->i", a)
        self.assertTrue(np.array_equal(a_mx, a_np))

        a = mx.arange(2 * 4).reshape(2, 4)
        a_mx = mx.einsum("ij->j", a)
        a_np = np.einsum("ij->j", a)
        self.assertTrue(np.array_equal(a_mx, a_np))

        a = mx.arange(2 * 2 * 2).reshape(2, 2, 2)
        a_mx = mx.einsum("iii->", a)
        a_np = np.einsum("iii->", a)
        self.assertTrue(np.array_equal(a_mx, a_np))

        a = mx.arange(2 * 2 * 3 * 3).reshape(2, 3, 2, 3)
        a_mx = mx.einsum("ijij->j", a)
        a_np = np.einsum("ijij->j", a)
        self.assertTrue(np.array_equal(a_mx, a_np))

        # Test some simple transposes
        a = mx.arange(2 * 4).reshape(2, 4)
        a_mx = mx.einsum("ij", a)
        a_np = np.einsum("ij", a)
        self.assertTrue(np.array_equal(a_mx, a_np))

        a = mx.arange(2 * 4).reshape(2, 4)
        a_mx = mx.einsum("ij->ji", a)
        a_np = np.einsum("ij->ji", a)
        self.assertTrue(np.array_equal(a_mx, a_np))

        a = mx.arange(2 * 3 * 4).reshape(2, 3, 4)
        a_mx = mx.einsum("ijk->jki", a)
        a_np = np.einsum("ijk->jki", a)
        self.assertTrue(np.array_equal(a_mx, a_np))

    def test_two_input_einsum(self):

        # Matmul
        a = mx.full((2, 8), 1.0)
        b = mx.full((8, 2), 1.0)
        a_mx = mx.einsum("ik,kj", a, b)
        a_np = np.einsum("ik,kj", a, b)
        self.assertTrue(np.array_equal(a_mx, a_np))

        # Matmul + transpose
        a = mx.full((2, 8), 1.0)
        b = mx.full((8, 3), 1.0)
        a_mx = mx.einsum("ik,kj->ji", a, b)
        a_np = np.einsum("ik,kj->ji", a, b)
        self.assertTrue(np.array_equal(a_mx, a_np))

        # Inner product
        a = mx.full((4,), 1.0)
        b = mx.full((4,), 1.0)
        a_mx = mx.einsum("i,i", a, b)
        a_np = np.einsum("i,i", a, b)
        self.assertTrue(np.array_equal(a_mx, a_np))

        # Outer product
        a = mx.full((4,), 0.5)
        b = mx.full((6,), 2.0)
        a_mx = mx.einsum("i,j->ij", a, b)
        a_np = np.einsum("i,j->ij", a, b)
        self.assertTrue(np.array_equal(a_mx, a_np))

        # Elementwise multiply
        a = mx.full((2, 8), 1.0)
        b = mx.full((2, 8), 1.0)
        a_mx = mx.einsum("ij,ij->ij", a, b)
        a_np = np.einsum("ij,ij->ij", a, b)
        self.assertTrue(np.array_equal(a_mx, a_np))

        # Medley
        a = mx.full((2, 8, 3, 5), 1.0)
        b = mx.full((3, 7, 5, 2), 1.0)
        a_mx = mx.einsum("abcd,fgda->bfca", a, b)
        a_np = np.einsum("abcd,fgda->bfca", a, b)
        self.assertTrue(np.array_equal(a_mx, a_np))

    def test_sum_first(self):
        a = mx.full((5, 8), 1.0)
        b = mx.full((8, 2), 1.0)
        a_mx = mx.einsum("ab,bc->c", a, b)
        a_np = np.einsum("ab,bc->c", a, b)
        self.assertTrue(np.array_equal(a_mx, a_np))

    def test_broadcasting(self):
        a = mx.full((5, 1), 1.0)
        b = mx.full((8, 2), 1.0)
        a_mx = mx.einsum("ab,bc->c", a, b)
        return
        a_np = np.einsum("ab,bc->c", a, b)
        self.assertTrue(np.array_equal(a_mx, a_np))

        a = mx.random.uniform(shape=(5, 1, 3, 1))
        b = mx.random.uniform(shape=(1, 7, 1, 2))
        a_mx = mx.einsum("abcd,cdab->abcd", a, b)
        a_np = np.einsum("abcd,cdab->abcd", a, b)
        self.assertTrue(np.allclose(a_mx, a_np))

    def test_attention(self):
        q = mx.random.uniform(shape=(2, 3, 4, 5))
        k = mx.random.uniform(shape=(2, 3, 4, 5))
        v = mx.random.uniform(shape=(2, 3, 4, 5))

        s = mx.einsum("itjk,iujk->ijtu", q, k)
        out_mx = mx.einsum("ijtu,iujk->itjk", s, v)

        s = np.einsum("itjk,iujk->ijtu", q, k)
        out_np = np.einsum("ijtu,iujk->itjk", s, v)

        self.assertTrue(np.allclose(out_mx, out_np))

    def test_multi_input_einsum(self):
        a = mx.ones((3, 4, 5))
        out_mx = mx.einsum("ijk,lmk,ijf->lf", a, a, a)
        out_np = np.einsum("ijk,lmk,ijf->lf", a, a, a)
        self.assertTrue(np.allclose(out_mx, out_np))

    def test_opt_einsum_test_cases(self):
        # Test cases from
        # https://github.com/dgasmith/opt_einsum/blob/c826bb7df16f470a69f7bf90598fc27586209d11/opt_einsum/tests/test_contract.py#L11
        tests = [
            # Test hadamard-like products
            "a,ab,abc->abc",
            "a,b,ab->ab",
            # Test index-transformations
            "ea,fb,gc,hd,abcd->efgh",
            "ea,fb,abcd,gc,hd->efgh",
            "abcd,ea,fb,gc,hd->efgh",
            # Test complex contractions
            "acdf,jbje,gihb,hfac,gfac,gifabc,hfac",
            "cd,bdhe,aidb,hgca,gc,hgibcd,hgac",
            "abhe,hidj,jgba,hiab,gab",
            "bde,cdh,agdb,hica,ibd,hgicd,hiac",
            "chd,bde,agbc,hiad,hgc,hgi,hiad",
            "chd,bde,agbc,hiad,bdi,cgh,agdb",
            "bdhe,acad,hiab,agac,hibd",
            # Test collapse
            "ab,ab,c->",
            "ab,ab,c->c",
            "ab,ab,cd,cd->",
            "ab,ab,cd,cd->ac",
            "ab,ab,cd,cd->cd",
            "ab,ab,cd,cd,ef,ef->",
            # Test outer prodcuts
            "ab,cd,ef->abcdef",
            "ab,cd,ef->acdf",
            "ab,cd,de->abcde",
            "ab,cd,de->be",
            "ab,bcd,cd->abcd",
            "ab,bcd,cd->abd",
            # Random test cases that have previously failed
            "eb,cb,fb->cef",
            "dd,fb,be,cdb->cef",
            "bca,cdb,dbf,afc->",
            "dcc,fce,ea,dbf->ab",
            "fdf,cdd,ccd,afe->ae",
            "abcd,ad",
            "ed,fcd,ff,bcf->be",
            "baa,dcf,af,cde->be",
            "bd,db,eac->ace",
            "fff,fae,bef,def->abd",
            "efc,dbc,acf,fd->abe",
            # Inner products
            "ab,ab",
            "ab,ba",
            "abc,abc",
            "abc,bac",
            "abc,cba",
            # GEMM test cases
            "ab,bc",
            "ab,cb",
            "ba,bc",
            "ba,cb",
            "abcd,cd",
            "abcd,ab",
            "abcd,cdef",
            "abcd,cdef->feba",
            "abcd,efdc",
            # Inner then dot
            "aab,bc->ac",
            "ab,bcc->ac",
            "aab,bcc->ac",
            "baa,bcc->ac",
            "aab,ccb->ac",
            # Randomly build test caes
            "aab,fa,df,ecc->bde",
            "ecb,fef,bad,ed->ac",
            "bcf,bbb,fbf,fc->",
            "bb,ff,be->e",
            "bcb,bb,fc,fff->",
            "fbb,dfd,fc,fc->",
            "afd,ba,cc,dc->bf",
            "adb,bc,fa,cfc->d",
            "bbd,bda,fc,db->acf",
            "dba,ead,cad->bce",
            "aef,fbc,dca->bde",
        ]

        size_dict = dict(zip("abcdefghij", [2, 3, 4, 5, 2, 3, 4, 5, 2, 3]))

        def inputs_for_case(test_case):
            inputs = test_case.split("->")[0].split(",")
            return [
                mx.random.uniform(shape=tuple(size_dict[c] for c in inp))
                for inp in inputs
            ]

        for test_case in tests:
            inputs = inputs_for_case(test_case)
            np_out = np.einsum(test_case, *inputs)
            mx_out = mx.einsum(test_case, *inputs)
            self.assertTrue(np.allclose(mx_out, np_out, rtol=1e-4, atol=1e-4))

    def test_ellipses(self):
        size_dict = dict(zip("abcdefghij", [2, 3, 4, 5, 2, 3, 4, 5, 2, 3]))

        def inputs_for_case(test_case):
            inputs = test_case.split("->")[0].split(",")
            return [
                mx.random.uniform(shape=tuple(size_dict[c] for c in inp))
                for inp in inputs
            ]

        tests = [
            ("abc->ab", "...c->..."),
            ("abcd->ad", "a...d->..."),
            ("abij,abgj->abig", "...ij,...gj->...ig"),
            ("abij,abgj->abig", "...ij,...gj->..."),
            ("abhh->abh", "...hh->...h"),
            ("abhh->abh", "...hh->...h"),
            ("bch,abcj->abchj", "...h,...j->...hj"),
        ]
        for test_case in tests:
            inputs = inputs_for_case(test_case[0])
            np_out = np.einsum(test_case[1], *inputs)
            mx_out = mx.einsum(test_case[1], *inputs)
            self.assertTrue(np.allclose(mx_out, np_out, rtol=1e-4, atol=1e-4))

        error_tests = [
            ("abc,abc->ab", "a...b...c,a...b...c->abc"),
        ]
        for test_case in error_tests:
            inputs = inputs_for_case(test_case[0])
            with self.assertRaises(ValueError):
                mx.einsum(test_case[1], *inputs)


if __name__ == "__main__":
    unittest.main()
