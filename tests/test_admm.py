"""Tests for the batched ADMM decoder."""

import unittest

import numpy as np
import torch
from scipy.optimize import LinearConstraint, minimize

from qudec.admm import (
    AdmmOsdDecoder,
    admm_ldr_solve,
    admm_solve_batch,
    parity_polytope_projection,
)
from qudec.codes import gross_code, logicals, steane_code
from qudec.lp import solve_lp
from qudec.noise import sample_iid_errors


def reference_projection(a, b):
    """Project a onto the parity polytope via the explicit inequalities."""
    d = len(a)
    normals, bounds = [], []
    for mask in range(1 << d):
        if (mask.bit_count() % 2) != (1 - b) % 2:
            continue
        n = np.array([1.0 if (mask >> i) & 1 else -1.0 for i in range(d)])
        normals.append(n)
        bounds.append(mask.bit_count() - 1)
    constraints = LinearConstraint(np.array(normals), -np.inf, bounds)
    res = minimize(lambda x: 0.5 * ((x - a) ** 2).sum(), np.clip(a, 0, 1),
                   jac=lambda x: x - a, bounds=[(0, 1)] * d,
                   constraints=constraints, method="SLSQP",
                   options={"maxiter": 1000, "ftol": 1e-12})
    if not res.success:
        raise AssertionError(f"reference projection failed: {res.message}")
    return res.x


class TestParityProjection(unittest.TestCase):
    def test_known_cases(self):
        mask = torch.ones(1, 3, dtype=torch.bool)
        cases = [
            ([[0.9, 0.1, 0.1]], 1, [[0.9, 0.1, 0.1]]),
            ([[0.9, 0.8, 0.6]], 0, [[0.8, 0.7, 0.5]]),
            ([[0.0, 0.0, 0.0]], 1, [[1 / 3, 1 / 3, 1 / 3]]),
            ([[1.2, -0.3, 0.7]], 0, [[0.95, 0.0, 0.95]]),
            ([[0.75, 1.294, 1.051]], 0, [[0.385, 0.929, 0.686]]),
        ]
        for z, b, expected in cases:
            out = parity_polytope_projection(
                torch.tensor([z]), torch.tensor([[b]]), mask)
            self.assertTrue(
                torch.allclose(out, torch.tensor([expected]), atol=2e-3),
                f"z={z} b={b} got {out.tolist()} want {expected}")
        mask4 = torch.ones(1, 4, dtype=torch.bool)
        out = parity_polytope_projection(
            torch.tensor([[[0.9, 0.1, 0.1, 0.1]]]), torch.tensor([[0]]), mask4)
        self.assertTrue(torch.allclose(
            out, torch.tensor([[[0.75, 0.25, 0.25, 0.25]]]), atol=2e-3))

    def test_matches_reference(self):
        rng = np.random.default_rng(7)
        for d in (3, 4, 6, 8):
            mask = torch.ones(1, d, dtype=torch.bool)
            for b in (0, 1):
                for _ in range(8):
                    a = rng.uniform(-3.0, 4.0, d)
                    ref = reference_projection(a, b)
                    out = parity_polytope_projection(
                        torch.from_numpy(a.reshape(1, 1, d)).float(),
                        torch.tensor([[b]]), mask).numpy()[0][0]
                    self.assertLess(
                        float(np.abs(out - ref).max()), 3e-5,
                        f"d={d} b={b} a={np.round(a, 3)} got "
                        f"{np.round(out, 3)} ref {np.round(ref, 3)}")

    def test_high_weight_vertex_is_projected_despite_legacy_cap(self):
        z = torch.tensor([[[1., 1., 1., 1., 1., 1., 1., 0.]]])
        out = parity_polytope_projection(
            z, torch.tensor([[0]]), torch.ones(1, 8, dtype=torch.bool), max_r=4)
        expected = torch.tensor([[[.875] * 7 + [.125]]])
        self.assertTrue(torch.allclose(out, expected, atol=2e-6))

    def test_padding_and_degree_one(self):
        mask = torch.tensor([[True, False, False], [True, True, False]])
        z = torch.tensor([[[20., -100., 500.], [2., -2., 100.]]])
        out = parity_polytope_projection(z, torch.tensor([[0, 0]]), mask)
        self.assertTrue(torch.allclose(
            out, torch.tensor([[[0., 0., 0.], [0., 0., 0.]]]), atol=2e-6))
        odd = parity_polytope_projection(z, torch.tensor([[1, 1]]), mask)
        self.assertTrue(torch.allclose(
            odd, torch.tensor([[[1., 0., 0.], [1., 0., 0.]]]), atol=2e-6))

    def test_empty_checks(self):
        mask = torch.zeros(1, 0, dtype=torch.bool)
        z = torch.zeros(2, 1, 0)
        out = parity_polytope_projection(z, torch.zeros(2, 1), mask)
        self.assertEqual(out.shape, z.shape)
        with self.assertRaisesRegex(ValueError, "empty check"):
            parity_polytope_projection(z, torch.ones(2, 1), mask)
        with self.assertRaisesRegex(ValueError, "binary parities"):
            parity_polytope_projection(torch.zeros(1, 1, 1),
                                       torch.tensor([[0.9]]),
                                       torch.ones(1, 1, dtype=torch.bool))


class TestAdmm(unittest.TestCase):
    def test_admm_matches_exact_lp_on_steane(self):
        h_x, h_z = steane_code()
        for sy in range(8):
            s = np.array([(sy >> j) & 1 for j in range(3)], np.int8)
            x_lp = solve_lp(h_z, s)
            x_ad = admm_solve_batch(
                h_z, s.reshape(1, -1), max_iter=800, device="cpu").numpy()[0]
            # the LP can have multiple optima (integral vs fractional),
            # so compare optimal values, not solutions
            self.assertLess(
                abs(float(x_ad.sum() - x_lp.sum())), 0.02,
                f"syndrome {s}: admm obj {x_ad.sum():.4f} lp obj "
                f"{x_lp.sum():.4f}")

    def test_admm_decoder_close_to_mld_on_steane(self):
        h_x, h_z = steane_code()
        l_x, l_z = logicals(h_x, h_z)
        ad = AdmmOsdDecoder(h_x, h_z, l_x, l_z, osd_order=1, max_iter=1000)
        n = 7
        patterns = np.array(
            [[(i >> j) & 1 for j in range(n)] for i in range(1 << n)],
            dtype=np.int8)
        syn = (h_z @ patterns.T) % 2
        best = {}
        for i, e in enumerate(patterns):
            key = syn[:, i].tobytes()
            if key not in best or int(e.sum()) < best[key][0]:
                best[key] = (int(e.sum()), e)
        empty = np.zeros((1 << n, 3), np.int8)
        c_ad, _ = ad.decode_corrections(syn.T.astype(np.int8), empty)
        mld_fail = 0
        dec_fail = 0
        for i, e in enumerate(patterns):
            key = syn[:, i].tobytes()
            e_hat = best[key][1]
            if int(((l_z @ (e ^ e_hat)) % 2).item()):
                mld_fail += 1
            if int(((l_z @ (e ^ c_ad[i])) % 2).item()):
                dec_fail += 1
        self.assertLessEqual(dec_fail, mld_fail + 4)

    def test_admm_valid_on_gross(self):
        h_x, h_z = gross_code()
        l_x, l_z = logicals(h_x, h_z)
        ad = AdmmOsdDecoder(h_x, h_z, l_x, l_z, osd_order=1, max_iter=200)
        n = h_x.shape[1]
        e_x, e_z = sample_iid_errors(n, 0.05, 30, "depolarizing", seed=5)
        sx = (h_z @ e_x.T) % 2
        sz = (h_x @ e_z.T) % 2
        c_x, c_z = ad.decode_corrections(
            sx.T.astype(np.int8), sz.T.astype(np.int8))
        self.assertTrue(np.all((h_z @ c_x.T) % 2 == sx))
        self.assertTrue(np.all((h_x @ c_z.T) % 2 == sz))


class TestAdmmVariants(unittest.TestCase):
    def test_empty_batches_and_empty_variables(self):
        for solver in (admm_solve_batch, admm_ldr_solve):
            x = solver(np.ones((1, 2), np.int8), np.zeros((0, 1), np.int8),
                       device="cpu")
            self.assertEqual(x.shape, (0, 2))
            x = solver(np.zeros((1, 0), np.int8), np.zeros((2, 1), np.int8),
                       device="cpu")
            self.assertEqual(x.shape, (2, 0))

    def test_ldr_zero_and_signed_objectives(self):
        h = np.ones((1, 2), np.int8)
        for weights, expected in [(np.zeros(2), [.5, .5]),
                                   (np.array([-1., -2.]), [0., 1.])]:
            x = admm_ldr_solve(h, np.ones((1, 1), np.int8),
                               c_vec=weights, outer=1, device="cpu").numpy()
            np.testing.assert_allclose(x, [expected], atol=2e-5)

    def test_ldr_integrality_stop_also_requires_feasibility(self):
        x = admm_ldr_solve(np.ones((1, 3), np.int8),
                           np.ones((1, 1), np.int8), max_iter=1,
                           outer=2, device="cpu").numpy()
        self.assertGreater(float(x.sum()), 0.1)

    def test_isolated_columns_signed_and_zero_weights(self):
        h = np.array([[1, 0, 0]], dtype=np.int8)
        x = admm_solve_batch(h, np.array([[1]], np.int8),
                             c_vec=np.array([0., -2., 0.]), device="cpu").numpy()
        self.assertTrue(np.all(np.isfinite(x)))
        np.testing.assert_allclose(x, [[1., 1., 0.]], atol=2e-5)
        empty = admm_solve_batch(np.zeros((0, 3), np.int8),
                                 np.zeros((2, 0), np.int8),
                                 c_vec=np.array([1., -1., 0.]), device="cpu")
        np.testing.assert_array_equal(empty.numpy(), [[0., 1., 0.]] * 2)

    def test_batch_partition_does_not_change_solution(self):
        h, _ = steane_code()
        s = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 1]], np.int8)
        batched = admm_solve_batch(h, s, device="cpu").numpy()
        separate = np.concatenate([
            admm_solve_batch(h, row[None, :], device="cpu").numpy() for row in s])
        np.testing.assert_allclose(batched, separate, atol=1e-7)

    def test_quadratic_weights_are_not_rescaled_independently(self):
        # With g != 0 the original c and g define the same quadratic.
        # Normalizing c alone would turn this minimizer .5 into .25.
        h = np.array([[1, 1]], np.int8)
        x = admm_solve_batch(h, np.array([[1]], np.int8),
                             c_vec=np.array([-2., -2.]),
                             g_vec=np.array([4., 4.]), device="cpu").numpy()
        np.testing.assert_allclose(x, [[.5, .5]], atol=2e-5)

    def test_weighted_pushes_low_weight_column(self):
        h = np.array([[1, 1]], dtype=np.int8)
        s = np.array([[1]], dtype=np.int8)
        x_plain = admm_solve_batch(h, s, max_iter=500,
                                   device="cpu").numpy()[0]
        w = np.array([0.2, 10.0], dtype=np.float32)
        x_w = admm_solve_batch(h, s, c_vec=w, max_iter=500,
                               device="cpu").numpy()[0]
        self.assertLess(x_w[1], x_plain[1])

    def test_weighted_does_not_collapse(self):
        # large LLR weights used to clip every x to zero on the first
        # update, locking the iteration at the trivial fixed point
        h = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8)
        s = np.array([[1, 1]], dtype=np.int8)
        w = np.array([5.0, 4.6, 5.0], dtype=np.float32)
        x = admm_solve_batch(h, s, c_vec=w, max_iter=1000,
                               device="cpu").numpy()[0]
        self.assertGreater(float(x.sum()), 0.5)

    def test_ldr_smoke(self):
        h_x, h_z = steane_code()
        for sy in range(8):
            s = np.array([(sy >> j) & 1 for j in range(3)], np.int8)
            x = admm_ldr_solve(h_z, s.reshape(1, -1), max_iter=300,
                               outer=3, device="cpu").numpy()[0]
            self.assertTrue(np.all((x >= 0) & (x <= 1)))


if __name__ == "__main__":
    unittest.main()
