"""Tests for the batched ADMM decoder."""

import unittest

import numpy as np
import torch
from scipy.optimize import minimize

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
    cons = []
    for mask in range(1 << d):
        if (mask.bit_count() % 2) != (1 - b) % 2:
            continue
        n = np.array([1.0 if (mask >> i) & 1 else -1.0 for i in range(d)])
        cons.append({"type": "ineq",
                     "fun": lambda x, n=n, m=mask: m.bit_count() - 1 - n @ x})
    res = minimize(lambda x: ((x - a) ** 2).sum(), np.clip(a, 0, 1),
                   bounds=[(0, 1)] * d, constraints=cons, method="trust-constr",
                   options={"maxiter": 2000, "gtol": 1e-12, "xtol": 1e-14})
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
        for d in (3, 4):
            mask = torch.ones(1, d, dtype=torch.bool)
            for b in (0, 1):
                for _ in range(8):
                    a = rng.uniform(-0.5, 1.5, d)
                    ref = reference_projection(a, b)
                    out = parity_polytope_projection(
                        torch.tensor([[a]], dtype=torch.float32),
                        torch.tensor([[b]]), mask).numpy()[0][0]
                    self.assertLess(
                        float(np.abs(out - ref).max()), 2e-2,
                        f"d={d} b={b} a={np.round(a, 3)} got "
                        f"{np.round(out, 3)} ref {np.round(ref, 3)}")


class TestAdmm(unittest.TestCase):
    def test_admm_matches_exact_lp_on_steane(self):
        h_x, h_z = steane_code()
        for sy in range(8):
            s = np.array([(sy >> j) & 1 for j in range(3)], np.int8)
            x_lp = solve_lp(h_z, s)
            x_ad = admm_solve_batch(
                h_z, s.reshape(1, -1), max_iter=800).numpy()[0]
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
    def test_weighted_pushes_low_weight_column(self):
        h = np.array([[1, 1]], dtype=np.int8)
        s = np.array([[1]], dtype=np.int8)
        x_plain = admm_solve_batch(h, s, max_iter=500).numpy()[0]
        w = np.array([0.2, 10.0], dtype=np.float32)
        x_w = admm_solve_batch(h, s, c_vec=w, max_iter=500).numpy()[0]
        self.assertLess(x_w[1], x_plain[1])

    def test_ldr_smoke(self):
        h_x, h_z = steane_code()
        for sy in range(8):
            s = np.array([(sy >> j) & 1 for j in range(3)], np.int8)
            x = admm_ldr_solve(h_z, s.reshape(1, -1), max_iter=300,
                               outer=3).numpy()[0]
            self.assertTrue(np.all((x >= 0) & (x <= 1)))


if __name__ == "__main__":
    unittest.main()
