"""Tests for code constructions and GF(2) linear algebra."""

import unittest

import numpy as np

from qudec.codes import (
    code_info,
    gf2_rank,
    gross_code,
    logicals,
    medium_code,
    repetition_code,
    steane_code,
)


class TestConstructions(unittest.TestCase):
    def test_steane(self):
        h_x, h_z = steane_code()
        self.assertEqual(h_x.shape, (3, 7))
        self.assertEqual(code_info(h_x, h_z), 1)
        self.assertTrue(np.all((h_x @ h_z.T) % 2 == 0))

    def test_repetition(self):
        h_x, h_z = repetition_code(5)
        self.assertEqual(h_x.shape, (0, 5))
        self.assertEqual(h_z.shape, (4, 5))
        self.assertEqual(code_info(h_x, h_z), 1)
        self.assertTrue(np.all((h_x @ h_z.T) % 2 == 0))

    def test_gross_code(self):
        h_x, h_z = gross_code()
        self.assertEqual(h_x.shape, (72, 144))
        self.assertEqual(code_info(h_x, h_z), 12)
        self.assertTrue(np.all((h_x @ h_z.T) % 2 == 0))

    def test_medium_code(self):
        h_x, h_z = medium_code()
        self.assertEqual(h_x.shape, (36, 72))
        self.assertEqual(code_info(h_x, h_z), 12)
        self.assertTrue(np.all((h_x @ h_z.T) % 2 == 0))

    def test_logicals_steane(self):
        h_x, h_z = steane_code()
        l_x, l_z = logicals(h_x, h_z)
        self.assertEqual(l_x.shape, (1, 7))
        self.assertEqual(l_z.shape, (1, 7))
        self.assertTrue(np.all((h_z @ l_x.T) % 2 == 0))
        self.assertTrue(np.all((h_x @ l_z.T) % 2 == 0))
        self.assertEqual(gf2_rank((l_x @ l_z.T) % 2), 1)

    def test_logicals_gross(self):
        h_x, h_z = gross_code()
        l_x, l_z = logicals(h_x, h_z)
        self.assertEqual(l_x.shape, (12, 144))
        self.assertEqual(l_z.shape, (12, 144))
        self.assertTrue(np.all((h_z @ l_x.T) % 2 == 0))
        self.assertTrue(np.all((h_x @ l_z.T) % 2 == 0))
        self.assertEqual(gf2_rank((l_x @ l_z.T) % 2), 12)


if __name__ == "__main__":
    unittest.main()
