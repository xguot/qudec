# qudec

BP+OSD, LP+OSD, and ADMM+OSD decoders for 72- and 144-qubit bivariate bicycle codes.
Requires Python 3.10+.

```sh
python -m pip install -e '.[circuit]'
python -m unittest discover -s tests

# 72 qubits, 3 noisy rounds, 32 shots at 2% noise
python scripts/bench_phenom.py 72 3 32 0.02 --variants bp,admm-w
```
