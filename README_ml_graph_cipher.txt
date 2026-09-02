ML-guided bipartite graph-distance cipher prototype
===================================================

Files
-----
ml_guided_graph_cipher.py       Complete runnable Python implementation.
ml_graph_cipher_algorithm.tex   LaTeX equations + two algorithms for a manuscript.
requirements_ml_graph_cipher.txt Python dependencies.

Quick start
-----------
python -m pip install -r requirements_ml_graph_cipher.txt
python ml_guided_graph_cipher.py --samples 600 --candidates 300

Use your own plaintext corpus
-----------------------------
python ml_guided_graph_cipher.py \
    --corpus corpus.txt \
    --samples 2000 \
    --candidates 1000 \
    --dataset-out graph_cipher_dataset.csv \
    --importance-out feature_importance.csv

For a real experiment, use a sufficiently large and fixed evaluation corpus, keep a
held-out set of graph/key configurations, and report the individual diagnostics as
well as the composite screening score. The code is a research prototype, not a claim
of cryptographic security.
