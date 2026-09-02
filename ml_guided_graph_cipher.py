#!/usr/bin/env python3
"""ML-guided optimization of a bipartite graph-distance cipher.

This is a research prototype, not production cryptography.

Core idea
---------
1. Build candidate bipartite graphs on letters A..Z and number vertices 1..26.
2. Choose two distinct number landmarks.
3. Convert landmark distances into a graph-dependent cyclic permutation pi.
4. Use a *position-only* graph keystream so decryption does not require knowing
   the plaintext symbol in advance.
5. Evaluate empirical cipher diagnostics.
6. Train a regression model f_theta(z_G) -> Q to predict the empirical quality
   of unseen graph/key configurations.
7. Use the trained model to screen many new candidates, then verify the best
   candidates by actual encryption diagnostics.

The composite quality score Q is only an experimental proxy; it is NOT a proof
of cryptographic security.
"""

from __future__ import annotations

import argparse
import math
import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ALPHABET = string.ascii_uppercase
N = 26

LetterNode = str
NumberNode = int
Landmarks = Tuple[int, int]


@dataclass
class Candidate:
    graph: nx.Graph
    landmarks: Landmarks
    a: int
    b: int
    features: Dict[str, float]
    predicted_quality: float | None = None
    observed_metrics: Dict[str, float] | None = None


def build_candidate_graph(n_extra_edges: int = 6, seed: int | None = None) -> nx.Graph:
    """Construct a connected bipartite graph based on the draft's prototype.

    Base rule: letter X_i is adjacent to number i+1 and the next number
    cyclically. Random extra letter-number edges are then added.
    """
    rng = random.Random(seed)
    G = nx.Graph()

    letters = list(ALPHABET)
    numbers = list(range(1, N + 1))
    G.add_nodes_from(letters, bipartite=0, kind="letter")
    G.add_nodes_from(numbers, bipartite=1, kind="number")

    # Connected 52-cycle-like base structure.
    for i, X in enumerate(letters):
        v1 = i + 1
        v2 = (i + 1) % N + 1
        G.add_edge(X, v1)
        G.add_edge(X, v2)

    possible = [(X, v) for X in letters for v in numbers if not G.has_edge(X, v)]
    rng.shuffle(possible)
    for X, v in possible[: max(0, int(n_extra_edges))]:
        G.add_edge(X, v)

    return G


def build_draft_graph() -> nx.Graph:
    """Construct the specific worked graph described in the attached draft."""
    G = build_candidate_graph(n_extra_edges=0, seed=0)
    extras = [("A", 10), ("F", 18), ("K", 23), ("P", 5), ("T", 14), ("Y", 3)]
    G.add_edges_from(extras)
    return G


def distance_signature(G: nx.Graph, X: str, landmarks: Landmarks) -> Tuple[int, int]:
    w1, w2 = landmarks
    return nx.shortest_path_length(G, X, w1), nx.shortest_path_length(G, X, w2)


def draft_score(G: nx.Graph, X: str, landmarks: Landmarks) -> float:
    """Original draft score S(X)=100*d1+d2+i/1000."""
    d1, d2 = distance_signature(G, X, landmarks)
    i = ALPHABET.index(X)
    return 100.0 * d1 + d2 + i / 1000.0


def graph_order(G: nx.Graph, landmarks: Landmarks) -> List[str]:
    """Deterministic graph ordering.

    We use the transparent lexicographic form (d1,d2,i), which implements the
    same priority as the draft score for this 26-letter prototype.
    """
    return sorted(
        ALPHABET,
        key=lambda X: (*distance_signature(G, X, landmarks), ALPHABET.index(X)),
    )


def build_permutation(G: nx.Graph, landmarks: Landmarks) -> Tuple[Dict[str, str], Dict[str, str], List[str]]:
    order = graph_order(G, landmarks)
    pi = {order[k]: order[(k + 1) % N] for k in range(N)}
    inv_pi = {v: k for k, v in pi.items()}
    return pi, inv_pi, order


def graph_keystream_period(
    G: nx.Graph,
    order: Sequence[str],
    landmarks: Landmarks,
    a: int = 2,
    b: int = 3,
) -> List[int]:
    """Precompute the 26-symbol graph-derived, position-only keystream period.

    For j=1,...,26, let X_j be the j-th letter in the graph ordering and set
        k_G(j) = [a d(X_j,w1) + b d(X_j,w2) + j] mod 26.
    Since both X_j and j mod 26 repeat, the shift sequence has period 26.
    """
    shifts: List[int] = []
    for position in range(1, N + 1):
        anchor = order[position - 1]
        d1, d2 = distance_signature(G, anchor, landmarks)
        shifts.append(int((a * d1 + b * d2 + position) % N))
    return shifts


def graph_keystream_shift(
    period: Sequence[int],
    position: int,
) -> int:
    return int(period[(position - 1) % N])


def encrypt(
    plaintext: str,
    G: nx.Graph,
    landmarks: Landmarks,
    a: int = 2,
    b: int = 3,
) -> str:
    pi, _, order = build_permutation(G, landmarks)
    period = graph_keystream_period(G, order, landmarks, a=a, b=b)
    out: List[str] = []
    pos = 0
    for ch in plaintext.upper():
        if ch not in ALPHABET:
            out.append(ch)
            continue
        pos += 1
        k = graph_keystream_shift(period, pos)
        y = ALPHABET.index(pi[ch])
        out.append(ALPHABET[(y + k) % N])
    return "".join(out)


def decrypt(
    ciphertext: str,
    G: nx.Graph,
    landmarks: Landmarks,
    a: int = 2,
    b: int = 3,
) -> str:
    _, inv_pi, order = build_permutation(G, landmarks)
    period = graph_keystream_period(G, order, landmarks, a=a, b=b)
    out: List[str] = []
    pos = 0
    for ch in ciphertext.upper():
        if ch not in ALPHABET:
            out.append(ch)
            continue
        pos += 1
        k = graph_keystream_shift(period, pos)
        y = ALPHABET[(ALPHABET.index(ch) - k) % N]
        out.append(inv_pi[y])
    return "".join(out)


def clean_letters(text: str) -> str:
    return "".join(ch for ch in text.upper() if ch in ALPHABET)


def normalized_entropy(text: str) -> float:
    s = clean_letters(text)
    if not s:
        return 0.0
    counts = np.array([s.count(ch) for ch in ALPHABET], dtype=float)
    p = counts[counts > 0] / len(s)
    H = -np.sum(p * np.log2(p))
    return float(H / math.log2(N))


def uniformity_score(text: str) -> float:
    """1 - total-variation distance from the uniform 26-symbol distribution."""
    s = clean_letters(text)
    if not s:
        return 0.0
    counts = np.array([s.count(ch) for ch in ALPHABET], dtype=float)
    p = counts / len(s)
    u = np.full(N, 1.0 / N)
    tv = 0.5 * np.abs(p - u).sum()
    return float(max(0.0, 1.0 - tv))


def mutual_information_normalized(plaintext: str, ciphertext: str) -> float:
    """Empirical I(P;C)/H(P) for aligned alphabetic characters."""
    p = clean_letters(plaintext)
    c = clean_letters(ciphertext)
    n = min(len(p), len(c))
    if n == 0:
        return 0.0
    p, c = p[:n], c[:n]

    joint = np.zeros((N, N), dtype=float)
    for x, y in zip(p, c):
        joint[ALPHABET.index(x), ALPHABET.index(y)] += 1.0
    joint /= n
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)

    mi = 0.0
    for i in range(N):
        for j in range(N):
            if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += joint[i, j] * math.log2(joint[i, j] / (px[i] * py[j]))

    nz = px[px > 0]
    hp = -float(np.sum(nz * np.log2(nz)))
    if hp <= 1e-12:
        return 0.0
    return float(min(1.0, max(0.0, mi / hp)))


def perturb_landmarks(landmarks: Landmarks) -> Landmarks:
    """Deterministically change one landmark by one cyclic step.

    This gives a reproducible small key perturbation for an empirical
    sensitivity diagnostic.
    """
    w1, w2 = landmarks
    w2_new = (w2 % N) + 1
    if w2_new == w1:
        w2_new = (w2_new % N) + 1
    return w1, w2_new


def hamming_fraction(a: str, b: str) -> float:
    aa = clean_letters(a)
    bb = clean_letters(b)
    n = min(len(aa), len(bb))
    if n == 0:
        return 0.0
    return sum(x != y for x, y in zip(aa[:n], bb[:n])) / n


def empirical_metrics(
    G: nx.Graph,
    landmarks: Landmarks,
    plaintext: str,
    a: int = 2,
    b: int = 3,
) -> Dict[str, float]:
    cipher = encrypt(plaintext, G, landmarks, a=a, b=b)

    ent = normalized_entropy(cipher)
    uni = uniformity_score(cipher)
    mi = mutual_information_normalized(plaintext, cipher)

    # Reproducible small change in the landmark component of the key.
    perturbed = perturb_landmarks(landmarks)
    cipher2 = encrypt(plaintext, G, perturbed, a=a, b=b)
    sensitivity = hamming_fraction(cipher, cipher2)

    # Experimental proxy only; weights are explicit and tunable.
    Q = 0.30 * ent + 0.25 * uni + 0.25 * (1.0 - mi) + 0.20 * sensitivity

    return {
        "entropy_norm": ent,
        "uniformity": uni,
        "mi_norm": mi,
        "key_sensitivity": sensitivity,
        "quality": float(Q),
    }

def graph_features(G: nx.Graph, landmarks: Landmarks, a: int = 2, b: int = 3) -> Dict[str, float]:
    w1, w2 = landmarks
    letter_deg = np.array([G.degree(X) for X in ALPHABET], dtype=float)
    number_deg = np.array([G.degree(v) for v in range(1, N + 1)], dtype=float)

    d1 = np.array([nx.shortest_path_length(G, X, w1) for X in ALPHABET], dtype=float)
    d2 = np.array([nx.shortest_path_length(G, X, w2) for X in ALPHABET], dtype=float)
    signatures = list(zip(d1.astype(int), d2.astype(int)))

    _, _, order = build_permutation(G, landmarks)
    order_idx = np.array([ALPHABET.index(x) for x in order], dtype=float)
    cyclic_next = np.roll(order_idx, -1)
    mean_order_jump = float(np.mean(np.abs(cyclic_next - order_idx)) / (N - 1))

    pi, _, _ = build_permutation(G, landmarks)
    displacement = np.mean(
        [abs(ALPHABET.index(pi[x]) - ALPHABET.index(x)) / (N - 1) for x in ALPHABET]
    )

    # Graph/key-only descriptors of the resulting 26-step keystream.
    period = graph_keystream_period(G, order, landmarks, a=a, b=b)
    shift_counts = np.bincount(np.asarray(period, dtype=int), minlength=N).astype(float)
    shift_p = shift_counts[shift_counts > 0] / N
    shift_entropy = -float(np.sum(shift_p * np.log2(shift_p))) / math.log2(N)
    shift_unique = len(set(period)) / N
    shift_std = float(np.std(period) / (N - 1))

    # Structural sensitivity of the graph-induced permutation to a small
    # landmark-key perturbation; this is computed without encrypting a corpus.
    perturbed_lm = perturb_landmarks(landmarks)
    pi2, _, _ = build_permutation(G, perturbed_lm)
    order_sensitivity = sum(pi[x] != pi2[x] for x in ALPHABET) / N

    return {
        "edge_count": float(G.number_of_edges()),
        "extra_edge_count": float(G.number_of_edges() - 2 * N),
        "letter_degree_mean": float(letter_deg.mean()),
        "letter_degree_std": float(letter_deg.std()),
        "letter_degree_max": float(letter_deg.max()),
        "number_degree_mean": float(number_deg.mean()),
        "number_degree_std": float(number_deg.std()),
        "number_degree_max": float(number_deg.max()),
        "landmark_distance": float(nx.shortest_path_length(G, w1, w2)),
        "d1_mean": float(d1.mean()),
        "d1_std": float(d1.std()),
        "d2_mean": float(d2.mean()),
        "d2_std": float(d2.std()),
        "distance_pair_uniqueness": float(len(set(signatures)) / N),
        "graph_diameter": float(nx.diameter(G)),
        "avg_shortest_path": float(nx.average_shortest_path_length(G)),
        "mean_order_jump": mean_order_jump,
        "mean_permutation_displacement": float(displacement),
        "keystream_entropy": float(shift_entropy),
        "keystream_unique_fraction": float(shift_unique),
        "keystream_shift_std": float(shift_std),
        "permutation_landmark_sensitivity": float(order_sensitivity),
        "landmark_1": float(w1),
        "landmark_2": float(w2),
        "key_a": float(a),
        "key_b": float(b),
    }


def random_landmarks(rng: random.Random) -> Landmarks:
    w1, w2 = rng.sample(range(1, N + 1), 2)
    return int(w1), int(w2)


def generate_dataset(
    plaintext: str,
    n_samples: int = 1000,
    extra_min: int = 0,
    extra_max: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: List[Dict[str, float]] = []

    for _ in range(n_samples):
        n_extra = rng.randint(extra_min, extra_max)
        graph_seed = rng.randint(0, 2**31 - 1)
        G = build_candidate_graph(n_extra_edges=n_extra, seed=graph_seed)
        landmarks = random_landmarks(rng)
        a = rng.randint(1, N - 1)
        b = rng.randint(1, N - 1)

        row = graph_features(G, landmarks, a=a, b=b)
        row.update(empirical_metrics(G, landmarks, plaintext, a=a, b=b))
        row["graph_seed"] = float(graph_seed)
        rows.append(row)

    return pd.DataFrame(rows)


def train_quality_model(
    df: pd.DataFrame,
    seed: int = 42,
) -> Tuple[RandomForestRegressor, List[str], Dict[str, float], pd.DataFrame]:
    target = "quality"
    excluded = {
        target,
        "entropy_norm",
        "uniformity",
        "mi_norm",
        "key_sensitivity",
        "graph_seed",
    }
    feature_cols = [c for c in df.columns if c not in excluded]

    X = df[feature_cols]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed
    )

    model = RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    report = {
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_test, pred))),
        "R2": float(r2_score(y_test, pred)),
    }

    importance = pd.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False, ignore_index=True)

    return model, feature_cols, report, importance


def screen_candidates(
    model: RandomForestRegressor,
    feature_cols: Sequence[str],
    plaintext: str,
    n_candidates: int = 500,
    extra_min: int = 0,
    extra_max: int = 30,
    seed: int = 2026,
) -> List[Candidate]:
    rng = random.Random(seed)
    candidates: List[Candidate] = []

    for _ in range(n_candidates):
        n_extra = rng.randint(extra_min, extra_max)
        G = build_candidate_graph(n_extra, seed=rng.randint(0, 2**31 - 1))
        landmarks = random_landmarks(rng)
        a = rng.randint(1, N - 1)
        b = rng.randint(1, N - 1)
        feats = graph_features(G, landmarks, a=a, b=b)
        X = pd.DataFrame([[feats[c] for c in feature_cols]], columns=feature_cols)
        pred = float(model.predict(X)[0])
        candidates.append(Candidate(G, landmarks, a, b, feats, predicted_quality=pred))

    candidates.sort(key=lambda c: c.predicted_quality or -math.inf, reverse=True)

    # Verify the top candidate with the actual diagnostics.
    if candidates:
        candidates[0].observed_metrics = empirical_metrics(
            candidates[0].graph, candidates[0].landmarks, plaintext,
            a=candidates[0].a, b=candidates[0].b
        )
    return candidates


def default_plaintext() -> str:
    # A deterministic fallback for a runnable demo. For research, pass a much
    # larger representative corpus using --corpus.
    text = (
        "MATHEMATICS STUDIES PATTERNS STRUCTURE CHANGE AND RELATION. "
        "GRAPH THEORY REPRESENTS OBJECTS AND THEIR CONNECTIONS. "
        "MACHINE LEARNING CAN MODEL EMPIRICAL RELATIONS BETWEEN GRAPH "
        "CHARACTERISTICS AND MEASURED CIPHER BEHAVIOUR. "
    )
    return text * 80


def load_plaintext(path: str | None) -> str:
    if path is None:
        return default_plaintext()
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def demo_draft_graph() -> None:
    G = build_draft_graph()
    landmarks = (1, 14)
    plaintext = "MATHEMATICS"
    cipher = encrypt(plaintext, G, landmarks)
    recovered = decrypt(cipher, G, landmarks)
    print("Draft graph demo")
    print("----------------")
    print("Landmarks:", landmarks)
    print("Graph order:", " ".join(graph_order(G, landmarks)))
    print("Plaintext :", plaintext)
    print("Ciphertext:", cipher)
    print("Recovered :", recovered)
    assert recovered == plaintext


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=str, default=None, help="Plaintext corpus .txt file")
    parser.add_argument("--samples", type=int, default=600, help="Training configurations")
    parser.add_argument("--candidates", type=int, default=300, help="New graphs to screen")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-out", type=str, default="graph_cipher_dataset.csv")
    parser.add_argument("--importance-out", type=str, default="feature_importance.csv")
    args = parser.parse_args()

    demo_draft_graph()
    plaintext = load_plaintext(args.corpus)

    print("\nGenerating ML dataset ...")
    df = generate_dataset(plaintext, n_samples=args.samples, seed=args.seed)
    df.to_csv(args.dataset_out, index=False)
    print(f"Saved {len(df)} rows to {args.dataset_out}")

    print("\nTraining f_theta(z_G) -> Q ...")
    model, feature_cols, report, importance = train_quality_model(df, seed=args.seed)
    importance.to_csv(args.importance_out, index=False)
    for k, v in report.items():
        print(f"{k:>5s}: {v:.6f}")

    print("\nTop feature importances")
    print(importance.head(10).to_string(index=False))

    print("\nScreening unseen candidate graphs ...")
    candidates = screen_candidates(
        model,
        feature_cols,
        plaintext,
        n_candidates=args.candidates,
        seed=args.seed + 100,
    )
    best = candidates[0]
    print("Best predicted quality:", round(float(best.predicted_quality), 6))
    print("Landmarks:", best.landmarks)
    print("Edges:", best.graph.number_of_edges())
    print("Coefficients (a,b):", (best.a, best.b))
    print("Graph order:", " ".join(graph_order(best.graph, best.landmarks)))
    if best.observed_metrics:
        print("Observed diagnostics:")
        for k, v in best.observed_metrics.items():
            print(f"  {k:>16s}: {v:.6f}")

    test_word = "MATHEMATICS"
    ct = encrypt(test_word, best.graph, best.landmarks, a=best.a, b=best.b)
    pt = decrypt(ct, best.graph, best.landmarks, a=best.a, b=best.b)
    print("\nVerification example")
    print("Plaintext :", test_word)
    print("Ciphertext:", ct)
    print("Recovered :", pt)
    assert pt == test_word


if __name__ == "__main__":
    main()
