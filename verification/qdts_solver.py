#!/usr/bin/env python3
"""
QDTS Reference Solver - Python Implementation

A reference implementation of the QDTSSolver algorithm for verification.
This matches the C++ SuperCollider UGen implementation exactly.

Based on:
Kendall, G.S., Haworth, C., and Cadiz, R.F. (2014).
"Sound Synthesis with Auditory Distortion Products."
Computer Music Journal 38(4).

Usage:
    python qdts_solver.py

Or as a module:
    from qdts_solver import QDTSSolver
    solver = QDTSSolver()
    amplitudes, error = solver.solve([1.0, 0.5, 0.33, 0.25])
"""

import numpy as np
from typing import Tuple, List, Optional
import json


class QDTSSolver:
    """
    Solver for Quadratic Distortion Tone Spectra.

    Given target harmonic amplitudes T = [T1, T2, ..., Tn],
    finds sinusoid amplitudes A = [A1, A2, ..., An+1] such that
    the quadratic distortion products match the targets.

    A1 is always fixed at 1.0 (before normalization).
    """

    TOLERANCE = 0.0001
    MAX_NEWTON_ITER = 33
    MAX_RESTARTS = 100
    MAX_HARMONICS = 16

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize solver.

        Args:
            seed: Random seed for reproducibility (None for random)
        """
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def S(X: np.ndarray) -> np.ndarray:
        """Left shift: S([1,2,3]) = [2,3,0]"""
        n = len(X)
        shifted = np.zeros(n)
        shifted[:-1] = X[1:]
        return shifted

    @staticmethod
    def R(X: np.ndarray) -> np.ndarray:
        """Right shift: R([1,2,3]) = [0,1,2]"""
        n = len(X)
        shifted = np.zeros(n)
        shifted[1:] = X[:-1]
        return shifted

    @classmethod
    def SN(cls, X: np.ndarray, N: int) -> np.ndarray:
        """Apply S (left shift) N times"""
        result = X.copy()
        for _ in range(N):
            result = cls.S(result)
        return result

    @classmethod
    def RN(cls, X: np.ndarray, N: int) -> np.ndarray:
        """Apply R (right shift) N times"""
        result = X.copy()
        for _ in range(N):
            result = cls.R(result)
        return result

    @classmethod
    def A(cls, X: np.ndarray) -> np.ndarray:
        """
        Compute achieved spectrum from amplitudes.

        Given X = [A2, A3, ..., An+1] (n values),
        returns T = [T1, T2, ..., Tn] where Ti = sum of Aj * A(j+i).

        This implements equation (17) from the paper with A1 = 1.
        """
        n = len(X)
        # Y = [1, X[0], X[1], ..., X[n-1]] = [A1, A2, ..., An+1]
        Y = np.zeros(n + 1)
        Y[0] = 1.0
        Y[1:] = X

        # Z[i] = Y · S^(i+1)(Y) = sum of Aj * A(j+i+1)
        Z = np.zeros(n)
        for i in range(n):
            Z[i] = np.dot(Y, cls.SN(Y, i + 1))

        return Z

    @classmethod
    def F(cls, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Residual function: F(X, T) = A(X) - T"""
        return cls.A(X) - T

    @classmethod
    def DF(cls, X: np.ndarray) -> np.ndarray:
        """
        Jacobian matrix of F with respect to X.

        DF[i,j] = d(A(X)[i]) / d(X[j])
        """
        n = len(X)
        Z = np.zeros((n, n))

        for i in range(n - 1):
            temp = cls.SN(X, i + 1) + cls.RN(X, i + 1)
            Z[i, :] = temp

        # Add identity matrix
        Z += np.eye(n)

        return Z

    def _random_vector(self, n: int) -> np.ndarray:
        """Generate random vector with values in [-1, 1]"""
        return self.rng.uniform(-1.0, 1.0, n)

    def solve(self, targets: List[float], normalize: bool = True) -> Tuple[np.ndarray, float]:
        """
        Solve for amplitudes that produce target spectrum.

        Args:
            targets: Target harmonic amplitudes [T1, T2, ..., Tn]
            normalize: If True, divide amplitudes by sqrt(n+1)

        Returns:
            Tuple of (amplitudes, error) where:
            - amplitudes: [A1, A2, ..., An+1] (n+1 values)
            - error: L2 squared error of the solution
        """
        n = len(targets)
        if n > self.MAX_HARMONICS:
            n = self.MAX_HARMONICS
            targets = targets[:n]

        init_T = np.array(targets, dtype=float)
        T = init_T.copy()
        ones = np.ones(n)

        # Initial guess: random values in [0, 1]
        X = 0.5 * (self._random_vector(n) + ones)

        # Best solution tracking
        temporal_solution = np.full(n, 0.5)

        success = False
        escape = False
        j = 0

        # Phase 1: Perturbation strategy
        while not success and not escape:
            i = 0
            j += 1

            # Newton iteration
            residual = self.F(X, T)
            while np.dot(residual, residual) > self.TOLERANCE and i < self.MAX_NEWTON_ITER:
                i += 1
                df = self.DF(X)
                # X_new = DF^(-1) * (DF*X - F(X,T))
                X = np.linalg.lstsq(df, df @ X - residual, rcond=None)[0]
                residual = self.F(X, T)

            # Save best solution (evaluated against original targets)
            current_error = np.dot(self.F(X, init_T), self.F(X, init_T))
            best_error = np.dot(self.F(temporal_solution, init_T), self.F(temporal_solution, init_T))
            if current_error < best_error:
                temporal_solution = X.copy()

            if np.dot(residual, residual) <= self.TOLERANCE:
                success = True
            else:
                if j > self.MAX_RESTARTS:
                    escape = True
                else:
                    # Restart with perturbed target and new initial condition
                    T = init_T + 0.01 * self._random_vector(n)
                    X = 0.5 * (ones + self._random_vector(n))

        # Phase 2: Cumulative perturbation (if phase 1 failed)
        if not success:
            X = 0.5 * (self._random_vector(n) + ones)
            T = init_T.copy()
            j = 0

            while j < self.MAX_RESTARTS:
                i = 0
                j += 1

                residual = self.F(X, T)
                while np.dot(residual, residual) > self.TOLERANCE and i < self.MAX_NEWTON_ITER:
                    i += 1
                    df = self.DF(X)
                    X = np.linalg.lstsq(df, df @ X - residual, rcond=None)[0]
                    residual = self.F(X, T)

                # Save best solution
                current_error = np.dot(self.F(X, init_T), self.F(X, init_T))
                best_error = np.dot(self.F(temporal_solution, init_T), self.F(temporal_solution, init_T))
                if current_error < best_error:
                    temporal_solution = X.copy()

                if np.dot(residual, residual) <= self.TOLERANCE:
                    break
                else:
                    # Cumulative perturbation
                    T = T + 0.01 * self._random_vector(n)
                    X = 0.5 * (ones + self._random_vector(n))

        # Build output amplitudes [A1, A2, ..., An+1]
        solution = temporal_solution
        amplitudes = np.zeros(n + 1)
        amplitudes[0] = 1.0  # A1 is always 1
        amplitudes[1:] = solution

        # Calculate final error
        error_vec = self.F(solution, init_T)
        error = np.dot(error_vec, error_vec)

        # Normalize if requested
        if normalize:
            normalizer = 1.0 / np.sqrt(n + 1)
            amplitudes *= normalizer

        return amplitudes, error

    def verify_solution(self, amplitudes: np.ndarray, targets: List[float],
                       normalized: bool = True) -> dict:
        """
        Verify a solution by computing achieved spectrum.

        Args:
            amplitudes: [A1, A2, ..., An+1] from solve()
            targets: Original target spectrum
            normalized: Whether amplitudes are normalized

        Returns:
            Dictionary with verification results
        """
        n = len(targets)

        # Denormalize if needed
        if normalized:
            normalizer = 1.0 / np.sqrt(n + 1)
            amps = amplitudes / normalizer
        else:
            amps = amplitudes

        # X = [A2, A3, ..., An+1]
        X = amps[1:]

        # Compute achieved spectrum
        achieved = self.A(X)

        # Compute errors
        targets_arr = np.array(targets)
        absolute_errors = np.abs(achieved - targets_arr)
        relative_errors = absolute_errors / np.maximum(targets_arr, 1e-10)
        l2_error = np.sum((achieved - targets_arr) ** 2)

        return {
            'targets': targets_arr.tolist(),
            'achieved': achieved.tolist(),
            'absolute_errors': absolute_errors.tolist(),
            'relative_errors': relative_errors.tolist(),
            'l2_error': float(l2_error),
            'max_absolute_error': float(np.max(absolute_errors)),
            'max_relative_error': float(np.max(relative_errors)),
            'converged': bool(l2_error < self.TOLERANCE)
        }


def test_basic():
    """Basic functionality test"""
    print("=" * 60)
    print("QDTS Reference Solver - Basic Test")
    print("=" * 60)

    solver = QDTSSolver(seed=42)

    # Test 1: Sawtooth spectrum
    print("\n1. Sawtooth spectrum [1, 0.5, 0.33, 0.25]")
    targets = [1.0, 0.5, 0.33, 0.25]
    amplitudes, error = solver.solve(targets)
    verification = solver.verify_solution(amplitudes, targets)

    print(f"   Amplitudes: {amplitudes}")
    print(f"   Error: {error:.6f}")
    print(f"   Achieved: {verification['achieved']}")
    print(f"   Converged: {verification['converged']}")

    # Test 2: Square wave (odd harmonics)
    print("\n2. Square wave [1, 0, 0.33, 0]")
    targets = [1.0, 0.01, 0.33, 0.01]  # Use 0.01 instead of 0
    amplitudes, error = solver.solve(targets)
    verification = solver.verify_solution(amplitudes, targets)

    print(f"   Amplitudes: {amplitudes}")
    print(f"   Error: {error:.6f}")
    print(f"   Converged: {verification['converged']}")

    # Test 3: 8 harmonics
    print("\n3. 8-harmonic sawtooth")
    targets = [1/i for i in range(1, 9)]
    amplitudes, error = solver.solve(targets)
    verification = solver.verify_solution(amplitudes, targets)

    print(f"   Amplitudes: {amplitudes}")
    print(f"   Error: {error:.6f}")
    print(f"   Converged: {verification['converged']}")

    return True


def test_systematic():
    """Systematic testing of various spectra"""
    print("\n" + "=" * 60)
    print("QDTS Reference Solver - Systematic Tests")
    print("=" * 60)

    solver = QDTSSolver(seed=123)
    results = []

    test_cases = [
        ("Sawtooth-4", [1.0, 0.5, 0.333, 0.25]),
        ("Sawtooth-8", [1/i for i in range(1, 9)]),
        ("Square-4", [1.0, 0.01, 0.333, 0.01]),
        ("Triangle-4", [1.0, 0.01, 0.111, 0.01]),
        ("Flat-4", [1.0, 1.0, 1.0, 1.0]),
        ("Decay-4", [1.0, 0.8, 0.6, 0.4]),
        ("Bright-4", [0.5, 0.75, 1.0, 0.75]),
        ("Formant-A", [1.0, 0.8, 0.3, 0.5, 0.2, 0.1]),
        ("Formant-E", [1.0, 0.5, 0.7, 0.3, 0.4, 0.2]),
        ("Random-6", list(np.random.uniform(0.1, 1.0, 6))),
    ]

    print(f"\n{'Test Name':<15} {'N':>3} {'Error':>12} {'Max Err':>10} {'Status':<10}")
    print("-" * 55)

    for name, targets in test_cases:
        amplitudes, error = solver.solve(targets)
        verification = solver.verify_solution(amplitudes, targets)
        status = "✓ PASS" if verification['converged'] else "✗ FAIL"

        print(f"{name:<15} {len(targets):>3} {error:>12.6f} {verification['max_absolute_error']:>10.6f} {status:<10}")

        results.append({
            'name': name,
            'targets': targets,
            'amplitudes': amplitudes.tolist(),
            'error': error,
            'verification': verification
        })

    return results


def export_test_vectors(filename: str = "test_vectors.json"):
    """Export test vectors for comparison with SC implementation"""
    print("\n" + "=" * 60)
    print(f"Exporting test vectors to {filename}")
    print("=" * 60)

    solver = QDTSSolver(seed=42)  # Fixed seed for reproducibility

    test_cases = [
        ("sawtooth_4", [1.0, 0.5, 0.333, 0.25]),
        ("sawtooth_8", [1/i for i in range(1, 9)]),
        ("square_4", [1.0, 0.01, 0.333, 0.01]),
        ("triangle_4", [1.0, 0.01, 0.111, 0.01]),
        ("flat_4", [1.0, 1.0, 1.0, 1.0]),
        ("bright_4", [0.5, 0.75, 1.0, 0.75]),
        ("formant_6", [1.0, 0.8, 0.3, 0.5, 0.2, 0.1]),
    ]

    vectors = {}
    for name, targets in test_cases:
        amplitudes, error = solver.solve(targets, normalize=True)
        amplitudes_raw, _ = solver.solve(targets, normalize=False)
        verification = solver.verify_solution(amplitudes, targets)

        vectors[name] = {
            'targets': targets,
            'amplitudes_normalized': amplitudes.tolist(),
            'amplitudes_raw': amplitudes_raw.tolist(),
            'error': float(error),
            'achieved': verification['achieved'],
            'converged': bool(verification['converged'])
        }

    with open(filename, 'w') as f:
        json.dump(vectors, f, indent=2)

    print(f"Exported {len(vectors)} test vectors")
    return vectors


if __name__ == "__main__":
    # Run tests
    test_basic()
    results = test_systematic()

    # Export test vectors
    vectors = export_test_vectors()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
