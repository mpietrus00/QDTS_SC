/*
 * QDTS_SC - Quadratic Distortion Tone Spectra for SuperCollider
 *
 * A solver for a particular system of equations related to the synthesis
 * of Auditory Distortion Products. Such equation was first proposed in:
 * "Kendall, G.S., Haworth, C., and Cadiz, R.F. (2014). Sound Synthesis with
 * Auditory Distortion Products. Computer Music Journal 38(4)".
 *
 * Port from Max/MSP implementation by Gutierrez, E and Cadiz, R.
 */

#include "SC_PlugIn.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <random>

using namespace Eigen;

// Type alias for random number generator
using RNG = std::mt19937;

// InterfaceTable contains pointers to functions in SuperCollider's API
static InterfaceTable *ft;

// Maximum number of harmonics supported
static const int MAX_HARMONICS = 16;

// Declare the UGen class
struct QDTSSolver : public Unit {
    VectorXd X;              // Current solution vector
    VectorXd T;              // Target vector (original)
    VectorXd init_T;         // Initial target (for error calculation)
    VectorXd solution;       // Best solution found
    VectorXd temporal_solution; // Temporary solution for comparison

    int numHarmonics;        // Number of harmonics
    float estimationError;   // L^2 error of the estimation
    bool needsUpdate;        // Flag to trigger recalculation

    RNG rng;                 // Random number generator
};

// Function prototypes
static VectorXd S(const VectorXd& X);
static VectorXd R(const VectorXd& X);
static VectorXd SN(const VectorXd& X, int N);
static VectorXd RN(const VectorXd& X, int N);
static VectorXd A(const VectorXd& X);
static VectorXd F(const VectorXd& X, const VectorXd& T);
static MatrixXd DF(const VectorXd& X);

// S function: Left shift. For example, S(1,2,3)=(2,3,0)
static VectorXd S(const VectorXd& X) {
    int N = X.rows();
    VectorXd shifted = VectorXd::Zero(N);
    for (int i = 0; i < N - 1; i++) {
        shifted(i) = X(i + 1);
    }
    shifted(N - 1) = 0;
    return shifted;
}

// R function: Right shift. For example, R(1,2,3)=(0,1,2)
static VectorXd R(const VectorXd& X) {
    int N = X.rows();
    VectorXd shifted = VectorXd::Zero(N);
    for (int i = 0; i < N - 1; i++) {
        shifted(i + 1) = X(i);
    }
    shifted(0) = 0;
    return shifted;
}

// SN(X,n) applies S n times to the vector X
static VectorXd SN(const VectorXd& X, int N) {
    VectorXd nshifted = X;
    for (int i = 0; i < N; i++) {
        nshifted = S(nshifted);
    }
    return nshifted;
}

// RN(X,n) applies R n times to the vector X
static VectorXd RN(const VectorXd& X, int N) {
    VectorXd nshifted = X;
    for (int i = 0; i < N; i++) {
        nshifted = R(nshifted);
    }
    return nshifted;
}

// The function A generates the right part of equation (17) in the paper with A_1=1
// A:R^N -> R^N, where N+1 is the number of sinusoids and N is the number of target harmonics
static VectorXd A(const VectorXd& X) {
    int N = X.rows();
    VectorXd Y = VectorXd::Zero(N + 1);
    Y(0) = 1;
    for (int i = 1; i < N + 1; i++) {
        Y(i) = X(i - 1);
    }
    VectorXd Z = VectorXd::Zero(N);
    for (int i = 0; i < N; i++) {
        Z(i) = Y.dot(SN(Y, i + 1));
    }
    return Z;
}

// The function F is the system (17) equals 0 with A_1=1
static VectorXd F(const VectorXd& X, const VectorXd& T) {
    return A(X) - T;
}

// The Jacobian matrix
static MatrixXd DF(const VectorXd& X) {
    int N = X.rows();
    MatrixXd Z = MatrixXd::Zero(N, N);
    VectorXd temp;
    for (int i = 0; i < N - 1; i++) {
        temp = SN(X, i + 1) + RN(X, i + 1);
        for (int j = 0; j < N; j++) {
            Z(i, j) = temp(j);
        }
    }
    Z += MatrixXd::Identity(N, N);
    return Z;
}

// Generate random vector with values in [0, 1]
static VectorXd randomVector(int n, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    VectorXd v(n);
    for (int i = 0; i < n; i++) {
        v(i) = dist(rng);
    }
    return v;
}

// Main solver function using Newton's method with restarts
static void solveSystem(QDTSSolver* unit) {
    int n = unit->numHarmonics;
    const double TOLERANCE = 0.0001;
    const int MAX_NEWTON_ITER = 33;
    const int MAX_RESTARTS = 100;

    // Initialize target vectors
    VectorXd T = unit->init_T;
    VectorXd init_T = unit->init_T;

    // Initial guess: random values in [0, 1]
    VectorXd ones = VectorXd::Ones(n);
    VectorXd X = 0.5 * (randomVector(n, unit->rng) + ones);

    // Temporal solution for comparison
    VectorXd temporal_solution = VectorXd::Constant(n, 0.5);

    int success = 0;
    int escape = 0;
    int j = 0;

    // First attempt with perturbation strategy
    while (success == 0 && escape == 0) {
        int i = 0;
        j++;

        // Newton iteration
        while (F(X, T).dot(F(X, T)) > TOLERANCE && i < MAX_NEWTON_ITER) {
            i++;
            MatrixXd df = DF(X);
            X = df.colPivHouseholderQr().solve(df * X - F(X, T));
        }

        // Save best solution
        if (F(X, init_T).dot(F(X, init_T)) < F(temporal_solution, init_T).dot(F(temporal_solution, init_T))) {
            temporal_solution = X;
        }

        if (F(X, T).dot(F(X, T)) <= TOLERANCE) {
            success = 1;
            escape = 0;
        } else {
            if (j > MAX_RESTARTS) {
                success = 0;
                escape = 1;
            } else {
                // Restart with perturbed target and new initial condition
                T = init_T + 0.01 * randomVector(n, unit->rng);
                X = 0.5 * (ones + randomVector(n, unit->rng));
            }
        }
    }

    // Second attempt if first failed - cumulative perturbation
    if (success == 0) {
        X = 0.5 * (randomVector(n, unit->rng) + ones);
        T = init_T;

        int success_2 = 0;
        j = 0;

        while (success_2 == 0 && j < MAX_RESTARTS) {
            int i = 0;
            j++;

            while (F(X, T).dot(F(X, T)) > TOLERANCE && i < MAX_NEWTON_ITER) {
                i++;
                MatrixXd df = DF(X);
                X = df.colPivHouseholderQr().solve(df * X - F(X, T));
            }

            // Save best solution
            if (F(X, init_T).dot(F(X, init_T)) < F(temporal_solution, init_T).dot(F(temporal_solution, init_T))) {
                temporal_solution = X;
            }

            if (F(X, T).dot(F(X, T)) <= TOLERANCE) {
                success_2 = 1;
            } else {
                // Cumulative perturbation (different from first attempt)
                T = T + 0.01 * randomVector(n, unit->rng);
                X = 0.5 * (ones + randomVector(n, unit->rng));
            }
        }
    }

    // Use best solution found
    unit->solution = temporal_solution;

    // Calculate estimation error
    VectorXd error = F(unit->solution, init_T);
    unit->estimationError = static_cast<float>(error.dot(error));
}

// UGen function declarations
static void QDTSSolver_Ctor(QDTSSolver* unit);
static void QDTSSolver_Dtor(QDTSSolver* unit);
static void QDTSSolver_next_k(QDTSSolver* unit, int inNumSamples);

// Plugin constructor
static void QDTSSolver_Ctor(QDTSSolver* unit) {
    // Get number of harmonics from first input (capped at MAX_HARMONICS)
    unit->numHarmonics = static_cast<int>(IN0(0));
    unit->numHarmonics = std::max(1, std::min(unit->numHarmonics, MAX_HARMONICS));

    int n = unit->numHarmonics;

    // Initialize random number generator
    new (&unit->rng) std::mt19937(std::random_device{}());

    // Initialize vectors using placement new
    new (&unit->X) VectorXd(n);
    new (&unit->T) VectorXd(n);
    new (&unit->init_T) VectorXd(n);
    new (&unit->solution) VectorXd(n);
    new (&unit->temporal_solution) VectorXd(n);

    // Read target values from inputs (inputs 1 onwards are target harmonics)
    for (int i = 0; i < n; ++i) {
        unit->init_T(i) = IN0(1 + i);
        unit->T(i) = unit->init_T(i);
    }

    unit->estimationError = 0.0f;
    unit->needsUpdate = true;

    // Set calculation function
    SETCALC(QDTSSolver_next_k);

    // Calculate first output
    QDTSSolver_next_k(unit, 1);
}

// Plugin destructor
static void QDTSSolver_Dtor(QDTSSolver* unit) {
    // Explicitly call destructors for Eigen objects and RNG
    unit->X.~VectorXd();
    unit->T.~VectorXd();
    unit->init_T.~VectorXd();
    unit->solution.~VectorXd();
    unit->temporal_solution.~VectorXd();
    unit->rng.~RNG();
}

// Control-rate calculation function
static void QDTSSolver_next_k(QDTSSolver* unit, int inNumSamples) {
    int n = unit->numHarmonics;

    // Check if target values have changed
    bool changed = false;
    for (int i = 0; i < n; ++i) {
        double newVal = IN0(1 + i);
        if (std::abs(unit->init_T(i) - newVal) > 1e-10) {
            unit->init_T(i) = newVal;
            unit->T(i) = newVal;
            changed = true;
        }
    }

    // Recalculate if needed
    if (changed || unit->needsUpdate) {
        solveSystem(unit);
        unit->needsUpdate = false;
    }

    // Normalize output (as in Max/MSP version)
    float normalizer = 1.0f / std::sqrt(static_cast<float>(n + 1));

    // Output 0: constant 1 * normalizer (A_1 = 1)
    OUT0(0) = normalizer;

    // Outputs 1 to n: solution values * normalizer
    for (int i = 0; i < n; ++i) {
        OUT0(i + 1) = static_cast<float>(unit->solution(i)) * normalizer;
    }

    // Output n+1: estimation error
    OUT0(n + 1) = unit->estimationError;
}

// Entry point
PluginLoad(QDTSSolver) {
    ft = inTable;
    DefineDtorUnit(QDTSSolver);
}
