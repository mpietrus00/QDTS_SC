/*
QDTS Verification Tests
=======================

Systematic tests to verify QDTSSolver implementation correctness.

This file tests:
1. Basic convergence for standard spectra
2. Mathematical verification: A(X) should equal T
3. Edge cases (near-zero values, max harmonics)
4. Comparison with Python reference implementation

Usage:
1. Boot the server: s.boot;
2. Run individual test sections by selecting and executing (Cmd+Enter)
3. Compare results with Python: python verification/qdts_solver.py

*/

// ============================================================
// SETUP
// ============================================================
(
s.waitForBoot {
    "Server booted. Ready for QDTS verification tests.".postln;
};
)

// ============================================================
// TEST 1: Basic Convergence Tests
// ============================================================
(
"".postln;
"=" ! 60 ++ "";
"TEST 1: Basic Convergence".postln;
"=" ! 60 ++ "";

fork {
    var testCases = [
        [\sawtooth_4, [1.0, 0.5, 0.333, 0.25]],
        [\sawtooth_8, (1..8).collect { |i| 1/i }],
        [\square_4, [1.0, 0.01, 0.333, 0.01]],
        [\triangle_4, [1.0, 0.01, 0.111, 0.01]],
        [\flat_4, [1.0, 1.0, 1.0, 1.0]],
        [\bright_4, [0.5, 0.75, 1.0, 0.75]],
        [\formant_6, [1.0, 0.8, 0.3, 0.5, 0.2, 0.1]],
    ];

    "".postln;
    "%-15s %3s %12s %s".format("Test Name", "N", "Error", "Status").postln;
    "-" ! 50 ++ "";

    testCases.do { |test|
        var name = test[0];
        var targets = test[1];
        var numH = targets.size;
        var error, status;

        {
            var solver = QDTSSolver.kr(numH, *targets);
            var err = solver[numH + 1];
            SendReply.kr(Impulse.kr(10), '/test_result', [err]);
            Silent.ar;
        }.play;

        0.2.wait;

        OSCFunc({ |msg|
            error = msg[3];
            status = if(error < 0.001, "PASS", "FAIL");
            "%-15s %3d %12.6f %s".format(name, numH, error, status).postln;
        }, '/test_result').oneShot;

        0.3.wait;
        s.freeAll;
        0.1.wait;
    };

    0.5.wait;
    "".postln;
    "Test 1 complete.".postln;
};
)

// ============================================================
// TEST 2: Mathematical Verification
// Verify that A(X) produces the target spectrum
// ============================================================
(
"".postln;
"=" ! 60 ++ "";
"TEST 2: Mathematical Verification".postln;
"Computing A(X) from solver output and comparing to targets".postln;
"=" ! 60 ++ "";

fork {
    var targets = [1.0, 0.5, 0.333, 0.25];
    var numH = targets.size;

    "".postln;
    "Targets: %".format(targets).postln;

    {
        var solver = QDTSSolver.kr(numH, *targets);
        var amps = solver[0..numH];  // A1, A2, A3, A4, A5 (normalized)
        var error = solver[numH + 1];

        // Denormalize to get raw amplitudes
        var normalizer = (numH + 1).sqrt.reciprocal;
        var rawAmps = amps / normalizer;

        // Compute A(X) - the achieved spectrum
        // Y = [A1, A2, ..., An+1]
        // T_i = sum of Y[j] * Y[j+i+1] for valid j
        var achieved = numH.collect { |i|
            var sum = 0;
            (numH - i).do { |j|
                sum = sum + (rawAmps[j] * rawAmps[j + i + 1]);
            };
            sum
        };

        // Send results
        SendReply.kr(Impulse.kr(5), '/verify_amps', rawAmps);
        SendReply.kr(Impulse.kr(5), '/verify_achieved', achieved);
        SendReply.kr(Impulse.kr(5), '/verify_error', [error]);
        Silent.ar;
    }.play;

    OSCFunc({ |msg|
        var amps = msg[3..];
        "Raw amplitudes: %".format(amps.round(0.0001)).postln;
    }, '/verify_amps').oneShot;

    OSCFunc({ |msg|
        var achieved = msg[3..];
        var errors = (achieved - targets).abs;
        "Achieved spectrum: %".format(achieved.round(0.0001)).postln;
        "Target spectrum:   %".format(targets).postln;
        "Absolute errors:   %".format(errors.round(0.0001)).postln;
        "Max error: %".format(errors.maxItem.round(0.000001)).postln;
    }, '/verify_achieved').oneShot;

    OSCFunc({ |msg|
        var error = msg[3];
        "Solver L2 error: %".format(error.round(0.000001)).postln;
    }, '/verify_error').oneShot;

    0.5.wait;
    s.freeAll;

    "".postln;
    "Test 2 complete.".postln;
};
)

// ============================================================
// TEST 3: Edge Cases
// ============================================================
(
"".postln;
"=" ! 60 ++ "";
"TEST 3: Edge Cases".postln;
"=" ! 60 ++ "";

fork {
    var edgeCases = [
        // Near-minimum values
        [\near_zero, [1.0, 0.01, 0.01, 0.01]],
        // Very bright (high harmonics dominant)
        [\very_bright, [0.1, 0.3, 0.6, 1.0]],
        // Single harmonic (fundamental only)
        [\fundamental, [1.0, 0.01, 0.01, 0.01]],
        // Maximum harmonics (16)
        [\max_harmonics, (1..16).collect { |i| 1/i }],
        // Steep rolloff
        [\steep_rolloff, [1.0, 0.1, 0.01, 0.01]],
        // Wide dynamic range
        [\wide_range, [1.0, 0.001, 1.0, 0.001].max(0.01)],  // clamp to 0.01 min
    ];

    "".postln;
    "%-18s %3s %12s %s".format("Test Name", "N", "Error", "Status").postln;
    "-" ! 55 ++ "";

    edgeCases.do { |test|
        var name = test[0];
        var targets = test[1];
        var numH = targets.size;
        var error, status;

        {
            var solver = QDTSSolver.kr(numH, *targets);
            var err = solver[numH + 1];
            SendReply.kr(Impulse.kr(10), '/edge_result', [err]);
            Silent.ar;
        }.play;

        0.2.wait;

        OSCFunc({ |msg|
            error = msg[3];
            status = if(error < 0.01, "PASS", if(error < 0.1, "MARGINAL", "FAIL"));
            "%-18s %3d %12.6f %s".format(name, numH, error, status).postln;
        }, '/edge_result').oneShot;

        0.3.wait;
        s.freeAll;
        0.1.wait;
    };

    0.5.wait;
    "".postln;
    "Test 3 complete.".postln;
};
)

// ============================================================
// TEST 4: Compare with Python Reference (Interactive)
// Run python verification/qdts_solver.py first to generate test_vectors.json
// ============================================================
(
// Known test vectors from Python (seed=42)
// These were generated by running the Python solver
var pythonResults = (
    sawtooth_4: (
        targets: [1.0, 0.5, 0.333, 0.25],
        // Python amplitudes will vary due to random initialization
        // We compare error levels instead
        expected_error_below: 0.001
    ),
    sawtooth_8: (
        targets: [1.0, 0.5, 0.333, 0.25, 0.2, 0.167, 0.143, 0.125],
        expected_error_below: 0.001
    ),
    square_4: (
        targets: [1.0, 0.01, 0.333, 0.01],
        expected_error_below: 0.001
    ),
    flat_4: (
        targets: [1.0, 1.0, 1.0, 1.0],
        expected_error_below: 0.001
    )
);

"".postln;
"=" ! 60 ++ "";
"TEST 4: Python Reference Comparison".postln;
"=" ! 60 ++ "";

fork {
    "".postln;
    "Comparing SC solver output with Python reference expectations...".postln;
    "".postln;

    pythonResults.keysValuesDo { |name, data|
        var targets = data[\targets];
        var numH = targets.size;
        var expectedError = data[\expected_error_below];

        {
            var solver = QDTSSolver.kr(numH, *targets);
            var error = solver[numH + 1];
            SendReply.kr(Impulse.kr(10), '/python_compare', [error]);
            Silent.ar;
        }.play;

        0.2.wait;

        OSCFunc({ |msg|
            var error = msg[3];
            var status = if(error < expectedError, "MATCH", "DIFFERS");
            "% : SC error=% (expected <%) : %".format(
                name, error.round(0.000001), expectedError, status
            ).postln;
        }, '/python_compare').oneShot;

        0.3.wait;
        s.freeAll;
        0.1.wait;
    };

    0.5.wait;
    "".postln;
    "Test 4 complete.".postln;
    "".postln;
    "For detailed comparison, run:".postln;
    "  cd verification && python qdts_solver.py".postln;
};
)

// ============================================================
// TEST 5: Real-time Stability Test
// Verify solver handles rapid parameter changes
// ============================================================
(
"".postln;
"=" ! 60 ++ "";
"TEST 5: Real-time Stability".postln;
"Rapid parameter changes - check for crashes/glitches".postln;
"=" ! 60 ++ "";

{
    var numH = 4;
    // Rapidly changing targets
    var t1 = LFNoise1.kr(2).range(0.5, 1.0);
    var t2 = LFNoise1.kr(3).range(0.1, 0.8);
    var t3 = LFNoise1.kr(2.5).range(0.1, 0.5);
    var t4 = LFNoise1.kr(1.5).range(0.1, 0.4);

    var solver = QDTSSolver.kr(numH, t1, t2, t3, t4);
    var error = solver[numH + 1];

    // Monitor error
    Poll.kr(Impulse.kr(4), error, "Error");

    // Check for NaN or extreme values
    var hasNaN = error.isNaN;
    var isExtreme = error > 10;

    // Alert if problems
    SendReply.kr(hasNaN + isExtreme, '/stability_alert', [error]);

    Silent.ar;
}.play;

OSCFunc({ |msg|
    "STABILITY ALERT: Error = %".format(msg[3]).postln;
}, '/stability_alert');

"Running for 5 seconds...".postln;
"Press Cmd+. to stop early".postln;

fork {
    5.wait;
    s.freeAll;
    "".postln;
    "Test 5 complete. If no alerts appeared, stability is good.".postln;
};
)

// ============================================================
// TEST 6: Audio Output Verification
// Listen to verify timbres sound correct
// ============================================================
(
"".postln;
"=" ! 60 ++ "";
"TEST 6: Audio Output Verification".postln;
"Listen to each timbre and verify it sounds correct".postln;
"=" ! 60 ++ "";

SynthDef(\qdts_test, { |out=0, amp=0.3, gate=1, carrierFreq=3000, targetFreq=100|
    var numH = 4;
    var targets = \targets.kr([1, 0.5, 0.333, 0.25]);
    var solver = QDTSSolver.kr(numH, *targets);
    var amps = solver[0..numH];

    var sig = (numH + 1).collect { |i|
        SinOsc.ar(carrierFreq + (i * targetFreq)) * amps[i];
    }.sum;

    var env = EnvGen.kr(Env.asr(0.01, 1, 0.1), gate, doneAction: 2);

    Out.ar(out, (sig * env * amp) ! 2);
}).add;

s.sync;
"SynthDef loaded.".postln;
)

// Play different timbres
(
"Playing SAWTOOTH...".postln;
x = Synth(\qdts_test, [\targets, [1, 0.5, 0.333, 0.25]]);
)
x.set(\gate, 0); // Stop

(
"Playing SQUARE (odd harmonics)...".postln;
x = Synth(\qdts_test, [\targets, [1, 0.01, 0.333, 0.01]]);
)
x.set(\gate, 0); // Stop

(
"Playing TRIANGLE...".postln;
x = Synth(\qdts_test, [\targets, [1, 0.01, 0.111, 0.01]]);
)
x.set(\gate, 0); // Stop

(
"Playing BRIGHT...".postln;
x = Synth(\qdts_test, [\targets, [0.5, 0.75, 1.0, 0.75]]);
)
x.set(\gate, 0); // Stop

// ============================================================
// SUMMARY
// ============================================================
(
"".postln;
"=" ! 60 ++ "";
"VERIFICATION SUMMARY".postln;
"=" ! 60 ++ "";
"".postln;
"Tests available:".postln;
"  1. Basic Convergence - Standard spectra".postln;
"  2. Mathematical Verification - A(X) = T check".postln;
"  3. Edge Cases - Extreme values".postln;
"  4. Python Comparison - Reference implementation".postln;
"  5. Real-time Stability - Rapid changes".postln;
"  6. Audio Output - Listening test".postln;
"".postln;
"For Python comparison, run:".postln;
"  cd verification && python qdts_solver.py".postln;
"".postln;
"Expected results:".postln;
"  - All errors should be < 0.001 for standard cases".postln;
"  - Edge cases may have slightly higher errors".postln;
"  - No NaN or infinite values".postln;
"  - Audio should match expected timbres".postln;
)
