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

Solver : MultiOutUGen {
    *kr { |numHarmonics = 8 ... targets|
        // Outputs: [A_1 (normalized), A_2, ..., A_n+1, error]
        // Total outputs = numHarmonics + 2
        ^this.multiNew('control', numHarmonics, *targets);
    }

    *new { |numHarmonics = 8 ... targets|
        ^this.kr(numHarmonics, *targets);
    }

    init { |... theInputs|
        inputs = theInputs;
        // Outputs: n+1 amplitudes + 1 error = numHarmonics + 2
        ^this.initOutputs(theInputs[0].asInteger + 2, 'control');
    }

    checkInputs {
        var numHarmonics = inputs[0];
        if(numHarmonics.rate != 'scalar') {
            ^"numHarmonics must be a scalar value";
        };
        ^this.checkValidInputs;
    }
}

/*
 * QDTS - High-level interface for Quadratic Distortion Tone Spectra synthesis
 *
 * This class provides a convenient way to synthesize sounds using
 * Auditory Distortion Products (ADPs).
 */

QDTS {
    var <numHarmonics;
    var <targets;
    var <baseFreq;

    *new { |numHarmonics = 8, baseFreq = 440|
        ^super.new.init(numHarmonics, baseFreq);
    }

    init { |n, freq|
        numHarmonics = n.clip(1, 16);
        baseFreq = freq;
        // Default target: sawtooth-like spectrum
        targets = Array.fill(numHarmonics, { |i| 1.0 / (i + 1) });
    }

    // Set target spectrum from an array of harmonic amplitudes
    targets_ { |targetArray|
        if(targetArray.size != numHarmonics) {
            "Target array size (%) must match numHarmonics (%)".format(
                targetArray.size, numHarmonics
            ).warn;
            ^this;
        };
        targets = targetArray;
    }

    // Set individual target harmonic (0-indexed)
    setTarget { |index, value|
        if(index >= 0 and: { index < numHarmonics }) {
            targets[index] = value;
        } {
            "Index % out of range (0-%)".format(index, numHarmonics - 1).warn;
        };
    }

    // Get the solver UGen (returns array of [amplitudes, error])
    solver {
        ^Solver.kr(numHarmonics, *targets);
    }

    // Get just the amplitudes (without error)
    amplitudes {
        var sig = Solver.kr(numHarmonics, *targets);
        ^sig[0..numHarmonics];
    }

    // Get the estimation error
    error {
        var sig = Solver.kr(numHarmonics, *targets);
        ^sig[numHarmonics + 1];
    }

    // Synthesize audio using the solved amplitudes
    // Returns a mix of sinusoids at harmonic frequencies
    ar { |amp = 0.1|
        var amps = this.amplitudes;
        var sines = Array.fill(numHarmonics + 1, { |i|
            SinOsc.ar(baseFreq * (i + 1)) * amps[i];
        });
        ^Mix(sines) * amp;
    }

    // Synthesize with frequency modulation
    arFM { |modFreq = 1, modDepth = 0, amp = 0.1|
        var amps = this.amplitudes;
        var freqMod = SinOsc.kr(modFreq) * modDepth;
        var sines = Array.fill(numHarmonics + 1, { |i|
            SinOsc.ar((baseFreq + freqMod) * (i + 1)) * amps[i];
        });
        ^Mix(sines) * amp;
    }

    // Common target spectra presets

    *sawtooth { |numHarmonics = 8, baseFreq = 440|
        var qdts = this.new(numHarmonics, baseFreq);
        qdts.targets = Array.fill(numHarmonics, { |i| 1.0 / (i + 1) });
        ^qdts;
    }

    *square { |numHarmonics = 8, baseFreq = 440|
        var qdts = this.new(numHarmonics, baseFreq);
        qdts.targets = Array.fill(numHarmonics, { |i|
            if((i + 1).odd) { 1.0 / (i + 1) } { 0 };
        });
        ^qdts;
    }

    *triangle { |numHarmonics = 8, baseFreq = 440|
        var qdts = this.new(numHarmonics, baseFreq);
        qdts.targets = Array.fill(numHarmonics, { |i|
            if((i + 1).odd) {
                ((-1) ** ((i) / 2)) / ((i + 1) ** 2);
            } { 0 };
        });
        ^qdts;
    }

    // Print current state
    printOn { |stream|
        stream << "QDTS(numHarmonics: %, baseFreq: %, targets: %)".format(
            numHarmonics, baseFreq, targets.round(0.001)
        );
    }
}
