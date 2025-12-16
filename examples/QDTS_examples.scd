/*
QDTS_SC - Quadratic Distortion Tone Spectra for SuperCollider
==============================================================

Examples demonstrating the QDTSSolver UGen for Auditory Distortion Product synthesis.

Based on: Kendall, G.S., Haworth, C., and Cadiz, R.F. (2014).
"Sound Synthesis with Auditory Distortion Products." Computer Music Journal 38(4).
*/

// Boot the server first
s.boot;

// ============================================================================
// 1. BASIC USAGE
// ============================================================================

// Simple test - verify the UGen is working
(
{
    var targets = [1, 0.5, 0.33, 0.25];
    var sig = QDTSSolver.kr(4, *targets);
    sig.poll(2); // Poll outputs at 2 Hz
    Silent.ar;
}.play;
)
// Outputs: [amp0, amp1, amp2, amp3, amp4, error]
// The error should be very small (< 0.001) indicating good convergence

// Basic synthesis with carrier and target pitch
(
{
    var numPartials = 4;
    var carrierPitch = 1000;
    var targetPitch = 100;
    var targets = [1, 0.5, 0.33, 0.25];

    var amps = QDTSSolver.kr(numPartials, *targets);
    var freqs = (numPartials + 1).collect {|i| carrierPitch + (i * targetPitch) };
    var sines = (numPartials + 1).collect {|i| SinOsc.ar(freqs[i]) * amps[i] };

    sines.sum * 0.2 ! 2
}.play;
)

// ============================================================================
// 2. SYNTHDEF WITH ENVELOPE
// ============================================================================

(
SynthDef(\qdtsEnv, {|out = 0, carrierPitch = 1280, targetPitch = 130,
    amp = 0.2, gate = 1, atk = 0.01, dec = 0.1, sus = 0.7, rel = 0.5,
    t0 = 1, t1 = 0.5, t2 = 0.33, t3 = 0.25|

    // IMPORTANT: Clamp targets to avoid 0 (causes solver instability)
    var t0s = t0.max(0.01);
    var t1s = t1.max(0.01);
    var t2s = t2.max(0.01);
    var t3s = t3.max(0.01);

    var env = EnvGen.kr(Env.adsr(atk, dec, sus, rel), gate, doneAction: 2);
    var amps = QDTSSolver.kr(4, t0s, t1s, t2s, t3s);
    var freqs = 5.collect {|i| carrierPitch + (i * targetPitch) };
    var sines = 5.collect {|i| SinOsc.ar(freqs[i]) * amps[i] };

    // Limiter for safety
    var sig = Limiter.ar(sines.sum * env * amp, 0.9);

    Out.ar(out, sig ! 2);
}).add;
)

// Test the SynthDef
x = Synth(\qdtsEnv, [\carrierPitch, 1000, \targetPitch, 100]);
x.set(\gate, 0); // Release

// ============================================================================
// 3. REAL-TIME PARAMETER CONTROL
// ============================================================================

(
SynthDef(\qdtsDynamic, {|out = 0, carrierPitch = 1280, targetPitch = 130, amp = 0.2,
    t0 = 1, t1 = 0.5, t2 = 0.33, t3 = 0.25|

    var t0s = t0.max(0.01);
    var t1s = t1.max(0.01);
    var t2s = t2.max(0.01);
    var t3s = t3.max(0.01);

    var amps = QDTSSolver.kr(4, t0s, t1s, t2s, t3s);
    var freqs = 5.collect {|i| carrierPitch + (i * targetPitch) };
    var sines = 5.collect {|i| SinOsc.ar(freqs[i]) * amps[i] };

    Out.ar(out, Limiter.ar(sines.sum * amp, 0.9) ! 2);
}).add;
)

// Play and modulate
x = Synth(\qdtsDynamic);

// Try different spectra
x.set(\t0, 1, \t1, 0.01, \t2, 0.5, \t3, 0.01);    // Square-ish (odd harmonics)
x.set(\t0, 1, \t1, 0.5, \t2, 0.33, \t3, 0.25);    // Sawtooth
x.set(\t0, 1, \t1, 1, \t2, 1, \t3, 1);            // Bright/harsh
x.set(\t0, 1, \t1, 0.1, \t2, 0.05, \t3, 0.02);    // Mellow

// Non-standard spectra (upper harmonic dominant)
x.set(\t0, 0.2, \t1, 1, \t2, 0.1, \t3, 0.1);      // 2nd harmonic dominant (hollow)
x.set(\t0, 0.1, \t1, 0.1, \t2, 1, \t3, 0.1);      // 3rd harmonic dominant (nasal)
x.set(\t0, 0.3, \t1, 0.3, \t2, 0.3, \t3, 1);      // 4th harmonic dominant
x.set(\t0, 0.5, \t1, 0.8, \t2, 1, \t3, 0.6);      // Peak at 3rd, gradual rolloff

// Change pitch
x.set(\carrierPitch, 2000, \targetPitch, 150);
x.set(\carrierPitch, 500, \targetPitch, 50);

x.free;

// ============================================================================
// 4. PATTERN EXAMPLES
// ============================================================================

// Rhythmic pattern with varying spectrum
(
Pdef(\qdtsRhythm,
    Pbind(
        \instrument, \qdtsEnv,
        \dur, Pseq([0.25, 0.25, 0.5, 0.25, 0.25, 0.25, 0.25], inf),
        \carrierPitch, Pseq([800, 1000, 1200, 1600], inf),
        \targetPitch, Pwhite(60, 150),
        \amp, 0.15,
        \atk, 0.01,
        \dec, 0.1,
        \sus, 0.6,
        \rel, 0.3,
        // Vary spectrum per note
        \t0, 1,
        \t1, Pwhite(0.2, 0.8),
        \t2, Pwhite(0.1, 0.5),
        \t3, Pwhite(0.05, 0.3),
    )
).play;
)

Pdef(\qdtsRhythm).stop;

// Melodic pattern with scale
(
Pdef(\qdtsMelody,
    Pbind(
        \instrument, \qdtsEnv,
        \scale, Scale.minor,
        \degree, Pseq([0, 2, 4, 5, 7, 5, 4, 2], inf),
        \octave, Pseq([4, 4, 4, 4, 5, 5, 4, 4], inf),
        \dur, 0.25,
        // Map pitch to carrier/target
        \carrierPitch, Pfunc {|ev|
            var midi = (ev[\degree] + 60 + ((ev[\octave] - 4) * 12));
            midi.midicps * 4
        },
        \targetPitch, Pfunc {|ev|
            var midi = (ev[\degree] + 60 + ((ev[\octave] - 4) * 12));
            midi.midicps / 2
        },
        \amp, 0.12,
        \atk, 0.02,
        \dec, 0.15,
        \sus, 0.5,
        \rel, 0.4,
        \t0, 1,
        \t1, Pseg([0.2, 0.7, 0.2], [4, 4], \sin, inf),
        \t2, 0.33,
        \t3, 0.2,
    )
).play;
)

Pdef(\qdtsMelody).stop;

// Ambient pad
(
Pdef(\qdtsPad,
    Pbind(
        \instrument, \qdtsEnv,
        \dur, Pwhite(2.0, 4.0),
        \legato, 1.5,
        \carrierPitch, Pwhite(400, 800),
        \targetPitch, Pwhite(40, 80),
        \amp, 0.08,
        \atk, Pwhite(0.5, 2.0),
        \dec, 0.3,
        \sus, 0.7,
        \rel, Pwhite(1.0, 3.0),
        \t0, 1,
        \t1, Pwhite(0.3, 0.6),
        \t2, Pwhite(0.1, 0.4),
        \t3, Pwhite(0.05, 0.2),
    )
).play;
)

Pdef(\qdtsPad).stop;

// ============================================================================
// 5. LFO-MODULATED SPECTRUM
// ============================================================================

(
SynthDef(\qdtsLFO, {|out = 0, carrierPitch = 1280, targetPitch = 130, amp = 0.2,
    lfoRate = 0.3|

    // LFOs modulate the target spectrum
    var t0 = LFNoise1.kr(lfoRate).range(0.5, 1);
    var t1 = LFNoise1.kr(lfoRate * 0.7).range(0.1, 0.8);
    var t2 = LFNoise1.kr(lfoRate * 0.5).range(0.05, 0.5);
    var t3 = LFNoise1.kr(lfoRate * 0.3).range(0.02, 0.3);

    var amps = QDTSSolver.kr(4, t0, t1, t2, t3);
    var freqs = 5.collect {|i| carrierPitch + (i * targetPitch) };
    var sines = 5.collect {|i| SinOsc.ar(freqs[i]) * amps[i] };

    Out.ar(out, Limiter.ar(sines.sum * amp, 0.9) ! 2);
}).add;
)

y = Synth(\qdtsLFO, [\lfoRate, 0.2]);
y.set(\carrierPitch, 1000, \targetPitch, 80);
y.set(\lfoRate, 0.5);
y.free;

// ============================================================================
// 6. MORE HARMONICS (8 partials)
// ============================================================================

(
SynthDef(\qdts8, {|out = 0, carrierPitch = 1280, targetPitch = 130, amp = 0.15,
    gate = 1, atk = 0.01, dec = 0.1, sus = 0.7, rel = 0.5|

    var numPartials = 8;
    // Sawtooth-like spectrum: 1/n for each harmonic
    var targets = (1..numPartials).collect {|n| (1/n).max(0.01) };

    var env = EnvGen.kr(Env.adsr(atk, dec, sus, rel), gate, doneAction: 2);
    var amps = QDTSSolver.kr(numPartials, *targets);
    var freqs = (numPartials + 1).collect {|i| carrierPitch + (i * targetPitch) };
    var sines = (numPartials + 1).collect {|i| SinOsc.ar(freqs[i]) * amps[i] };

    Out.ar(out, Limiter.ar(sines.sum * env * amp, 0.9) ! 2);
}).add;
)

z = Synth(\qdts8, [\carrierPitch, 800, \targetPitch, 100]);
z.set(\gate, 0);

// ============================================================================
// 7. USING THE QDTS HELPER CLASS
// ============================================================================

// The QDTS class provides a higher-level interface

// Check if it's loaded
QDTS.class;

// Create with presets
(
{
    var qdts = QDTS.sawtooth(8, 440);
    qdts.ar(0.2) ! 2
}.play;
)

(
{
    var qdts = QDTS.square(8, 440);
    qdts.ar(0.2) ! 2
}.play;
)

(
{
    var qdts = QDTS.triangle(8, 440);
    qdts.ar(0.2) ! 2
}.play;
)

// ============================================================================
// 8. FORMANT SHAPING (Vowel-like Sounds)
// ============================================================================

/*
Formants are resonant frequencies that give vowels their characteristic sound.
By shaping the harmonic spectrum, QDTS can approximate vowel qualities.

These are artistic approximations - not precise formant synthesis, but they
capture the essential spectral character of each vowel.
*/

(
SynthDef(\qdtsFormant, {|out = 0, carrierPitch = 1500, targetPitch = 150,
    amp = 0.2, t0 = 1, t1 = 0.5, t2 = 0.33, t3 = 0.25, t4 = 0.2, t5 = 0.16|

    var amps = QDTSSolver.kr(6,
        t0.max(0.01), t1.max(0.01), t2.max(0.01),
        t3.max(0.01), t4.max(0.01), t5.max(0.01)
    );
    var freqs = 7.collect {|i| carrierPitch + (i * targetPitch) };
    var sines = 7.collect {|i| SinOsc.ar(freqs[i]) * amps[i] };

    Out.ar(out, Limiter.ar(sines.sum * amp, 0.9) ! 2);
}).add;
)

// Create synth
x = Synth(\qdtsFormant, [\carrierPitch, 1500, \targetPitch, 150]);

// Vowel presets - execute each line to hear the difference

// "AH" (as in "father") - open, strong low harmonics
x.set(\t0, 1, \t1, 0.9, \t2, 0.5, \t3, 0.3, \t4, 0.15, \t5, 0.1);

// "EE" (as in "see") - bright, emphasis on higher harmonics
x.set(\t0, 1, \t1, 0.3, \t2, 0.2, \t3, 0.5, \t4, 0.7, \t5, 0.4);

// "OO" (as in "boot") - dark, rapid rolloff
x.set(\t0, 1, \t1, 0.4, \t2, 0.1, \t3, 0.05, \t4, 0.02, \t5, 0.01);

// "EH" (as in "bed") - mid-range emphasis
x.set(\t0, 1, \t1, 0.7, \t2, 0.8, \t3, 0.4, \t4, 0.2, \t5, 0.1);

// "OH" (as in "go") - rounded, low-mid emphasis
x.set(\t0, 1, \t1, 0.8, \t2, 0.6, \t3, 0.2, \t4, 0.1, \t5, 0.05);

// "IH" (as in "bit") - between EE and EH
x.set(\t0, 1, \t1, 0.5, \t2, 0.4, \t3, 0.6, \t4, 0.4, \t5, 0.2);

x.free;

// ----------------------------------------
// Animated vowel sequence
// ----------------------------------------
(
SynthDef(\qdtsVowelSeq, {|out = 0, carrierPitch = 1500, targetPitch = 150, amp = 0.2,
    t0 = 1, t1 = 0.5, t2 = 0.33, t3 = 0.25, t4 = 0.2, t5 = 0.16|

    var amps = QDTSSolver.kr(6,
        t0.max(0.01), t1.max(0.01), t2.max(0.01),
        t3.max(0.01), t4.max(0.01), t5.max(0.01)
    );
    var freqs = 7.collect {|i| carrierPitch + (i * targetPitch) };
    var sines = 7.collect {|i| SinOsc.ar(freqs[i]) * amps[i] };

    Out.ar(out, Limiter.ar(sines.sum * amp, 0.9) ! 2);
}).add;
)

// Vowel sequence: AH -> EE -> OO -> EH -> OH (loops)
(
~vowels = [
    [1, 0.9, 0.5, 0.3, 0.15, 0.1],   // AH
    [1, 0.3, 0.2, 0.5, 0.7, 0.4],    // EE
    [1, 0.4, 0.1, 0.05, 0.02, 0.01], // OO
    [1, 0.7, 0.8, 0.4, 0.2, 0.1],    // EH
    [1, 0.8, 0.6, 0.2, 0.1, 0.05]    // OH
];
~vowelNames = ["AH", "EE", "OO", "EH", "OH"];

~vowelSynth = Synth(\qdtsVowelSeq);
~vowelRoutine = fork {
    var idx = 0;
    loop {
        var v = ~vowels[idx];
        (~vowelNames[idx] ++ " vowel").postln;
        ~vowelSynth.set(\t0, v[0], \t1, v[1], \t2, v[2], \t3, v[3], \t4, v[4], \t5, v[5]);
        idx = (idx + 1) % ~vowels.size;
        1.5.wait;  // Hold each vowel for 1.5 seconds
    }
};
)

// Stop vowel sequence
(~vowelRoutine.stop; ~vowelSynth.free;)

// ============================================================================
// 9. BEATING AND ROUGHNESS
// ============================================================================

/*
Two QDTS voices slightly detuned create beating (slow) or roughness (fast).
- Detune < 5 Hz: Slow beating, phasing effect
- Detune 5-15 Hz: Fast beating, tremolo-like
- Detune 15-30 Hz: Roughness, dissonance
- Detune > 30 Hz: Separate pitches emerge
*/

(
SynthDef(\qdtsBeating, {|out = 0, carrierPitch = 1500, targetPitch = 100,
    amp = 0.2, detune = 2|

    // Two QDTS voices with slight detune
    var targets = [1, 0.5, 0.33, 0.25];

    var amps1 = QDTSSolver.kr(4, *targets);
    var amps2 = QDTSSolver.kr(4, *targets);

    var freqs1 = 5.collect {|i| carrierPitch + (i * targetPitch) };
    var freqs2 = 5.collect {|i| (carrierPitch + detune) + (i * targetPitch) };

    var sines1 = 5.collect {|i| SinOsc.ar(freqs1[i]) * amps1[i] };
    var sines2 = 5.collect {|i| SinOsc.ar(freqs2[i]) * amps2[i] };

    var sig = (sines1.sum + sines2.sum) * 0.5;

    Out.ar(out, Limiter.ar(sig * amp, 0.9) ! 2);
}).add;
)

// Slow beating (2 Hz) - gentle phasing
x = Synth(\qdtsBeating, [\detune, 2]);

// Faster beating (8 Hz) - tremolo-like
x.set(\detune, 8);

// Roughness zone (20 Hz) - dissonant, tense
x.set(\detune, 20);

// Very rough (35 Hz) - harsh, two pitches emerging
x.set(\detune, 35);

// Back to gentle
x.set(\detune, 1.5);

x.free;

// ----------------------------------------
// Stereo beating (detuned left/right)
// ----------------------------------------
(
SynthDef(\qdtsStereoBeating, {|out = 0, carrierPitch = 1500, targetPitch = 100,
    amp = 0.2, detune = 3|

    var targets = [1, 0.5, 0.33, 0.25];

    var amps1 = QDTSSolver.kr(4, *targets);
    var amps2 = QDTSSolver.kr(4, *targets);

    // Left channel: base pitch
    var freqsL = 5.collect {|i| carrierPitch + (i * targetPitch) };
    var sinesL = 5.collect {|i| SinOsc.ar(freqsL[i]) * amps1[i] };

    // Right channel: detuned
    var freqsR = 5.collect {|i| (carrierPitch + detune) + (i * targetPitch) };
    var sinesR = 5.collect {|i| SinOsc.ar(freqsR[i]) * amps2[i] };

    var sigL = sinesL.sum;
    var sigR = sinesR.sum;

    Out.ar(out, Limiter.ar([sigL, sigR] * amp, 0.9));
}).add;
)

// Creates a widening stereo effect
y = Synth(\qdtsStereoBeating, [\detune, 3]);
y.set(\detune, 5);   // Wider
y.set(\detune, 0.5); // Subtle
y.free;

// ----------------------------------------
// Beating with different spectra per voice
// ----------------------------------------
(
SynthDef(\qdtsSpectralBeating, {|out = 0, carrierPitch = 1500, targetPitch = 100,
    amp = 0.2, detune = 2|

    // Voice 1: Sawtooth-like
    var amps1 = QDTSSolver.kr(4, 1, 0.5, 0.33, 0.25);
    // Voice 2: Hollow (2nd harmonic dominant)
    var amps2 = QDTSSolver.kr(4, 0.3, 1, 0.2, 0.1);

    var freqs1 = 5.collect {|i| carrierPitch + (i * targetPitch) };
    var freqs2 = 5.collect {|i| (carrierPitch + detune) + (i * targetPitch) };

    var sines1 = 5.collect {|i| SinOsc.ar(freqs1[i]) * amps1[i] };
    var sines2 = 5.collect {|i| SinOsc.ar(freqs2[i]) * amps2[i] };

    var sig = (sines1.sum + sines2.sum) * 0.5;

    Out.ar(out, Limiter.ar(sig * amp, 0.9) ! 2);
}).add;
)

// Two different timbres beating against each other
z = Synth(\qdtsSpectralBeating, [\detune, 2]);
z.set(\detune, 5);
z.free;

// ============================================================================
// 10. SPECTRAL ENVELOPE (Per-harmonic dynamics)
// ============================================================================

/*
Different attack/decay times per harmonic creates evolving timbres.
- Fast attack on fundamentals, slow on upper harmonics = "bloom" effect
- Slow attack on all = pad-like
- Fast decay on upper harmonics = natural instrument-like decay
*/

(
SynthDef(\qdtsSpectralEnv, {|out = 0, carrierPitch = 1500, targetPitch = 100,
    amp = 0.3, gate = 1|

    // Different envelope times per harmonic
    var atkTimes = [0.01, 0.05, 0.1, 0.2, 0.3];   // Fundamental fastest
    var relTimes = [0.8, 0.6, 0.4, 0.3, 0.2];     // Upper harmonics decay faster

    var envs = 5.collect {|i|
        EnvGen.kr(Env.perc(atkTimes[i], relTimes[i]), gate)
    };

    // Static target spectrum
    var targets = [1, 0.6, 0.4, 0.25, 0.15];
    var amps = QDTSSolver.kr(5, *targets);

    var freqs = 6.collect {|i| carrierPitch + (i * targetPitch) };
    // Apply per-harmonic envelopes
    var sines = 6.collect {|i| SinOsc.ar(freqs[i]) * amps[i] * envs[i.min(4)] };

    Out.ar(out, Limiter.ar(sines.sum * amp, 0.9) ! 2);
}).add;
)

// Trigger notes - upper harmonics bloom in after attack
Synth(\qdtsSpectralEnv, [\carrierPitch, 1200, \targetPitch, 100]);
Synth(\qdtsSpectralEnv, [\carrierPitch, 1500, \targetPitch, 120]);
Synth(\qdtsSpectralEnv, [\carrierPitch, 1800, \targetPitch, 80]);

// ----------------------------------------
// Reverse spectral envelope (upper harmonics attack first)
// ----------------------------------------
(
SynthDef(\qdtsReverseEnv, {|out = 0, carrierPitch = 1500, targetPitch = 100,
    amp = 0.3, gate = 1|

    // Upper harmonics attack first, fundamental last
    var atkTimes = [0.3, 0.2, 0.1, 0.05, 0.01];
    var relTimes = [0.5, 0.5, 0.5, 0.5, 0.5];

    var envs = 5.collect {|i|
        EnvGen.kr(Env.perc(atkTimes[i], relTimes[i]), gate)
    };

    var targets = [1, 0.6, 0.4, 0.25, 0.15];
    var amps = QDTSSolver.kr(5, *targets);

    var freqs = 6.collect {|i| carrierPitch + (i * targetPitch) };
    var sines = 6.collect {|i| SinOsc.ar(freqs[i]) * amps[i] * envs[i.min(4)] };

    Out.ar(out, Limiter.ar(sines.sum * amp, 0.9) ! 2);
}).add;
)

// "Reverse bloom" - brightness arrives before body
Synth(\qdtsReverseEnv, [\carrierPitch, 1400, \targetPitch, 110]);

// ----------------------------------------
// Sustained with spectral shimmer
// ----------------------------------------
(
SynthDef(\qdtsShimmer, {|out = 0, carrierPitch = 1500, targetPitch = 100,
    amp = 0.2, gate = 1, atk = 0.3, rel = 1.0|

    var mainEnv = EnvGen.kr(Env.asr(atk, 1, rel), gate, doneAction: 2);

    // Each harmonic has slight independent movement
    var shimmer = 5.collect {|i|
        LFNoise1.kr(0.5 + (i * 0.2)).range(0.7, 1.0)
    };

    var targets = [1, 0.5, 0.33, 0.25, 0.2];
    var amps = QDTSSolver.kr(5, *targets);

    var freqs = 6.collect {|i| carrierPitch + (i * targetPitch) };
    var sines = 6.collect {|i| SinOsc.ar(freqs[i]) * amps[i] * shimmer[i.min(4)] };

    Out.ar(out, Limiter.ar(sines.sum * mainEnv * amp, 0.9) ! 2);
}).add;
)

// Sustained shimmering tone
x = Synth(\qdtsShimmer);
x.set(\gate, 0);  // Release

// ----------------------------------------
// Pattern: Blooming melody
// ----------------------------------------
(
Pdef(\bloomMelody,
    Pbind(
        \type, \on,
        \instrument, \qdtsSpectralEnv,
        \dur, 0.5,
        \carrierPitch, Pseq([1200, 1350, 1500, 1350, 1200, 1100, 1200, 1350], inf),
        \targetPitch, Pseq([100, 110, 120, 110, 100, 90, 100, 110], inf),
        \amp, 0.25
    )
).play;
)

Pdef(\bloomMelody).stop;

// ----------------------------------------
// Pattern: Reverse bloom chords
// ----------------------------------------
(
Pdef(\reverseChords,
    Pbind(
        \type, \on,
        \instrument, \qdtsReverseEnv,
        \dur, 1.0,
        \carrierPitch, Pseq([
            [1200, 1500, 1800],  // Chord 1
            [1100, 1400, 1700],  // Chord 2
            [1300, 1600, 1900],  // Chord 3
            [1000, 1300, 1600]   // Chord 4
        ], inf).flatten,
        \targetPitch, 100,
        \amp, 0.2,
        \strum, 0.05  // Slight strum effect
    )
).play;
)

Pdef(\reverseChords).stop;

// ----------------------------------------
// Pattern: Random spectral blooms
// ----------------------------------------
(
Pdef(\randomBlooms,
    Pbind(
        \type, \on,
        \instrument, \qdtsSpectralEnv,
        \dur, Pwhite(0.3, 0.8),
        \carrierPitch, Pwhite(1000, 2000),
        \targetPitch, Pwhite(80, 150),
        \amp, 0.2
    )
).play;
)

Pdef(\randomBlooms).stop;

// ============================================================================
// 11. GALLOPING RHYTHM (Tempo-dependent streaming)
// ============================================================================

/*
Classic auditory scene analysis demo: ABA_ABA_ pattern
- At slow tempos: hear integrated "gallop" rhythm (da-da-DUM)
- At fast tempos: segregates into two separate streams (A-A-A... and B...B...)

Based on: Bregman, A.S. (1990). Auditory Scene Analysis. MIT Press.
*/

(
SynthDef(\qdtsGallop, {|out = 0, carrierPitch = 1000, targetPitch = 100,
    amp = 0.15, atk = 0.005, rel = 0.1|

    var env = EnvGen.kr(Env.perc(atk, rel), doneAction: 2);
    var amps = QDTSSolver.kr(4, 1, 0.5, 0.33, 0.25);
    var freqs = 5.collect {|i| carrierPitch + (i * targetPitch) };
    var sines = 5.collect {|i| SinOsc.ar(freqs[i]) * amps[i] };

    Out.ar(out, Limiter.ar(sines.sum * env * amp, 0.9) ! 2);
}).add;
)

// ----------------------------------------
// Slow gallop - hear as integrated ABA rhythm
// ----------------------------------------
(
~gallopA = 1000;  // Low pitch
~gallopB = 1600;  // High pitch

~gallopSlow = fork {
    loop {
        Synth(\qdtsGallop, [\carrierPitch, ~gallopA, \rel, 0.15]);
        0.3.wait;
        Synth(\qdtsGallop, [\carrierPitch, ~gallopB, \rel, 0.15]);
        0.3.wait;
        Synth(\qdtsGallop, [\carrierPitch, ~gallopA, \rel, 0.15]);
        0.3.wait;
        0.3.wait;  // Rest
    }
};
)

~gallopSlow.stop;

// ----------------------------------------
// Fast gallop - segregates into two streams
// ----------------------------------------
(
~gallopFast = fork {
    loop {
        Synth(\qdtsGallop, [\carrierPitch, ~gallopA, \rel, 0.06]);
        0.1.wait;
        Synth(\qdtsGallop, [\carrierPitch, ~gallopB, \rel, 0.06]);
        0.1.wait;
        Synth(\qdtsGallop, [\carrierPitch, ~gallopA, \rel, 0.06]);
        0.1.wait;
        0.1.wait;  // Rest
    }
};
)

~gallopFast.stop;

// ----------------------------------------
// Accelerating gallop - hear the transition
// ----------------------------------------
(
~gallopAccel = fork {
    var dur = 0.35;
    100.do {
        Synth(\qdtsGallop, [\carrierPitch, ~gallopA, \rel, dur * 0.5]);
        dur.wait;
        Synth(\qdtsGallop, [\carrierPitch, ~gallopB, \rel, dur * 0.5]);
        dur.wait;
        Synth(\qdtsGallop, [\carrierPitch, ~gallopA, \rel, dur * 0.5]);
        dur.wait;
        dur.wait;  // Rest
        dur = (dur * 0.97).max(0.08);  // Gradually speed up
        ("Tempo: " ++ (60 / dur / 4).round(1) ++ " BPM").postln;
    };
    "Done - did you hear the streams separate?".postln;
};
)

~gallopAccel.stop;

// ============================================================================
// 12. PITCH GLIDE (Carrier sweeps)
// ============================================================================

/*
Sweeping the carrier pitch while maintaining spectral shape creates
glissando effects. The perceived pitch follows the target pitch spacing,
while the carrier determines the register/brightness.
*/

(
SynthDef(\qdtsGlide, {|out = 0, carrierStart = 1000, carrierEnd = 2000,
    targetPitch = 100, amp = 0.2, dur = 2|

    var carrierPitch = Line.kr(carrierStart, carrierEnd, dur, doneAction: 2);
    var env = EnvGen.kr(Env.linen(0.05, dur - 0.1, 0.05));

    var amps = QDTSSolver.kr(4, 1, 0.5, 0.33, 0.25);
    var freqs = 5.collect {|i| carrierPitch + (i * targetPitch) };
    var sines = 5.collect {|i| SinOsc.ar(freqs[i]) * amps[i] };

    Out.ar(out, Limiter.ar(sines.sum * env * amp, 0.9) ! 2);
}).add;
)

// Rising glide
Synth(\qdtsGlide, [\carrierStart, 800, \carrierEnd, 2000, \dur, 2]);

// Falling glide
Synth(\qdtsGlide, [\carrierStart, 2500, \carrierEnd, 800, \dur, 3]);

// Fast sweep
Synth(\qdtsGlide, [\carrierStart, 1000, \carrierEnd, 3000, \dur, 0.5]);

// ----------------------------------------
// Exponential glide (more musical)
// ----------------------------------------
(
SynthDef(\qdtsExpGlide, {|out = 0, carrierStart = 1000, carrierEnd = 2000,
    targetPitch = 100, amp = 0.2, dur = 2|

    var carrierPitch = XLine.kr(carrierStart, carrierEnd, dur, doneAction: 2);
    var env = EnvGen.kr(Env.linen(0.05, dur - 0.1, 0.05));

    var amps = QDTSSolver.kr(4, 1, 0.5, 0.33, 0.25);
    var freqs = 5.collect {|i| carrierPitch + (i * targetPitch) };
    var sines = 5.collect {|i| SinOsc.ar(freqs[i]) * amps[i] };

    Out.ar(out, Limiter.ar(sines.sum * env * amp, 0.9) ! 2);
}).add;
)

// Exponential rise (octave glide)
Synth(\qdtsExpGlide, [\carrierStart, 1000, \carrierEnd, 2000, \dur, 2]);

// ----------------------------------------
// Vibrato (pitch oscillation)
// ----------------------------------------
(
SynthDef(\qdtsVibrato, {|out = 0, carrierPitch = 1500, targetPitch = 100,
    amp = 0.2, vibRate = 5, vibDepth = 30|

    var vibrato = SinOsc.kr(vibRate) * vibDepth;
    var pitch = carrierPitch + vibrato;

    var amps = QDTSSolver.kr(4, 1, 0.5, 0.33, 0.25);
    var freqs = 5.collect {|i| pitch + (i * targetPitch) };
    var sines = 5.collect {|i| SinOsc.ar(freqs[i]) * amps[i] };

    Out.ar(out, Limiter.ar(sines.sum * amp, 0.9) ! 2);
}).add;
)

// Gentle vibrato
x = Synth(\qdtsVibrato, [\vibRate, 5, \vibDepth, 20]);

// Wide, slow vibrato
x.set(\vibRate, 3, \vibDepth, 50);

// Fast, narrow vibrato (shimmer)
x.set(\vibRate, 8, \vibDepth, 10);

x.free;

// ----------------------------------------
// Pattern: Gliding melody
// ----------------------------------------
(
Pdef(\glideMelody,
    Pbind(
        \type, \on,
        \instrument, \qdtsExpGlide,
        \dur, 0.8,
        \carrierStart, Pseq([1000, 1200, 1400, 1200], inf),
        \carrierEnd, Pseq([1200, 1400, 1600, 1000], inf),
        \targetPitch, 100,
        \amp, 0.2
    )
).play;
)

Pdef(\glideMelody).stop;

// ----------------------------------------
// Siren effect (continuous up/down)
// ----------------------------------------
(
SynthDef(\qdtsSiren, {|out = 0, carrierLow = 800, carrierHigh = 2000,
    targetPitch = 100, amp = 0.2, rate = 0.5|

    var carrierPitch = SinOsc.kr(rate).range(carrierLow, carrierHigh);

    var amps = QDTSSolver.kr(4, 1, 0.5, 0.33, 0.25);
    var freqs = 5.collect {|i| carrierPitch + (i * targetPitch) };
    var sines = 5.collect {|i| SinOsc.ar(freqs[i]) * amps[i] };

    Out.ar(out, Limiter.ar(sines.sum * amp, 0.9) ! 2);
}).add;
)

// Slow siren
x = Synth(\qdtsSiren, [\rate, 0.3]);

// Faster siren
x.set(\rate, 1);

// Ambulance-style
x.set(\rate, 2, \carrierLow, 1200, \carrierHigh, 1800);

x.free;

// ============================================================================
// 13. TIPS AND TROUBLESHOOTING
// ============================================================================

/*
IMPORTANT NOTES:

1. TARGET VALUE RANGE
   Target values (t0, t1, t2...) represent RELATIVE amplitude ratios.
   - Minimum: > 0 (use 0.01 as safe minimum to avoid instability)
   - Maximum: No strict limit, but typical range is 0.01 to 2.0
   - Values can exceed 1.0 (e.g., t1=1.5 means 2nd harmonic is 1.5x the 1st)
   Always clamp: target.max(0.01) or target.clip(0.01, 2.0)

   Common spectra (fundamental dominant):
   - [1, 0.5, 0.33, 0.25]  - Sawtooth (1/n rolloff)
   - [1, 0.01, 0.5, 0.01]  - Square-ish (odd harmonics)
   - [1, 1, 1, 1]          - Equal harmonics (bright)
   - [1, 0.1, 0.05, 0.02]  - Mellow/sine-like

   Non-standard spectra (upper harmonic dominant):
   - [0.2, 1, 0.1, 0.1]    - 2nd harmonic dominant (hollow)
   - [0.1, 0.1, 1, 0.1]    - 3rd harmonic dominant (nasal)
   - [0.3, 0.3, 0.3, 1]    - 4th harmonic dominant
   - [0.5, 0.8, 1, 0.6]    - Peak at 3rd, gradual rolloff

2. MONITOR ERROR OUTPUT
   The last output of QDTSSolver is the estimation error.
   Values < 0.001 indicate good convergence.

   {
       var amps = QDTSSolver.kr(4, 1, 0.5, 0.33, 0.25);
       var error = amps[5]; // Last output
       error.poll;
       Silent.ar;
   }.play;

3. USE LIMITER
   For safety, wrap output in Limiter.ar(signal, 0.9)

4. CARRIER vs TARGET PITCH
   - Carrier: The base frequency (optimal range: 1-5 kHz)
   - Target: The difference between successive partials (the perceived pitch)
   - Frequencies generated: carrier, carrier+target, carrier+2*target, etc.

5. NUMBER OF HARMONICS
   More harmonics = richer sound but more CPU
   Maximum supported: 16 harmonics
*/

// Cleanup
Pdef.all.do(_.stop);
