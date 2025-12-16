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
// Vowel morphing with mouse control
// ----------------------------------------
// X-axis: EE (left) to OO (right)
// Interpolates between two stable vowel presets

(
SynthDef(\qdtsVowelMorph, {|out = 0, carrierPitch = 1500, targetPitch = 150, amp = 0.2|
    // Heavy smoothing on mouse to prevent solver instability
    var mx = Lag2.kr(MouseX.kr(0, 1), 0.5);

    // EE vowel (bright): [1, 0.3, 0.2, 0.5, 0.7, 0.4]
    // OO vowel (dark):   [1, 0.4, 0.15, 0.1, 0.08, 0.05]
    // Simple crossfade between two stable presets
    var t0 = 1;
    var t1 = LinLin.kr(mx, 0, 1, 0.3, 0.4);
    var t2 = LinLin.kr(mx, 0, 1, 0.2, 0.15);
    var t3 = LinLin.kr(mx, 0, 1, 0.5, 0.1);
    var t4 = LinLin.kr(mx, 0, 1, 0.7, 0.08);
    var t5 = LinLin.kr(mx, 0, 1, 0.4, 0.05);

    // Smooth the solver outputs too
    var amps = QDTSSolver.kr(6, t0, t1, t2, t3, t4, t5);
    var smoothAmps = 7.collect {|i| Lag.kr(amps[i], 0.1) };

    var freqs = 7.collect {|i| carrierPitch + (i * targetPitch) };
    var sines = 7.collect {|i| SinOsc.ar(freqs[i]) * smoothAmps[i] };

    Out.ar(out, Limiter.ar(sines.sum * amp, 0.9) ! 2);
}).add;
)

// Move mouse left-right: EE (bright) <-> OO (dark)
y = Synth(\qdtsVowelMorph);
y.free;

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
// 9. TIPS AND TROUBLESHOOTING
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
