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
// 8. TIPS AND TROUBLESHOOTING
// ============================================================================

/*
IMPORTANT NOTES:

1. TARGET VALUE RANGE
   Target values (t0, t1, t2...) represent RELATIVE amplitude ratios.
   - Minimum: > 0 (use 0.01 as safe minimum to avoid instability)
   - Maximum: No strict limit, but typical range is 0.01 to 2.0
   - Values can exceed 1.0 (e.g., t1=1.5 means 2nd harmonic is 1.5x the 1st)
   Always clamp: target.max(0.01) or target.clip(0.01, 2.0)

   Common spectra:
   - [1, 0.5, 0.33, 0.25]  - Sawtooth (1/n rolloff)
   - [1, 0.01, 0.5, 0.01]  - Square-ish (odd harmonics)
   - [0.5, 1, 0.5, 0.25]   - 2nd harmonic dominant
   - [1, 1, 1, 1]          - Equal harmonics (bright)
   - [1, 0.1, 0.05, 0.02]  - Mellow/sine-like

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

// ============================================================================
// 9. BREGMAN AUDITORY SCENE ANALYSIS EXAMPLES
// ============================================================================

/*
Bregman's Auditory Scene Analysis (ASA) describes how we perceptually organize
sound into coherent streams. QDTS is particularly interesting for ASA because
the phantom ADPs can either fuse with carriers or segregate into distinct streams.

Key ASA principles demonstrated below:
- Stream segregation vs. fusion (frequency proximity)
- Sequential streaming (tempo-dependent grouping)
- Common fate (synchronized changes bind sounds)
- Harmonic capture (ADPs fusing with carrier complex)

Reference: Bregman, A.S. (1990). Auditory Scene Analysis. MIT Press.
*/

// ----------------------------------------
// 9a. STREAM SEGREGATION: Two QDTS sources
// ----------------------------------------
// When carrier frequencies are far apart, we hear two separate streams.
// When close together, they fuse into one complex sound.

(
SynthDef(\qdtsStream, {|out = 0, carrierPitch = 1000, targetPitch = 100,
    amp = 0.15, pan = 0, gate = 1, atk = 0.005, rel = 0.1|

    var env = EnvGen.kr(Env.perc(atk, rel), gate, doneAction: 2);
    var amps = QDTSSolver.kr(4, 1, 0.5, 0.33, 0.25);
    var freqs = 5.collect {|i| carrierPitch + (i * targetPitch) };
    var sines = 5.collect {|i| SinOsc.ar(freqs[i]) * amps[i] };

    Out.ar(out, Pan2.ar(Limiter.ar(sines.sum * env * amp, 0.9), pan));
}).add;
)

// SEGREGATED: Two distinct streams (carriers far apart: 800 Hz vs 2400 Hz)
// You should hear HIGH and LOW as separate melodic lines
(
Pdef(\asaSegregated,
    Ppar([
        // Low stream
        Pbind(
            \type, \on,  // Don't schedule release (synth self-frees)
            \instrument, \qdtsStream,
            \targetPitch, 80,
            \dur, 0.25,
            \amp, 0.12,
            \pan, -0.5,
            \carrierPitch, Pseq([800, 840, 880, 900], inf)
        ),
        // High stream
        Pbind(
            \type, \on,
            \instrument, \qdtsStream,
            \targetPitch, 120,
            \dur, 0.25,
            \amp, 0.10,
            \pan, 0.5,
            \carrierPitch, Pseq([2610, 2550, 2520, 2460], inf)
        )
    ])
).play;
)

Pdef(\asaSegregated).stop;

// FUSED: Single complex stream (carriers close: 1000 Hz vs 1200 Hz)
// The two sources blend into one rich, complex timbre
(
Pdef(\asaFused,
    Ppar([
        Pbind(
            \type, \on,
            \instrument, \qdtsStream,
            \targetPitch, 100,
            \dur, 0.25,
            \amp, 0.10,
            \pan, -0.3,
            \carrierPitch, Pseq([1000, 1030, 1060, 1075], inf)
        ),
        Pbind(
            \type, \on,
            \instrument, \qdtsStream,
            \targetPitch, 100,
            \dur, 0.25,
            \amp, 0.10,
            \pan, 0.3,
            \carrierPitch, Pseq([1200, 1230, 1260, 1275], inf)  // Close to first stream
        )
    ])
).play;
)

Pdef(\asaFused).stop;

// ----------------------------------------
// 9b. GALLOPING RHYTHM: Tempo-dependent streaming
// ----------------------------------------
// Classic ASA demo: ABA_ABA_ pattern
// At slow tempos: hear integrated gallop (da-da-DUM)
// At fast tempos: hear two separate streams (A-A-A... and B...B...)

(
SynthDef(\qdtsGallop, {|out = 0, carrierPitch = 1000, targetPitch = 100,
    amp = 0.15, gate = 1, atk = 0.005, rel = 0.08|

    var env = EnvGen.kr(Env.perc(atk, rel), gate, doneAction: 2);
    var amps = QDTSSolver.kr(4, 1, 0.6, 0.35, 0.2);
    var freqs = 5.collect {|i| carrierPitch + (i * targetPitch) };
    var sines = 5.collect {|i| SinOsc.ar(freqs[i]) * amps[i] };

    Out.ar(out, Limiter.ar(sines.sum * env * amp, 0.9) ! 2);
}).add;
)

// Slow gallop - should hear integrated ABA pattern
(
Pdef(\gallopSlow,
    Pbind(
        \type, \on,
        \instrument, \qdtsGallop,
        \dur, 0.3,  // Slow tempo promotes integration
        \carrierPitch, Pseq([1000, 1600, 1000, \rest], inf),  // A B A _
        \targetPitch, 100,
        \amp, 0.12,
        \rel, 0.15
    )
).play;
)

Pdef(\gallopSlow).stop;

// Fast gallop - streams segregate into A-A-A and B-B-B
(
Pdef(\gallopFast,
    Pbind(
        \type, \on,
        \instrument, \qdtsGallop,
        \dur, 0.1,  // Fast tempo promotes segregation
        \carrierPitch, Pseq([1000, 1600, 1000, \rest], inf),  // A B A _
        \targetPitch, 100,
        \amp, 0.12,
        \rel, 0.06
    )
).play;
)

Pdef(\gallopFast).stop;

// Interactive: Gradually speed up to hear transition from fusion to segregation
(
Pdef(\gallopAccel,
    Pbind(
        \type, \on,
        \instrument, \qdtsGallop,
        \dur, Pseg([0.35, 0.08], [15], \exp),  // Accelerate over 15 seconds
        \carrierPitch, Pseq([1000, 1600, 1000, \rest], inf),
        \targetPitch, 100,
        \amp, 0.12,
        \rel, Pkey(\dur) * 0.6
    )
).play;
)

Pdef(\gallopAccel).stop;

// ----------------------------------------
// 9c. COMMON FATE: Synchronized spectral change
// ----------------------------------------
// Sounds that change together (common fate) are grouped as one source.
// Here, two QDTS voices with synchronized spectral modulation fuse,
// while asynchronous modulation causes segregation.

(
SynthDef(\qdtsCommonFate, {|out = 0, carrierPitch = 1000, targetPitch = 100,
    amp = 0.15, pan = 0, t1mod = 0.5, t2mod = 0.3|

    // Spectrum controlled by arguments (can be synchronized or not)
    var t0 = 1;
    var t1 = t1mod.max(0.01);
    var t2 = t2mod.max(0.01);
    var t3 = 0.2;

    var amps = QDTSSolver.kr(4, t0, t1, t2, t3);
    var freqs = 5.collect {|i| carrierPitch + (i * targetPitch) };
    var sines = 5.collect {|i| SinOsc.ar(freqs[i]) * amps[i] };

    Out.ar(out, Pan2.ar(Limiter.ar(sines.sum * amp, 0.9), pan));
}).add;
)

// COMMON FATE: Both voices modulate spectrum in sync → perceived as ONE source
(
~commonFate1 = Synth(\qdtsCommonFate, [\carrierPitch, 1000, \targetPitch, 100, \pan, -0.4]);
~commonFate2 = Synth(\qdtsCommonFate, [\carrierPitch, 1400, \targetPitch, 100, \pan, 0.4]);

// Update both with same modulation (language-side LFO)
~commonFateRoutine = fork {
    var phase = 0;
    loop {
        var mod = sin(phase).linlin(-1, 1, 0.1, 0.8);  // Triangle-ish wave
        ~commonFate1.set(\t1mod, mod, \t2mod, mod * 0.6);
        ~commonFate2.set(\t1mod, mod, \t2mod, mod * 0.6);
        phase = phase + 0.03;  // ~0.3 Hz at 50ms update rate
        0.05.wait;
    }
}
)

// Stop common fate demo
(~commonFateRoutine.stop; ~commonFate1.free; ~commonFate2.free;)

// INDEPENDENT FATE: Different modulation → perceived as TWO sources
(
~indep1 = Synth(\qdtsCommonFate, [\carrierPitch, 1000, \targetPitch, 100, \pan, -0.4]);
~indep2 = Synth(\qdtsCommonFate, [\carrierPitch, 1400, \targetPitch, 100, \pan, 0.4]);

// Update with different modulation rates
~indepRoutine = fork {
    var phase1 = 0, phase2 = 0;
    loop {
        phase1 = phase1 + 0.02;
        phase2 = phase2 + 0.035;  // Different rate
        ~indep1.set(\t1mod, sin(phase1).linlin(-1, 1, 0.1, 0.8));
        ~indep2.set(\t1mod, sin(phase2).linlin(-1, 1, 0.1, 0.8));
        0.05.wait;
    }
}
)

// Stop independent fate demo
(~indepRoutine.stop; ~indep1.free; ~indep2.free;)

// ----------------------------------------
// 9d. HARMONIC SIEVE: Capturing stray partials
// ----------------------------------------
// A harmonic complex can "capture" nearby frequencies, pulling them
// into the percept. QDTS ADPs demonstrate this by creating phantom
// tones that fuse with the carrier harmonics.

(
SynthDef(\qdtsCapture, {|out = 0, carrierPitch = 1000, targetPitch = 100,
    amp = 0.15, detuneAmt = 0|

    var amps = QDTSSolver.kr(4, 1, 0.5, 0.33, 0.25);
    // Slight detuning breaks harmonic relationship
    var freqs = 5.collect {|i| carrierPitch + (i * targetPitch) + (i * detuneAmt) };
    var sines = 5.collect {|i| SinOsc.ar(freqs[i]) * amps[i] };

    Out.ar(out, Limiter.ar(sines.sum * amp, 0.9) ! 2);
}).add;
)

// Pure harmonic - all partials fuse into single pitch percept
x = Synth(\qdtsCapture, [\carrierPitch, 1000, \targetPitch, 100, \detuneAmt, 0]);

// Slight detune - still fuses (captured by harmonic template)
x.set(\detuneAmt, 3);

// More detune - starts to separate (roughness, beating)
x.set(\detuneAmt, 8);

// Strong detune - clearly separate components
x.set(\detuneAmt, 15);

x.free;

// ----------------------------------------
// 9e. CONTINUITY ILLUSION with QDTS
// ----------------------------------------
// Alternating QDTS tone with noise bursts.
// At right levels, the tone seems to continue "through" the noise.

(
SynthDef(\qdtsContinuity, {|out = 0, carrierPitch = 1200, targetPitch = 80,
    amp = 0.15, noiseAmp = 0.3, gapDur = 0.15|

    var toneEnv = LFPulse.kr(1 / (gapDur * 2), 0, 0.5);
    var noiseEnv = 1 - toneEnv;

    var amps = QDTSSolver.kr(4, 1, 0.5, 0.33, 0.25);
    var freqs = 5.collect {|i| carrierPitch + (i * targetPitch) };
    var tone = 5.collect {|i| SinOsc.ar(freqs[i]) * amps[i] }.sum;

    var noise = PinkNoise.ar;

    var sig = (tone * toneEnv * amp) + (noise * noiseEnv * noiseAmp);

    Out.ar(out, Limiter.ar(sig, 0.9) ! 2);
}).add;
)

// The QDTS tone appears to continue through the noise gaps
y = Synth(\qdtsContinuity, [\noiseAmp, 0.25, \gapDur, 0.12]);

// Adjust noise level - too quiet breaks the illusion
y.set(\noiseAmp, 0.1);  // Illusion weakens
y.set(\noiseAmp, 0.3);  // Illusion restored

// Longer gaps make illusion harder to maintain
y.set(\gapDur, 0.25);

y.free;

// ----------------------------------------
// 9f. MELODIC STREAMING: Van Noorden boundaries
// ----------------------------------------
// Demonstrates the temporal coherence and fission boundaries
// Two interleaved melodies that segregate based on pitch distance

(
SynthDef(\qdtsMelStream, {|out = 0, carrierPitch = 1000, targetPitch = 80,
    amp = 0.12, atk = 0.005, rel = 0.1, gate = 1|

    var env = EnvGen.kr(Env.perc(atk, rel), gate, doneAction: 2);
    var amps = QDTSSolver.kr(4, 1, 0.5, 0.33, 0.25);
    var freqs = 5.collect {|i| carrierPitch + (i * targetPitch) };
    var sines = 5.collect {|i| SinOsc.ar(freqs[i]) * amps[i] };

    Out.ar(out, Limiter.ar(sines.sum * env * amp, 0.9) ! 2);
}).add;
)

// Small pitch interval - hear as single integrated melody
(
Pdef(\streamNarrow,
    Pbind(
        \type, \on,
        \instrument, \qdtsMelStream,
        \dur, 0.15,
        // Alternating between two pitch levels (small interval)
        \carrierPitch, Pseq([1000, 1100, 1000, 1100, 1000, 1100, 1200, 1100], inf),
        \targetPitch, 80,
        \amp, 0.1
    )
).play;
)

Pdef(\streamNarrow).stop;

// Large pitch interval - segregates into two streams
(
Pdef(\streamWide,
    Pbind(
        \type, \on,
        \instrument, \qdtsMelStream,
        \dur, 0.15,
        // Alternating between two pitch levels (large interval)
        \carrierPitch, Pseq([800, 1800, 800, 1800, 800, 1800, 900, 1800], inf),
        \targetPitch, 80,
        \amp, 0.1
    )
).play;
)

Pdef(\streamWide).stop;

// Gradually increasing interval - hear transition from integration to segregation
(
Pdef(\streamExpand,
    Pbind(
        \type, \on,
        \instrument, \qdtsMelStream,
        \dur, 0.15,
        \carrierPitch, Pfunc({
            var base = 1000;
            var interval = ~streamInterval ? 100;
            [base, base + interval].choose
        }),
        \targetPitch, 80,
        \amp, 0.1
    )
).play;

// Slowly increase interval
~streamExpandRoutine = fork {
    ~streamInterval = 100;
    100.do {|i|
        ~streamInterval = 100 + (i * 10);
        ("Interval: " ++ ~streamInterval ++ " Hz").postln;
        1.wait;
    }
}
)

Pdef(\streamExpand).stop;

// Cleanup
Pdef.all.do(_.stop);
