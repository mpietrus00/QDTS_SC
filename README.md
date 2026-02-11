# QDTS_SC - Quadratic Distortion Tone Spectra for SuperCollider

A SuperCollider UGen plugin for synthesizing sounds using Auditory Distortion Products (ADPs).

## Overview

QDTS_SC implements a solver for a system of equations related to the synthesis of Auditory Distortion Products (ADPs). The method allows for controlled generation of quadratic difference tone spectra, enabling resynthesis of target timbres as auditory illusions.

Based on:

> Gutiérrez, E., Haworth, C., and Cádiz, R. (2024). *Generating Sonic Phantoms with Quadratic Difference Tone Spectrum Synthesis*. Computer Music Journal 47(3):1-16.

> Kendall, G.S., Haworth, C., and Cádiz, R.F. (2014). *Sound Synthesis with Auditory Distortion Products*. Computer Music Journal 38(4).

## Features

- **QDTSSolver UGen**: Real-time Newton's method solver running on the server
- **Dynamic spectrum control**: Modulate target harmonics in real-time
- **Normalized output**: Amplitudes are normalized for safe mixing
- **Error output**: Monitor solver convergence quality

## Installation

1. Download the latest release for your platform from the [Releases](https://github.com/mpietrus00/QDTS_SC/releases) page

2. Copy the `QDTS_SC` folder to your SuperCollider Extensions directory:
   - **macOS**: `~/Library/Application Support/SuperCollider/Extensions/`
   - **Linux**: `~/.local/share/SuperCollider/Extensions/`
   - **Windows**: `%USERPROFILE%\AppData\Local\SuperCollider\Extensions\`

3. Recompile the class library: `Language → Recompile Class Library` (or Cmd/Ctrl+Shift+L)

4. Reboot the server

## Quick Start

```supercollider
// Boot server
s.boot;

// Basic test
(
{
    var targets = [1, 0.5, 0.33, 0.25];
    var amps = QDTSSolver.kr(4, *targets);
    var carrierPitch = 1000;
    var targetPitch = 100;
    var freqs = 5.collect {|i| carrierPitch + (i * targetPitch) };
    var sines = 5.collect {|i| SinOsc.ar(freqs[i]) * amps[i] };
    sines.sum * 0.2 ! 2
}.play;
)
```

## Usage

### QDTSSolver UGen

```supercollider
QDTSSolver.kr(numHarmonics, *targets)
```

**Inputs:**
- `numHarmonics`: Number of target harmonics (1-16)
- `targets`: Array of target harmonic amplitudes

**Outputs:** Array of `numHarmonics + 2` values:
- `[0]` to `[numHarmonics]`: Solved amplitudes (normalized)
- `[numHarmonics + 1]`: Estimation error (lower = better convergence)

### QDTS Helper Class

```supercollider
// Create with presets
~qdts = QDTS.sawtooth(8, 440);
~qdts = QDTS.square(8, 440);
~qdts = QDTS.triangle(8, 440);

// Custom targets
~qdts = QDTS(8, 440);
~qdts.targets = [1, 0.5, 0.33, 0.25, 0.2, 0.166, 0.142, 0.125];

// Synthesize
{ ~qdts.ar(0.2) }.play;
```

## Important Notes

### Target Values

**Avoid setting targets to exactly 0** - this can cause solver instability. Always use a small minimum value:

```supercollider
var t0s = t0.max(0.01);
var t1s = t1.max(0.01);
```

### Carrier and Target Pitch

- **Carrier pitch**: Base frequency, should be in **1-5 kHz range** (typically ~2.5kHz)
- **Target pitch**: Difference frequency (the perceived pitch)
- **Resulting frequencies**: carrier, carrier+target, carrier+2*target, ...

### Perceptual Notes

- **Max 16 harmonics**: More than 16 QDT harmonics unlikely to be effective
- **Continuous sounds**: Work better than transients for perceiving QDTs
- **Loudspeakers preferred**: QDTs easier to hear over speakers than headphones
- **Fatigue**: Keep durations under 2 minutes at high levels

## Examples

See `examples/QDTS_examples.scd` for comprehensive usage examples including:
- Basic synthesis and SynthDefs
- Pattern integration (Pbind, Pdef)
- Real-time parameter control
- LFO-modulated spectrum
- Formant shaping (vowel-like sounds)
- Beating and roughness effects
- Spectral envelope (per-harmonic dynamics)
- Galloping rhythm patterns
- Pitch glide and vibrato

## License

GPL-3.0 - See LICENSE file for details.

## References

- Gutiérrez, E., Haworth, C., and Cádiz, R. (2024). Generating Sonic Phantoms with Quadratic Difference Tone Spectrum Synthesis. *Computer Music Journal* 47(3):1-16.
- Kendall, G.S., Haworth, C., and Cádiz, R.F. (2014). Sound Synthesis with Auditory Distortion Products. *Computer Music Journal* 38(4).

## Credits

- Original Max/MSP implementation: Gutiérrez, E. and Cádiz, R.
- SuperCollider port: Marcin Pietruszewski
