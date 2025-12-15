# QDTS_SC - Quadratic Distortion Tone Spectra for SuperCollider

A SuperCollider UGen plugin for synthesizing sounds using Auditory Distortion Products (ADPs).

## Overview

QDTS_SC implements a solver for a system of equations related to the synthesis of Auditory Distortion Products, as proposed in:

> Kendall, G.S., Haworth, C., and Cadiz, R.F. (2014). *Sound Synthesis with Auditory Distortion Products*. Computer Music Journal 38(4).

This plugin is a port from the Max/MSP implementation by Gutierrez, E. and Cadiz, R.

## Features

- **QDTSSolver UGen**: Real-time Newton's method solver running on the server
- **Dynamic spectrum control**: Modulate target harmonics in real-time
- **Normalized output**: Amplitudes are normalized for safe mixing
- **Error output**: Monitor solver convergence quality

## Requirements

- SuperCollider 3.9+
- CMake 3.12+
- Eigen3 library
- SuperCollider source code (for plugin headers)

## Installation

### macOS / Linux

```bash
# Clone the repository
git clone https://github.com/mpietrus00/QDTS_SC.git
cd QDTS_SC

# Install Eigen (macOS)
brew install eigen

# Install Eigen (Ubuntu/Debian)
# sudo apt-get install libeigen3-dev

# Clone SuperCollider source (for headers)
git clone --depth 1 https://github.com/supercollider/supercollider.git ~/supercollider

# Build
mkdir build && cd build
cmake .. -DSC_PATH=~/supercollider
make

# Install (copy to Extensions)
mkdir -p ~/Library/Application\ Support/SuperCollider/Extensions/QDTS_SC  # macOS
# mkdir -p ~/.local/share/SuperCollider/Extensions/QDTS_SC  # Linux

cp plugins/QDTSSolver.scx ~/Library/Application\ Support/SuperCollider/Extensions/QDTS_SC/
cp ../sc-classes/*.sc ~/Library/Application\ Support/SuperCollider/Extensions/QDTS_SC/
cp ../HelpSource/Classes/QDTSSolver.schelp ~/Library/Application\ Support/SuperCollider/Extensions/QDTS_SC/
```

### Architecture Notes (macOS)

If you get an architecture mismatch error, rebuild for your SC version:

```bash
# For Intel/Rosetta
cmake .. -DSC_PATH=~/supercollider -DCMAKE_OSX_ARCHITECTURES=x86_64

# For Apple Silicon native
cmake .. -DSC_PATH=~/supercollider -DCMAKE_OSX_ARCHITECTURES=arm64

# Universal binary
cmake .. -DSC_PATH=~/supercollider -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64"
```

## Quick Start

```supercollider
// Boot server and recompile class library after installation
Server.local.boot;

// Basic test
(
{
    var targets = [1, 0.5, 0.33, 0.25];
    var amps = QDTSSolver.kr(4, *targets);
    var freqs = [800, 900, 1000, 1100, 1200];
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

**Avoid setting targets to exactly 0** - this can cause solver instability and loud output. Always use a small minimum value:

```supercollider
var t0s = t0.max(0.01);
var t1s = t1.max(0.01);
// etc.
```

### Carrier and Target Pitch

The synthesis uses frequency relationships:
- **Carrier pitch**: Base frequency (f_c)
- **Target pitch**: Difference frequency (f_d)
- **Resulting frequencies**: f_c, f_c + f_d, f_c + 2*f_d, ...

```supercollider
(
SynthDef(\qdts, {|carrierPitch = 1280, targetPitch = 130|
    var amps = QDTSSolver.kr(4, 1, 0.5, 0.33, 0.25);
    var freqs = 5.collect {|i| carrierPitch + (i * targetPitch) };
    var sines = 5.collect {|i| SinOsc.ar(freqs[i]) * amps[i] };
    Out.ar(0, sines.sum * 0.2 ! 2);
}).add;
)
```

## Examples

See `examples/QDTS_examples.scd` for comprehensive usage examples including:
- Basic synthesis
- Pattern integration
- Real-time spectrum morphing
- Envelope control

## License

[Add your license here]

## References

- Kendall, G.S., Haworth, C., and Cadiz, R.F. (2014). Sound Synthesis with Auditory Distortion Products. *Computer Music Journal* 38(4).

## Credits

- Original Max/MSP implementation: Gutierrez, E. and Cadiz, R.
- SuperCollider port: [Your name]
