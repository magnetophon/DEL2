
# DEL2

A rhythm delay with space.

## Current features

- Tap a rhythm into your delay using midi
- Each delay tap has:
  - a non-linear filters
- The filter parameters are pair of settings, and each tap's settings are an interpolation between these two, depending on the velocity

## Planned features

- Tap the rhythm using your computer keyboard or side-channel audio
- Expand the DSP per tap to:
  - two non-linear filters
  - a reverb
  - a panner, using either:
    - level
    - haas + eq + level 
    - HRTF
- Compressor and ducker on the output
- Optional individual settings per tap
- Optional auxiliary outputs per tap
