
# DEL2

A rhythm delay with space.

## Planned features

- Tap a rhythm into your delay, 
  using midi, keyboard or side-channel audio.
- Each delay tap has:
  - two non-linear filters
  - a reverb
  - a panner, using either:
    - level
    - haas + eq + level 
    - HRTF
- The parameters of the above effects can be:
  - All the same, using one instance of the effects
    This has the lowest CPU usage.
  - A pair of settings, and each tap's settings are an interpolation between these two.
    This is more flexible, but also more CPU intensive.
  - Individual settings for each tap.
    This is the most flexible, but also the most tedious to dail in.
