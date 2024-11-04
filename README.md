
# DEL2

A delay that you play like an instrument.

<p align=”center”>
    <img src="images/DEL2.png" alt="screenshot">
</p>

## Current features

- Tap a rhythm into it using midi
- Each delay tap has a separate moog style filter
  - The filter parameters are pair of settings, and each tap's settings is an interpolation between these two, depending on the velocity
  - You can also link the cutoff frequency to the note
- A haas panner that you can link to the note 
- 4 triggers, controlled by a learnable midi note:
  - mute input
  - mute output
  - clear taps
  - lock taps
- There are two modes for the input and output mute: toggle-mode and direct-mode:
  - In toggle-mode, the trigger note for mute toggles the respective mute
  - In direct-mode, the trigger note switches it on when you press it, and off when you let it go
    The note also (retroactively) turns the other mute off when you press it.

## Planned features

- Expand the DSP per tap to:
  - two non-linear filters
  - a reverb
  - a HRTF panner
  - a pitch shifter that shifts the difference between the initial timing tap note and the current tap note
  - a Karplus-Strong per tap, at the pitch of the note
- Compressor and ducker on the output
- Optional individual settings per tap
- Optional auxiliary outputs per tap
- BPM sync, scale and quantization
- Tap the rhythm using your computer keyboard or side-channel audio
- A third mode for the mutes:
  - In instrument-mode, the first note you play will be silent, and start recording a new pattern, as always.
    The next eight notes will be like playing an sampler, that plays the sample you started recording at the beginning of the pattern.
    When you let go of a note, the tap will start the release phase.
    The state of the mute switches will determine if a note turns on and off the input, the output, or both.
    When you switch to instrument-mode, it turns on "mute in" and turns off "mute out", so by default the notes you play

## Thanks

This plugin would not have been possible without the following projects:
- [NIH-plug](https://github.com/robbert-vdh/nih-plug)
- [Vizia](https://github.com/vizia/vizia)
- [va-filter](https://github.com/Fredemus/va-filter)
- [synfx-dsp](https://github.com/WeirdConstructor/synfx-dsp)

I would like to thank [Robbert van der Helm](https://github.com/robbert-vdh), [Fredemus](https://github.com/Fredemus) and [WeirdConstructor](https://github.com/WeirdConstructor) for their fantastic support and feedback!   
