# DEL2

The delay that you play.

https://github.com/user-attachments/assets/cd501d5a-0c7d-404d-b970-8f6272594568

In VST3 and CLAP format, for Linux, Mac and Windows.


## Features & Quickstart

- [Midi rhythm delay](#Recording-a-rhythm)  
You record a delay rhythm by playing midi notes into the plugin.  
The first note starts counting, and each note after that creates a delay tap, with a maximum of 16. 

- Independent effects per tap:  
  - A [Moog style filter](#Filters) with creamy drive
  The parameters are (optionally) linked to velocity and note.
  - [Panning](#Panning) linked to the note
- Four [midi triggers](#Triggers):
  - mute input
  - mute output
  - clear taps
  - lock taps
  There are two modes for the input and output mute: "toggle" and "direct":
    - Toggle-mode is how most mutes work: the triggers turn the mute on when it's off and vice versa.
    - Direct-mode is more like an instrument: the mute is on by default and the trigger turns it off only as long as you hold the trigger note.
- The rhythm pattern is saved alongside the other parameters in the preset and DAW session.
- [Stereo from end to end](#Faux-HRTF-panning).


## Manual

### Installation
Download the zip from [here](https://github.com/magnetophon/DEL2/releases/tag/V0.3.0), unpack it, and copy the .vst3 or .clap to your plugin folder.  
Mac users will have to [disable GateKeeper](https://disable-gatekeeper.github.io/)

### Before you start
- Connect a midi source.  
This can be an instrument, a foot-pedal, a sequencer, or anything else that sends midi.
- Connect an audio source.  
Most DAWs allow you to send a live or recorded audio track into the plugin, alongside the midi from the previous step.  
You could also put an instrument plugin before DEL2 and use the audio from that.

### Recording a rhythm
The first note starts counting, and each note after that creates a delay tap.  
The **max tap** slider sets how long a tap can take.  
The pink line in the graph shows how much time there is left to add new taps to the rhythm  
Every time you add a tap, the counter resets.  
**min tap** is the minimum time between two taps, mainly for de-bouncing, but it can also be used creatively.  
**sync** lets you choose between a free running delay, or one synced to the host tempo.  
BPM mode doesn't quantize the tap timing; it just changes the duration of the taps when the host tempo changes.  
**listen to** lets you choose which midi channel the rhythm notes come from: "any channel", or 1 to 16

### Filters
There's a set of filter parameters for low velocity and one for high velocity.  
Each individual filter follows the velocity of the tap.  
You can also link the note to the cutoff frequency, to play melodies with the resonance.  
**vel>cut** sets how much influence the velocity has on the cutoff.  
**note>cut** sets how much influence the notes have on the cutoff.  
**drive** lets you adjust how much the filter distorts  
If the input is loud, high drive can sound quieter than low drive.  
**mode** lets you choose between low-pass, high-pass, and notch filters of various steepness.

### Panning
By playing lower or higher notes, you pan each tap to the left or right.  
The first note determines the center of the stereo image   
The **offset** slider adjusts that center.  
**panning amount** controls how much each note is panned.  


##### Faux HRTF panning
Normal panners operate on the level of each input.  
Let's say we want to pan left.  
A regular stereo panner would either turn down the right channel, or mix some of the right input into the left output.  
When you hard-pan, the first method will turn off the right input, and the second method will have a mono mix of both channels in the left output.  

The panners in DEL2 work differently.  
When you pan left, they do three things:
- Add a bit more delay to the right input.
This is called the [Haas Effect](https://www.izotope.com/en/learn/what-is-the-haas-effect.html).  
- Make the right side slightly duller  
- Make it a bit quieter.  
Together they emulate a [HRTF](https://en.wikipedia.org/wiki/Head-related_transfer_function)".  

The upside is that the signal retains it's stereo-width and sounds that where hard-panned in the input don't get lost.  
The downside is that you get comb filtering when you listen in mono.  
Since this panner is only on the delay and not on any of the main channels of your mix, the stereo width you get is more than worth it.


### Triggers
To set which note controls the trigger, just click the button and play a note.  
  - There are two modes for the input and output mute: "toggle" and "direct":  
    - **Toggle-mode** is how most mutes work: the triggers turn the mute on when it's off and vice versa.  
    - **Direct-mode** is more like an instrument: the mute is on by default and the trigger turns it off only as long as you hold the trigger note.  
it also (retroactively) turns the other mute off when you press it.  
  **attack** and **release** set how long it takes for the mute to turn on or off.  
  
### Other parameters
**dry/wet** mix between the unaltered dry signal and the effect.  
**wet gain**this is a post effect make up gain.  
**global drive*** lets you adjust the overall amount of distortion.  


## Thanks

This plugin would not have been possible without the following projects:
- [NIH-plug](https://github.com/robbert-vdh/nih-plug)
- [Vizia](https://github.com/vizia/vizia)
- [va-filter](https://github.com/Fredemus/va-filter)
- [synfx-dsp](https://github.com/WeirdConstructor/synfx-dsp)

I would like to thank [Robbert van der Helm](https://github.com/robbert-vdh), [Dr George Atkinson](https://github.com/geom3trik), [Fredemus](https://github.com/Fredemus) and [WeirdConstructor](https://github.com/WeirdConstructor) for their fantastic support and feedback!   
