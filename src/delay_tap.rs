use crate::DSP_BLOCK_SIZE;
use nih_plug::prelude::*;

#[derive(Debug, Clone)]
pub struct DelayTap {
    /// The identifier for this delay tap. Polyphonic modulation events are linked to a delay tap based on
    /// these IDs. If the host doesn't provide these IDs, then this is computed through
    /// `compute_fallback_delay_tap_id()`. In that case polyphonic modulation will not work, but the
    /// basic note events will still have an effect.
    pub delay_tap_id: i32,
    /// The note's channel, in `0..16`. Only used for the delay tap terminated event.
    pub channel: u8,
    /// The note's key/note, in `0..128`. Only used for the delay tap terminated event.
    pub note: u8,
    /// The delay taps internal ID. Each delay tap has an internal delay tap ID one higher than the previous
    /// delay tap. This is used to steal the last delay tap in case all 16 delay taps are in use.
    pub internal_delay_tap_id: u64,
    /// The note's velocity. This is used to interpollate it's dsp parameters.
    pub velocity: f32,
    /// Whether the key has been released and the delay tap is in its release stage. The delay tap will be
    /// terminated when the amplitude envelope hits 0 while the note is releasing.
    pub releasing: bool,
    /// Fades between 0 and 1 with timings based on the global attack and release settings.
    pub amp_envelope: Smoother<f32>,
    /// If this delay tap has polyphonic gain modulation applied, then this contains the normalized
    /// offset and a smoother.
    pub delay_tap_gain: Option<(f32, Smoother<f32>)>,

    /// Are we currently muting? To determine if we need to trigger the amp envelope,
    pub is_muted: bool,
    /// Which of the NUM_TAPS taps is this.
    pub tap_index: usize,
    /// The delayed audio of this tap.
    /// Used to apply the envelopes and filters, and other DSP to.
    pub delayed_audio_l: [f32; DSP_BLOCK_SIZE],
    pub delayed_audio_r: [f32; DSP_BLOCK_SIZE],
}
