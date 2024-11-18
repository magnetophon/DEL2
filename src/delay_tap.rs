use nih_plug::prelude::*;
use std::sync::Arc;
use synfx_dsp::fh_va::FilterParams;
use synfx_dsp::fh_va::LadderFilter;

#[derive(Debug, Clone)]
pub struct DelayTap {
    pub delayed_audio_l: Box<[f32]>,
    pub delayed_audio_r: Box<[f32]>,

    pub filter_params: Arc<FilterParams>,
    pub ladders: LadderFilter,
    pub mute_in_delayed: Box<[bool]>,
    pub amp_envelopes: Smoother<f32>,

    /// The delay taps internal ID. Each delay tap has an internal delay tap ID one higher than the previous
    /// delay tap. This is used to steal the last delay tap in case all 16 delay taps are in use.
    pub internal_id: u64,
    /// The taps delay time.
    /// A new tap will be created if the delay_time and note are not the same as one that is currently playing.
    pub delay_time: u32,
    /// The note's channel, in `0..16`. Only used for the delay tap terminated event.
    // TODO: include this in the decision whether to start a new tap
    // TODO: make a channel select in the gui, where the default is: all
    pub channel: u8,
    /// The note's key/note, in `0..128`. Only used for the delay tap terminated event.
    pub note: u8,
    /// The note's velocity. This is used to interpollate it's dsp parameters.
    pub velocity: f32,
    /// Whether the key has been released and the delay tap is in its release stage. The delay tap will be
    /// terminated when the amplitude envelope hits 0 while the note is releasing.
    pub releasing: bool,
    /// Fades between 0 and 1 with timings based on the global attack and release settings.
    pub amp_envelope: Smoother<f32>,

    /// Are we currently muting? To determine if we need to trigger the amp envelope,
    pub is_muted: bool,
}
