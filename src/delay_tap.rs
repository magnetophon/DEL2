use nih_plug::prelude::*;
use std::sync::Arc;
use synfx_dsp::fh_va::FilterParams;
use synfx_dsp::fh_va::LadderFilter;

use crate::{MAX_BLOCK_SIZE, NO_LEARNED_NOTE};

#[derive(Debug, Clone)]
pub struct DelayTap {
    pub delayed_audio_l: Box<[f32]>,
    pub delayed_audio_r: Box<[f32]>,

    pub filter_params: Arc<FilterParams>,
    pub ladders: LadderFilter,
    pub mute_in_delayed: Box<[bool]>,
    /// Fades between 0 and 1 with timings based on the global attack and release settings.
    pub amp_envelope: Smoother<f32>,

    /// The delay taps internal ID. Each delay tap has an internal delay tap ID one higher than the previous
    /// delay tap. This is used to steal the last delay tap in case all 16 delay taps are in use.
    pub internal_id: u64,
    /// The taps delay time.
    /// A new tap will be created if the `delay_time` and note are not the same as one that is currently playing.
    pub delay_time: u32,
    /// The note's key/note, in `0..128`. Only used for the delay tap terminated event.
    pub note: u8,
    /// The note's velocity. This is used to interpollate it's dsp parameters.
    pub velocity: f32,
    /// Whether the key has been released and the delay tap is in its release stage. The delay tap will be
    /// terminated when the amplitude envelope hits 0 while the note is releasing.
    pub releasing: bool,

    /// Are we currently muting? To determine if we need to trigger the amp envelope,
    pub is_muted: bool,
    /// Is set to true when the tap is created and false  is created
    /// and false when the amplitude envelope hits 0 while the note is releasing.
    pub is_alive: bool,
    /// Only used for indexing into tap_meters
    pub meter_index: usize,
}

impl DelayTap {
    pub fn new(filter_params: Arc<FilterParams>) -> Self {
        Self {
            delayed_audio_l: vec![0.0; MAX_BLOCK_SIZE].into_boxed_slice(),
            delayed_audio_r: vec![0.0; MAX_BLOCK_SIZE].into_boxed_slice(),
            filter_params: filter_params.clone(),
            ladders: LadderFilter::new(filter_params),
            mute_in_delayed: vec![false; MAX_BLOCK_SIZE].into_boxed_slice(),
            amp_envelope: Smoother::new(SmoothingStyle::Linear(13.0)),
            internal_id: 0,
            delay_time: 0,
            note: NO_LEARNED_NOTE,
            velocity: 0.0,
            releasing: false,
            is_muted: true,
            is_alive: false,
            meter_index: 0,
        }
    }

    pub fn init(
        &mut self,
        amp_envelope: Smoother<f32>,
        internal_id: u64,
        delay_time: u32,
        note: u8,
        velocity: f32,
        index: usize,
        new_is_alive: bool,
    ) {
        self.amp_envelope = amp_envelope;
        self.internal_id = internal_id;
        self.delay_time = delay_time;
        self.note = note;
        self.velocity = velocity;
        self.releasing = false;
        self.is_alive = new_is_alive;
        // self.meter_index = index;
    }
}
