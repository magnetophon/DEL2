// #![allow(non_snake_case)]
// #![deny(clippy::all)]
#![warn(clippy::pedantic)]
// #![warn(clippy::cargo)]
#![warn(clippy::nursery)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::cognitive_complexity)]
#![allow(clippy::type_complexity)]
#![feature(portable_simd)]
#![feature(get_mut_unchecked)]
use array_init::array_init;
use bit_mask_ring_buf::BMRingBuf;
use default_boxed::DefaultBoxed;
use nih_plug::params::persist::PersistentField;
use nih_plug::prelude::*;
use nih_plug_vizia::ViziaState;
use std::ops::Index;
use std::simd::f32x4;
use std::sync::atomic::{
    AtomicBool, AtomicI32, AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering,
};
use std::sync::Arc;
use synfx_dsp::fh_va::{FilterParams, LadderMode};

mod delay_tap;
mod editor;
mod svf_simper;
use delay_tap::DelayTap;
use svf_simper::SVFSimper;

// max seconds per tap
const MAX_TAP_SECONDS: usize = 20;
const NUM_TAPS: usize = 16;
const TOTAL_DELAY_SECONDS: usize = MAX_TAP_SECONDS * NUM_TAPS;
const MAX_SAMPLE_RATE: usize = 192_000;
const TOTAL_DELAY_SAMPLES: usize = TOTAL_DELAY_SECONDS * MAX_SAMPLE_RATE;
// this seems to be the number JUCE is using
const MAX_BLOCK_SIZE: usize = 32768;
const PEAK_METER_DECAY_MS: f64 = 150.0;
// abuse the difference in range between u8 and midi notes for special meaning
const NO_LEARNED_NOTE: u8 = 128;
const LEARNING: u8 = 129;
// action trigger indexes
// should be 0..7 because of AtomicByteArray size
const MUTE_IN: usize = 0;
const MUTE_OUT: usize = 1;
const CLEAR_TAPS: usize = 2;
const LOCK_TAPS: usize = 3;
// another value with a special meaning
const NO_GUI_SMOOTHING: f32 = f32::MAX;

const MAX_HAAS_MS: f32 = 30.0;
const PANNER_EQ_FREQ: f32 = 5_000.0;
const MIN_EQ_GAIN: f32 = -13.0;
const MIN_PAN_GAIN: f32 = -3.0;
const DC_HP_FREQ: f32 = 80.0;
const DEFAULT_TEMPO: f32 = 60.0;

pub struct Del2 {
    params: Arc<Del2Params>,

    /// The effect's delay taps. Inactive delay taps will be set to `None` values.
    delay_taps: [DelayTap; NUM_TAPS],
    /// The next internal delay tap ID, used only to figure out the oldest delay tap for "voice stealing".
    /// This is incremented by one each time a delay tap is created.
    next_internal_id: u64,

    // delay write buffer
    delay_buffer: [BMRingBuf<f32>; 2],
    mute_in_delay_buffer: BMRingBuf<bool>,
    mute_in_delay_temp_buffer: Box<[bool]>,
    dc_filter: SVFSimper<4>,

    // for the smoothers
    dry_wet_block: Box<[f32]>,
    drive_main_block: Box<[f32]>,
    drive_mod_block: Box<[f32]>,
    delay_tap_amp_envelope_block: Box<[f32]>,
    delay_tap_smoothed_offset_l_block: Box<[f32]>,
    delay_tap_smoothed_offset_r_block: Box<[f32]>,
    delay_tap_eq_gain_l_block: Box<[f32]>,
    delay_tap_eq_gain_r_block: Box<[f32]>,
    delay_tap_pan_gain_block: Box<[f32]>,
    drive_main_smoother: Smoother<f32>,
    peak_meter_decay_weight: f32,
    input_meter: Arc<AtomicF32>,
    output_meter: Arc<AtomicF32>,
    tap_meters: Arc<AtomicF32Array>,
    meter_indexes: Arc<AtomicUsizeArray>,
    delay_write_index: isize,
    is_learning: Arc<AtomicBool>,
    // for which control are we learning?
    learning_index: Arc<AtomicUsize>,
    learned_notes: Arc<AtomicByteArray>,
    last_learned_notes: Arc<AtomicByteArray>,
    last_played_notes: Arc<LastPlayedNotes>,
    samples_since_last_event: u32,
    timing_last_event: u32,
    min_tap_samples: u32,
    delay_buffer_size: u32,
    counting_state: CountingState,
    should_update_filter: Arc<AtomicBool>,
    enabled_actions: Arc<AtomicBoolArray>,
    running_delay_tempo: f32,
    first_process_after_reset: bool,
}

/// All the parameters
#[derive(Params)]
pub struct Del2Params {
    #[persist = "editor-state"]
    editor_state: Arc<ViziaState>,
    #[nested(group = "global")]
    global: GlobalParams,
    #[nested(group = "taps")]
    pub taps: TapsParams,
    #[persist = "learned-notes"]
    learned_notes: ArcAtomicByteArray,
    #[persist = "enabled-actions"]
    enabled_actions: ArcAtomicBoolArray,
    #[persist = "delay-times"]
    delay_times: AtomicF32Array,
    #[persist = "velocities"]
    velocities: AtomicF32Array,
    #[persist = "notes"]
    notes: AtomicU8Array,
    #[persist = "tap-counter"]
    tap_counter: Arc<AtomicUsize>,
    old_nr_taps: Arc<AtomicUsize>,
    current_time: Arc<AtomicF32>,
    max_tap_time: Arc<AtomicF32>,
    #[persist = "first-note"]
    first_note: Arc<AtomicU8>,
    // for the horizontal gui smoother
    previous_time_scaling_factor: Arc<AtomicF32>,
    previous_note_heights: AtomicF32Array,
    previous_first_note_height: Arc<AtomicF32>,
    previous_panning_center_height: Arc<AtomicF32>,
    previous_pan_foreground_lengths: AtomicF32Array,
    previous_pan_background_lengths: AtomicF32Array,
    // for making the gui smoother frame rate independent
    last_frame_time: AtomicU64,
    // the rate we are nunning at now
    sample_rate: AtomicF32,
    host_tempo: AtomicF32,
    time_sig_numerator: AtomicI32,
    learning_start_time: AtomicU64,
    #[persist = "preset_tempo"]
    preset_tempo: AtomicF32,
}

/// Contains the global parameters.
#[derive(Params)]
struct GlobalParams {
    #[id = "dry_wet"]
    dry_wet: FloatParam,
    #[id = "mute_is_toggle"]
    mute_is_toggle: BoolParam,
    #[id = "attack_ms"]
    attack_ms: FloatParam,
    #[id = "release_ms"]
    release_ms: FloatParam,
    #[id = "min_tap_milliseconds"]
    pub min_tap_milliseconds: FloatParam,
    #[id = "max_tap_ms"]
    pub max_tap_ms: FloatParam,
    #[id = "channel"]
    pub channel: IntParam,
    #[id = "sync"]
    pub sync: BoolParam,
}

impl GlobalParams {
    pub fn new(enabled_actions: Arc<AtomicBoolArray>, learned_notes: Arc<AtomicByteArray>) -> Self {
        Self {
            dry_wet: FloatParam::new("mix", 0.5, FloatRange::Linear { min: 0.0, max: 1.0 })
                .with_unit("%")
                .with_smoother(SmoothingStyle::Linear(15.0))
                .with_value_to_string(formatters::v2s_f32_percentage(0))
                .with_string_to_value(formatters::s2v_f32_percentage()),
            mute_is_toggle: BoolParam::new("mute mode", true)
                .with_value_to_string(Arc::new(|value| {
                    String::from(if value { "toggle" } else { "direct" })
                }))
                .with_callback(Arc::new(move |value| {
                    if !value {
                        let mute_in_note = learned_notes.load(MUTE_IN);
                        if mute_in_note != LEARNING && mute_in_note != NO_LEARNED_NOTE {
                            enabled_actions.store(MUTE_IN, true);
                        }
                        enabled_actions.store(MUTE_OUT, false);
                    }
                })),

            attack_ms: FloatParam::new(
                "Attack",
                13.0,
                FloatRange::Skewed {
                    min: 1.0,
                    max: 5_000.0,
                    factor: FloatRange::skew_factor(-1.2),
                },
            )
            .with_value_to_string(Del2::v2s_f32_ms_then_s(3))
            .with_string_to_value(Del2::s2v_f32_s_or_ms_if_bigger_than(5_000.0)),
            release_ms: FloatParam::new(
                "Release",
                13.0,
                FloatRange::Skewed {
                    min: 5.0,
                    max: 20_000.0,
                    factor: FloatRange::skew_factor(-1.5),
                },
            )
            .with_value_to_string(Del2::v2s_f32_ms_then_s(3))
            .with_string_to_value(Del2::s2v_f32_s_or_ms_if_bigger_than(20_000.0)),
            min_tap_milliseconds: FloatParam::new(
                "min tap",
                13.0,
                FloatRange::Skewed {
                    min: 1.0,
                    max: 1000.0,
                    factor: FloatRange::skew_factor(-1.5),
                },
            )
            .with_value_to_string(Del2::v2s_f32_ms_then_s(3))
            .with_string_to_value(Del2::s2v_f32_s_or_ms_if_bigger_than(1000.0)),
            max_tap_ms: FloatParam::new(
                "max tap",
                3030.0,
                FloatRange::Skewed {
                    min: 500.0,
                    max: (MAX_TAP_SECONDS * 1000) as f32,
                    factor: FloatRange::skew_factor(-0.8),
                },
            )
            .with_value_to_string(Del2::v2s_f32_ms_then_s(3))
            .with_string_to_value(Del2::s2v_f32_s_or_ms_if_bigger_than(
                MAX_TAP_SECONDS as f32 * 1000.0,
            )),

            channel: IntParam::new(
                "channel",
                0, // 0 means: any channel
                IntRange::Linear { min: 0, max: 16 },
            )
            .with_value_to_string(Del2::v2s_i32_channel())
            .with_string_to_value(Del2::s2v_i32_channel()),

            sync: BoolParam::new("sync", true).with_value_to_string(Arc::new(|value| {
                String::from(if value { "bpm" } else { "free" })
            })),
        }
    }
}

/// Contains the high and low tap parameters.
#[derive(Params)]
pub struct TapsParams {
    #[id = "panning_offset"]
    pub panning_offset: FloatParam,
    #[id = "panning_amount"]
    pub panning_amount: FloatParam,
    #[id = "filter_type"]
    pub filter_type: EnumParam<MyLadderMode>,
    #[id = "use_note_frequency"]
    use_note_frequency: BoolParam,

    #[id = "cutoff"]
    pub cutoff_main: FloatParam,
    #[id = "res"]
    pub res_main: FloatParam,
    #[id = "drive"]
    pub drive_main: FloatParam,

    #[id = "cutoff mod"]
    pub cutoff_mod: FloatParam,
    #[id = "res mod"]
    pub res_mod: FloatParam,
    #[id = "drive mod"]
    pub drive_mod: FloatParam,

    // for note modulation
    #[id = "cutoff_octave"]
    pub cutoff_octave: IntParam,
    #[id = "cutoff_transpose"]
    pub cutoff_transpose: IntParam,
}

impl TapsParams {
    pub fn new(should_update_filter: Arc<AtomicBool>) -> Self {
        Self {
            panning_offset: FloatParam::new(
                "panning offset",
                0.0,
                FloatRange::SymmetricalSkewed {
                    min: -1.0,
                    max: 1.0,
                    factor: FloatRange::skew_factor(-1.8),
                    center: 0.0,
                },
            )
            .with_unit(" %")
            .with_value_to_string(formatters::v2s_f32_percentage(0))
            .with_string_to_value(formatters::s2v_f32_percentage()),

            panning_amount: FloatParam::new(
                "panning amount",
                0.13,
                FloatRange::SymmetricalSkewed {
                    min: -1.0,
                    max: 1.0,
                    factor: FloatRange::skew_factor(-1.42),
                    center: 0.0,
                },
            )
            .with_unit(" %")
            .with_value_to_string(formatters::v2s_f32_percentage(0))
            .with_string_to_value(formatters::s2v_f32_percentage()),
            filter_type: EnumParam::new("filter type".to_string(), MyLadderMode::lp6())
                .with_callback(Arc::new({
                    let should_update_filter = should_update_filter.clone();
                    move |_| should_update_filter.store(true, Ordering::Release)
                })),

            use_note_frequency: BoolParam::new("cutoff mode".to_string(), false)
                .with_callback(Arc::new({
                    let should_update_filter = should_update_filter.clone();
                    move |_| should_update_filter.store(true, Ordering::Release)
                }))
                .with_value_to_string(Arc::new(|value| {
                    String::from(if value { "note" } else { "velocity" })
                }))
                .with_callback(Arc::new({
                    let should_update_filter = should_update_filter.clone();
                    move |_| should_update_filter.store(true, Ordering::Release)
                })),

            cutoff_main: FloatParam::new(
                "cutoff".to_string(),
                1_690.0,
                FloatRange::Skewed {
                    min: 10.0,
                    max: 18_000.0,
                    factor: FloatRange::skew_factor(-1.6),
                },
            )
            .with_value_to_string(Del2::v2s_f32_hz_then_khz_three_digits())
            .with_string_to_value(formatters::s2v_f32_hz_then_khz())
            .with_callback(Arc::new({
                let should_update_filter = should_update_filter.clone();
                move |_| should_update_filter.store(true, Ordering::Release)
            })),
            res_main: FloatParam::new(
                "resonance".to_string(),
                0.13,
                FloatRange::Linear { min: 0., max: 1. },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(2))
            .with_callback(Arc::new({
                let should_update_filter = should_update_filter.clone();
                move |_| should_update_filter.store(true, Ordering::Release)
            })),
            drive_main: FloatParam::new(
                "drive".to_string(),
                0.0,
                FloatRange::Skewed {
                    min: 0.0,
                    max: 42.0,
                    // This makes the range appear as if it was linear when displaying the values as
                    // decibels
                    factor: FloatRange::skew_factor(-1.0),
                },
            )
            .with_step_size(0.1)
            .with_unit(" dB"),

            cutoff_mod: FloatParam::new(
                "cutoff mod".to_string(),
                0.13,
                FloatRange::Linear {
                    min: -1.0,
                    max: 1.0,
                },
            )
            .with_unit("%")
            .with_value_to_string(formatters::v2s_f32_percentage(0))
            .with_string_to_value(formatters::s2v_f32_percentage())
            .with_callback(Arc::new({
                let should_update_filter = should_update_filter.clone();
                move |_| should_update_filter.store(true, Ordering::Release)
            })),
            res_mod: FloatParam::new(
                "resonance mod".to_string(),
                0.0,
                FloatRange::Linear {
                    min: -1.0,
                    max: 1.0,
                },
            )
            .with_unit("%")
            .with_value_to_string(formatters::v2s_f32_percentage(0))
            .with_string_to_value(formatters::s2v_f32_percentage())
            .with_callback(Arc::new({
                let should_update_filter = should_update_filter.clone();
                move |_| should_update_filter.store(true, Ordering::Release)
            })),

            drive_mod: FloatParam::new(
                "drive mod".to_string(),
                13.0,
                FloatRange::SymmetricalSkewed {
                    min: -42.0,
                    max: 42.0,
                    factor: FloatRange::skew_factor(-1.0),
                    center: 0.0,
                },
            )
            .with_step_size(0.1)
            .with_unit(" dB"),

            cutoff_octave: IntParam::new("cutoff octave", 0, IntRange::Linear { min: -4, max: 4 })
                .with_callback(Arc::new({
                    let should_update_filter = should_update_filter.clone();
                    move |_| should_update_filter.store(true, Ordering::Release)
                })),

            cutoff_transpose: IntParam::new(
                "cutoff transpose",
                0,
                IntRange::Linear { min: -12, max: 12 },
            )
            .with_callback(Arc::new({
                move |_| should_update_filter.store(true, Ordering::Release)
            })),
        }
    }
}

#[derive(PartialEq)]
enum CountingState {
    TimeOut,
    CountingInBuffer,
    CountingAcrossBuffer,
}

impl Default for Del2 {
    fn default() -> Self {
        let should_update_filter = Arc::new(AtomicBool::new(false));
        let learned_notes = Arc::new(AtomicByteArray::new(NO_LEARNED_NOTE));
        let enabled_actions = Arc::new(AtomicBoolArray::new());

        Self {
            params: Arc::new(Del2Params::new(
                should_update_filter.clone(),
                enabled_actions.clone(),
                learned_notes.clone(),
            )),

            delay_taps: array_init(|_| DelayTap::new(Arc::new(FilterParams::new()))),
            next_internal_id: 0,

            delay_buffer: [
                BMRingBuf::<f32>::from_len(TOTAL_DELAY_SAMPLES),
                BMRingBuf::<f32>::from_len(TOTAL_DELAY_SAMPLES),
            ],
            mute_in_delay_buffer: BMRingBuf::<bool>::from_len(TOTAL_DELAY_SAMPLES),
            mute_in_delay_temp_buffer: bool::default_boxed_array::<MAX_BLOCK_SIZE>(),
            dc_filter: SVFSimper::new(DC_HP_FREQ, 0.0, 48000.0),

            dry_wet_block: f32::default_boxed_array::<MAX_BLOCK_SIZE>(),
            drive_main_block: f32::default_boxed_array::<MAX_BLOCK_SIZE>(),
            drive_mod_block: f32::default_boxed_array::<MAX_BLOCK_SIZE>(),
            delay_tap_amp_envelope_block: f32::default_boxed_array::<MAX_BLOCK_SIZE>(),
            delay_tap_smoothed_offset_l_block: f32::default_boxed_array::<MAX_BLOCK_SIZE>(),
            delay_tap_smoothed_offset_r_block: f32::default_boxed_array::<MAX_BLOCK_SIZE>(),
            delay_tap_eq_gain_l_block: f32::default_boxed_array::<MAX_BLOCK_SIZE>(),
            delay_tap_eq_gain_r_block: f32::default_boxed_array::<MAX_BLOCK_SIZE>(),
            delay_tap_pan_gain_block: f32::default_boxed_array::<MAX_BLOCK_SIZE>(),
            drive_main_smoother: Smoother::new(SmoothingStyle::Linear(13.0)),

            peak_meter_decay_weight: 1.0,
            input_meter: Arc::new(AtomicF32::new(util::MINUS_INFINITY_DB)),
            output_meter: Arc::new(AtomicF32::new(util::MINUS_INFINITY_DB)),
            tap_meters: AtomicF32Array(array_init(|_| {
                Arc::new(AtomicF32::new(util::MINUS_INFINITY_DB))
            }))
            .into(),
            meter_indexes: AtomicUsizeArray(array_init(|_| Arc::new(AtomicUsize::new(0)))).into(),
            delay_write_index: 0,
            is_learning: Arc::new(AtomicBool::new(false)),
            learning_index: Arc::new(AtomicUsize::new(0)),
            learned_notes,
            last_learned_notes: Arc::new(AtomicByteArray::new(NO_LEARNED_NOTE)),
            last_played_notes: Arc::new(LastPlayedNotes::new()),
            samples_since_last_event: 0,
            timing_last_event: 0,
            min_tap_samples: 0,
            delay_buffer_size: 0,
            counting_state: CountingState::TimeOut,
            should_update_filter,
            enabled_actions,
            running_delay_tempo: DEFAULT_TEMPO,
            first_process_after_reset: true,
        }
    }
}

impl Del2Params {
    fn new(
        should_update_filter: Arc<AtomicBool>,
        enabled_actions: Arc<AtomicBoolArray>,
        learned_notes: Arc<AtomicByteArray>,
    ) -> Self {
        Self {
            editor_state: editor::default_state(),
            taps: TapsParams::new(should_update_filter),
            global: GlobalParams::new(enabled_actions.clone(), learned_notes.clone()),
            learned_notes: ArcAtomicByteArray(learned_notes),
            enabled_actions: ArcAtomicBoolArray(enabled_actions),

            tap_counter: Arc::new(AtomicUsize::new(0)),
            old_nr_taps: Arc::new(AtomicUsize::new(0)),
            delay_times: AtomicF32Array(array_init(|_| Arc::new(AtomicF32::new(0.0)))),
            velocities: AtomicF32Array(array_init(|_| Arc::new(AtomicF32::new(0.0)))),
            notes: AtomicU8Array(array_init(|_| Arc::new(AtomicU8::new(0)))),
            current_time: Arc::new(AtomicF32::new(0.0)),
            max_tap_time: Arc::new(AtomicF32::new(0.0)),
            first_note: Arc::new(AtomicU8::new(NO_LEARNED_NOTE)),
            previous_time_scaling_factor: Arc::new(AtomicF32::new(0.0)),
            previous_note_heights: AtomicF32Array(array_init(|_| Arc::new(AtomicF32::new(0.0)))),
            previous_first_note_height: Arc::new(AtomicF32::new(0.0)),
            previous_panning_center_height: Arc::new(AtomicF32::new(0.0)),
            previous_pan_foreground_lengths: AtomicF32Array(array_init(|_| {
                Arc::new(AtomicF32::new(0.0))
            })),
            previous_pan_background_lengths: AtomicF32Array(array_init(|_| {
                Arc::new(AtomicF32::new(0.0))
            })),
            last_frame_time: AtomicU64::new(0),
            sample_rate: 1.0.into(),
            host_tempo: (-1.0).into(),
            time_sig_numerator: (-1).into(),
            learning_start_time: AtomicU64::new(0),
            preset_tempo: (DEFAULT_TEMPO).into(),
        }
    }
}

impl Plugin for Del2 {
    const NAME: &'static str = "DEL2";
    const VENDOR: &'static str = "magnetophon";
    const URL: &'static str = env!("CARGO_PKG_HOMEPAGE");
    const EMAIL: &'static str = "bart@magnetophon.nl";

    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    // The first audio IO layout is used as the default. The other layouts may be selected either
    // explicitly or automatically by the host or the user depending on the plugin API/backend.
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(2),
        main_output_channels: NonZeroU32::new(2),

        aux_input_ports: &[],
        aux_output_ports: &[],

        // Individual ports and the layout as a whole can be named here. By default these names
        // are generated as needed. This layout will be called 'Stereo', while a layout with
        // only one input and output channel would be called 'Mono'.
        names: PortNames::const_default(),
    }];

    const MIDI_INPUT: MidiConfig = MidiConfig::Basic;
    const MIDI_OUTPUT: MidiConfig = MidiConfig::None;

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    // If the plugin can send or receive SysEx messages, it can define a type to wrap around those
    // messages here. The type implements the `SysExMessage` trait, which allows conversion to and
    // from plain byte buffers.
    type SysExMessage = ();
    // More advanced plugins can use this to run expensive background tasks. See the field's
    // documentation for more information. `()` means that the plugin does not have any background
    // tasks.
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        editor::create(
            editor::Data {
                params: self.params.clone(),
                input_meter: self.input_meter.clone(),
                output_meter: self.output_meter.clone(),
                tap_meters: self.tap_meters.clone(),
                meter_indexes: self.meter_indexes.clone(),
                is_learning: self.is_learning.clone(),
                learning_index: self.learning_index.clone(),
                learned_notes: self.learned_notes.clone(),
                last_learned_notes: self.last_learned_notes.clone(),
                last_played_notes: self.last_played_notes.clone(),
                enabled_actions: self.enabled_actions.clone(),
                show_full_parameters: false, // default to the easy editor
            },
            self.params.editor_state.clone(),
        )
    }

    // After this function [`reset()`][Self::reset()] will always be called. If you need to clear
    // state, such as filters or envelopes, then you should do so in that function instead.
    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        // Set the sample rate from the buffer configuration
        let sample_rate = buffer_config.sample_rate;
        self.params.sample_rate.store(sample_rate, Ordering::SeqCst);
        // After `PEAK_METER_DECAY_MS` milliseconds of pure silence, the peak meter's value should
        // have dropped by 12 dB
        self.peak_meter_decay_weight =
            0.25f64.powf((f64::from(sample_rate) * PEAK_METER_DECAY_MS / 1000.0).recip()) as f32;
        // Calculate and set the delay buffer size
        self.set_delay_buffer_size(buffer_config);
        self.dc_filter.reset(DC_HP_FREQ, 0.0, sample_rate);

        // Initialize filter parameters for each tap
        self.initialize_filter_parameters();
        for delay_tap in &mut self.delay_taps {
            delay_tap
                .shelving_eq
                .reset(PANNER_EQ_FREQ, 0.0, sample_rate);
        }
        true
    }

    fn reset(&mut self) {
        let tap_counter = self.params.tap_counter.load(Ordering::SeqCst);

        // first make room in the array
        self.delay_taps.iter_mut().for_each(|delay_tap| {
            delay_tap.is_alive = false;
            delay_tap.amp_envelope.reset(0.0);
            delay_tap.ladders.s = [f32x4::splat(0.); 4];
        });

        self.drive_main_smoother.reset(0.0);
        // then fill the array
        for tap_index in 0..tap_counter {
            // nih_log!("reset tap_counter: {tap_counter}");
            self.start_tap(tap_index, self.params.preset_tempo.load(Ordering::SeqCst));
        }
        self.first_process_after_reset = true;
        self.counting_state = CountingState::TimeOut;
        self.timing_last_event = 0;
        self.samples_since_last_event = 0;

        // reset learning system
        self.is_learning.store(false, Ordering::SeqCst);
        self.params.learning_start_time.store(0, Ordering::SeqCst);
        for i in 0..8 {
            self.last_learned_notes.store(i, self.learned_notes.load(i));
        }

        // don't smooth the gui for the new taps
        for i in (self.params.old_nr_taps.load(Ordering::SeqCst) + 1)..tap_counter {
            self.params.previous_note_heights[i].store(NO_GUI_SMOOTHING, Ordering::SeqCst);
            self.params.previous_pan_foreground_lengths[i]
                .store(NO_GUI_SMOOTHING, Ordering::SeqCst);
            self.params.previous_pan_background_lengths[i]
                .store(NO_GUI_SMOOTHING, Ordering::SeqCst);
        }
    }

    #[allow(clippy::float_cmp)]
    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        self.update_peak_meter(buffer, &self.input_meter);
        self.update_min_max_tap_time();

        let num_samples = buffer.samples();

        // let sample_rate = context.transport().sample_rate;
        let sample_rate = self.params.sample_rate.load(Ordering::SeqCst);

        let sync = self.params.global.sync.value();
        let host_tempo = context
            .transport()
            .tempo
            .unwrap_or_else(|| f64::from(DEFAULT_TEMPO)) as f32;
        // since we don't know the tempo in reset, we need to do this here, instead of in reset.
        if self.first_process_after_reset {
            self.running_delay_tempo = host_tempo;
            self.params.preset_tempo.store(host_tempo, Ordering::SeqCst);
            self.first_process_after_reset = false;
        }
        let preset_tempo = self.params.preset_tempo.load(Ordering::SeqCst);
        let running_delay_tempo = self.running_delay_tempo;
        if (sync && running_delay_tempo != host_tempo)
            || (!sync && running_delay_tempo != preset_tempo)
        {
            self.running_delay_tempo = if sync { host_tempo } else { preset_tempo };
            // nih_log!(
            // "tempo change from {} to {host_tempo}",
            // self.running_delay_tempo
            // );

            // convert the delay times for the new tempo
            // we do this in  process() and not in reset(),
            // because if the user changes the tempo in the song,
            // we want the delay time to change with it.

            // the graph should remain stationary after the recalculation
            self.params
                .previous_time_scaling_factor
                .store(NO_GUI_SMOOTHING, Ordering::SeqCst);

            let tap_counter = self.params.tap_counter.load(Ordering::SeqCst);

            // first start fading out the current delay taps
            self.start_release_for_all_delay_taps();
            for tap_index in 0..tap_counter {
                self.start_tap(tap_index, self.running_delay_tempo);
            }
        }

        self.params
            .sample_rate
            .store(context.transport().sample_rate, Ordering::SeqCst);
        self.params.host_tempo.store(host_tempo, Ordering::SeqCst);
        self.params.time_sig_numerator.store(
            context.transport().time_sig_numerator.unwrap_or(-1),
            Ordering::SeqCst,
        );
        let mut next_event = context.next_event();
        let mut block_start: usize = 0;
        let mut block_end: usize = MAX_BLOCK_SIZE.min(num_samples);

        // dc_block the audio and write it into the delay
        self.write_into_delay(buffer);

        let panning_center = self
            .params
            .taps
            .panning_offset
            .value()
            .mul_add(
                127.0,
                f32::from(self.params.first_note.load(Ordering::SeqCst)),
            )
            .clamp(0.0, 127.0);

        let panning_amount = self.params.taps.panning_amount.value();

        while block_start < num_samples {
            let old_nr_taps = self.params.tap_counter.load(Ordering::SeqCst);

            self.params.old_nr_taps.store(old_nr_taps, Ordering::SeqCst);
            // First of all, handle all note events that happen at the start of the block, and cut
            // the block short if another event happens before the end of it.
            'events: loop {
                match next_event {
                    // If the event happens now, then we'll keep processing events
                    Some(event) if (event.timing() as usize) <= block_start => {
                        match event {
                            NoteEvent::NoteOn {
                                timing,
                                channel,
                                note,
                                velocity,
                                ..
                            } => {
                                let listen_to = self.params.global.channel.value() - 1;
                                if listen_to as u8 == channel || listen_to == -1 {
                                    self.store_note_on_in_delay_data(timing, note, velocity);
                                    let tap_counter =
                                        self.params.tap_counter.load(Ordering::SeqCst);
                                    if tap_counter > old_nr_taps {
                                        // nih_log!("process: added a tap: {} > {old_nr_taps}", tap_counter);
                                        self.start_tap(tap_counter - 1, host_tempo);
                                    }
                                }
                            }
                            NoteEvent::NoteOff { channel, note, .. } => {
                                let listen_to = self.params.global.channel.value() - 1;
                                if listen_to as u8 == channel || listen_to == -1 {
                                    self.store_note_off_in_delay_data(note);
                                }
                            }
                            _ => (),
                        };
                        next_event = context.next_event();
                    }
                    // If the event happens before the end of the block, then the block should be cut
                    // short so the next block starts at the event
                    Some(event) if (event.timing() as usize) < block_end => {
                        block_end = event.timing() as usize;
                        break 'events;
                    }
                    _ => break 'events,
                }
            }

            let block_len = block_end - block_start;
            self.prepare_for_delay(block_len);

            if self
                .should_update_filter
                .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                self.update_filters();
            }

            self.set_mute_for_all_delay_taps();

            // Calculate dry mix and update gains
            let dry_wet_block = &mut self.dry_wet_block[..block_len];

            self.params
                .global
                .dry_wet
                .smoothed
                .next_block(dry_wet_block, block_len);

            let output = buffer.as_slice();
            for (value_idx, sample_idx) in (block_start..block_end).enumerate() {
                let dry = 1.0 - dry_wet_block[value_idx];
                output[0][sample_idx] *= dry;
                output[1][sample_idx] *= dry;
            }

            let drive_main_block = &mut self.drive_main_block[..block_len];
            self.delay_taps
                .iter_mut()
                .filter(|tap| tap.is_alive)
                .enumerate()
                .for_each(|(meter_index, delay_tap)| {
                    // nih_log!("delay_tap.is_audible: {}", delay_tap.is_audible);

                    delay_tap
                        .amp_envelope
                        .next_block(&mut self.delay_tap_amp_envelope_block, block_len);

                    let delay_time = delay_tap.delay_time as isize;
                    let write_index = self.delay_write_index;

                    let read_index = (write_index - (delay_time - 1)) as f32;

                    let velocity = delay_tap.velocity;
                    let velocity_factor = (velocity - 0.5) * 2.0; // Scale 0.0 to 1.0 range to -1.0 to 1.0.
                    let taps_params = &self.params.taps;
                    let drive_main = taps_params.drive_main.value();
                    let drive_mod = taps_params.drive_mod.value();
                    self.drive_main_smoother
                        .set_target(sample_rate, util::db_to_gain_fast(drive_main));
                    self.drive_main_smoother
                        .next_block(drive_main_block, block_len);

                    let drive_mod_block = &mut self.drive_mod_block[..block_len];
                    let drive_mod_smoother = &delay_tap.drive_mod_smoother;
                    drive_mod_smoother.set_target(
                        sample_rate,
                        util::db_to_gain_fast(drive_mod * velocity_factor),
                    );
                    drive_mod_smoother.next_block(drive_mod_block, block_len);

                    self.mute_in_delay_buffer.read_into(
                        &mut delay_tap.mute_in_delayed,
                        write_index - (delay_time - 1),
                    );

                    if delay_tap.is_audible {
                        let pan = ((f32::from(delay_tap.note) - panning_center) * panning_amount)
                            .clamp(-1.0, 1.0);
                        let (offset_l, offset_r) = Self::pan_to_haas_samples(pan, sample_rate);
                        let is_new_tap = meter_index + 1 > old_nr_taps;
                        if is_new_tap {
                            // nih_log!("reset offset for {meter_index}");
                            delay_tap.smoothed_offset_l.reset(offset_l);
                            delay_tap.smoothed_offset_r.reset(offset_r);
                        } else {
                            delay_tap
                                .smoothed_offset_l
                                .set_target(sample_rate, offset_l);
                            delay_tap
                                .smoothed_offset_r
                                .set_target(sample_rate, offset_r);
                        }

                        let calculate_and_set_gain = |target: &mut Smoother<f32>, min_gain: f32| {
                            let gain_value = util::db_to_gain_fast((min_gain * pan).min(0.0));
                            if is_new_tap {
                                target.reset(gain_value);
                            } else {
                                target.set_target(sample_rate, gain_value);
                            }
                        };

                        calculate_and_set_gain(&mut delay_tap.eq_gain_l, MIN_EQ_GAIN);
                        calculate_and_set_gain(&mut delay_tap.eq_gain_r, MIN_EQ_GAIN * -1.0);

                        // don't use calculate_and_set_gain here, cause we don't want it clamped
                        let pan_gain_value = util::db_to_gain_fast(MIN_PAN_GAIN * pan);
                        if is_new_tap {
                            delay_tap.pan_gain.reset(pan_gain_value);
                        } else {
                            delay_tap.pan_gain.set_target(sample_rate, pan_gain_value);
                        }
                        delay_tap
                            .smoothed_offset_l
                            .next_block(&mut self.delay_tap_smoothed_offset_l_block, block_len);
                        delay_tap
                            .smoothed_offset_r
                            .next_block(&mut self.delay_tap_smoothed_offset_r_block, block_len);
                        delay_tap
                            .eq_gain_l
                            .next_block(&mut self.delay_tap_eq_gain_l_block, block_len);
                        delay_tap
                            .eq_gain_r
                            .next_block(&mut self.delay_tap_eq_gain_r_block, block_len);
                        delay_tap
                            .pan_gain
                            .next_block(&mut self.delay_tap_pan_gain_block, block_len);
                        for (value_idx, sample_idx) in (block_start..block_end).enumerate() {
                            let pre_filter_gain =
                                drive_main_block[value_idx] * drive_mod_block[value_idx];

                            // if self.delay_tap_amp_envelope_block[value_idx] != 1.0 && self.delay_tap_amp_envelope[value_idx] != 0.0 {
                            // nih_log!("self.delay_tap_amp_envelope_block[value_idx]: {}", self.delay_tap_amp_envelope[value_idx]);
                            // }
                            delay_tap.delayed_audio_l[sample_idx] =
                                self.delay_buffer[0].lin_interp_f32(
                                    read_index - self.delay_tap_smoothed_offset_l_block[value_idx]
                                        + value_idx as f32,
                                ) * pre_filter_gain;
                            delay_tap.delayed_audio_r[sample_idx] =
                                self.delay_buffer[1].lin_interp_f32(
                                    read_index - self.delay_tap_smoothed_offset_r_block[value_idx]
                                        + value_idx as f32,
                                ) * pre_filter_gain;
                        }
                        // HQ mode:
                        for i in block_start..block_end {
                            let frame = f32x4::from_array([
                                delay_tap.delayed_audio_l[i],
                                delay_tap.delayed_audio_r[i],
                                0.0,
                                0.0,
                            ]);

                            let gain_values = f32x4::from_array([
                                self.delay_tap_eq_gain_l_block[i],
                                self.delay_tap_eq_gain_r_block[i],
                                0.0,
                                0.0,
                            ]);

                            let frame_filtered = *delay_tap.ladders.tick_pivotal(frame).as_array();
                            // let frame_out = *delay_tap.ladders.tick_linear(frame).as_array();
                            let frame_out = delay_tap
                                .shelving_eq
                                .highshelf(frame_filtered.into(), gain_values);

                            delay_tap.delayed_audio_l[i] = frame_out[0];
                            delay_tap.delayed_audio_r[i] = frame_out[1];
                        }

                        // LQ mode:
                        //     for i in (block_start..block_end).step_by(2) {
                        //         let frame = f32x4::from_array([
                        //             delay_tap.delayed_audio_l[i],
                        //             delay_tap.delayed_audio_r[i],
                        //             delay_tap.delayed_audio_l.get(i + 1).copied().unwrap_or(0.0),
                        //             delay_tap.delayed_audio_r.get(i + 1).copied().unwrap_or(0.0),
                        //         ]);

                        //         let gain_values = f32x4::from_array([
                        //             self.delay_tap_eq_gain_l_block[i],
                        //             self.delay_tap_eq_gain_r_block[i],
                        //             self.delay_tap_eq_gain_l_block
                        //                 .get(i + 1)
                        //                 .copied()
                        //                 .unwrap_or(1.0),
                        //             self.delay_tap_eq_gain_r_block
                        //                 .get(i + 1)
                        //                 .copied()
                        //                 .unwrap_or(1.0),
                        //         ]);

                        //         let frame_filtered =
                        //         // frame;
                        //             *delay_tap.ladders.tick_pivotal(frame).as_array();
                        //         // *delay_tap.ladders.tick_linear(frame).as_array();
                        //         let frame_out = delay_tap
                        //             .shelving_eq
                        //             .highshelf(frame_filtered.into(), gain_values);

                        //         delay_tap.delayed_audio_l[i] = frame_out[0];
                        //         delay_tap.delayed_audio_r[i] = frame_out[1];
                        //         if i + 1 < block_end {
                        //             delay_tap.delayed_audio_l[i + 1] = frame_out[2];
                        //             delay_tap.delayed_audio_r[i + 1] = frame_out[3];
                        //         }
                        //     }
                    }

                    // Process the output and meter updates
                    let mut amplitude = 0.0;
                    for (value_idx, sample_idx) in (block_start..block_end).enumerate() {
                        let post_filter_gain = dry_wet_block[value_idx]
                            * self.delay_tap_amp_envelope_block[value_idx]
                            / drive_main_block[value_idx];
                        let left = delay_tap.delayed_audio_l[sample_idx]
                            * post_filter_gain
                            * self.delay_tap_pan_gain_block[value_idx];
                        let right = delay_tap.delayed_audio_r[sample_idx] * post_filter_gain
                            / self.delay_tap_pan_gain_block[value_idx];
                        output[0][sample_idx] += left;
                        output[1][sample_idx] += right;
                        amplitude += (left.abs() + right.abs()) * 0.5;
                    }

                    if self.params.editor_state.is_open() {
                        let weight = self.peak_meter_decay_weight * 0.91;

                        amplitude = (amplitude / block_len as f32).min(1.0);
                        let current_peak_meter =
                            self.tap_meters[meter_index].load(Ordering::Relaxed);
                        let new_peak_meter = if amplitude > current_peak_meter {
                            // nih_log!(
                            // "process: self.meter_indexes[{meter_index}]: {}",
                            // self.meter_indexes[meter_index].load(Ordering::Relaxed)
                            // );
                            amplitude
                        } else {
                            current_peak_meter.mul_add(weight, amplitude * (1.0 - weight))
                        };

                        self.tap_meters[meter_index].store(new_peak_meter, Ordering::Relaxed);
                    }
                    if self.delay_tap_amp_envelope_block[0] == 0.0 {
                        if delay_tap.releasing {
                            delay_tap.is_alive = false;
                            // nih_log!("killed");
                        }
                        delay_tap.is_audible = false;
                    } else {
                        delay_tap.is_audible = true;
                    }
                });

            block_start = block_end;
            block_end = (block_start + MAX_BLOCK_SIZE).min(num_samples);
        }

        self.update_peak_meter(buffer, &self.output_meter);

        ProcessStatus::Normal
    }
}

impl Del2 {
    fn update_min_max_tap_time(&mut self) {
        let sample_rate = self.params.sample_rate.load(Ordering::SeqCst);
        self.params.max_tap_time.store(
            self.params.global.max_tap_ms.value() * 0.001,
            Ordering::SeqCst,
        );
        self.min_tap_samples =
            (sample_rate * self.params.global.min_tap_milliseconds.value() * 0.001) as u32;
    }

    fn write_into_delay(&mut self, buffer: &mut Buffer) {
        for (_, block) in buffer.iter_blocks(buffer.samples()) {
            let block_len = block.samples();
            let mut block_channels = block.into_iter();

            let out_l = block_channels.next().expect("Left output channel missing");
            let out_r = block_channels.next().expect("Right output channel missing");

            let write_index = self.delay_write_index;
            // Process each sample in the block
            for i in 0..block_len {
                let frame = f32x4::from_array([out_l[i], out_r[i], 0.0, 0.0]);
                // filter to remove dc-offset, since offsets can make the non lin filter models behave weirdly
                let filtered_frame = self.dc_filter.highpass(frame);

                self.delay_buffer[0][write_index + i as isize] = filtered_frame[0];
                self.delay_buffer[1][write_index + i as isize] = filtered_frame[1];
            }

            let mute_in_value = if self.params.global.mute_is_toggle.value() {
                self.enabled_actions.load(MUTE_IN)
            } else {
                !self.is_playing_action(MUTE_IN) || self.enabled_actions.load(MUTE_OUT)
            };
            for elem in &mut self.mute_in_delay_temp_buffer {
                *elem = mute_in_value;
            }
            self.mute_in_delay_buffer
                .write_latest(&self.mute_in_delay_temp_buffer[..block_len], write_index);

            self.delay_write_index =
                (write_index + block_len as isize) % self.delay_buffer_size as isize;
        }
    }

    fn store_note_on_in_delay_data(&mut self, timing: u32, note: u8, velocity: f32) {
        // Load values that are used multiple times at the start
        let mut tap_counter = self.params.tap_counter.load(Ordering::SeqCst);
        let is_tap_slot_available = tap_counter < NUM_TAPS;
        let is_delay_note = !self.learned_notes.contains(note);
        let is_learning = self.is_learning.load(Ordering::SeqCst);
        let taps_unlocked = !self.enabled_actions.load(LOCK_TAPS);
        let min_tap_samples = self.min_tap_samples;
        let sample_rate = self.params.sample_rate.load(Ordering::SeqCst);
        let sample_rate_recip = sample_rate.recip();

        if !is_delay_note {
            self.last_played_notes.note_on(note);
        }

        if self.is_playing_action(LOCK_TAPS) {
            self.enabled_actions.toggle(LOCK_TAPS);
            self.last_played_notes
                .note_off(self.learned_notes.load(LOCK_TAPS));
        }

        let mut should_record_tap =
            is_delay_note && is_tap_slot_available && taps_unlocked && !is_learning;

        match self.counting_state {
            CountingState::TimeOut => {
                if is_delay_note && !is_learning && taps_unlocked {
                    // If in TimeOut state, reset and start new counting phase
                    self.clear_taps(timing, true);
                    self.params.first_note.store(note, Ordering::SeqCst);
                    self.params.preset_tempo.store(
                        self.params.host_tempo.load(Ordering::SeqCst),
                        Ordering::SeqCst,
                    );
                }
            }
            CountingState::CountingInBuffer => {
                // Validate and record a new tap within the buffer
                if timing - self.timing_last_event > min_tap_samples && should_record_tap {
                    self.samples_since_last_event = timing - self.timing_last_event;
                    self.timing_last_event = timing;
                } else {
                    should_record_tap = false; // Debounce or max taps reached, ignore tap
                }
            }
            CountingState::CountingAcrossBuffer => {
                // Handle cross-buffer taps timing
                if self.samples_since_last_event + timing > min_tap_samples && should_record_tap {
                    self.samples_since_last_event += timing;
                    self.timing_last_event = timing;
                    self.counting_state = CountingState::CountingInBuffer;
                } else {
                    should_record_tap = false; // Debounce or max taps reached, ignore tap
                }
            }
        }

        if is_learning {
            self.is_learning.store(false, Ordering::SeqCst);
            let index = self.learning_index.load(Ordering::SeqCst);
            self.learned_notes.store(index, note);
            self.last_learned_notes.store(index, note);
            self.last_played_notes.note_off(note);
        }

        // Check for timeout condition and reset if necessary
        if self.samples_since_last_event
            > (self.params.max_tap_time.load(Ordering::SeqCst) * sample_rate) as u32
        {
            self.counting_state = CountingState::TimeOut;
            self.timing_last_event = 0;
            self.samples_since_last_event = 0;
        } else if should_record_tap
            && self.counting_state != CountingState::TimeOut
            && self.samples_since_last_event > 0
            && velocity > 0.0
        {
            // Update tap information with timing and velocity
            if tap_counter > 0 {
                self.params.delay_times[tap_counter].store(
                    (self.samples_since_last_event as f32).mul_add(
                        sample_rate_recip,
                        self.params.delay_times[tap_counter - 1].load(Ordering::SeqCst),
                    ),
                    Ordering::SeqCst,
                );
            } else {
                self.params.delay_times[tap_counter].store(
                    self.samples_since_last_event as f32 * sample_rate_recip,
                    Ordering::SeqCst,
                );
            }
            self.params.velocities[tap_counter].store(velocity, Ordering::SeqCst);
            self.params.notes[tap_counter].store(note, Ordering::SeqCst);

            tap_counter += 1;
            self.params.tap_counter.store(tap_counter, Ordering::SeqCst);
            if tap_counter == NUM_TAPS {
                self.counting_state = CountingState::TimeOut;
                self.timing_last_event = 0;
                self.samples_since_last_event = 0;
            }
        }
        // Handle ActionTrigger events
        // LOCK_TAPS is handled at the start

        if !is_learning {
            let mute_in_note = self.learned_notes.load(MUTE_IN);
            let mute_out_note = self.learned_notes.load(MUTE_OUT);
            let is_toggle = self.params.global.mute_is_toggle.value();

            if note == mute_in_note {
                if is_toggle {
                    self.enabled_actions.toggle(MUTE_IN);
                } else {
                    self.enabled_actions.store(MUTE_IN, false);
                    self.enabled_actions.store(MUTE_OUT, false);
                }
            }
            if note == mute_out_note {
                if is_toggle {
                    self.enabled_actions.toggle(MUTE_OUT);
                } else {
                    self.enabled_actions.store(MUTE_IN, false);
                    self.enabled_actions.store(MUTE_OUT, false);
                }
            }
            if self.is_playing_action(CLEAR_TAPS) {
                self.clear_taps(timing, false);
            }
        }
    }

    fn store_note_off_in_delay_data(&self, note: u8) {
        // if we are in direct mode
        if !self.params.global.mute_is_toggle.value() {
            // mute the in and out
            for &mute in &[MUTE_IN, MUTE_OUT] {
                if note == self.learned_notes.load(mute) {
                    self.enabled_actions.store(mute, true);
                }
            }
        }
        self.last_played_notes.note_off(note);
    }

    fn clear_taps(&mut self, timing: u32, restart: bool) {
        self.enabled_actions.store(LOCK_TAPS, false);
        self.params.tap_counter.store(0, Ordering::SeqCst);

        self.params
            .previous_time_scaling_factor
            .store(NO_GUI_SMOOTHING, Ordering::SeqCst);
        self.params
            .previous_first_note_height
            .store(NO_GUI_SMOOTHING, Ordering::SeqCst);
        self.params
            .previous_panning_center_height
            .store(NO_GUI_SMOOTHING, Ordering::SeqCst);
        for i in 0..NUM_TAPS {
            self.params.previous_note_heights[i].store(NO_GUI_SMOOTHING, Ordering::SeqCst);
            self.params.previous_pan_foreground_lengths[i]
                .store(NO_GUI_SMOOTHING, Ordering::SeqCst);
            self.params.previous_pan_background_lengths[i]
                .store(NO_GUI_SMOOTHING, Ordering::SeqCst);
        }

        self.start_release_for_all_delay_taps();
        if restart {
            if self.params.global.mute_is_toggle.value() {
                self.enabled_actions.store(MUTE_IN, false);
            }
            self.counting_state = CountingState::CountingInBuffer;
            self.timing_last_event = timing;
        } else {
            self.counting_state = CountingState::TimeOut;
            self.timing_last_event = 0;
            self.samples_since_last_event = 0;
            self.params
                .first_note
                .store(NO_LEARNED_NOTE, Ordering::SeqCst);
        }
    }

    fn prepare_for_delay(&mut self, buffer_samples: usize) {
        let tap_counter = self.params.tap_counter.load(Ordering::SeqCst);
        let sample_rate = self.params.sample_rate.load(Ordering::SeqCst);
        let sample_rate_recip = sample_rate.recip();
        match self.counting_state {
            CountingState::TimeOut => {}
            CountingState::CountingInBuffer => {
                // Use saturating_sub to safely handle potential overflow
                self.samples_since_last_event =
                    (buffer_samples as u32).saturating_sub(self.timing_last_event);
                self.counting_state = CountingState::CountingAcrossBuffer;
            }
            CountingState::CountingAcrossBuffer => {
                self.samples_since_last_event += buffer_samples as u32;
            }
        }

        if self.samples_since_last_event
            > (self.params.max_tap_time.load(Ordering::SeqCst) * sample_rate) as u32
        {
            self.counting_state = CountingState::TimeOut;
            self.timing_last_event = 0;
            self.samples_since_last_event = 0;

            if tap_counter == 0 {
                self.params
                    .first_note
                    .store(NO_LEARNED_NOTE, Ordering::SeqCst);
            }
        }

        let samples_since_last_event = self.samples_since_last_event as f32;

        // Calculate the current time based on whether there are taps
        let current_time = if self.counting_state == CountingState::TimeOut {
            0.0
        } else if tap_counter > 0 {
            let last_delay_time = self.params.delay_times[tap_counter - 1].load(Ordering::SeqCst);
            samples_since_last_event.mul_add(sample_rate_recip, last_delay_time)
        } else {
            samples_since_last_event * sample_rate_recip
        };

        // Store the computed current time
        self.params
            .current_time
            .store(current_time, Ordering::SeqCst);
    }

    fn update_filters(&mut self) {
        // Extract params once to avoid repeated access
        let taps_params = &self.params.taps;
        let use_note_frequency = &taps_params.use_note_frequency.value();
        // let taps_params = &taps_params.velocity_high;

        // Pre-compute static values for the iteration
        let cutoff_main = taps_params.cutoff_main.value();
        let cutoff_mod = taps_params.cutoff_mod.value();
        let cutoff_octave = taps_params.cutoff_octave.value();
        let cutoff_transpose = taps_params.cutoff_transpose.value();
        let res_main = taps_params.res_main.value();
        let res_mod = taps_params.res_mod.value();
        let filter_type = &self.params.taps.filter_type.value();

        // Use par_iter_mut() if the collection is large enough to benefit from parallelization
        self.delay_taps
            .iter_mut()
            .filter(|tap| tap.is_alive)
            .for_each(|delay_tap| {
                let velocity = delay_tap.velocity;
                let velocity_factor = (velocity - 0.5) * 2.0; // Scale 0.0 to 1.0 range to -1.0 to 1.0.

                // Safety: We know this is unique as we have &mut access to delay_tap
                let filter_params = unsafe { Arc::get_mut_unchecked(&mut delay_tap.filter_params) };

                // Compute filter parameters
                let res = res_mod.mul_add(velocity_factor, res_main).clamp(0.0, 1.0);
                // cutoff_mod is in octaves, 8.0 octaves gives most of the range.
                let velocity_cutoff =
                    Self::modulate_log(cutoff_main, cutoff_mod * 8.0 * velocity_factor);
                let note_cutoff = util::f32_midi_note_to_freq(
                    f32::from(delay_tap.note)
                        + (cutoff_octave * 12) as f32
                        + cutoff_transpose as f32,
                );

                let cutoff = if *use_note_frequency {
                    note_cutoff
                } else {
                    velocity_cutoff
                }
                .clamp(10.0, 18_000.0);

                // Batch update filter parameters
                filter_params.set_resonance(res);
                filter_params.set_frequency(cutoff);
                filter_params.drive = 1.0;
                filter_params.ladder_mode = filter_type.0;

                // Update filter mix mode
                delay_tap.ladders.set_mix(filter_type.0);
            });
    }

    #[inline]
    fn _lerp(a: f32, b: f32, x: f32) -> f32 {
        (b - a).mul_add(x, a)
    }
    #[inline]
    fn _log_interpolate(a: f32, b: f32, x: f32) -> f32 {
        a * (b / a).powf(x)
    }

    #[inline]
    fn modulate_log(main: f32, modulation: f32) -> f32 {
        main * modulation.exp2()
    }
    // Takes a pan value and gives a delay offset, in samples
    // instead of adding delay, it subtracts delay from the other channel,
    // so we stay under the maximum delay value
    // TODO: this approach breaks down for delay taps shorter then the haas delay
    #[inline]
    fn pan_to_haas_samples(pan: f32, sample_rate: f32) -> (f32, f32) {
        let delay_samples = pan.abs() * (MAX_HAAS_MS * 0.001) * sample_rate;
        if pan > 0.0 {
            (0.0, -delay_samples) // Pan right: delay left
        } else {
            (-delay_samples, 0.0) // Pan left: delay right
        }
    }

    // for fn initialize():
    #[inline]
    fn calculate_buffer_size(buffer_size: u32, total_delay_samples: f64) -> u32 {
        ((total_delay_samples / f64::from(buffer_size)).ceil() as u32 * buffer_size)
            .next_power_of_two()
    }

    fn set_delay_buffer_size(&mut self, buffer_config: &BufferConfig) {
        let sample_rate = self.params.sample_rate.load(Ordering::SeqCst);
        let total_delay_samples = f64::from(sample_rate) * TOTAL_DELAY_SECONDS as f64;
        let min_size = Self::calculate_buffer_size(
            buffer_config.min_buffer_size.unwrap_or(1),
            total_delay_samples,
        );
        let max_size =
            Self::calculate_buffer_size(buffer_config.max_buffer_size, total_delay_samples);

        self.delay_buffer_size = u32::max(min_size, max_size);

        // Apply the calculated size to the delay buffers
        self.delay_buffer
            .iter_mut()
            .for_each(|buffer| buffer.clear_set_len(self.delay_buffer_size as usize));
        self.mute_in_delay_buffer
            .clear_set_len(self.delay_buffer_size as usize);
    }

    fn initialize_filter_parameters(&mut self) {
        for delay_tap in &mut self.delay_taps {
            let filter_params = unsafe { Arc::get_mut_unchecked(&mut delay_tap.filter_params) };
            filter_params.set_sample_rate(self.params.sample_rate.load(Ordering::SeqCst));
        }
    }

    fn update_peak_meter(&self, buffer: &mut Buffer, peak_meter: &AtomicF32) {
        // Access samples using the iterator
        for channel_samples in buffer.iter_samples() {
            let num_samples = channel_samples.len();
            let mut amplitude = 0.0;

            for sample in channel_samples {
                amplitude += sample.abs() * 0.5;
            }

            if self.params.editor_state.is_open() {
                amplitude = (amplitude / num_samples as f32).min(1.0);
                let current_peak_meter = peak_meter.load(Ordering::Relaxed);
                let new_peak_meter = if amplitude > current_peak_meter {
                    amplitude
                } else {
                    current_peak_meter.mul_add(
                        self.peak_meter_decay_weight,
                        amplitude * (1.0 - self.peak_meter_decay_weight),
                    )
                };

                peak_meter.store(new_peak_meter, Ordering::Relaxed);
            }
        }
    }

    #[inline]
    fn is_playing_action(&self, index: usize) -> bool {
        self.last_played_notes
            .is_playing(self.learned_notes.load(index))
    }

    /// Format a `f32` Hertz value as a rounded `Hz` below 1000 Hz, and as a rounded `kHz` value above
    /// 1000 Hz. This already includes the unit.
    fn v2s_f32_hz_then_khz_three_digits() -> Arc<dyn Fn(f32) -> String + Send + Sync> {
        Arc::new(move |value| {
            if value < 100.0 {
                format!("{value:.1} Hz")
            } else if value < 1000.0 {
                format!("{value:.0} Hz")
            } else if value < 10000.0 {
                format!("{:.2} kHz", value / 1000.0)
            } else {
                format!("{:.1} kHz", value / 1000.0)
            }
        })
    }

    fn v2s_f32_ms_then_s(total_digits: usize) -> Arc<dyn Fn(f32) -> String + Send + Sync> {
        Arc::new(move |value| format_time(value * 0.001, total_digits))
    }
    fn _v2s_f32_ms_then_s(total_digits: usize) -> Arc<dyn Fn(f32) -> String + Send + Sync> {
        Arc::new(move |value| {
            if value < 1000.0 {
                // Calculate the number of digits after the decimal to maintain a total of three digits
                let digits_after_decimal = (total_digits - value.trunc().to_string().len())
                    .max(0)
                    .min(total_digits - 1); // Ensure it's between 0 and 2
                format!("{value:.digits_after_decimal$} ms")
            } else {
                let seconds = value / 1000.0;
                // Same logic for seconds
                let digits_after_decimal =
                    (total_digits - seconds.trunc().to_string().len()).max(0);
                format!("{seconds:.digits_after_decimal$} s")
            }
        })
    }

    // TODO: test, when skia vizia fixes the bug where it won't go into text mode
    fn s2v_f32_s_or_ms_if_bigger_than(
        max_val: f32,
    ) -> Arc<dyn Fn(&str) -> Option<f32> + Send + Sync> {
        Arc::new(move |string| {
            let string = string.trim().to_lowercase();

            if let Some(ms_value_str) = string
                .strip_suffix("ms")
                .or_else(|| string.strip_suffix(" ms"))
            {
                // Explicitly specified as milliseconds
                return ms_value_str.trim().parse::<f32>().ok();
            }

            if let Some(s_value_str) = string
                .strip_suffix("s")
                .or_else(|| string.strip_suffix(" s"))
            {
                // Explicitly specified as seconds
                return s_value_str.trim().parse::<f32>().ok().map(|s| s * 1000.0);
            }

            // If no specification, interpret as seconds and check
            string.parse::<f32>().ok().map(|val| {
                let ms_value = val * 1000.0;
                if ms_value < max_val {
                    ms_value // Treat it as milliseconds
                } else {
                    val // Keep as seconds
                }
            })
        })
    }

    fn _s2v_f32_ms_then_s() -> Arc<dyn Fn(&str) -> Option<f32> + Send + Sync> {
        Arc::new(move |string| {
            let string = string.trim().to_lowercase();
            if let Some(ms_value_str) = string
                .strip_suffix("ms")
                .or_else(|| string.strip_suffix(" ms"))
            {
                return ms_value_str.trim().parse::<f32>().ok();
            }
            if let Some(s_value_str) = string
                .strip_suffix("s")
                .or_else(|| string.strip_suffix(" s"))
            {
                return s_value_str.trim().parse::<f32>().ok().map(|s| s * 1000.0);
            }
            string.parse::<f32>().ok()
        })
    }
    fn _v2s_i32_note() -> Arc<dyn Fn(i32) -> String + Send + Sync> {
        Arc::new(move |value| {
            let note_nr = value as u8; // Convert the floating-point value to the nearest u8
            if value < 0 {
                "initial note".to_string()
            } else {
                let note_name = util::NOTES[(note_nr % 12) as usize];
                let octave = (note_nr / 12) as i8 - 1; // Correct the octave calculation
                format!("{note_name}{octave}") // Ensure correct value formatting
            }
        })
    }
    fn _s2v_i32_note() -> Arc<dyn Fn(&str) -> Option<i32> + Send + Sync> {
        Arc::new(move |string| {
            let trimmed_string = string.trim().to_lowercase();

            // Check if the string contains specific keywords
            let keywords = ["first", "note", "pattern"];
            if keywords
                .iter()
                .any(|&keyword| trimmed_string.contains(keyword))
            {
                return Some(-1);
            }
            let len = trimmed_string.len();
            if len < 2 {
                // if it's short, return to default: "first note of pattern"
                return Some(-1);
            }

            // The note part could be one or two characters, based on whether it includes a sharp or flat (e.g., "C", "C#", "D")
            let is_sharp_or_flat =
                trimmed_string.get(1..2) == Some("#") || trimmed_string.get(1..2) == Some("b");
            let note_length = if is_sharp_or_flat { 2 } else { 1 };

            // Extract note and octave
            let note_name = &trimmed_string[..note_length];
            let octave_part = &trimmed_string[note_length..];

            // Parse the octave
            if let Ok(octave) = octave_part.parse::<i32>() {
                if let Some(note_index) = util::NOTES
                    .iter()
                    .position(|&n| n.eq_ignore_ascii_case(note_name))
                {
                    return Some(note_index as i32 + (octave + 1) * 12);
                }
            }

            None
        })
    }

    fn v2s_i32_channel() -> Arc<dyn Fn(i32) -> String + Send + Sync> {
        Arc::new(move |value| {
            if value < 1 {
                "any channel".to_string()
            } else {
                format!("channel {value}")
            }
        })
    }
    fn s2v_i32_channel() -> Arc<dyn Fn(&str) -> Option<i32> + Send + Sync> {
        Arc::new(move |string| {
            // Retain only numeric characters from the input string
            let numeric_string: String = string.chars().filter(char::is_ascii_digit).collect();

            // Attempt to parse the string as an i32
            if let Ok(number) = numeric_string.parse::<i32>() {
                // Check if the number is within the desired range
                if (1..=16).contains(&number) {
                    return Some(number);
                }
            }
            // If parsing fails or the number is out of range, return 0
            Some(0)
        })
    }

    fn start_tap(&mut self, new_index: usize, host_tempo: f32) {
        let sample_rate = self.params.sample_rate.load(Ordering::SeqCst);
        let global_params = &self.params.global;

        let conversion_factor = self.params.preset_tempo.load(Ordering::SeqCst) / host_tempo;

        // nih_log!("conversion_factor: {conversion_factor}");
        let delay_samples = (self.params.delay_times[new_index].load(Ordering::SeqCst)
            * sample_rate
            * conversion_factor) as u32;
        let note = self.params.notes[new_index].load(Ordering::SeqCst);
        let velocity = self.params.velocities[new_index].load(Ordering::SeqCst);
        let velocity_factor = (velocity - 0.5) * 2.0; // Scale 0.0 to 1.0 range to -1.0 to 1.0.
        let drive_mod = self.params.taps.drive_mod.value();
        // don't smooth the drive when we start a new tap
        self.delay_taps[new_index]
            .drive_mod_smoother
            .reset(util::db_to_gain_fast(drive_mod * velocity_factor));

        self.should_update_filter.store(true, Ordering::Release);
        if global_params.mute_is_toggle.value() {
            self.enabled_actions.store(MUTE_OUT, false);
        }

        let amp_envelope = Smoother::new(SmoothingStyle::Linear(global_params.attack_ms.value()));
        if global_params.mute_is_toggle.value() {
            amp_envelope.set_target(sample_rate, 1.0);
        }

        let mut found_inactive = None;
        let mut found_oldest = None;
        let mut oldest_id = u64::MAX;
        let mut found_inactive_index = None;
        let mut found_oldest_index = None;

        for (index, delay_tap) in self.delay_taps.iter_mut().enumerate() {
            if delay_tap.is_alive {
                // Recycle an old tap if `delay_samples` and `note` match
                // nih_log!("delay_tap.delay_time == delay_samples: {}, {delay_samples}", delay_tap.delay_time);
                if delay_tap.delay_time == delay_samples && delay_tap.note == note {
                    delay_tap.velocity = velocity;
                    delay_tap.releasing = false;
                    delay_tap.amp_envelope.style =
                        SmoothingStyle::Linear(self.params.global.attack_ms.value());
                    delay_tap.amp_envelope.set_target(sample_rate, 1.0);
                    self.meter_indexes[new_index].store(index, Ordering::Relaxed);
                    // delay_tap.meter_index = new_index;
                    // nih_log!("recycled tap {index}");
                    return;
                } else if delay_tap.internal_id < oldest_id {
                    // Track the oldest active tap
                    oldest_id = delay_tap.internal_id;
                    found_oldest = Some(delay_tap);
                    found_oldest_index = Some(index);
                }
            } else if found_inactive.is_none() {
                // Track the first non-active tap
                found_inactive = Some(delay_tap);
                found_inactive_index = Some(index);
                // Stop iterating when we found an inactive tap
                break;
            }
        }

        if let Some(delay_tap) = found_inactive {
            // Initialize the non-active tap
            // nih_log!("start_tap: inactive tap: {}", found_inactive_index.unwrap());
            delay_tap.init(
                amp_envelope,
                self.next_internal_id,
                delay_samples,
                note,
                velocity,
            );
            self.next_internal_id = self.next_internal_id.wrapping_add(1);
            self.meter_indexes[new_index].store(found_inactive_index.unwrap(), Ordering::Relaxed);
        } else if let Some(oldest_delay_tap) = found_oldest {
            // nih_log!("start_tap: oldest tap: {}", found_oldest_index.unwrap());
            // Replace the oldest tap if needed
            oldest_delay_tap.init(
                amp_envelope,
                self.next_internal_id,
                delay_samples,
                note,
                velocity,
            );
            self.next_internal_id = self.next_internal_id.wrapping_add(1);
            self.meter_indexes[new_index].store(found_oldest_index.unwrap(), Ordering::Relaxed);
        }
    }

    /// Start the release process for all delay taps by changing their amplitude envelope.
    fn start_release_for_all_delay_taps(&mut self) {
        for delay_tap in &mut self.delay_taps {
            delay_tap.releasing = true;
            delay_tap.amp_envelope.style =
                SmoothingStyle::Linear(self.params.global.release_ms.value());
            delay_tap
                .amp_envelope
                .set_target(self.params.sample_rate.load(Ordering::SeqCst), 0.0);
        }
    }

    /// Set mute for all delay taps by changing their amplitude envelope.
    fn set_mute_for_all_delay_taps(&mut self) {
        let is_playing_mute_out = self.is_playing_action(MUTE_OUT);

        // for (tap_index, delay_tap) in self.delay_taps.iter_mut().enumerate() {
        for delay_tap in &mut self.delay_taps {
            let is_toggle = self.params.global.mute_is_toggle.value();
            let mute_in_delayed = delay_tap.mute_in_delayed[0];
            let mute_out = self.enabled_actions.load(MUTE_OUT);

            let new_mute = if is_toggle {
                mute_in_delayed || mute_out
            } else if is_playing_mute_out != mute_out {
                !is_playing_mute_out
            } else {
                mute_in_delayed
            };
            let muted = new_mute || !delay_tap.is_alive;
            let needs_toggle = delay_tap.is_muted != muted;

            // if delay_tap.is_alive {
            // nih_log!("delay_tap.is_muted: {tap_index} {}", delay_tap.is_muted);
            // }
            if needs_toggle {
                if muted {
                    delay_tap.amp_envelope.style =
                        SmoothingStyle::Linear(self.params.global.release_ms.value());
                    delay_tap
                        .amp_envelope
                        .set_target(self.params.sample_rate.load(Ordering::SeqCst), 0.0);
                    // nih_log!("set target {tap_index} 0.0");
                    // don't set releasing, since we want the tap to stay alive
                    // delay_tap.releasing = true;
                } else {
                    delay_tap.amp_envelope.style =
                        SmoothingStyle::Linear(self.params.global.attack_ms.value());
                    delay_tap
                        .amp_envelope
                        .set_target(self.params.sample_rate.load(Ordering::SeqCst), 1.0);
                    // nih_log!("set target 1.0");
                    delay_tap.is_alive = true;
                    delay_tap.is_audible = true;
                    delay_tap.releasing = false;
                }
                delay_tap.is_muted = muted;
            }
        }
    }
}

impl ClapPlugin for Del2 {
    const CLAP_ID: &'static str = "https://magnetophon.nl/DEL2";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("A rhythm delay with space.");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;

    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Delay,
        ClapFeature::Filter,
        ClapFeature::Distortion,
        ClapFeature::Mixing,
        ClapFeature::Stereo,
        // TODO: Mono,
        // TODO: Sampler,
        // TODO: PitchShifter,
        // TODO: Compressor,
        // TODO: Limiter,
        // TODO: Reverb,
        // TODO: Tremolo,
        // TODO: MultiEffects,
    ];
}

impl Vst3Plugin for Del2 {
    const VST3_CLASS_ID: [u8; 16] = *b"magnetophon/DEL2";

    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Delay,
        Vst3SubCategory::Filter,
        Vst3SubCategory::Distortion,
        Vst3SubCategory::Spatial,
        Vst3SubCategory::Stereo,
    ];
}

#[derive(PartialEq, Debug, Clone, Copy)]
pub struct MyLadderMode(LadderMode);

impl MyLadderMode {
    // Define the order of modes for interpolation
    const fn sequence() -> &'static [LadderMode] {
        &[
            LadderMode::LP6,
            LadderMode::LP12,
            LadderMode::LP18,
            LadderMode::LP24,
            LadderMode::BP12,
            LadderMode::BP24,
            LadderMode::HP6,
            LadderMode::HP12,
            LadderMode::HP18,
            LadderMode::HP24,
            LadderMode::N12,
        ]
    }

    // TODO: remove?
    #[allow(dead_code)]
    fn index(self) -> Option<usize> {
        Self::sequence().iter().position(|&mode| mode == self.0)
    }

    fn _lerp(start: Self, end: Self, t: f32) -> LadderMode {
        let start_index = start.index().unwrap_or(0);
        let end_index = end.index().unwrap_or(Self::sequence().len() - 1);

        let t = t.clamp(0.0, 1.0);

        let interpolated_index = t
            .mul_add(end_index as f32 - start_index as f32, start_index as f32)
            .round() as usize;

        Self::from_index(interpolated_index).0
    }

    const fn lp6() -> Self {
        Self(LadderMode::LP6)
    }
}

impl Enum for MyLadderMode {
    fn variants() -> &'static [&'static str] {
        &[
            "LP6", "LP12", "LP18", "LP24", "BP12", "BP24", "HP6", "HP12", "HP18", "HP24", "N12",
        ]
    }

    fn ids() -> Option<&'static [&'static str]> {
        Some(&[
            "lp6_id", "lp12_id", "lp18_id", "lp24_id", "bp12_id", "bp24_id", "hp6_id", "hp12_id",
            "hp18_id", "hp24_id", "n12_id",
        ])
    }

    fn to_index(self) -> usize {
        match self.0 {
            LadderMode::LP6 => 0,
            LadderMode::LP12 => 1,
            LadderMode::LP18 => 2,
            LadderMode::LP24 => 3,
            LadderMode::BP12 => 4,
            LadderMode::BP24 => 5,
            LadderMode::HP6 => 6,
            LadderMode::HP12 => 7,
            LadderMode::HP18 => 8,
            LadderMode::HP24 => 9,
            LadderMode::N12 => 10,
        }
    }

    fn from_index(index: usize) -> Self {
        Self(match index {
            0 => LadderMode::LP6,
            1 => LadderMode::LP12,
            2 => LadderMode::LP18,
            3 => LadderMode::LP24,
            4 => LadderMode::BP12,
            5 => LadderMode::BP24,
            6 => LadderMode::HP6,
            7 => LadderMode::HP12,
            8 => LadderMode::HP18,
            9 => LadderMode::HP24,
            10 => LadderMode::N12,
            _ => panic!("Invalid index for LadderMode"),
        })
    }
}
pub struct AtomicBoolArray {
    data: AtomicU8,
}

impl AtomicBoolArray {
    const fn new() -> Self {
        Self {
            data: AtomicU8::new(0),
        }
    }

    fn load(&self, index: usize) -> bool {
        assert!(index < 8, "Index out of bounds");
        let mask = 1 << index;
        self.data.load(Ordering::SeqCst) & mask != 0
    }
    fn store(&self, index: usize, value: bool) {
        assert!(index < 8, "Index out of bounds");
        let mask = 1 << index;
        self.data
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| {
                Some(if value {
                    v | mask // Set bit
                } else {
                    v & !mask // Clear bit
                })
            })
            .expect("Atomic update failed");
    }

    fn load_u8(&self) -> u8 {
        self.data.load(Ordering::SeqCst)
    }
    fn store_u8(&self, new_value: u8) {
        self.data.store(new_value, Ordering::SeqCst);
    }

    fn toggle(&self, index: usize) {
        assert!(index < 8, "Index out of bounds");
        let mask = 1 << index;
        self.data.fetch_xor(mask, Ordering::SeqCst);
    }
}
pub struct AtomicByteArray {
    data: AtomicU64,
}

impl AtomicByteArray {
    fn new(initial_byte: u8) -> Self {
        let u64_value = u64::from(initial_byte)
            | (u64::from(initial_byte) << 8)
            | (u64::from(initial_byte) << 16)
            | (u64::from(initial_byte) << 24)
            | (u64::from(initial_byte) << 32)
            | (u64::from(initial_byte) << 40)
            | (u64::from(initial_byte) << 48)
            | (u64::from(initial_byte) << 56);

        Self {
            data: AtomicU64::new(u64_value),
        }
    }

    fn load(&self, index: usize) -> u8 {
        assert!(index < 8, "Index out of bounds");
        let value = self.data.load(Ordering::SeqCst);
        ((value >> (index * 8)) & 0xFF) as u8
    }

    fn store(&self, index: usize, byte: u8) {
        assert!(index < 8, "Index out of bounds");
        let mask = !(0xFFu64 << (index * 8));
        let new_value =
            (self.data.load(Ordering::SeqCst) & mask) | ((u64::from(byte)) << (index * 8));
        self.data.store(new_value, Ordering::SeqCst);
    }
    fn load_u64(&self) -> u64 {
        self.data.load(Ordering::SeqCst)
    }
    fn store_u64(&self, new_value: u64) {
        self.data.store(new_value, Ordering::SeqCst);
    }
    fn contains(&self, byte: u8) -> bool {
        let value = self.data.load(Ordering::SeqCst);
        let byte_u64 = u64::from(byte);

        for i in 0..8 {
            let shifted_byte = (value >> (i * 8)) & 0xFF;
            if shifted_byte == byte_u64 {
                return true;
            }
        }
        false
    }
}

// Create a newtype for your Arc<AtomicByteArray>
struct ArcAtomicByteArray(Arc<AtomicByteArray>);

impl PersistentField<'_, u64> for ArcAtomicByteArray {
    fn set(&self, new_value: u64) {
        self.0.store_u64(new_value);
    }

    fn map<F, R>(&self, f: F) -> R
    where
        F: Fn(&u64) -> R,
    {
        let value = self.0.load_u64();
        f(&value)
    }
}

struct ArcAtomicBoolArray(Arc<AtomicBoolArray>);

impl PersistentField<'_, u8> for ArcAtomicBoolArray {
    fn set(&self, new_value: u8) {
        self.0.store_u8(new_value);
    }

    fn map<F, R>(&self, f: F) -> R
    where
        F: Fn(&u8) -> R,
    {
        let value = self.0.load_u8();
        f(&value)
    }
}
struct AtomicU8Array([Arc<AtomicU8>; NUM_TAPS]);
pub struct AtomicUsizeArray([Arc<AtomicUsize>; NUM_TAPS]);
#[allow(dead_code)]
struct AtomicU32Array([Arc<AtomicU32>; NUM_TAPS]);
pub struct AtomicF32Array([Arc<AtomicF32>; NUM_TAPS]);

// Implement PersistentField for AtomicU8Array
impl PersistentField<'_, [u8; NUM_TAPS]> for AtomicU8Array {
    fn set(&self, new_values: [u8; NUM_TAPS]) {
        for (atomic, &new_value) in self.0.iter().zip(new_values.iter()) {
            atomic.store(new_value, Ordering::SeqCst);
        }
    }

    fn map<F, R>(&self, f: F) -> R
    where
        F: Fn(&[u8; NUM_TAPS]) -> R,
    {
        let values: [u8; NUM_TAPS] = self
            .0
            .iter()
            .map(|arc| arc.load(Ordering::SeqCst))
            .collect::<Vec<u8>>()
            .try_into()
            .unwrap_or_else(|_| panic!("Size mismatch"));
        f(&values)
    }
}

impl PersistentField<'_, [u32; NUM_TAPS]> for AtomicU32Array {
    fn set(&self, new_values: [u32; NUM_TAPS]) {
        for (atomic, &new_value) in self.0.iter().zip(new_values.iter()) {
            atomic.store(new_value, Ordering::SeqCst);
        }
    }

    fn map<F, R>(&self, f: F) -> R
    where
        F: Fn(&[u32; NUM_TAPS]) -> R,
    {
        let values: [u32; NUM_TAPS] = self
            .0
            .iter()
            .map(|arc| arc.load(Ordering::SeqCst))
            .collect::<Vec<u32>>()
            .try_into()
            .unwrap_or_else(|_| panic!("Size mismatch"));
        f(&values)
    }
}

impl PersistentField<'_, [f32; NUM_TAPS]> for AtomicF32Array {
    fn set(&self, new_values: [f32; NUM_TAPS]) {
        for (atomic, &new_value) in self.0.iter().zip(new_values.iter()) {
            atomic.store(new_value, Ordering::SeqCst);
        }
    }

    fn map<F, R>(&self, f: F) -> R
    where
        F: Fn(&[f32; NUM_TAPS]) -> R,
    {
        let values: [f32; NUM_TAPS] = self
            .0
            .iter()
            .map(|arc| arc.load(Ordering::SeqCst))
            .collect::<Vec<f32>>()
            .try_into()
            .unwrap_or_else(|_| panic!("Size mismatch"));
        f(&values)
    }
}

// Implement the Index trait to allow for array-style access
macro_rules! impl_index_for_atomic_array {
    ($atomic_array_type:ident, $atomic_type:ty) => {
        impl Index<usize> for $atomic_array_type {
            type Output = Arc<$atomic_type>;

            fn index(&self, index: usize) -> &Self::Output {
                &self.0[index]
            }
        }
    };
}

// Apply the macro to different types
impl_index_for_atomic_array!(AtomicU8Array, AtomicU8);
impl_index_for_atomic_array!(AtomicUsizeArray, AtomicUsize);
impl_index_for_atomic_array!(AtomicU32Array, AtomicU32);
impl_index_for_atomic_array!(AtomicF32Array, AtomicF32);

// Represents the last played notes with capabilities for managing active notes.
pub struct LastPlayedNotes {
    state: AtomicU8,
    notes: AtomicByteArray,
    sequence: AtomicByteArray,
    current_sequence: AtomicU8,
    active_notes: AtomicBoolArray, // Tracks active status of notes
}

impl LastPlayedNotes {
    /// Constructs a new instance of `LastPlayedNotes`.
    fn new() -> Self {
        // Initializes the state, notes, sequence, and current_sequence fields.
        Self {
            state: AtomicU8::new(0),
            notes: AtomicByteArray::new(0),
            sequence: AtomicByteArray::new(0),
            current_sequence: AtomicU8::new(1),
            active_notes: AtomicBoolArray::new(), // Initialize new field
        }
    }

    /// Handles the 'note on' event.
    fn note_on(&self, note: u8) {
        let mut current_state = self.state.load(Ordering::SeqCst);

        // Check if the note is already in the table and reactivate if so.
        if let Some(index) = (0..8).find(|&i| self.notes.load(i) == note) {
            // Update sequence and reactivate the note.
            self.sequence
                .store(index, self.current_sequence.fetch_add(1, Ordering::SeqCst));
            self.active_notes.store(index, true); // Mark as active

            // Ensure it's marked as active in the state.
            while let Err(actual_state) = self.state.compare_exchange_weak(
                current_state,
                current_state | (1 << index), // Set this index as active
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                current_state = actual_state;
            }
            return;
        }

        loop {
            // Try to find an empty slot in the current state.
            if let Some(index) = (0..8).find(|&i| (current_state & (1 << i)) == 0) {
                // Attempt to occupy this empty spot.
                while let Err(actual_state) = self.state.compare_exchange_weak(
                    current_state,
                    current_state | (1 << index),
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    current_state = actual_state;
                }

                // Store the note and its sequence once the position is successfully claimed.
                self.notes.store(index, note);
                self.sequence
                    .store(index, self.current_sequence.fetch_add(1, Ordering::SeqCst));
                self.active_notes.store(index, true); // Mark as active
                break;
            }
        }
    }

    /// Handles the 'note off' event.
    fn note_off(&self, note: u8) {
        let mut current_state = self.state.load(Ordering::SeqCst);

        if let Some(index) = (0..8).find(|&i| self.notes.load(i) == note) {
            // Calculate new state after disabling the note at the found index.
            loop {
                let new_state = current_state & !(1 << index);
                match self.state.compare_exchange_weak(
                    current_state,
                    new_state,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => {
                        self.sequence.store(index, 0); // Reset sequence
                        self.active_notes.store(index, false); // Mark as inactive
                        break;
                    }
                    Err(actual_state) => current_state = actual_state,
                }
            }
        }
    }

    /// Checks if a note is currently being played.
    fn is_playing(&self, note: u8) -> bool {
        for i in 0..8 {
            if self.notes.load(i) == note {
                return self.active_notes.load(i);
            }
        }
        false
    }

    /// Print the notes for testing purposes.
    fn _print_notes(&self) {
        // Width used for formatting alignment
        const WIDTH: usize = 4;

        for i in 0..8 {
            let note = self.notes.load(i);
            if self.is_playing(note) {
                // Print active notes
                print!("{note:>WIDTH$}");
            } else {
                // Print placeholder for inactive notes
                print!("{:>WIDTH$}", "_");
            }
        }
        println!();
    }
}
fn format_time(seconds: f32, total_digits: usize) -> String {
    if seconds < 1.0 {
        let milliseconds = seconds * 1000.0;
        let digits_after_decimal = (total_digits - milliseconds.trunc().to_string().len())
            .max(0)
            .min(total_digits - 1);
        format!("{milliseconds:.digits_after_decimal$} ms")
    } else {
        let digits_after_decimal = (total_digits - seconds.trunc().to_string().len()).max(0);
        format!("{seconds:.digits_after_decimal$} s")
    }
}

nih_export_clap!(Del2);
nih_export_vst3!(Del2);
