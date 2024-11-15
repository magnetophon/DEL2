/*
TODO:

- smooth all dsp params (at the end of the interpolation?)

- evaluate filters:
https://github.com/AquaEBM/svf
https://github.com/neodsp/simper-filter
 also simper, with simd
// https://github.com/SamiPerttu/fundsp/blob/801ad0a0c97838f9744d0dbe506512215f780b7d/src/svf.rs#L2

- make mutes sample-accurate

- optional: live-mode / daw-mode switch
  - compensate for host latency by adjusting the delay read index
 */

// #![allow(non_snake_case)]
// #![deny(clippy::all)]
#![warn(clippy::pedantic)]
// #![warn(clippy::cargo)]
#![warn(clippy::nursery)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]
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
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;
use synfx_dsp::fh_va::{FilterParams, LadderFilter, LadderMode};

mod delay_tap;
mod editor;
use delay_tap::DelayTap;

// max seconds per tap
const MAX_TAP_SECONDS: usize = 10;
const NUM_TAPS: usize = 8;
const TOTAL_DELAY_SECONDS: usize = MAX_TAP_SECONDS * NUM_TAPS;
const MAX_SAMPLE_RATE: usize = 192_000;
const TOTAL_DELAY_SAMPLES: usize = TOTAL_DELAY_SECONDS * MAX_SAMPLE_RATE;
const VELOCITY_LOW_NAME_PREFIX: &str = "low velocity";
const VELOCITY_HIGH_NAME_PREFIX: &str = "high velocity";
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
const MAX_HAAS_MS: f32 = 5.0;

// Polyphonic modulation works by assigning integer IDs to parameters. Pattern matching on these in
// `PolyModulation` and `MonoAutomation` events makes it possible to easily link these events to the
// correct parameter.
const GAIN_POLY_MOD_ID: u32 = 0;

struct Del2 {
    params: Arc<Del2Params>,

    /// The effect's delay taps. Inactive delay taps will be set to `None` values.
    delay_taps: [Option<DelayTap>; NUM_TAPS],
    /// The next internal delay tap ID, used only to figure out the oldest delay tap for "voice stealing".
    /// This is incremented by one each time a delay tap is created.
    next_internal_id: u64,

    filter_params: [Arc<FilterParams>; NUM_TAPS],
    ladders: [LadderFilter; NUM_TAPS],

    // delay write buffer
    delay_buffer: [BMRingBuf<f32>; 2],
    delayed_audio_l: [Box<[f32]>; NUM_TAPS],
    delayed_audio_r: [Box<[f32]>; NUM_TAPS],
    mute_in_delay_buffer: BMRingBuf<bool>,
    mute_in_delayed: [Box<[bool]>; NUM_TAPS],
    mute_in_delay_temp_buffer: Box<[bool]>,

    // for the smoothers
    gain: Box<[f32]>,
    dry_wet: Box<[f32]>,
    output_gain: Box<[f32]>,
    global_drive: Box<[f32]>,
    delay_tap_gain: Box<[f32]>,
    delay_tap_amp_envelope: Box<[f32]>,

    // N counters to know where in the fade in we are: 0 is the start
    amp_envelopes: [Smoother<f32>; NUM_TAPS],
    sample_rate: f32,
    peak_meter_decay_weight: f32,
    input_meter: Arc<AtomicF32>,
    output_meter: Arc<AtomicF32>,
    tap_meters: Arc<AtomicF32Array>,
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
    delay_times: AtomicU32Array,
    #[persist = "velocities"]
    velocities: AtomicF32Array,
    #[persist = "pans"]
    pans: AtomicF32Array,
    #[persist = "notes"]
    notes: AtomicU8Array,
    #[persist = "current-tap"]
    current_tap: Arc<AtomicUsize>,
    current_time: Arc<AtomicU32>,
    max_tap_samples: Arc<AtomicU32>,
    #[persist = "first-note"]
    first_note: Arc<AtomicU8>,
    previous_time_scaling_factor: Arc<AtomicF32>,
    previous_note_heights: AtomicF32Array,
    previous_first_note_height: Arc<AtomicF32>,

    /// A voice's gain. This can be polyphonically modulated.
    #[id = "gain"]
    gain: FloatParam,
}

/// Contains the global parameters.
#[derive(Params)]
struct GlobalParams {
    #[id = "dry_wet"]
    dry_wet: FloatParam,
    #[id = "output_gain"]
    pub output_gain: FloatParam,
    #[id = "global_drive"]
    pub global_drive: FloatParam,
    #[id = "mute_is_toggle"]
    mute_is_toggle: BoolParam,
    #[id = "attack_ms"]
    attack_ms: FloatParam,
    #[id = "release_ms"]
    release_ms: FloatParam,
    #[id = "min_tap_milliseconds"]
    pub min_tap_milliseconds: FloatParam,
    #[id = "max_tap_seconds"]
    pub max_tap_seconds: FloatParam,
}

impl GlobalParams {
    pub fn new(enabled_actions: Arc<AtomicBoolArray>, learned_notes: Arc<AtomicByteArray>) -> Self {
        Self {
            dry_wet: FloatParam::new("mix", 1.0, FloatRange::Linear { min: 0.0, max: 1.0 })
                .with_unit("%")
                .with_smoother(SmoothingStyle::Linear(15.0))
                .with_value_to_string(formatters::v2s_f32_percentage(0))
                .with_string_to_value(formatters::s2v_f32_percentage()),
            output_gain: FloatParam::new(
                "out gain",
                util::db_to_gain(0.0),
                FloatRange::Skewed {
                    min: util::db_to_gain(-30.0),
                    max: util::db_to_gain(30.0),
                    factor: FloatRange::gain_skew_factor(-30.0, 30.0),
                },
            )
            .with_smoother(SmoothingStyle::Logarithmic(50.0))
            .with_unit(" dB")
            .with_value_to_string(formatters::v2s_f32_gain_to_db(1))
            .with_string_to_value(formatters::s2v_f32_gain_to_db()),
            global_drive: FloatParam::new(
                "drive",
                util::db_to_gain(0.0),
                FloatRange::Skewed {
                    min: util::db_to_gain(-30.0),
                    max: util::db_to_gain(30.0),
                    factor: FloatRange::gain_skew_factor(-30.0, 30.0),
                },
            )
            .with_smoother(SmoothingStyle::Logarithmic(50.0))
            .with_unit(" dB")
            .with_value_to_string(formatters::v2s_f32_gain_to_db(1))
            .with_string_to_value(formatters::s2v_f32_gain_to_db()),
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
                    min: 0.0,
                    max: 5_000.0,
                    factor: FloatRange::skew_factor(-1.2),
                },
            )
            .with_value_to_string(Del2::v2s_f32_ms_then_s(3))
            .with_string_to_value(Del2::s2v_f32_ms_then_s()),
            release_ms: FloatParam::new(
                "Release",
                420.0,
                FloatRange::Skewed {
                    min: 5.0,
                    max: 20_000.0,
                    factor: FloatRange::skew_factor(-1.5),
                },
            )
            .with_value_to_string(Del2::v2s_f32_ms_then_s(3))
            .with_string_to_value(Del2::s2v_f32_ms_then_s()),
            min_tap_milliseconds: FloatParam::new(
                "min tap",
                13.0,
                FloatRange::Skewed {
                    min: 1.0,
                    max: 1000.0,
                    factor: FloatRange::skew_factor(-1.5),
                },
            )
            .with_step_size(0.1)
            .with_unit(" ms"),
            max_tap_seconds: FloatParam::new(
                "max tap",
                3.03,
                FloatRange::Skewed {
                    min: 0.5,
                    max: MAX_TAP_SECONDS as f32,
                    factor: FloatRange::skew_factor(-0.8),
                },
            )
            .with_step_size(0.01)
            .with_unit(" s"),
        }
    }
}

/// Contains the high and low tap parameters.
#[derive(Params)]
pub struct TapsParams {
    #[id = "panning_center"]
    pub panning_center: FloatParam,
    #[id = "panning_amount"]
    pub panning_amount: FloatParam,
    #[id = "note_to_cutoff_amount"]
    pub note_to_cutoff_amount: FloatParam,
    #[id = "velocity_to_cutoff_amount"]
    pub velocity_to_cutoff_amount: FloatParam,

    #[nested(id_prefix = "velocity_low", group = "velocity_low")]
    pub velocity_low: Arc<FilterGuiParams>,
    #[nested(id_prefix = "velocity_high", group = "velocity_high")]
    pub velocity_high: Arc<FilterGuiParams>,
}

impl TapsParams {
    pub fn new(should_update_filter: Arc<AtomicBool>) -> Self {
        Self {
            panning_center: FloatParam::new(
                "panning center",
                -1.0,
                FloatRange::Linear {
                    min: -1.0,
                    max: 127.0,
                },
            )
            .with_value_to_string(Del2::v2s_f32_note())
            .with_string_to_value(Del2::s2v_f32_note()),
            panning_amount: FloatParam::new(
                "panning_amount",
                0.0,
                FloatRange::SymmetricalSkewed {
                    min: -1.0,
                    max: 1.0,
                    factor: FloatRange::skew_factor(-1.2),
                    center: 0.0,
                },
            )
            .with_unit(" %")
            .with_value_to_string(formatters::v2s_f32_percentage(0))
            .with_string_to_value(formatters::s2v_f32_percentage()),
            note_to_cutoff_amount: FloatParam::new(
                "note -> cutoff",
                0.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_unit(" %")
            .with_value_to_string(formatters::v2s_f32_percentage(0))
            .with_string_to_value(formatters::s2v_f32_percentage()),
            velocity_to_cutoff_amount: FloatParam::new(
                "velocity -> cutoff",
                1.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_unit(" %")
            .with_value_to_string(formatters::v2s_f32_percentage(0))
            .with_string_to_value(formatters::s2v_f32_percentage()),
            velocity_low: Arc::new(FilterGuiParams::new(
                VELOCITY_LOW_NAME_PREFIX,
                should_update_filter.clone(),
                124.0,                  // Default cutoff for velocity_low
                0.7,                    // Default res for velocity_low
                util::db_to_gain(13.0), // Default drive for velocity_low
                MyLadderMode::lp6(),    // Default mode for velocity_low
            )),
            velocity_high: Arc::new(FilterGuiParams::new(
                VELOCITY_HIGH_NAME_PREFIX,
                should_update_filter,
                6_000.0,               // Default cutoff for velocity_high
                0.4,                   // Default res for velocity_high
                util::db_to_gain(6.0), // Default drive for velocity_high
                MyLadderMode::lp6(),   // Default mode for velocity_high
            )),
        }
    }
}

/// This struct contains the parameters for either the high or low tap. The `Params`
/// trait is implemented manually to avoid copy-pasting parameters for both types of compressor.
/// Both versions will have a parameter ID and a parameter name prefix to distinguish them.
#[derive(Params)]
pub struct FilterGuiParams {
    #[id = "cutoff"]
    pub cutoff: FloatParam,
    #[id = "res"]
    pub res: FloatParam,
    #[id = "drive"]
    pub drive: FloatParam,
    #[id = "mode"]
    pub mode: EnumParam<MyLadderMode>,
}

impl FilterGuiParams {
    pub fn new(
        name_prefix: &str,
        should_update_filter: Arc<AtomicBool>,
        default_cutoff: f32,
        default_res: f32,
        default_drive: f32,
        default_mode: MyLadderMode,
    ) -> Self {
        Self {
            cutoff: FloatParam::new(
                format!("{name_prefix} cutoff"),
                default_cutoff, // Use the passed default value
                FloatRange::Skewed {
                    min: 10.0,
                    max: 20_000.0,
                    factor: FloatRange::skew_factor(-1.6),
                },
            )
            .with_value_to_string(formatters::v2s_f32_hz_then_khz(1))
            .with_string_to_value(formatters::s2v_f32_hz_then_khz())
            .with_callback(Arc::new({
                let should_update_filter = should_update_filter.clone();
                move |_| should_update_filter.store(true, Ordering::Release)
            })),
            res: FloatParam::new(
                format!("{name_prefix} res"),
                default_res, // Use the passed default value
                FloatRange::Linear { min: 0., max: 1. },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(2))
            .with_callback(Arc::new({
                let should_update_filter = should_update_filter.clone();
                move |_| should_update_filter.store(true, Ordering::Release)
            })),
            drive: FloatParam::new(
                format!("{name_prefix} drive"),
                default_drive, // Use the passed default value
                FloatRange::Skewed {
                    min: util::db_to_gain(0.0),
                    max: util::db_to_gain(30.0),
                    // This makes the range appear as if it was linear when displaying the values as
                    // decibels
                    factor: FloatRange::gain_skew_factor(0.0, 30.0),
                },
            )
            .with_unit(" dB")
            .with_value_to_string(formatters::v2s_f32_gain_to_db(1))
            .with_string_to_value(formatters::s2v_f32_gain_to_db())
            .with_callback(Arc::new({
                let should_update_filter = should_update_filter.clone();
                move |_| should_update_filter.store(true, Ordering::Release)
            })),

            mode: EnumParam::new(format!("{name_prefix} mode"), default_mode) // Use the passed default value
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
        let filter_params = array_init(|_| Arc::new(FilterParams::new()));
        let should_update_filter = Arc::new(AtomicBool::new(false));
        let learned_notes = Arc::new(AtomicByteArray::new(NO_LEARNED_NOTE));
        let last_learned_notes = Arc::new(AtomicByteArray::new(NO_LEARNED_NOTE));
        let enabled_actions = Arc::new(AtomicBoolArray::new());
        let ladders: [LadderFilter; NUM_TAPS] =
            array_init(|i| LadderFilter::new(filter_params[i].clone()));
        let amp_envelopes = array_init::array_init(|_| Smoother::none());

        let tap_meters = AtomicF32Array(array_init::array_init(|_| {
            Arc::new(AtomicF32::new(util::MINUS_INFINITY_DB))
        }))
        .into();
        // let delayed_audio_l = array_init(|_| f32::default_boxed_array::<MAX_BLOCK_SIZE>());
        // let delayed_audio_r = array_init(|_| f32::default_boxed_array::<MAX_BLOCK_SIZE>());
        let delayed_audio_l = array_init(|_| vec![0.0; MAX_BLOCK_SIZE].into_boxed_slice());
        let delayed_audio_r = array_init(|_| vec![0.0; MAX_BLOCK_SIZE].into_boxed_slice());
        let mute_in_delayed = array_init(|_| vec![false; MAX_BLOCK_SIZE].into_boxed_slice());

        Self {
            params: Arc::new(Del2Params::new(
                should_update_filter.clone(),
                enabled_actions.clone(),
                learned_notes.clone(),
            )),

            delay_taps: [0; NUM_TAPS].map(|_| None),
            next_internal_id: 0,

            filter_params,
            ladders,
            delay_buffer: [
                BMRingBuf::<f32>::from_len(TOTAL_DELAY_SAMPLES),
                BMRingBuf::<f32>::from_len(TOTAL_DELAY_SAMPLES),
            ],
            delayed_audio_l,
            delayed_audio_r,

            mute_in_delay_buffer: BMRingBuf::<bool>::from_len(TOTAL_DELAY_SAMPLES),
            mute_in_delayed,
            mute_in_delay_temp_buffer: bool::default_boxed_array::<MAX_BLOCK_SIZE>(),

            gain: f32::default_boxed_array::<MAX_BLOCK_SIZE>(),
            dry_wet: f32::default_boxed_array::<MAX_BLOCK_SIZE>(),
            output_gain: f32::default_boxed_array::<MAX_BLOCK_SIZE>(),
            global_drive: f32::default_boxed_array::<MAX_BLOCK_SIZE>(),
            delay_tap_gain: f32::default_boxed_array::<MAX_BLOCK_SIZE>(),
            delay_tap_amp_envelope: f32::default_boxed_array::<MAX_BLOCK_SIZE>(),

            amp_envelopes,
            sample_rate: 1.0,
            peak_meter_decay_weight: 1.0,
            input_meter: Arc::new(AtomicF32::new(util::MINUS_INFINITY_DB)),
            output_meter: Arc::new(AtomicF32::new(util::MINUS_INFINITY_DB)),
            tap_meters,
            delay_write_index: 0,
            is_learning: Arc::new(AtomicBool::new(false)),
            learning_index: Arc::new(AtomicUsize::new(0)),
            learned_notes,
            last_learned_notes,
            last_played_notes: Arc::new(LastPlayedNotes::new()),
            samples_since_last_event: 0,
            timing_last_event: 0,
            min_tap_samples: 0,
            delay_buffer_size: 0,
            counting_state: CountingState::TimeOut,
            should_update_filter,
            enabled_actions,
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

            current_tap: Arc::new(AtomicUsize::new(0)),
            delay_times: AtomicU32Array(array_init::array_init(|_| Arc::new(AtomicU32::new(0)))),
            velocities: AtomicF32Array(array_init::array_init(|_| Arc::new(AtomicF32::new(0.0)))),
            pans: AtomicF32Array(array_init::array_init(|_| Arc::new(AtomicF32::new(0.0)))),
            notes: AtomicU8Array(array_init::array_init(|_| Arc::new(AtomicU8::new(0)))),
            current_time: Arc::new(AtomicU32::new(0)),
            max_tap_samples: Arc::new(AtomicU32::new(0)),
            first_note: Arc::new(AtomicU8::new(NO_LEARNED_NOTE)),
            previous_time_scaling_factor: Arc::new(AtomicF32::new(0.0)),
            previous_note_heights: AtomicF32Array(array_init::array_init(|_| {
                Arc::new(AtomicF32::new(0.0))
            })),
            previous_first_note_height: Arc::new(AtomicF32::new(0.0)),

            gain: FloatParam::new(
                "Gain",
                util::db_to_gain(0.0),
                // Because we're representing gain as decibels the range is already logarithmic
                FloatRange::Linear {
                    min: util::db_to_gain(-36.0),
                    max: util::db_to_gain(0.0),
                },
            )
            // This enables polyphonic mdoulation for this parameter by representing all related
            // events with this ID. After enabling this, the plugin **must** start sending
            // `VoiceTerminated` events to the host whenever a voice has ended.
            .with_poly_modulation_id(GAIN_POLY_MOD_ID)
            .with_smoother(SmoothingStyle::Logarithmic(5.0))
            .with_unit(" dB")
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .with_string_to_value(formatters::s2v_f32_gain_to_db()),
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
                is_learning: self.is_learning.clone(),
                learning_index: self.learning_index.clone(),
                learned_notes: self.learned_notes.clone(),
                last_learned_notes: self.last_learned_notes.clone(),
                last_played_notes: self.last_played_notes.clone(),
                enabled_actions: self.enabled_actions.clone(),
            },
            self.params.editor_state.clone(),
        )
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        // Set the sample rate from the buffer configuration
        self.sample_rate = buffer_config.sample_rate;

        // After `PEAK_METER_DECAY_MS` milliseconds of pure silence, the peak meter's value should
        // have dropped by 12 dB
        self.peak_meter_decay_weight = 0.25f64
            .powf((f64::from(self.sample_rate) * PEAK_METER_DECAY_MS / 1000.0).recip())
            as f32;
        // Calculate and set the delay buffer size
        self.set_delay_buffer_size(buffer_config);

        // Initialize filter parameters for each tap
        self.initialize_filter_parameters();

        true
    }

    fn reset(&mut self) {
        for tap in 0..NUM_TAPS {
            self.ladders[tap].s = [f32x4::splat(0.); 4];
            self.amp_envelopes[tap].reset(0.0);
        }
        self.delay_taps.fill(None);

        for tap_index in 0..self
            .params
            .current_tap
            .load(std::sync::atomic::Ordering::SeqCst)
        {
            self.load_and_configure_tap(self.sample_rate, 0, None, tap_index);
        }

        // Indicate filter update needed
        self.should_update_filter.store(true, Ordering::Release);
        // Reset buffers and envelopes here. This can be called from the audio thread and may not
        // allocate. You can remove this function if you do not need it.
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        self.update_peak_meter(buffer, &self.input_meter);

        self.update_min_max_tap_samples();

        let num_samples = buffer.samples();
        let sample_rate = context.transport().sample_rate;

        let mut next_event = context.next_event();
        let mut block_start: usize = 0;
        let mut block_end: usize = MAX_BLOCK_SIZE.min(num_samples);

        // write the audio buffer into the delay
        self.write_into_delay(buffer);

        while block_start < num_samples {
            let old_nr_taps = self
                .params
                .current_tap
                .load(std::sync::atomic::Ordering::SeqCst);

            // First of all, handle all note events that happen at the start of the block, and cut
            // the block short if another event happens before the end of it. To handle polyphonic
            // modulation for new notes properly, we'll keep track of the next internal note index
            // at the block's start. If we receive polyphonic modulation that matches a delay tap that
            // has an internal note ID that's great than or equal to this one, then we should start
            // the note's smoother at the new value instead of fading in from the global value.
            let this_sample_internal_id_start = self.next_internal_id;
            'events: loop {
                match next_event {
                    // If the event happens now, then we'll keep processing events
                    Some(event) if (event.timing() as usize) <= block_start => {
                        // This synth doesn't support any of the polyphonic expression events. A
                        // real synth plugin however will want to support those.
                        match event {
                            NoteEvent::NoteOn {
                                timing,
                                voice_id,
                                channel,
                                note,
                                velocity,
                            } => {
                                self.store_note_on_in_delay_data(timing, note, velocity);
                                if self
                                    .params
                                    .current_tap
                                    .load(std::sync::atomic::Ordering::SeqCst)
                                    > old_nr_taps
                                {
                                    let tap_index = self
                                        .params
                                        .current_tap
                                        .load(std::sync::atomic::Ordering::SeqCst)
                                        - 1;
                                    self.load_and_configure_tap(
                                        sample_rate,
                                        // timing,
                                        channel,
                                        voice_id,
                                        tap_index,
                                    );
                                }
                            }
                            NoteEvent::NoteOff {
                                timing: _,
                                voice_id: _,
                                channel: _,
                                note,
                                velocity: _,
                            } => {
                                self.store_note_off_in_delay_data(note);
                            }
                            NoteEvent::Choke {
                                timing,
                                voice_id,
                                channel,
                                note,
                            } => {
                                self.choke_delay_taps(context, timing, voice_id, channel, note);
                            }
                            NoteEvent::PolyModulation {
                                timing: _,
                                voice_id,
                                poly_modulation_id,
                                normalized_offset,
                            } => {
                                // Polyphonic modulation events are matched to delay taps using the
                                // delay tap ID, and to parameters using the poly modulation ID. The
                                // host will probably send a modulation event every N samples. This
                                // will happen before the delay tap is active, and of course also after
                                // it has been terminated (because the host doesn't know that it
                                // will be). Because of that, we won't print any assertion failures
                                // when we can't find the delay tap index here.
                                if let Some(idx) = self.get_idx(voice_id) {
                                    let delay_tap = self.delay_taps[idx].as_mut().unwrap();

                                    match poly_modulation_id {
                                        GAIN_POLY_MOD_ID => {
                                            // This should either create a smoother for this
                                            // modulated parameter or update the existing one.
                                            // Notice how this uses the parameter's unmodulated
                                            // normalized value in combination with the normalized
                                            // offset to create the target plain value
                                            let target_plain_value = self
                                                .params
                                                .gain
                                                .preview_modulated(normalized_offset);
                                            let (_, smoother) =
                                                delay_tap.delay_tap_gain.get_or_insert_with(|| {
                                                    (
                                                        normalized_offset,
                                                        self.params.gain.smoothed.clone(),
                                                    )
                                                });

                                            // If this `PolyModulation` events happens on the
                                            // same sample as a delay tap's `NoteOn` event, then it
                                            // should immediately use the modulated value
                                            // instead of slowly fading in
                                            if delay_tap.internal_id
                                                >= this_sample_internal_id_start
                                            {
                                                smoother.reset(target_plain_value);
                                            } else {
                                                smoother
                                                    .set_target(sample_rate, target_plain_value);
                                            }
                                        }
                                        n => nih_debug_assert_failure!(
                                            "Polyphonic modulation sent for unknown poly \
                                             modulation ID {}",
                                            n
                                        ),
                                    }
                                }
                            }
                            NoteEvent::MonoAutomation {
                                timing: _,
                                poly_modulation_id,
                                normalized_value,
                            } => {
                                // Modulation always acts as an offset to the parameter's current
                                // automated value. So if the host sends a new automation value for
                                // a modulated parameter, the modulated values/smoothing targets
                                // need to be updated for all polyphonically modulated delay taps.
                                for delay_tap in
                                    self.delay_taps.iter_mut().filter_map(|v| v.as_mut())
                                {
                                    match poly_modulation_id {
                                        GAIN_POLY_MOD_ID => {
                                            let (normalized_offset, smoother) =
                                                match delay_tap.delay_tap_gain.as_mut() {
                                                    Some((o, s)) => (o, s),
                                                    // If the delay tap does not have existing
                                                    // polyphonic modulation, then there's nothing
                                                    // to do here. The global automation/monophonic
                                                    // modulation has already been taken care of by
                                                    // the framework.
                                                    None => continue,
                                                };
                                            let target_plain_value =
                                                self.params.gain.preview_plain(
                                                    normalized_value + *normalized_offset,
                                                );
                                            smoother.set_target(sample_rate, target_plain_value);
                                        }
                                        n => nih_debug_assert_failure!(
                                            "Automation event sent for unknown poly modulation ID \
                                             {}",
                                            n
                                        ),
                                    }
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
                    _ => {
                        break 'events;
                    }
                }
            }
            self.prepare_for_delay(block_end - block_start);

            if self
                .should_update_filter
                .compare_exchange(
                    true,
                    false,
                    std::sync::atomic::Ordering::SeqCst,
                    std::sync::atomic::Ordering::SeqCst,
                )
                .is_ok()
            {
                for tap in 0..self
                    .params
                    .current_tap
                    .load(std::sync::atomic::Ordering::SeqCst)
                {
                    self.update_filter(tap);
                }
            }

            self.set_mute_for_all_delay_taps(sample_rate);

            let block_len = block_end - block_start;

            // These are the smoothed global parameter values. These are used for delay taps that do not
            // have polyphonic modulation applied to them. With a plugin as simple as this it would
            // be possible to avoid this completely by simply always copying the smoother into the
            // delay tap's struct, but that may not be realistic when the plugin has hundreds of
            // parameters. The `delay_tap_*` arrays are scratch arrays that an individual delay tap can use.
            // for poly modulation
            self.params
                .gain
                .smoothed
                .next_block(&mut self.gain, block_len);

            // not poly:
            self.params
                .global
                .dry_wet
                .smoothed
                .next_block(&mut self.dry_wet, block_len);
            self.params
                .global
                .output_gain
                .smoothed
                .next_block(&mut self.output_gain, block_len);
            self.params
                .global
                .global_drive
                .smoothed
                .next_block(&mut self.global_drive, block_len);

            let panning_center = if self.params.taps.panning_center.value() < 0.0 {
                f32::from(
                    self.params
                        .first_note
                        .load(std::sync::atomic::Ordering::SeqCst),
                )
            } else {
                self.params.taps.panning_center.value()
            };
            let panning_amount = self.params.taps.panning_amount.value();

            let output = buffer.as_slice();
            for (value_idx, sample_idx) in (block_start..block_end).enumerate() {
                let dry = 1.0 - self.dry_wet[value_idx];
                output[0][sample_idx] *= dry;
                output[1][sample_idx] *= dry;
            }

            for delay_tap in self.delay_taps.iter_mut().filter_map(|v| v.as_mut()) {
                let tap_index = delay_tap.tap_index;
                let note = self.params.notes[tap_index].load(std::sync::atomic::Ordering::SeqCst);

                let pan = ((f32::from(note) - panning_center) * panning_amount).clamp(-1.0, 1.0);
                self.params.pans[tap_index].store(pan, Ordering::SeqCst);

                let (offset_l, offset_r) = Self::pan_to_haas_samples(pan, sample_rate);

                let delay_time = self.params.delay_times[tap_index]
                    .load(std::sync::atomic::Ordering::SeqCst)
                    as isize;
                let write_index = self.delay_write_index;
                // delay_time - 1 because we are processing 2 samples at once in process_audio
                let read_index_l = write_index - (delay_time - 1 + offset_l);
                let read_index_r = write_index - (delay_time - 1 + offset_r);
                self.delay_buffer[0].read_into(&mut self.delayed_audio_l[tap_index], read_index_l);
                self.delay_buffer[1].read_into(&mut self.delayed_audio_r[tap_index], read_index_r);

                let drive = self.filter_params[tap_index].clone().drive;
                self.mute_in_delay_buffer.read_into(
                    &mut self.mute_in_delayed[tap_index],
                    write_index - (delay_time - 1),
                );
                // Depending on whether the delay tap has polyphonic modulation applied to it,
                // either the global parameter values are used, or the delay tap's smoother is used
                // to generate unique modulated values for that delay tap
                let gain = match &delay_tap.delay_tap_gain {
                    Some((_, smoother)) => {
                        smoother.next_block(&mut self.delay_tap_gain, block_len);
                        &self.delay_tap_gain
                    }
                    None => &self.gain,
                };

                // This is an exponential smoother repurposed as an AR envelope with values between
                // 0 and 1. When a note off event is received, this envelope will start fading out
                // again. When it reaches 0, we will terminate the delay tap.
                delay_tap
                    .amp_envelope
                    .next_block(&mut self.delay_tap_amp_envelope, block_len);

                for (value_idx, sample_idx) in (block_start..block_end).enumerate() {
                    let pre_filter_gain = self.global_drive[value_idx]
                        * gain[value_idx]
                        * self.delay_tap_amp_envelope[value_idx];

                    self.delayed_audio_l[tap_index][sample_idx] *= pre_filter_gain;
                    self.delayed_audio_r[tap_index][sample_idx] *= pre_filter_gain;
                }

                for i in (block_start..block_end).step_by(2) {
                    let frame = f32x4::from_array([
                        self.delayed_audio_l[tap_index][i],
                        self.delayed_audio_r[tap_index][i],
                        self.delayed_audio_l[tap_index]
                            .get(i + 1)
                            .copied()
                            .unwrap_or(0.0),
                        self.delayed_audio_r[tap_index]
                            .get(i + 1)
                            .copied()
                            .unwrap_or(0.0),
                    ]);
                    let processed = self.ladders[tap_index].tick_newton(frame);
                    let frame_out = *processed.as_array();
                    self.delayed_audio_l[tap_index][i] = frame_out[0];
                    self.delayed_audio_r[tap_index][i] = frame_out[1];
                    if i + 1 < block_end {
                        self.delayed_audio_l[tap_index][i + 1] = frame_out[2];
                        self.delayed_audio_r[tap_index][i + 1] = frame_out[3];
                    }
                }

                let mut amplitude = 0.0;
                for (value_idx, sample_idx) in (block_start..block_end).enumerate() {
                    let post_filter_gain = self.dry_wet[value_idx] * self.output_gain[value_idx]
                        / (drive * self.global_drive[value_idx]);
                    let left = self.delayed_audio_l[tap_index][sample_idx] * post_filter_gain;
                    let right = self.delayed_audio_r[tap_index][sample_idx] * post_filter_gain;
                    output[0][sample_idx] += left;
                    output[1][sample_idx] += right;
                    amplitude += left.abs(); // + right.abs();
                }

                if self.params.editor_state.is_open() {
                    amplitude = (amplitude / block_len as f32).abs();
                    let current_peak_meter =
                        self.tap_meters[tap_index].load(std::sync::atomic::Ordering::Relaxed);
                    let new_peak_meter = if amplitude > current_peak_meter {
                        amplitude
                    } else {
                        // println!("self.peak_meter_decay_weight: {}",self.peak_meter_decay_weight);

                        current_peak_meter.mul_add(0.8, amplitude * (1.0 - 0.8))
                        // current_peak_meter.mul_add(
                        //     self.peak_meter_decay_weight,
                        //     amplitude * (1.0 - self.peak_meter_decay_weight),
                        // )
                    };

                    self.tap_meters[tap_index]
                        .store(new_peak_meter, std::sync::atomic::Ordering::Relaxed);
                }
            }

            // Terminate delay taps whose release period has fully ended. This could be done as part of
            // the previous loop but this is simpler.
            for delay_tap in &mut self.delay_taps {
                match delay_tap {
                    Some(v) if v.releasing && v.amp_envelope.previous_value() == 0.0 => {
                        // This event is very important, as it allows the host to manage its own modulation
                        // delay taps
                        context.send_event(NoteEvent::VoiceTerminated {
                            timing: block_end as u32,
                            voice_id: Some(v.id),
                            channel: v.channel,
                            note: v.note,
                        });
                        *delay_tap = None;
                    }
                    _ => (),
                }
            }

            // And then just keep processing blocks until we've run out of buffer to fill
            block_start = block_end;
            block_end = (block_start + MAX_BLOCK_SIZE).min(num_samples);
        }

        self.update_peak_meter(buffer, &self.output_meter);

        ProcessStatus::Normal
    }
}

impl Del2 {
    fn update_min_max_tap_samples(&mut self) {
        let sample_rate = self.sample_rate;
        self.params.max_tap_samples.store(
            (sample_rate * self.params.global.max_tap_seconds.value()) as u32,
            Ordering::SeqCst,
        );
        self.min_tap_samples =
            (sample_rate * self.params.global.min_tap_milliseconds.value() * 0.001) as u32;
    }

    fn write_into_delay(&mut self, buffer: &mut Buffer) {
        for (_, block) in buffer.iter_blocks(buffer.samples()) {
            let block_len = block.samples();
            // TODO: assert needed?
            // assert!(block_len <= MAX_BLOCK_SIZE);
            let mut block_channels = block.into_iter();

            let out_l = block_channels.next().expect("Left output channel missing");
            let out_r = block_channels.next().expect("Right output channel missing");

            let write_index = self.delay_write_index;

            self.delay_buffer[0].write_latest(out_l, write_index);
            self.delay_buffer[1].write_latest(out_r, write_index);

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
        let is_tap_slot_available = self
            .params
            .current_tap
            .load(std::sync::atomic::Ordering::SeqCst)
            < NUM_TAPS;
        let is_delay_note = !self.learned_notes.contains(note);
        let is_learning = self.is_learning.load(Ordering::SeqCst);

        self.last_played_notes.note_on(note);

        if self.is_playing_action(LOCK_TAPS) {
            self.enabled_actions.toggle(LOCK_TAPS);
            self.last_played_notes
                .note_off(self.learned_notes.load(LOCK_TAPS));
        }
        let taps_unlocked = !self.enabled_actions.load(LOCK_TAPS);

        let mut should_record_tap =
            is_delay_note && is_tap_slot_available && taps_unlocked && !is_learning;

        match self.counting_state {
            CountingState::TimeOut => {
                if is_delay_note && !is_learning && taps_unlocked {
                    // If in TimeOut state, reset and start new counting phase
                    self.clear_taps(timing, true);
                    self.params.first_note.store(note, Ordering::SeqCst);
                }
            }
            CountingState::CountingInBuffer => {
                // Validate and record a new tap within the buffer
                if timing - self.timing_last_event > self.min_tap_samples && should_record_tap {
                    self.samples_since_last_event = timing - self.timing_last_event;
                    self.timing_last_event = timing;
                } else {
                    should_record_tap = false; // Debounce or max taps reached, ignore tap
                }
            }
            CountingState::CountingAcrossBuffer => {
                // Handle cross-buffer taps timing
                if self.samples_since_last_event + timing > self.min_tap_samples
                    && should_record_tap
                {
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

        let current_tap = self
            .params
            .current_tap
            .load(std::sync::atomic::Ordering::SeqCst);
        let mute_in_note = self.learned_notes.load(MUTE_IN);
        let mute_out_note = self.learned_notes.load(MUTE_OUT);

        // Check for timeout condition and reset if necessary
        if self.samples_since_last_event
            > self
                .params
                .max_tap_samples
                .load(std::sync::atomic::Ordering::SeqCst)
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
            if current_tap > 0 {
                self.params.delay_times[current_tap].store(
                    self.samples_since_last_event
                        + self.params.delay_times[current_tap - 1]
                            .load(std::sync::atomic::Ordering::SeqCst),
                    Ordering::SeqCst,
                );
            } else {
                self.params.delay_times[current_tap]
                    .store(self.samples_since_last_event, Ordering::SeqCst);
            }
            self.params.velocities[current_tap].store(velocity, Ordering::SeqCst);
            self.params.notes[current_tap].store(note, Ordering::SeqCst);

            self.params.current_tap.fetch_add(1, Ordering::SeqCst);

            // Indicate filter update needed
            self.should_update_filter.store(true, Ordering::Release);
        }
        // Handle ActionTrigger events
        // LOCK_TAPS is handled at the start
        if !is_learning
        // this is the value at the start of the fn, from before we adjusted the global one
        {
            let is_toggle = self.params.global.mute_is_toggle.value();

            if note == mute_in_note {
                if is_toggle {
                    self.enabled_actions.toggle(MUTE_IN);
                } else {
                    self.enabled_actions.store(MUTE_IN, false);
                    self.enabled_actions.store(MUTE_OUT, false);
                }
                // self.last_played_notes.note_off(note);
            }
            if note == mute_out_note {
                if is_toggle {
                    self.enabled_actions.toggle(MUTE_OUT);
                } else {
                    self.enabled_actions.store(MUTE_IN, false);
                    self.enabled_actions.store(MUTE_OUT, false);
                }
                // self.last_played_notes.note_off(note);
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
        self.params.current_tap.store(0, Ordering::SeqCst);

        self.params
            .previous_time_scaling_factor
            .store(0.0, Ordering::SeqCst);
        self.params
            .previous_first_note_height
            .store(0.0, Ordering::SeqCst);
        for i in 0..NUM_TAPS {
            self.params.previous_note_heights[i].store(0.0, Ordering::SeqCst);
        }

        self.start_release_for_all_delay_taps(self.sample_rate);
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
    fn load_and_configure_tap(
        &mut self,
        sample_rate: f32,
        // context: &mut (impl ProcessContext<Self> + nih_plug::prelude::ProcessContext<Self>),
        // timing: u32,
        channel: u8,
        voice_id: Option<i32>,
        tap_index: usize,
    ) {
        // Load atomic values corresponding to the tap index
        let delay_time = self.params.delay_times[tap_index].load(Ordering::SeqCst);
        let note = self.params.notes[tap_index].load(Ordering::SeqCst);
        let velocity = self.params.velocities[tap_index].load(Ordering::SeqCst);

        // Create amplitude envelope
        let amp_envelope =
            Smoother::new(SmoothingStyle::Linear(self.params.global.attack_ms.value()));

        // Conditionally reset and configure the amplitude envelope
        if self.params.global.mute_is_toggle.value() {
            amp_envelope.reset(0.0);
            amp_envelope.set_target(sample_rate, 1.0);
        }

        // Start delay tap and configure its properties
        let delay_tap = self.start_delay_tap(channel, note);
        delay_tap.velocity = velocity;
        delay_tap.amp_envelope = amp_envelope;
        delay_tap.tap_index = tap_index;
        delay_tap.is_muted = false;

        // Set a unique identifier for the delay tap
        delay_tap.id = voice_id.unwrap_or_else(|| compute_fallback_id(note, channel, delay_time));

        // delay_tap
    }
    fn prepare_for_delay(&mut self, buffer_samples: usize) {
        self.no_more_events(buffer_samples as u32);

        if self
            .params
            .current_tap
            .load(std::sync::atomic::Ordering::SeqCst)
            > 0
        {
            self.params.current_time.store(
                self.params.delay_times[self
                    .params
                    .current_tap
                    .load(std::sync::atomic::Ordering::SeqCst)
                    - 1]
                .load(std::sync::atomic::Ordering::SeqCst)
                    + self.samples_since_last_event,
                Ordering::SeqCst,
            );
        } else {
            self.params
                .current_time
                .store(self.samples_since_last_event, Ordering::SeqCst);
        }
        // FIXME: remove
        // self.delay_data_input.write(self.params.clone());
    }

    fn no_more_events(&mut self, buffer_samples: u32) {
        match self.counting_state {
            CountingState::TimeOut => {}
            CountingState::CountingInBuffer => {
                // TODO: is this correct?
                // timing_last_event is sometimes bigger than buffer_samples, so this overflows:
                // self.samples_since_last_event = buffer_samples as u32 - self.timing_last_event;
                self.samples_since_last_event =
                    buffer_samples.saturating_sub(self.timing_last_event);
                self.counting_state = CountingState::CountingAcrossBuffer;
            }
            CountingState::CountingAcrossBuffer => {
                self.samples_since_last_event += buffer_samples;
            }
        }

        if self.samples_since_last_event
            > self
                .params
                .max_tap_samples
                .load(std::sync::atomic::Ordering::SeqCst)
        {
            self.counting_state = CountingState::TimeOut;
            self.timing_last_event = 0;
            self.samples_since_last_event = 0;
            if self
                .params
                .current_tap
                .load(std::sync::atomic::Ordering::SeqCst)
                == 0
            {
                self.params
                    .first_note
                    .store(NO_LEARNED_NOTE, Ordering::SeqCst);
            }
        };
    }

    fn update_filter(&mut self, tap: usize) {
        let velocity = self.params.velocities[tap].load(std::sync::atomic::Ordering::SeqCst);
        let velocity_params = &self.params.taps;

        let filter_params = unsafe { Arc::get_mut_unchecked(&mut self.filter_params[tap]) };

        // Cache repeated calculations
        let low_params = &velocity_params.velocity_low;
        let high_params = &velocity_params.velocity_high;

        let res = Self::lerp(low_params.res.value(), high_params.res.value(), velocity);
        let velocity_cutoff = Self::log_interpolate(
            low_params.cutoff.value(),
            high_params.cutoff.value(),
            velocity,
        );
        let note_cutoff = util::midi_note_to_freq(
            self.params.notes[tap].load(std::sync::atomic::Ordering::SeqCst),
        );
        let cutoff = note_cutoff
            .mul_add(
                self.params.taps.note_to_cutoff_amount.value(),
                velocity_cutoff * self.params.taps.velocity_to_cutoff_amount.value(),
            )
            .clamp(10.0, 20_000.0);
        let drive_db = Self::lerp(
            util::gain_to_db(low_params.drive.value()),
            util::gain_to_db(high_params.drive.value()),
            velocity,
        );
        let drive = util::db_to_gain(drive_db);
        let mode = MyLadderMode::lerp(low_params.mode.value(), high_params.mode.value(), velocity);

        filter_params.set_resonance(res);
        filter_params.set_frequency(cutoff);
        filter_params.drive = drive;
        filter_params.ladder_mode = mode;
        self.ladders[tap].set_mix(mode);
    }

    fn lerp(a: f32, b: f32, x: f32) -> f32 {
        (b - a).mul_add(x, a)
    }
    fn log_interpolate(a: f32, b: f32, x: f32) -> f32 {
        a * (b / a).powf(x)
    }

    // Takes a pan value and gives a delay offset, in samples
    // instead of adding delay, it subtracts delay from the other channel,
    // so we stay under the maximum delay value
    fn pan_to_haas_samples(pan: f32, sample_rate: f32) -> (isize, isize) {
        let delay_samples = (pan.abs() * (MAX_HAAS_MS / 1000.0) * sample_rate) as isize;
        if pan > 0.0 {
            (0, -delay_samples) // Pan right: delay left
        } else {
            (-delay_samples, 0) // Pan left: delay right
        }
    }

    // for fn initialize():
    fn calculate_buffer_size(buffer_size: u32) -> u32 {
        ((TOTAL_DELAY_SAMPLES as f64 / f64::from(buffer_size)).ceil() as u32 * buffer_size)
            .next_power_of_two()
    }

    fn set_delay_buffer_size(&mut self, buffer_config: &BufferConfig) {
        let min_size = Self::calculate_buffer_size(buffer_config.min_buffer_size.unwrap_or(1));
        let max_size = Self::calculate_buffer_size(buffer_config.max_buffer_size);

        self.delay_buffer_size = u32::max(min_size, max_size);

        // Apply the calculated size to the delay buffers
        self.delay_buffer
            .iter_mut()
            .for_each(|buffer| buffer.clear_set_len(self.delay_buffer_size as usize));
    }

    fn initialize_filter_parameters(&mut self) {
        for tap in 0..NUM_TAPS {
            // Safety: Assumes exclusive access is guaranteed beforehand.
            let filter_params = unsafe { Arc::get_mut_unchecked(&mut self.filter_params[tap]) };
            filter_params.set_sample_rate(self.sample_rate);
        }
    }

    fn update_peak_meter(&self, buffer: &mut Buffer, peak_meter: &AtomicF32) {
        // Access samples using the iterator
        for channel_samples in buffer.iter_samples() {
            let mut amplitude = 0.0;
            let num_samples = channel_samples.len();

            for sample in channel_samples {
                // Process each sample (e.g., apply gain if necessary)
                amplitude += *sample;
            }

            if self.params.editor_state.is_open() {
                amplitude = (amplitude / num_samples as f32).abs();
                let current_peak_meter = peak_meter.load(std::sync::atomic::Ordering::Relaxed);
                let new_peak_meter = if amplitude > current_peak_meter {
                    amplitude
                } else {
                    current_peak_meter.mul_add(
                        self.peak_meter_decay_weight,
                        amplitude * (1.0 - self.peak_meter_decay_weight),
                    )
                };

                peak_meter.store(new_peak_meter, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    fn is_playing_action(&self, index: usize) -> bool {
        self.last_played_notes
            .is_playing(self.learned_notes.load(index))
    }

    fn v2s_f32_ms_then_s(total_digits: usize) -> Arc<dyn Fn(f32) -> String + Send + Sync> {
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

    fn s2v_f32_ms_then_s() -> Arc<dyn Fn(&str) -> Option<f32> + Send + Sync> {
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
    fn v2s_f32_note() -> Arc<dyn Fn(f32) -> String + Send + Sync> {
        Arc::new(move |value| {
            let note_nr = value.round() as u8; // Convert the floating-point value to the nearest u8
            if value < 0.0 {
                "first note of pattern".to_string()
            } else {
                let note_name = util::NOTES[(note_nr % 12) as usize];
                let octave = (note_nr / 12) as i8 - 1; // Correct the octave calculation
                format!("{note_name}{octave}") // Ensure correct value formatting
            }
        })
    }
    fn s2v_f32_note() -> Arc<dyn Fn(&str) -> Option<f32> + Send + Sync> {
        Arc::new(move |string| {
            let trimmed_string = string.trim().to_lowercase();

            // Check if the string contains specific keywords
            let keywords = ["first", "note", "pattern"];
            if keywords
                .iter()
                .any(|&keyword| trimmed_string.contains(keyword))
            {
                return Some(-1.0);
            }
            let len = trimmed_string.len();
            if len < 2 {
                // if it's short, return to default: "first note of pattern"
                return Some(-1.0);
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
                    return Some((note_index as i32 + (octave + 1) * 12) as f32);
                }
            }

            None
        })
    }
    /// Get the index of a delay tap by its delay tap ID, if the delay tap exists. This does not immediately
    /// return a reference to the delay tap to avoid lifetime issues.
    fn get_idx(&mut self, voice_id: i32) -> Option<usize> {
        self.delay_taps
            .iter_mut()
            .position(|delay_tap| matches!(delay_tap, Some(delay_tap) if delay_tap.id == voice_id))
    }

    /// Start a new delay tap with the given delay tap ID. If all `delay_taps` are currently in use, the oldest
    /// delay tap will be stolen. Returns a reference to the new delay tap.
    fn start_delay_tap(
        &mut self,
        // context: &mut impl ProcessContext<Self>,
        // sample_offset: u32,
        channel: u8,
        note: u8,
    ) -> &mut DelayTap {
        let new_delay_tap = DelayTap {
            id: 0,
            internal_id: self.next_internal_id,
            channel,
            note,
            velocity: 1.0,

            releasing: false,
            amp_envelope: Smoother::none(),

            delay_tap_gain: None,
            is_muted: false,
            tap_index: NUM_TAPS, // start with tap_index out of bounds, to make sure it gets set.
        };
        self.next_internal_id = self.next_internal_id.wrapping_add(1);

        if self.params.global.mute_is_toggle.value() {
            self.enabled_actions.store(MUTE_OUT, false);
        }

        // Can't use `.iter_mut().find()` here because nonlexical lifetimes don't apply to return
        // values
        if let Some(free_idx) = self
            .delay_taps
            .iter()
            .position(std::option::Option::is_none)
        {
            self.delay_taps[free_idx] = Some(new_delay_tap);
            self.delay_taps[free_idx].as_mut().unwrap()
        } else {
            // If there is no free delay tap, find and steal the oldest one
            // SAFETY: We can skip a lot of checked unwraps here since we already know all delay_taps are in
            //         use
            let oldest_delay_tap = unsafe {
                self.delay_taps
                    .iter_mut()
                    .min_by_key(|delay_tap| delay_tap.as_ref().unwrap_unchecked().internal_id)
                    .unwrap_unchecked()
            };

            // The stolen delay tap needs to be terminated so the host can reuse its modulation
            // resources
            // {
            // let oldest_delay_tap = oldest_delay_tap.as_ref().unwrap();
            // context.send_event(NoteEvent::VoiceTerminated {
            // timing: sample_offset,
            // voice_id: Some(oldest_delay_tap.id),
            // channel: oldest_delay_tap.channel,
            // note: oldest_delay_tap.note,
            // });
            // }

            *oldest_delay_tap = Some(new_delay_tap);
            oldest_delay_tap.as_mut().unwrap()
        }
    }

    /// Start the release process for all delay taps by changing their amplitude envelope.
    fn start_release_for_all_delay_taps(&mut self, sample_rate: f32) {
        for delay_tap in self.delay_taps.iter_mut().flatten() {
            delay_tap.releasing = true;
            delay_tap.amp_envelope.style =
                SmoothingStyle::Linear(self.params.global.release_ms.value());
            delay_tap.amp_envelope.set_target(sample_rate, 0.0);
        }
    }

    /// Set mute for all delay taps by changing their amplitude envelope.
    fn set_mute_for_all_delay_taps(&mut self, sample_rate: f32) {
        let is_playing_mute_out = self.is_playing_action(MUTE_OUT);

        for delay_tap in self.delay_taps.iter_mut().flatten() {
            let is_toggle = self.params.global.mute_is_toggle.value();
            let mute_in_delayed = self.mute_in_delayed[delay_tap.tap_index][0];
            let mute_out = self.enabled_actions.load(MUTE_OUT);

            let new_mute = if is_toggle {
                mute_in_delayed || mute_out
            } else if is_playing_mute_out != mute_out {
                !is_playing_mute_out
            } else {
                mute_in_delayed
            };

            if delay_tap.is_muted != new_mute {
                if new_mute {
                    delay_tap.amp_envelope.style =
                        SmoothingStyle::Linear(self.params.global.release_ms.value());
                    delay_tap.amp_envelope.set_target(sample_rate, 0.0);
                } else {
                    delay_tap.amp_envelope.style =
                        SmoothingStyle::Linear(self.params.global.attack_ms.value());
                    delay_tap.amp_envelope.set_target(sample_rate, 1.0);
                }
                delay_tap.is_muted = new_mute;
            }
        }
    }

    /// Immediately terminate one or more delay tap, removing it from the pool and informing the host
    /// that the delay tap has ended. If `id` is not provided, then this will terminate all
    /// matching `delay_taps`.
    fn choke_delay_taps(
        &mut self,
        context: &mut impl ProcessContext<Self>,
        sample_offset: u32,
        id: Option<i32>,
        channel: u8,
        note: u8,
    ) {
        for delay_tap in &mut self.delay_taps {
            match delay_tap {
                Some(DelayTap {
                    id: candidate_id,
                    channel: candidate_channel,
                    note: candidate_note,
                    ..
                }) if id == Some(*candidate_id)
                    || (channel == *candidate_channel && note == *candidate_note) =>
                {
                    context.send_event(NoteEvent::VoiceTerminated {
                        timing: sample_offset,
                        // Notice how we always send the terminated delay tap ID here
                        voice_id: Some(*candidate_id),
                        channel,
                        note,
                    });
                    *delay_tap = None;

                    if id.is_some() {
                        return;
                    }
                }
                _ => (),
            }
        }
    }
}

/// Compute a delay tap ID in case the host doesn't provide them. Polyphonic modulation will not work in
/// this case, but playing notes will.
const fn compute_fallback_id(note: u8, channel: u8, delay_time: u32) -> i32 {
    // Ensure inputs are within their valid ranges
    assert!(note <= 127, "note must be between 0 and 127");
    assert!(channel <= 15, "channel must be between 0 and 15");
    // The delay_time uses 21 bits, allowing values from 0 to 2,097,151.
    // At a sample rate of 48,000 samples per second (48 kHz), this translates to 2,097,151 / 48,000
    // which results in a maximum delay time of approximately 43.69 seconds.
    assert!(
        delay_time <= 2_097_151,
        "delay_time exceeds maximum storable value with 21 bits"
    );

    // Combine note, channel, and delay_time into a 32-bit integer
    (note as i32) | ((channel as i32) << 7) | ((delay_time as i32 & 0x1F_FFFF) << 11)
}

impl ClapPlugin for Del2 {
    const CLAP_ID: &'static str = "https://magnetophon.nl/DEL2";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("A rhythm delay with space.");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;

    // Don't forget to change these features
    const CLAP_FEATURES: &'static [ClapFeature] = &[ClapFeature::AudioEffect, ClapFeature::Stereo];
}

impl Vst3Plugin for Del2 {
    const VST3_CLASS_ID: [u8; 16] = *b"magnetophon/DEL2";

    // And also don't forget to change these categories
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Dynamics];
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

    fn index(self) -> Option<usize> {
        Self::sequence().iter().position(|&mode| mode == self.0)
    }

    fn lerp(start: Self, end: Self, t: f32) -> LadderMode {
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
struct AtomicU32Array([Arc<AtomicU32>; NUM_TAPS]);
pub struct AtomicF32Array([Arc<AtomicF32>; NUM_TAPS]);

// Implement PersistentField for AtomicU8Array
impl PersistentField<'_, [u8; 8]> for AtomicU8Array {
    fn set(&self, new_values: [u8; 8]) {
        for (atomic, &new_value) in self.0.iter().zip(new_values.iter()) {
            atomic.store(new_value, Ordering::SeqCst);
        }
    }

    fn map<F, R>(&self, f: F) -> R
    where
        F: Fn(&[u8; 8]) -> R,
    {
        let values: [u8; 8] = self
            .0
            .iter()
            .map(|arc| arc.load(Ordering::SeqCst))
            .collect::<Vec<u8>>()
            .try_into()
            .unwrap_or_else(|_| panic!("Size mismatch"));
        f(&values)
    }
}

impl PersistentField<'_, [u32; 8]> for AtomicU32Array {
    fn set(&self, new_values: [u32; 8]) {
        for (atomic, &new_value) in self.0.iter().zip(new_values.iter()) {
            atomic.store(new_value, Ordering::SeqCst);
        }
    }

    fn map<F, R>(&self, f: F) -> R
    where
        F: Fn(&[u32; 8]) -> R,
    {
        let values: [u32; 8] = self
            .0
            .iter()
            .map(|arc| arc.load(Ordering::SeqCst))
            .collect::<Vec<u32>>()
            .try_into()
            .unwrap_or_else(|_| panic!("Size mismatch"));
        f(&values)
    }
}

impl PersistentField<'_, [f32; 8]> for AtomicF32Array {
    fn set(&self, new_values: [f32; 8]) {
        for (atomic, &new_value) in self.0.iter().zip(new_values.iter()) {
            atomic.store(new_value, Ordering::SeqCst);
        }
    }

    fn map<F, R>(&self, f: F) -> R
    where
        F: Fn(&[f32; 8]) -> R,
    {
        let values: [f32; 8] = self
            .0
            .iter()
            .map(|arc| arc.load(Ordering::SeqCst))
            .collect::<Vec<f32>>()
            .try_into()
            .unwrap_or_else(|_| panic!("Size mismatch"));
        f(&values)
    }
}

// TODO: use this:
// impl<T> Index<usize> for AtomicArray<T> {
// type Output = Arc<T>;
// fn index(&self, index: usize) -> &Self::Output {
// &self.0[index]
// }
// }

// Implement the Index trait to allow for array-style access
impl Index<usize> for AtomicU8Array {
    type Output = Arc<AtomicU8>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}
impl Index<usize> for AtomicU32Array {
    type Output = Arc<AtomicU32>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl Index<usize> for AtomicF32Array {
    type Output = Arc<AtomicF32>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}
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
        let current_state = self.state.load(Ordering::SeqCst);

        // Check if the note is already in the table and reactivate if so.
        if let Some(index) = (0..8).find(|&i| self.notes.load(i) == note) {
            // Update sequence and reactivate the note.
            self.sequence
                .store(index, self.current_sequence.fetch_add(1, Ordering::SeqCst));
            self.active_notes.store(index, true); // Mark as active

            // Ensure it's marked as active in the state.
            loop {
                let new_state = current_state | (1 << index); // Set this index as active
                if self
                    .state
                    .compare_exchange_weak(
                        current_state,
                        new_state,
                        Ordering::SeqCst,
                        Ordering::SeqCst,
                    )
                    .is_ok()
                {
                    break;
                }
            }
            return;
        }

        // Loop until space is found in the notes array.
        loop {
            // Find first available spot in the notes array.
            if let Some(index) = (0..8).find(|i| (current_state & (1 << i)) == 0) {
                // Attempt to occupy this empty spot.
                let new_state = current_state | (1 << index);
                if self
                    .state
                    .compare_exchange_weak(
                        current_state,
                        new_state,
                        Ordering::SeqCst,
                        Ordering::SeqCst,
                    )
                    .is_ok()
                {
                    // Store the note and its sequence once the position is successfully claimed.
                    self.notes.store(index, note);
                    self.sequence
                        .store(index, self.current_sequence.fetch_add(1, Ordering::SeqCst));
                    self.active_notes.store(index, true); // Mark as active
                    break;
                }
            } else {
                // Overwrite the oldest active note
                let oldest_index = (0..8).min_by_key(|&i| self.sequence.load(i)).unwrap();
                self.notes.store(oldest_index, note);
                self.sequence.store(
                    oldest_index,
                    self.current_sequence.fetch_add(1, Ordering::SeqCst),
                );
                self.active_notes.store(oldest_index, true); // Mark as active
                break;
            }
        }
    }

    /// Handles the 'note off' event.
    fn note_off(&self, note: u8) {
        let current_state = self.state.load(Ordering::SeqCst);

        while let Some(index) = (0..8).find(|&i| self.notes.load(i) == note) {
            // Calculate new state after disabling the note at the found index.
            let new_state = current_state & !(1 << index);
            if self
                .state
                .compare_exchange_weak(current_state, new_state, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                self.sequence.store(index, 0); // Reset sequence
                self.active_notes.store(index, false); // Mark as inactive
                break;
            }
        }
    }

    /// Checks if a note is currently being played.
    fn is_playing(&self, note: u8) -> bool {
        // Find the index of the note and check if its spot in state is occupied.
        (0..8)
            .find(|&i| self.notes.load(i) == note)
            .map_or(false, |index| self.active_notes.load(index))
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

nih_export_clap!(Del2);
nih_export_vst3!(Del2);
