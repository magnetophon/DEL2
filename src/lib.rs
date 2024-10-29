/*
TODO:

- for triggers:
- all mutes with choice between latch and toggle
    - momentary mode:
in this mode, the taps are only heard until the key is released
    the counter starts when the note for this trigger is pressed

    - smooth all dsp params (at the end of the interpolation?)



Proper DSP structure:
- always use a big buffer size,
- make mutes sample-accurate

  Steps:
  - process_midi_events
    save a tuple of the timing of the tap start and end, and its state, separate for each tap, relative to the start of the nih buffer
    save the same tuples for the mute trigger state changes, separate for each tap, relative to the start of the tap
    save the timing of the envelope state changes (mutes), separate for each tap, relative to the start of the tap
  - fill up to NUM_TAPS big buffers of DSP_BLOCK_SIZE, beginning at the start of each tap
    - save the length of each input block, so we can split up the smoothing block at the output and do the post gain in smaller blocks, for minimum latency
  - for each tap:
        - do drive pre-gain on DSP buffers
          - has to be done here cause we want the gains after the delayline (so they react immediately) but before the filters
          - we save the smooted values as one big block
        - split up the buffers at mute events and do envelopes
          - after the delayline but before the filters
          - has to be done in split buffers so the envelopes are sample accurate.
        - recombine buffers into DSP_BLOCK_SIZE
        - do oversampling, DSP and downsampling
        - split up the DSP blocks and smoother blocks into the sizes we saved earlier
        - do drive post gain
  - mix taps into nih process buffers
  - do output gain


  - optional: live-mode / daw-mode switch
    - compensate for host latency by adjusting the delay read index (optional; when latency reporting is off?)
      take the actual buffer size into account?
    - make a switch to turn on and off latency compensation:
      - when it's off, the delays are at the correct time, but the gains are late
      - when it's on, the gains are also at the correct time, but the whole DAW is late!
 */

// #![allow(non_snake_case)]
#![feature(portable_simd)]
#![feature(get_mut_unchecked)]
use array_init::array_init;
use bit_mask_ring_buf::BMRingBuf;
use nih_plug::prelude::*;
use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::ViziaState;
use std::simd::f32x4;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use synfx_dsp::fh_va::{FilterParams, LadderFilter, LadderMode};
use triple_buffer::TripleBuffer;

mod delay_tap;
mod editor;
use delay_tap::DelayTap;

// max seconds per tap
const MAX_TAP_SECONDS: usize = 10;
const NUM_TAPS: usize = 8;
const TOTAL_DELAY_SECONDS: usize = MAX_TAP_SECONDS * NUM_TAPS;
const MAX_SAMPLE_RATE: usize = 192000;
const TOTAL_DELAY_SAMPLES: usize = TOTAL_DELAY_SECONDS * MAX_SAMPLE_RATE;
const VELOCITY_LOW_NAME_PREFIX: &str = "low velocity";
const VELOCITY_HIGH_NAME_PREFIX: &str = "high velocity";
// this seems to be the number JUCE is using
const MAX_BLOCK_SIZE: usize = 32768;
const DSP_BLOCK_SIZE: usize = 1024;
const PEAK_METER_DECAY_MS: f64 = 150.0;
const MAX_LEARNED_NOTES: usize = 8;
// abuse the difference in range between u8 and midi notes for special meaning
const NO_LEARNED_NOTE: u8 = 128;
const LEARNING: u8 = 129;
// action trigger indexes
// should be 0..7 because of AtomicByteArray size
const MUTE_IN: usize = 0;
const MUTE_OUT: usize = 1;
const RESET_TAPS: usize = 2;
const LOCK_TAPS: usize = 3;
const MAX_HAAS_MS: f32 = 5.0;

// Polyphonic modulation works by assigning integer IDs to parameters. Pattern matching on these in
// `PolyModulation` and `MonoAutomation` events makes it possible to easily link these events to the
// correct parameter.
const GAIN_POLY_MOD_ID: u32 = 0;

struct Del2 {
    params: Arc<Del2Params>,

    /// The effect's delay taps. Inactive delay taps will be set to `None` values.
    delay_taps: [Option<DelayTap>; NUM_TAPS as usize],
    /// The next internal delay tap ID, used only to figure out the oldest delay tap for "voice stealing".
    /// This is incremented by one each time a delay tap is created.
    next_internal_delay_tap_id: u64,

    filter_params: [Arc<FilterParams>; NUM_TAPS],
    ladders: [LadderFilter; NUM_TAPS],

    // delay write buffer
    delay_buffer: [BMRingBuf<f32>; 2],
    // delay read buffers
    // TODO: which vecs should be arrays?
    temp_l: Vec<f32>,
    temp_r: Vec<f32>,

    // stores the time of each target-change of the amp envelope, relative to the start of each tap
    // the state change at the start of the tap is implicit
    amp_envelope_states: [BMRingBuf<(u32, bool)>; NUM_TAPS],
    // rolling index into amp_envelope_states
    amp_envelope_write_index: [isize; NUM_TAPS],
    // when did each mute change, relative to the start of the nih buffer
    // so we can determine when the amp envelope should change state
    mute_in_states: BMRingBuf<(u32, bool)>,
    mute_out_states: BMRingBuf<(u32, bool)>,
    // where does each tap start, relative to the nih buffer
    tap_states: [(u32, bool); NUM_TAPS],
    mute_in_write_index: isize,
    mute_out_write_index: isize,
    tap_was_muted: [bool; NUM_TAPS],

    delay_data: SharedDelayData,
    delay_data_input: SharedDelayDataInput,
    delay_data_output: Arc<Mutex<SharedDelayDataOutput>>,
    // N counters to know where in the fade in we are: 0 is the start
    amp_envelopes: [Smoother<f32>; NUM_TAPS],
    envelope_block: [[f32; DSP_BLOCK_SIZE]; NUM_TAPS],
    sample_rate: f32,
    peak_meter_decay_weight: f32,
    input_meter: Arc<AtomicF32>,
    output_meter: Arc<AtomicF32>,
    delay_write_index: isize,
    is_learning: Arc<AtomicBool>,
    // for which control are we learning?
    learning_index: Arc<AtomicUsize>,
    learned_notes: Arc<AtomicByteArray>,
    last_played_notes: Arc<LastPlayedNotes>,
    samples_since_last_event: u32,
    timing_last_event: u32,
    min_tap_samples: u32,
    delay_buffer_size: u32,
    counting_state: CountingState,
    should_update_filter: Arc<AtomicBool>,
    enabled_actions: Arc<AtomicBoolArray>,
}

// for use in graph
#[derive(Clone)]
pub struct SharedDelayData {
    delay_times: [u32; NUM_TAPS],
    velocities: [f32; NUM_TAPS],
    notes: [u8; NUM_TAPS],
    current_tap: usize,
    current_time: u32,
    max_tap_samples: u32,
}
pub type SharedDelayDataInput = triple_buffer::Input<SharedDelayData>;
pub type SharedDelayDataOutput = triple_buffer::Output<SharedDelayData>;

impl Data for SharedDelayData {
    fn same(&self, other: &Self) -> bool {
        self.delay_times == other.delay_times
            && self.velocities == other.velocities
            && self.notes == other.notes
            && self.current_tap == other.current_tap
            && self.current_time == other.current_time
            && self.max_tap_samples == other.max_tap_samples
    }
}
/// All the parameters
#[derive(Params)]
struct Del2Params {
    #[persist = "editor-state"]
    editor_state: Arc<ViziaState>,
    #[nested(group = "global")]
    pub global: GlobalParams,
    #[nested(group = "taps")]
    pub taps: TapsParams,

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
    pub fn new(enabled_actions: Arc<AtomicBoolArray>) -> Self {
        GlobalParams {
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
                    String::from(if value { "toggle" } else { "instant" })
                }))
                .with_callback(Arc::new(move |value| {
                    if !value {
                        enabled_actions.store(MUTE_IN, true);
                        enabled_actions.store(MUTE_OUT, false);
                    }
                })),

            attack_ms: FloatParam::new(
                "Attack",
                10.0,
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
                20.0,
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
                10.0,
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
                3.0,
                FloatRange::Skewed {
                    min: 0.5,
                    max: MAX_TAP_SECONDS as f32,
                    factor: FloatRange::skew_factor(-0.8),
                },
            )
            .with_step_size(0.1)
            .with_unit(" s"),
        }
    }
}

/// Contains the high and low tap parameters.
#[derive(Params)]
pub struct TapsParams {
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
        TapsParams {
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
                should_update_filter.clone(),
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
        FilterGuiParams {
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
                    let should_update_filter = should_update_filter.clone();
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
        let initial_delay_data: SharedDelayData = SharedDelayData::default();
        let (delay_data_input, delay_data_output) = TripleBuffer::new(&initial_delay_data).split();

        let filter_params = array_init(|_| Arc::new(FilterParams::new()));
        let should_update_filter = Arc::new(AtomicBool::new(false));
        let enabled_actions = Arc::new(AtomicBoolArray::new());
        let ladders: [LadderFilter; NUM_TAPS] =
            array_init(|i| LadderFilter::new(filter_params[i].clone()));
        // in theory we could have a new mute value every sample
        // TODO: since it's storing just one u32 per mute state change, we can probably get away with a smaller array
        let amp_envelope_states: [BMRingBuf<(u32, bool)>; NUM_TAPS] =
            array_init::array_init(|_| BMRingBuf::from_len(DSP_BLOCK_SIZE));

        let amp_envelopes = array_init::array_init(|_| Smoother::none());
        Self {
            params: Arc::new(Del2Params::new(
                should_update_filter.clone(),
                enabled_actions.clone(),
            )),

            delay_taps: [0; NUM_TAPS as usize].map(|_| None),
            next_internal_delay_tap_id: 0,

            filter_params,
            ladders,
            delay_buffer: [
                BMRingBuf::<f32>::from_len(TOTAL_DELAY_SAMPLES),
                BMRingBuf::<f32>::from_len(TOTAL_DELAY_SAMPLES),
            ],
            temp_l: vec![0.0; MAX_BLOCK_SIZE],
            temp_r: vec![0.0; MAX_BLOCK_SIZE],

            amp_envelope_states,
            amp_envelope_write_index: [0; NUM_TAPS],
            mute_in_states: BMRingBuf::from_len(MAX_BLOCK_SIZE),
            mute_out_states: BMRingBuf::from_len(MAX_BLOCK_SIZE),
            tap_states: [(0, true); NUM_TAPS],
            mute_in_write_index: 0,
            mute_out_write_index: 0,
            tap_was_muted: [true; NUM_TAPS],
            delay_data: initial_delay_data,
            delay_data_input,
            delay_data_output: Arc::new(Mutex::new(delay_data_output)),
            amp_envelopes,
            envelope_block: [[0.0; DSP_BLOCK_SIZE]; NUM_TAPS],
            //TODO: make Option<u8>
            sample_rate: 1.0,
            peak_meter_decay_weight: 1.0,
            input_meter: Arc::new(AtomicF32::new(util::MINUS_INFINITY_DB)),
            output_meter: Arc::new(AtomicF32::new(util::MINUS_INFINITY_DB)),
            delay_write_index: 0,
            is_learning: Arc::new(AtomicBool::new(false)),
            learning_index: Arc::new(AtomicUsize::new(0)),
            learned_notes: Arc::new(AtomicByteArray::new()),
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

impl Default for SharedDelayData {
    fn default() -> Self {
        Self {
            delay_times: [0; NUM_TAPS],
            velocities: [0.0; NUM_TAPS],
            notes: [0; NUM_TAPS],
            current_tap: 0,
            current_time: 0,
            max_tap_samples: 0,
        }
    }
}

impl Del2Params {
    fn new(should_update_filter: Arc<AtomicBool>, enabled_actions: Arc<AtomicBoolArray>) -> Self {
        Self {
            editor_state: editor::default_state(),
            taps: TapsParams::new(should_update_filter.clone()),
            global: GlobalParams::new(enabled_actions.clone()),
            gain: FloatParam::new(
                "Gain",
                util::db_to_gain(-12.0),
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
                delay_data: self.delay_data_output.clone(),
                is_learning: self.is_learning.clone(),
                learning_index: self.learning_index.clone(),
                learned_notes: self.learned_notes.clone(),
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
        self.peak_meter_decay_weight =
            0.25f64.powf((self.sample_rate as f64 * PEAK_METER_DECAY_MS / 1000.0).recip()) as f32;
        // Resize temporary buffers for left and right channels to maximum buffer size
        // Either we resize here, or in the audio thread
        // If we don't, we are slower.
        self.resize_temp_buffers(buffer_config.max_buffer_size);

        // Calculate and set the delay buffer size
        self.set_delay_buffer_size(buffer_config);

        // Initialize filter parameters for each tap
        self.initialize_filter_parameters();

        for i in 0..MAX_LEARNED_NOTES {
            self.learned_notes.store(i, NO_LEARNED_NOTE);
        }

        true
    }

    fn reset(&mut self) {
        for tap in 0..NUM_TAPS {
            self.ladders[tap].s = [f32x4::splat(0.); 4];
            self.amp_envelopes[tap].reset(0.0);
        }
        self.delay_taps.fill(None);
        // Reset buffers and envelopes here. This can be called from the audio thread and may not
        // allocate. You can remove this function if you do not need it.
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        // self.update_min_max_tap_samples();
        // self.process_midi_events(context);
        // self.set_amp_envelope_states();
        // self.prepare_for_delay(buffer.samples());
        // if Del2::compare_exchange(&self.should_update_filter) {
        // for tap in 0..self.delay_data.current_tap {
        // self.update_filter(tap);
        // }
        // }
        // self.update_peak_meter(buffer, &self.input_meter);
        // self.process_audio_blocks(buffer);
        // self.update_peak_meter(buffer, &self.output_meter);

        /*



        */

        self.update_min_max_tap_samples();
        // write the audio buffer into the delay
        self.write_into_delay(buffer);

        let num_samples = buffer.samples();
        let sample_rate = context.transport().sample_rate;
        let output = buffer.as_slice();

        let mut next_event = context.next_event();
        let mut block_start: usize = 0;
        let mut block_end: usize = MAX_BLOCK_SIZE.min(num_samples);

        while block_start < num_samples {
            // First of all, handle all note events that happen at the start of the block, and cut
            // the block short if another event happens before the end of it. To handle polyphonic
            // modulation for new notes properly, we'll keep track of the next internal note index
            // at the block's start. If we receive polyphonic modulation that matches a delay tap that
            // has an internal note ID that's great than or equal to this one, then we should start
            // the note's smoother at the new value instead of fading in from the global value.
            let this_sample_internal_delay_tap_id_start = self.next_internal_delay_tap_id;
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
                                // This starts with the attack portion of the amplitude envelope
                                let amp_envelope = Smoother::new(SmoothingStyle::Exponential(
                                    self.params.global.attack_ms.value(),
                                ));
                                let delay_time = if self.delay_data.current_tap > 0 {
                                    self.delay_data.delay_times[self.delay_data.current_tap - 1]
                                        as isize
                                } else {
                                    0
                                };
                                amp_envelope.reset(0.0);
                                amp_envelope.set_target(sample_rate, 1.0);

                                {
                                    if self.delay_data.current_tap > 0 {
                                        let delay_tap = self.start_delay_tap(
                                            context, timing, voice_id, channel, note,
                                        );
                                        delay_tap.velocity = velocity;
                                        delay_tap.amp_envelope = amp_envelope;
                                        delay_tap.delay_time = delay_time;
                                    }
                                }
                            }
                            NoteEvent::NoteOff {
                                timing,
                                voice_id,
                                channel,
                                note,
                                velocity: _,
                            } => {
                                self.store_note_off_in_delay_data(timing, note);
                                // self.start_release_for_delay_taps(
                                // sample_rate,
                                // voice_id,
                                // channel,
                                // note,
                                // )
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
                                if let Some(delay_tap_idx) = self.get_delay_tap_idx(voice_id) {
                                    let delay_tap =
                                        self.delay_taps[delay_tap_idx].as_mut().unwrap();

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
                                            if delay_tap.internal_delay_tap_id
                                                >= this_sample_internal_delay_tap_id_start
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
                    _ => break 'events,
                }
            }

            self.prepare_for_delay(block_end - block_start);

            // We'll start with silence, and then add the output from the active delay taps
            output[0][block_start..block_end].fill(0.0);
            output[1][block_start..block_end].fill(0.0);

            // These are the smoothed global parameter values. These are used for delay taps that do not
            // have polyphonic modulation applied to them. With a plugin as simple as this it would
            // be possible to avoid this completely by simply always copying the smoother into the
            // delay tap's struct, but that may not be realistic when the plugin has hundreds of
            // parameters. The `delay_tap_*` arrays are scratch arrays that an individual delay tap can use.
            let block_len = block_end - block_start;
            let mut gain = [0.0; MAX_BLOCK_SIZE];
            let mut delay_tap_gain = [0.0; MAX_BLOCK_SIZE];
            let mut delay_tap_amp_envelope = [0.0; MAX_BLOCK_SIZE];
            self.params.gain.smoothed.next_block(&mut gain, block_len);

            // TODO: Some form of band limiting
            // TODO: Filter
            for delay_tap in self.delay_taps.iter_mut().filter_map(|v| v.as_mut()) {
                // delay_time - 1 because we are processing 2 samples at once in process_audio
                let read_index = self.delay_write_index - (delay_tap.delay_time - 1).max(0);
                self.delay_buffer[0].read_into(&mut delay_tap.delayed_audio_l, read_index);
                self.delay_buffer[1].read_into(&mut delay_tap.delayed_audio_r, read_index);

                // Depending on whether the delay tap has polyphonic modulation applied to it,
                // either the global parameter values are used, or the delay tap's smoother is used
                // to generate unique modulated values for that delay tap
                let gain = match &delay_tap.delay_tap_gain {
                    Some((_, smoother)) => {
                        smoother.next_block(&mut delay_tap_gain, block_len);
                        &delay_tap_gain
                    }
                    None => &gain,
                };

                // This is an exponential smoother repurposed as an AR envelope with values between
                // 0 and 1. When a note off event is received, this envelope will start fading out
                // again. When it reaches 0, we will terminate the delay tap.
                delay_tap
                    .amp_envelope
                    .next_block(&mut delay_tap_amp_envelope, block_len);

                for (value_idx, sample_idx) in (block_start..block_end).enumerate() {
                    let amp =
                        delay_tap.velocity * gain[value_idx] * delay_tap_amp_envelope[value_idx];

                    output[0][sample_idx] += delay_tap.delayed_audio_l[sample_idx] * amp;
                    output[1][sample_idx] += delay_tap.delayed_audio_r[sample_idx] * amp;
                }
            }

            // Terminate delay taps whose release period has fully ended. This could be done as part of
            // the previous loop but this is simpler.
            for delay_tap in self.delay_taps.iter_mut() {
                match delay_tap {
                    Some(v) if v.releasing && v.amp_envelope.previous_value() == 0.0 => {
                        // This event is very important, as it allows the host to manage its own modulation
                        // delay taps
                        context.send_event(NoteEvent::VoiceTerminated {
                            timing: block_end as u32,
                            voice_id: Some(v.delay_tap_id),
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

        ProcessStatus::Normal
    }
}

impl Del2 {
    fn update_min_max_tap_samples(&mut self) {
        let sample_rate = self.sample_rate as f32;
        self.delay_data.max_tap_samples =
            (sample_rate * self.params.global.max_tap_seconds.value()) as u32;
        self.min_tap_samples =
            (sample_rate * self.params.global.min_tap_milliseconds.value() * 0.001) as u32;
    }

    fn write_into_delay(&mut self, buffer: &mut Buffer) {
        for (_, block) in buffer.iter_blocks(buffer.samples()) {
            let block_len = block.samples();
            // TODO: assert needed?
            // assert!(block_len <= MAX_BLOCK_SIZE);
            // Either we resize here, or in the initialization fn
            // If we don't, we are slower.
            // self.resize_temp_buffers(block_len.try_into().unwrap());
            let mut block_channels = block.into_iter();

            let out_l = block_channels.next().expect("Left output channel missing");
            let out_r = block_channels.next().expect("Right output channel missing");

            let write_index = self.delay_write_index;

            // println!("out_l: {}", out_l[42]);

            self.delay_buffer[0].write_latest(out_l, write_index);
            self.delay_buffer[1].write_latest(out_r, write_index);
            self.delay_write_index =
                (write_index + block_len as isize) % self.delay_buffer_size as isize;
        }
    }

    fn read_into_delayed_audio(
        &mut self,
        delayed_audio_l: &mut [f32],
        delayed_audio_r: &mut [f32],
        delay_time: isize,
    ) {
        // delay_time - 1 because we are processing 2 samples at once in process_audio
        let read_index = self.delay_write_index - (delay_time - 1).max(0);

        self.delay_buffer[0].read_into(delayed_audio_l, read_index);
        self.delay_buffer[1].read_into(delayed_audio_r, read_index);
    }

    fn store_note_on_in_delay_data(&mut self, timing: u32, note: u8, velocity: f32) {
        let is_tap_slot_available = self.delay_data.current_tap < NUM_TAPS;
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
                    self.reset_taps(timing, true);
                }
            }
            CountingState::CountingInBuffer => {
                // Validate and record a new tap within the buffer
                if timing - self.timing_last_event > self.min_tap_samples && should_record_tap {
                    self.samples_since_last_event = timing - self.timing_last_event;
                    self.timing_last_event = timing;
                } else {
                    // continue; // Debounce or max taps reached, ignore tap
                    should_record_tap = false;
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
                    // continue; // Debounce or max taps reached, ignore tap
                    should_record_tap = false;
                }
            }
        }
        if is_learning {
            self.is_learning.store(false, Ordering::SeqCst);
            self.learned_notes
                .store(self.learning_index.load(Ordering::SeqCst), note);
            self.last_played_notes.note_off(note);
        }

        let current_tap = self.delay_data.current_tap;
        let mute_in_note = self.learned_notes.load(MUTE_IN);
        let mute_out_note = self.learned_notes.load(MUTE_OUT);

        // Check for timeout condition and reset if necessary
        if self.samples_since_last_event > self.delay_data.max_tap_samples {
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
                self.delay_data.delay_times[current_tap] =
                    self.samples_since_last_event + self.delay_data.delay_times[current_tap - 1];
            } else {
                self.delay_data.delay_times[current_tap] = self.samples_since_last_event;
            }
            self.delay_data.velocities[current_tap] = velocity;
            self.delay_data.notes[current_tap] = note;

            // self.mute_out(current_tap, false);
            self.delay_data.current_tap += 1;
            // self.delay_data.current_tap = self.delay_data.current_tap;

            // Indicate filter update needed
            self.should_update_filter.store(true, Ordering::Release);
        }
        // Handle ActionTrigger events
        // LOCK_TAPS is handled at the start
        if !is_learning
        // this is the value at the start of the fn, from before we adjusted the global one
        {
            let old_mute_in = self.enabled_actions.load(MUTE_IN);
            let old_mute_out = self.enabled_actions.load(MUTE_OUT);
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
            if self.is_playing_action(RESET_TAPS) {
                self.reset_taps(timing, false);
            }

            let new_mute_in = self.enabled_actions.load(MUTE_IN);
            let new_mute_out = self.enabled_actions.load(MUTE_OUT);

            if new_mute_in != old_mute_in {
                Self::update_mute_times(
                    &mut self.mute_in_states,
                    &mut self.mute_in_write_index,
                    (timing, new_mute_in),
                );
            }

            if new_mute_out != old_mute_out {
                Self::update_mute_times(
                    &mut self.mute_out_states,
                    &mut self.mute_out_write_index,
                    (timing, new_mute_out),
                );
            }
        }
    }

    fn store_note_off_in_delay_data(&mut self, timing: u32, note: u8) {
        // if we are in direct mode
        if !self.params.global.mute_is_toggle.value() {
            // mute the in and out
            for &mute in &[MUTE_IN, MUTE_OUT] {
                if note == self.learned_notes.load(mute) {
                    self.enabled_actions.store(mute, true);
                }
            }
            if note == self.learned_notes.load(MUTE_IN) {
                Self::update_mute_times(
                    &mut self.mute_in_states,
                    &mut self.mute_in_write_index,
                    (timing, true),
                );
            } else if note == self.learned_notes.load(MUTE_OUT) {
                Self::update_mute_times(
                    &mut self.mute_out_states,
                    &mut self.mute_out_write_index,
                    (timing, true),
                );
            }
        }
        self.last_played_notes.note_off(note);
    }

    fn process_midi_events(&mut self, context: &mut impl ProcessContext<Self>) {
        self.mute_in_write_index = 0;
        self.mute_out_write_index = 0;
        while let Some(event) = context.next_event() {
            match event {
                NoteEvent::NoteOn {
                    timing,
                    note,
                    velocity,
                    ..
                } => {
                    let is_tap_slot_available = self.delay_data.current_tap < NUM_TAPS;
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
                                self.reset_taps(timing, true);
                            }
                        }
                        CountingState::CountingInBuffer => {
                            // Validate and record a new tap within the buffer
                            if timing - self.timing_last_event > self.min_tap_samples
                                && should_record_tap
                            {
                                self.samples_since_last_event = timing - self.timing_last_event;
                                self.timing_last_event = timing;
                            } else {
                                // continue; // Debounce or max taps reached, ignore tap
                                should_record_tap = false;
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
                                // continue; // Debounce or max taps reached, ignore tap
                                should_record_tap = false;
                            }
                        }
                    }
                    if is_learning {
                        self.is_learning.store(false, Ordering::SeqCst);
                        self.learned_notes
                            .store(self.learning_index.load(Ordering::SeqCst), note);
                        self.last_played_notes.note_off(note);
                    }

                    let current_tap = self.delay_data.current_tap;
                    let mute_in_note = self.learned_notes.load(MUTE_IN);
                    let mute_out_note = self.learned_notes.load(MUTE_OUT);

                    // Check for timeout condition and reset if necessary
                    if self.samples_since_last_event > self.delay_data.max_tap_samples {
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
                            self.delay_data.delay_times[current_tap] = self
                                .samples_since_last_event
                                + self.delay_data.delay_times[current_tap - 1];
                        } else {
                            self.delay_data.delay_times[current_tap] =
                                self.samples_since_last_event;
                        }
                        self.delay_data.velocities[current_tap] = velocity;
                        self.delay_data.notes[current_tap] = note;

                        // self.mute_out(current_tap, false);
                        self.delay_data.current_tap += 1;
                        // self.delay_data.current_tap = self.delay_data.current_tap;

                        // Indicate filter update needed
                        self.should_update_filter.store(true, Ordering::Release);
                    }
                    // Handle ActionTrigger events
                    // LOCK_TAPS is handled at the start
                    if !is_learning
                    // this is the value at the start of the fn, from before we adjusted the global one
                    {
                        let old_mute_in = self.enabled_actions.load(MUTE_IN);
                        let old_mute_out = self.enabled_actions.load(MUTE_OUT);
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
                        if self.is_playing_action(RESET_TAPS) {
                            self.reset_taps(timing, false);
                        }

                        let new_mute_in = self.enabled_actions.load(MUTE_IN);
                        let new_mute_out = self.enabled_actions.load(MUTE_OUT);

                        if new_mute_in != old_mute_in {
                            Self::update_mute_times(
                                &mut self.mute_in_states,
                                &mut self.mute_in_write_index,
                                (timing, new_mute_in),
                            );
                        }

                        if new_mute_out != old_mute_out {
                            Self::update_mute_times(
                                &mut self.mute_out_states,
                                &mut self.mute_out_write_index,
                                (timing, new_mute_out),
                            );
                        }
                    }
                }
                // Handling NoteOff events
                NoteEvent::NoteOff { timing, note, .. } => {
                    if !self.params.global.mute_is_toggle.value() {
                        println!("direct mode!");
                        // if we are in direct mode
                        // mute the in and out
                        for &mute in &[MUTE_IN, MUTE_OUT] {
                            if note == self.learned_notes.load(mute) {
                                self.enabled_actions.store(mute, true);
                            }
                        }
                        if note == self.learned_notes.load(MUTE_IN) {
                            Self::update_mute_times(
                                &mut self.mute_in_states,
                                &mut self.mute_in_write_index,
                                (timing, true),
                            );
                        } else if note == self.learned_notes.load(MUTE_OUT) {
                            Self::update_mute_times(
                                &mut self.mute_out_states,
                                &mut self.mute_out_write_index,
                                (timing, true),
                            );
                        }
                    }
                    self.last_played_notes.note_off(note);
                }
                _ => {} // Handle other types of events if necessary
            }
        }
    }

    fn reset_taps(&mut self, timing: u32, restart: bool) {
        self.enabled_actions.store(LOCK_TAPS, false);
        self.delay_data.current_tap = 0;
        self.start_release_for_all_delay_taps(self.sample_rate);
        if self.params.global.mute_is_toggle.value() {
            self.enabled_actions.store(MUTE_IN, false);
            self.enabled_actions.store(MUTE_OUT, false);
        }
        if restart {
            self.counting_state = CountingState::CountingInBuffer;
            self.timing_last_event = timing;
        } else {
            self.counting_state = CountingState::TimeOut;
            self.timing_last_event = 0;
            self.samples_since_last_event = 0;
        }
    }

    fn update_mute_times(
        times: &mut BMRingBuf<(u32, bool)>,
        write_index: &mut isize,
        state: (u32, bool),
    ) {
        times.write_latest(&[state], *write_index);
        *write_index = (*write_index + 1) % DSP_BLOCK_SIZE as isize;
        // println!("state changed, state: {:?}, index: {}", state, *write_index);
    }

    fn prepare_for_delay(&mut self, buffer_samples: usize) {
        self.no_more_events(buffer_samples as u32);

        if self.delay_data.current_tap > 0 {
            self.delay_data.current_time = self.delay_data.delay_times
                [self.delay_data.current_tap - 1]
                + self.samples_since_last_event;
        } else {
            self.delay_data.current_time = self.samples_since_last_event;
        }
        self.delay_data_input.write(self.delay_data.clone());
    }

    fn no_more_events(&mut self, buffer_samples: u32) {
        match self.counting_state {
            CountingState::TimeOut => {}
            CountingState::CountingInBuffer => {
                self.samples_since_last_event = buffer_samples as u32 - self.timing_last_event;
                self.counting_state = CountingState::CountingAcrossBuffer;
            }
            CountingState::CountingAcrossBuffer => {
                self.samples_since_last_event += buffer_samples;
            }
        }

        if self.samples_since_last_event > self.delay_data.max_tap_samples {
            self.counting_state = CountingState::TimeOut;
            self.timing_last_event = 0;
            self.samples_since_last_event = 0;
        };
    }
    fn compare_exchange(atomic_bool: &AtomicBool) -> bool {
        atomic_bool
            .compare_exchange(
                true,
                false,
                std::sync::atomic::Ordering::SeqCst,
                std::sync::atomic::Ordering::SeqCst,
            )
            .is_ok()
    }
    fn update_filter(&mut self, tap: usize) {
        let velocity = self.delay_data.velocities[tap];
        let velocity_params = &self.params.taps;

        unsafe {
            let filter_params = Arc::get_mut_unchecked(&mut self.filter_params[tap]);

            // Cache repeated calculations
            let low_params = &velocity_params.velocity_low;
            let high_params = &velocity_params.velocity_high;

            let res = Del2::lerp(low_params.res.value(), high_params.res.value(), velocity);
            let velocity_cutoff = Del2::log_interpolate(
                low_params.cutoff.value(),
                high_params.cutoff.value(),
                velocity,
            );
            let note_cutoff = util::midi_note_to_freq(self.delay_data.notes[tap]);
            let cutoff = note_cutoff * self.params.taps.note_to_cutoff_amount.value()
                + velocity_cutoff * self.params.taps.velocity_to_cutoff_amount.value();
            let drive_db = Del2::lerp(
                util::gain_to_db(low_params.drive.value()),
                util::gain_to_db(high_params.drive.value()),
                velocity,
            );
            let drive = util::db_to_gain(drive_db);
            let mode =
                MyLadderMode::lerp(low_params.mode.value(), high_params.mode.value(), velocity);

            filter_params.set_resonance(res);
            filter_params.set_frequency(cutoff);
            filter_params.drive = drive;
            filter_params.ladder_mode = mode;
            self.ladders[tap].set_mix(mode);
        }
    }

    fn lerp(a: f32, b: f32, x: f32) -> f32 {
        a + (b - a) * x
    }
    fn log_interpolate(a: f32, b: f32, x: f32) -> f32 {
        a * (b / a).powf(x)
    }

    fn process_audio_blocks(&mut self, buffer: &mut Buffer) {
        for (_, block) in buffer.iter_blocks(buffer.samples()) {
            let block_len = block.samples();
            // TODO: assert needed?
            // assert!(block_len <= MAX_BLOCK_SIZE);
            // Either we resize here, or in the initialization fn
            // If we don't, we are slower.
            self.resize_temp_buffers(block_len.try_into().unwrap());
            let mut block_channels = block.into_iter();

            let out_l = block_channels.next().expect("Left output channel missing");
            let out_r = block_channels.next().expect("Right output channel missing");

            let write_index = self.delay_write_index;
            self.delay_buffer[0].write_latest(out_l, write_index);
            self.delay_buffer[1].write_latest(out_r, write_index);
            self.delay_write_index =
                (write_index + block_len as isize) % self.delay_buffer_size as isize;

            // TODO: no dry signal yet
            out_l.fill(0.0);
            out_r.fill(0.0);

            for tap in 0..NUM_TAPS {
                self.create_envelope_block(block_len, tap);
                if self.amp_envelopes[tap].is_smoothing() || (tap < self.delay_data.current_tap) {
                    self.read_tap_into_temp(tap);
                    self.process_tap(block_len, tap, out_l, out_r);
                }
            }
        }
    }

    fn read_tap_into_temp(&mut self, tap: usize) {
        let delay_time = self.delay_data.delay_times[tap] as isize;
        // delay_time - 1 because we are processing 2 samples at once in process_audio
        let read_index = self.delay_write_index - (delay_time - 1).max(0);

        self.delay_buffer[0].read_into(&mut self.temp_l, read_index);
        self.delay_buffer[1].read_into(&mut self.temp_r, read_index);
    }

    fn pan_to_haas_samples(pan: f32, sample_rate: f32) -> (i32, i32) {
        let delay_samples = (pan.abs() * (MAX_HAAS_MS / 1000.0) * sample_rate) as i32;
        if pan < 0.0 {
            (0, delay_samples) // Pan left: delay right
        } else {
            (delay_samples, 0) // Pan right: delay left
        }
    }

    fn process_tap(&mut self, block_len: usize, tap: usize, out_l: &mut [f32], out_r: &mut [f32]) {
        self.process_temp_with_envelope(block_len, tap);
        self.process_audio(block_len, tap, out_l, out_r);
    }

    fn set_amp_envelope_states(&mut self) {
        let out_index = self.mute_out_write_index;
        for tap in 0..NUM_TAPS {
            self.amp_envelope_write_index[tap] = 0;
        }
        // if there are any events
        if out_index > 0 {
            for tap in 0..NUM_TAPS {
                let (mute_out_state_slices, _) = self
                    .mute_out_states
                    .as_slices_len(0, out_index.try_into().unwrap());

                self.amp_envelope_states[tap]
                    .write_latest(&mute_out_state_slices, self.amp_envelope_write_index[tap]);
                self.amp_envelope_write_index[tap] =
                    (self.amp_envelope_write_index[tap] + out_index) % DSP_BLOCK_SIZE as isize;
            }
        }
    }

    fn create_envelope_block(&mut self, block_len: usize, tap: usize) {
        let index = self.amp_envelope_write_index[tap].try_into().unwrap();
        let mut block_start = 0;
        let mut block_end = block_len;
        // if no events
        if index == 0 {
            // println!("no events: tap {}, index: {}", tap, index);
            let tap_state = self.tap_states[tap].1;
            if tap_state != self.tap_was_muted[tap] {
                // println!("set tap {} to {}", tap, tap_state);
                self.mute_out(tap, tap_state);
                self.tap_was_muted[tap] = tap_state;
            }
            self.amp_envelopes[tap].next_block(
                &mut self.envelope_block[tap][block_start..block_end],
                block_end - block_start,
            );
        } else {
            let (end_time, state) = self.amp_envelope_states[tap].raw_at(0);
            block_end = *end_time as usize;
            // TODO: put back, or handle it on the next call?
            // if block_end == 0 {
            // block_end = block_len
            // }
            self.mute_out(tap, *state);
            self.amp_envelopes[tap].next_block(
                &mut self.envelope_block[tap][block_start..block_end],
                block_end - block_start,
            );
            for i in 1..index {
                println!("toggle tap {} at {} till {}", tap, block_start, block_end);
                let (start_time, state) = self.amp_envelope_states[tap].raw_at(i - 1);
                let (end_time, _) = self.amp_envelope_states[tap].raw_at(i);
                block_start = *start_time as usize;
                block_end = *end_time as usize;
                // println!("toggle tap {} to {} at {}", tap, state, block_start);
                self.mute_out(tap, *state);
                self.amp_envelopes[tap].next_block(
                    &mut self.envelope_block[tap][block_start..block_end],
                    block_end - block_start,
                );
            }
        }

        // debug
        // if self.envelope_block[tap][7] != 1.0 && self.envelope_block[tap][7] != 0.0 && tap == 0 {
        //     println!(
        //         "self.envelope_block[tap][7]: {}",
        //         self.envelope_block[tap][7]
        //     );
        // }
    }

    fn process_temp_with_envelope(&mut self, block_len: usize, tap: usize) {
        // Apply the envelope to each sample in the temporary buffers
        for i in 0..block_len {
            self.temp_l[i] *= self.envelope_block[tap][i];
            self.temp_r[i] *= self.envelope_block[tap][i];
        }
    }

    fn mute_out(&mut self, tap: usize, mute: bool) {
        let time_value = if mute {
            self.params.global.release_ms.value()
        } else {
            self.params.global.attack_ms.value()
        };
        let target_value = if mute { 0.0 } else { 1.0 };

        // if mute {
        // linear release, otherwise too short.
        //     self.amp_envelopes[tap].style = SmoothingStyle::Linear(time_value);
        // } else {
        // exponential attack
        self.amp_envelopes[tap].style = SmoothingStyle::Exponential(time_value);
        // }
        self.amp_envelopes[tap].set_target(self.sample_rate, target_value);
    }

    fn process_audio(
        &mut self,
        block_len: usize,
        tap: usize,
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) {
        // TODO: in this configuration, the filter does not work fully correctly.
        // You can't process a sample without having processed the sample that came before it, otherwise the filter states won't be correct.
        // The correct sollution, is to process 2 stereo taps at a time.
        // For that we need to feed two different parameter values to the filter, one for each tap.
        // No idea how...
        // Loop through each sample, processing two channels at a time
        for i in (0..block_len).step_by(2) {
            let output_gain1 = self.params.global.output_gain.smoothed.next();
            let output_gain2 = self.params.global.output_gain.smoothed.next();
            let drive = self.filter_params[tap].clone().drive;

            let pre_filter_gain1 = self.params.global.global_drive.smoothed.next();
            let pre_filter_gain2 = self.params.global.global_drive.smoothed.next();

            // Calculate post-filter gains, including the fade effect
            let post_filter_gain1 = output_gain1 / (drive * pre_filter_gain1);
            let post_filter_gain2 = output_gain2 / (drive * pre_filter_gain2);

            let mut frame = self.make_stereo_frame(i);

            // Apply global drive before filtering for each channel
            frame.as_mut_array()[0] *= pre_filter_gain1;
            frame.as_mut_array()[1] *= pre_filter_gain1;
            frame.as_mut_array()[2] *= pre_filter_gain2;
            frame.as_mut_array()[3] *= pre_filter_gain2;

            // Process the frame through the filter
            let processed = self.ladders[tap].tick_newton(frame);
            let mut frame_out = *processed.as_array();

            // Apply post-filter gains
            frame_out[0] *= post_filter_gain1;
            frame_out[1] *= post_filter_gain1;
            frame_out[2] *= post_filter_gain2;
            frame_out[3] *= post_filter_gain2;

            Del2::accumulate_processed_results(i, block_len, out_l, out_r, frame_out);
        }
    }

    fn make_stereo_frame(&self, index: usize) -> f32x4 {
        f32x4::from_array([
            self.temp_l[index],
            self.temp_r[index],
            self.temp_l.get(index + 1).copied().unwrap_or(0.0),
            self.temp_r.get(index + 1).copied().unwrap_or(0.0),
        ])
    }

    fn accumulate_processed_results(
        i: usize,
        block_len: usize,
        out_l: &mut [f32],
        out_r: &mut [f32],
        frame_out: [f32; 4],
    ) {
        out_l[i] += frame_out[0];
        out_r[i] += frame_out[1];
        if i + 1 < block_len {
            out_l[i + 1] += frame_out[2];
            out_r[i + 1] += frame_out[3];
        }
    }
    // TODO: when the fade time is long, there are bugs with taps not appearing, or fading out while fading in, etc.
    // more testing is needed
    // for fn initialize():

    // Either we resize in the audio thread, or in the initialization fn
    // If we don't, we are slower.
    fn resize_temp_buffers(&mut self, max_buffer_size: u32) {
        let max_size = max_buffer_size as usize;
        self.temp_l.resize(max_size, 0.0);
        self.temp_r.resize(max_size, 0.0);
        // for tap in 0..NUM_TAPS {
        //     self.envelope_block[tap].resize(max_size, 0.0);
        // }
    }

    fn calculate_buffer_size(&self, buffer_size: u32) -> u32 {
        ((TOTAL_DELAY_SAMPLES as f64 / buffer_size as f64).ceil() as u32 * buffer_size)
            .next_power_of_two()
    }

    fn set_delay_buffer_size(&mut self, buffer_config: &BufferConfig) {
        let min_size = self.calculate_buffer_size(buffer_config.min_buffer_size.unwrap_or(1));
        let max_size = self.calculate_buffer_size(buffer_config.max_buffer_size);

        self.delay_buffer_size = u32::max(min_size, max_size);

        // Apply the calculated size to the delay buffers
        self.delay_buffer
            .iter_mut()
            .for_each(|buffer| buffer.clear_set_len(self.delay_buffer_size as usize));
    }

    fn initialize_filter_parameters(&mut self) {
        for tap in 0..NUM_TAPS {
            unsafe {
                // Safety: Assumes exclusive access is guaranteed beforehand.
                let filter_params = Arc::get_mut_unchecked(&mut self.filter_params[tap]);
                filter_params.set_sample_rate(self.sample_rate);
            }
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

            // Example of condition dependent on editor or GUI state
            if self.params.editor_state.is_open() {
                amplitude = (amplitude / num_samples as f32).abs();
                let current_peak_meter = peak_meter.load(std::sync::atomic::Ordering::Relaxed);
                let new_peak_meter = if amplitude > current_peak_meter {
                    amplitude
                } else {
                    current_peak_meter * self.peak_meter_decay_weight
                        + amplitude * (1.0 - self.peak_meter_decay_weight)
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
                format!("{:.1$} ms", value, digits_after_decimal)
            } else {
                let seconds = value / 1000.0;
                // Same logic for seconds
                let digits_after_decimal =
                    (total_digits - seconds.trunc().to_string().len()).max(0);
                format!("{:.1$} s", seconds, digits_after_decimal)
            }
        })
    }

    fn s2v_f32_ms_then_s() -> Arc<dyn Fn(&str) -> Option<f32> + Send + Sync> {
        Arc::new(move |string| {
            let string = string.trim().to_lowercase();
            if let Some(ms_value_str) = string.strip_suffix("ms").or(string.strip_suffix(" ms")) {
                return ms_value_str.trim().parse::<f32>().ok();
            }
            if let Some(s_value_str) = string.strip_suffix("s").or(string.strip_suffix(" s")) {
                return s_value_str.trim().parse::<f32>().ok().map(|s| s * 1000.0);
            }
            string.parse::<f32>().ok()
        })
    }

    /*
     *
     *
     *
     *
     *
     */

    /// Get the index of a delay tap by its delay tap ID, if the delay tap exists. This does not immediately
    /// reutnr a reference to the delay tap to avoid lifetime issues.
    fn get_delay_tap_idx(&mut self, voice_id: i32) -> Option<usize> {
        self.delay_taps.iter_mut().position(
            |delay_tap| matches!(delay_tap, Some(delay_tap) if delay_tap.delay_tap_id == voice_id),
        )
    }

    /// Start a new delay tap with the given delay tap ID. If all delay_taps are currently in use, the oldest
    /// delay tap will be stolen. Returns a reference to the new delay tap.
    fn start_delay_tap(
        &mut self,
        context: &mut impl ProcessContext<Self>,
        sample_offset: u32,
        delay_tap_id: Option<i32>,
        channel: u8,
        note: u8,
    ) -> &mut DelayTap {
        let new_delay_tap = DelayTap {
            delay_tap_id: delay_tap_id
                .unwrap_or_else(|| compute_fallback_delay_tap_id(note, channel)),
            internal_delay_tap_id: self.next_internal_delay_tap_id,
            channel,
            note,
            velocity: 1.0,

            releasing: false,
            amp_envelope: Smoother::none(),

            delay_tap_gain: None,
            delay_time: 0,
            delayed_audio_l: [0.0; DSP_BLOCK_SIZE],
            delayed_audio_r: [0.0; DSP_BLOCK_SIZE],
        };
        self.next_internal_delay_tap_id = self.next_internal_delay_tap_id.wrapping_add(1);

        // Can't use `.iter_mut().find()` here because nonlexical lifetimes don't apply to return
        // values
        match self
            .delay_taps
            .iter()
            .position(|delay_tap| delay_tap.is_none())
        {
            Some(free_delay_tap_idx) => {
                self.delay_taps[free_delay_tap_idx] = Some(new_delay_tap);
                return self.delay_taps[free_delay_tap_idx].as_mut().unwrap();
            }
            None => {
                // If there is no free delay tap, find and steal the oldest one
                // SAFETY: We can skip a lot of checked unwraps here since we already know all delay_taps are in
                //         use
                let oldest_delay_tap = unsafe {
                    self.delay_taps
                        .iter_mut()
                        .min_by_key(|delay_tap| {
                            delay_tap.as_ref().unwrap_unchecked().internal_delay_tap_id
                        })
                        .unwrap_unchecked()
                };

                // The stolen delay tap needs to be terminated so the host can reuse its modulation
                // resources
                {
                    let oldest_delay_tap = oldest_delay_tap.as_ref().unwrap();
                    context.send_event(NoteEvent::VoiceTerminated {
                        timing: sample_offset,
                        voice_id: Some(oldest_delay_tap.delay_tap_id),
                        channel: oldest_delay_tap.channel,
                        note: oldest_delay_tap.note,
                    });
                }

                *oldest_delay_tap = Some(new_delay_tap);
                return oldest_delay_tap.as_mut().unwrap();
            }
        }
    }

    /// Start the release process for all delay taps by changing their amplitude envelope.
    fn start_release_for_all_delay_taps(&mut self, sample_rate: f32) {
        for delay_tap in self.delay_taps.iter_mut() {
            match delay_tap {
                Some(DelayTap {
                    releasing,
                    amp_envelope,
                    ..
                }) => {
                    *releasing = true;
                    amp_envelope.style =
                        SmoothingStyle::Exponential(self.params.global.release_ms.value());
                    amp_envelope.set_target(sample_rate, 0.0);
                }
                _ => (),
            }
        }
    }
    /// Start the release process for one or more delay tap by changing their amplitude envelope. If
    /// `delay_tap_id` is not provided, then this will terminate all matching delay_taps.
    fn start_release_for_delay_taps(
        &mut self,
        sample_rate: f32,
        delay_tap_id: Option<i32>,
        channel: u8,
        note: u8,
    ) {
        for delay_tap in self.delay_taps.iter_mut() {
            match delay_tap {
                Some(DelayTap {
                    delay_tap_id: candidate_delay_tap_id,
                    channel: candidate_channel,
                    note: candidate_note,
                    releasing,
                    amp_envelope,
                    ..
                }) if delay_tap_id == Some(*candidate_delay_tap_id)
                    || (channel == *candidate_channel && note == *candidate_note) =>
                {
                    *releasing = true;
                    amp_envelope.style =
                        SmoothingStyle::Exponential(self.params.global.release_ms.value());
                    amp_envelope.set_target(sample_rate, 0.0);

                    // If this targeted a single delay tap ID, we're done here. Otherwise there may be
                    // multiple overlapping delay_taps as we enabled support for that in the
                    // `PolyModulationConfig`.
                    if delay_tap_id.is_some() {
                        return;
                    }
                }
                _ => (),
            }
        }
    }

    /// Immediately terminate one or more delay tap, removing it from the pool and informing the host
    /// that the delay tap has ended. If `delay_tap_id` is not provided, then this will terminate all
    /// matching delay_taps.
    fn choke_delay_taps(
        &mut self,
        context: &mut impl ProcessContext<Self>,
        sample_offset: u32,
        delay_tap_id: Option<i32>,
        channel: u8,
        note: u8,
    ) {
        for delay_tap in self.delay_taps.iter_mut() {
            match delay_tap {
                Some(DelayTap {
                    delay_tap_id: candidate_delay_tap_id,
                    channel: candidate_channel,
                    note: candidate_note,
                    ..
                }) if delay_tap_id == Some(*candidate_delay_tap_id)
                    || (channel == *candidate_channel && note == *candidate_note) =>
                {
                    context.send_event(NoteEvent::VoiceTerminated {
                        timing: sample_offset,
                        // Notice how we always send the terminated delay tap ID here
                        voice_id: Some(*candidate_delay_tap_id),
                        channel,
                        note,
                    });
                    *delay_tap = None;

                    if delay_tap_id.is_some() {
                        return;
                    }
                }
                _ => (),
            }
        }
    }

    /*
     *
     *
     *
     *
     */
}

/// Compute a delay tap ID in case the host doesn't provide them. Polyphonic modulation will not work in
/// this case, but playing notes will.
const fn compute_fallback_delay_tap_id(note: u8, channel: u8) -> i32 {
    note as i32 | ((channel as i32) << 16)
}
/// Compute a voice ID in case the host doesn't provide them. Polyphonic modulation will not work in
/// this case, but playing notes will.
const fn compute_fallback_voice_id(note: u8, channel: u8) -> i32 {
    note as i32 | ((channel as i32) << 16)
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
    fn sequence() -> &'static [LadderMode] {
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

    fn index(&self) -> Option<usize> {
        Self::sequence().iter().position(|&mode| mode == self.0)
    }

    fn lerp(start: MyLadderMode, end: MyLadderMode, t: f32) -> LadderMode {
        let start_index = start.index().unwrap_or(0);
        let end_index = end.index().unwrap_or(Self::sequence().len() - 1);

        let t = t.max(0.0).min(1.0);

        let interpolated_index =
            (start_index as f32 + t * (end_index as f32 - start_index as f32)).round() as usize;

        Self::from_index(interpolated_index).0
    }

    fn lp6() -> Self {
        MyLadderMode(LadderMode::LP6)
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
        MyLadderMode(match index {
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
struct AtomicBoolArray {
    data: AtomicU8,
}

impl AtomicBoolArray {
    fn new() -> Self {
        Self {
            data: AtomicU8::new(0),
        }
    }

    #[inline(always)]
    fn load(&self, index: usize) -> bool {
        assert!(index < 8, "Index out of bounds");
        let mask = 1 << index;
        self.data.load(Ordering::SeqCst) & mask != 0
    }
    #[inline(always)]
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
    #[inline(always)]
    fn toggle(&self, index: usize) {
        assert!(index < 8, "Index out of bounds");
        let mask = 1 << index;
        self.data.fetch_xor(mask, Ordering::SeqCst);
    }
}

struct AtomicByteArray {
    data: AtomicU64,
}

impl AtomicByteArray {
    fn new() -> Self {
        Self {
            data: AtomicU64::new(0),
        }
    }

    #[inline(always)]
    fn load(&self, index: usize) -> u8 {
        assert!(index < 8, "Index out of bounds");
        let value = self.data.load(Ordering::SeqCst);
        ((value >> (index * 8)) & 0xFF) as u8
    }

    #[inline(always)]
    fn store(&self, index: usize, byte: u8) {
        assert!(index < 8, "Index out of bounds");
        let mask = !(0xFFu64 << (index * 8));
        let new_value = (self.data.load(Ordering::SeqCst) & mask) | ((byte as u64) << (index * 8));
        self.data.store(new_value, Ordering::SeqCst);
    }
    #[inline(always)]
    fn contains(&self, byte: u8) -> bool {
        let value = self.data.load(Ordering::SeqCst);
        let byte_u64 = byte as u64;

        for i in 0..8 {
            let shifted_byte = (value >> (i * 8)) & 0xFF;
            if shifted_byte == byte_u64 {
                return true;
            }
        }
        false
    }
}

// Represents the last played notes with capabilities for managing active notes.
struct LastPlayedNotes {
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
            notes: AtomicByteArray::new(),
            sequence: AtomicByteArray::new(),
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
                } else {
                    current_state = self.state.load(Ordering::SeqCst);
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
                } else {
                    // Reload state as previous compare_exchange was not successful.
                    current_state = self.state.load(Ordering::SeqCst);
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
        let mut current_state = self.state.load(Ordering::SeqCst);
        loop {
            // Check if the note exists among the recorded notes.
            if let Some(index) = (0..8).find(|&i| self.notes.load(i) == note) {
                // Calculate new state after disabling the note at the found index.
                let new_state = current_state & !(1 << index);
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
                    self.sequence.store(index, 0); // Reset sequence
                    self.active_notes.store(index, false); // Mark as inactive
                    break;
                } else {
                    // Reload state as previous compare_exchange was not successful.
                    current_state = self.state.load(Ordering::SeqCst);
                }
            } else {
                break;
            }
        }
    }

    /// Checks if a note is currently being played.
    fn is_playing(&self, note: u8) -> bool {
        // Find the index of the note and check if its spot in state is occupied.
        if let Some(index) = (0..8).find(|&i| self.notes.load(i) == note) {
            self.active_notes.load(index)
        } else {
            false
        }
    }

    /// Print the notes for testing purposes.
    fn _print_notes(&self) {
        // Width used for formatting alignment
        const WIDTH: usize = 4;

        for i in 0..8 {
            let note = self.notes.load(i);
            if self.is_playing(note) {
                // Print active notes
                print!("{:>WIDTH$}", note);
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
