// TODO:
//
// make a midi learn struct/button
// see: git@github.com:icsga/Yazz.git
// use it to make a few midi triggers:
// - mute out
// - mute in
// - mute both
// all mutes with choice between latch and toggle
// - reset pattern
// - stop listening for taps (so it can be used in a daw on an instrument track)
//
// add adsr envelopes
//
// #![allow(non_snake_case)]
#![feature(portable_simd)]
#![feature(get_mut_unchecked)]
use array_init::array_init;
use bit_mask_ring_buf::BMRingBuf;
use nih_plug::prelude::*;
use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::ViziaState;
use std::simd::f32x4;
use std::sync::atomic::Ordering;
use std::sync::{atomic::AtomicBool, Arc, Mutex};
use synfx_dsp::fh_va::{FilterParams, LadderFilter, LadderMode};
use triple_buffer::TripleBuffer;

mod editor;

// max seconds per tap
const MAX_TAP_SECONDS: usize = 10;
const MAX_NR_TAPS: usize = 8;
const TOTAL_DELAY_SECONDS: usize = MAX_TAP_SECONDS * MAX_NR_TAPS;
const MAX_SAMPLE_RATE: usize = 192000;
const TOTAL_DELAY_SAMPLES: usize = TOTAL_DELAY_SECONDS * MAX_SAMPLE_RATE;
const VELOCITY_LOW_NAME_PREFIX: &str = "low velocity";
const VELOCITY_HIGH_NAME_PREFIX: &str = "high velocity";
// this seems to be the number JUCE is using
const MAX_SOUNDCARD_BUFFER_SIZE: usize = 32768;

const PEAK_METER_DECAY_MS: f64 = 150.0;
const FADE_DURATION_SECONDS: f32 = 0.01; // 10ms

struct Del2 {
    params: Arc<Del2Params>,
    filter_params: [Arc<FilterParams>; MAX_NR_TAPS],
    ladders: [LadderFilter; MAX_NR_TAPS],

    // delay write buffer
    delay_buffer: [BMRingBuf<f32>; 2],
    // delay read buffers
    temp_l: Vec<f32>,
    temp_r: Vec<f32>,

    delay_data: DelayData,
    delay_data_input: DelayDataInput,
    delay_data_output: Arc<Mutex<DelayDataOutput>>,
    // N counters to know where in the fade in we are: 0 is the start
    fade_in_states: [usize; MAX_NR_TAPS],
    learned_notes: [u8; MAX_NR_TAPS],
    sample_rate: f32,
    /// Needed to normalize the peak meter's response based on the sample rate.
    peak_meter_decay_weight: f32,
    /// The current data for the peak meter. This is stored as an [`Arc`] so we can share it between
    /// the GUI and the audio processing parts. If you have more state to share, then it's a good
    /// idea to put all of that in a struct behind a single `Arc`.
    ///
    /// This is stored as voltage gain.
    /// This is stored as voltage gain.
    input_meter: Arc<AtomicF32>,
    output_meter: Arc<AtomicF32>,
    delay_write_index: usize,
    fade_samples: usize,
    // a counter to know where in the fade out we are: 0 is the start
    fade_out_state: usize,
    // same as delay_data.current_tap, but only reset after the fade out is done
    fade_out_tap: usize,
    // for which control are we learning?
    learning_index: usize,
    samples_since_last_event: u32,
    timing_last_event: u32,
    min_tap_samples: u32,
    delay_buffer_size: u32,
    counting_state: CountingState,
    should_update_filter: Arc<AtomicBool>,
}

// for use in graph
#[derive(Clone)]
pub struct DelayData {
    delay_times: [u32; MAX_NR_TAPS],
    velocities: [f32; MAX_NR_TAPS],
    notes: [u8; MAX_NR_TAPS],
    current_tap: usize,
    current_time: u32,
    max_tap_samples: u32,
}
pub type DelayDataInput = triple_buffer::Input<DelayData>;
pub type DelayDataOutput = triple_buffer::Output<DelayData>;

impl Data for DelayData {
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
    pub taps: DualFilterGuiParams,
}

/// Contains the global parameters.
#[derive(Params)]
pub struct GlobalParams {
    #[nested(id_prefix = "timing_params", group = "timing_params")]
    pub timing_params: Arc<TimingParams>,
    #[nested(id_prefix = "gain_params", group = "gain_params")]
    pub gain_params: Arc<GainParams>,
}

impl GlobalParams {
    pub fn new() -> Self {
        GlobalParams {
            timing_params: Arc::new(TimingParams::new()),
            gain_params: Arc::new(GainParams::new()),
        }
    }
}

/// This struct contains the parameters for either the high or low tap. The `Params`
/// trait is implemented manually to avoid copy-pasting parameters for both types of compressor.
/// Both versions will have a parameter ID and a parameter name prefix to distinguish them.
#[derive(Params)]
pub struct TimingParams {
    #[id = "max_tap_seconds"]
    pub max_tap_seconds: FloatParam,
    #[id = "min_tap_milliseconds"]
    pub min_tap_milliseconds: FloatParam,
}

impl TimingParams {
    /// Create a new [`TapSetParams`] object with a prefix for all parameter names.
    pub fn new() -> Self {
        TimingParams {
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
        }
    }
}

#[derive(Params)]
pub struct GainParams {
    #[id = "output_gain"]
    pub output_gain: FloatParam,
    #[id = "global_drive"]
    pub global_drive: FloatParam,
}

impl GainParams {
    /// Create a new [`TapSetParams`] object with a prefix for all parameter names.
    pub fn new() -> Self {
        GainParams {
            // This gain is stored as linear gain. NIH-plug comes with useful conversion functions
            // to treat these kinds of parameters as if we were dealing with decibels. Storing this
            // as decibels is easier to work with, but requires a conversion for every sample.
            output_gain: FloatParam::new(
                "out gain",
                util::db_to_gain(0.0),
                FloatRange::Skewed {
                    min: util::db_to_gain(-30.0),
                    max: util::db_to_gain(30.0),
                    // This makes the range appear as if it was linear when displaying the values as
                    // decibels
                    factor: FloatRange::gain_skew_factor(-30.0, 30.0),
                },
            )
            // Because the gain parameter is stored as linear gain instead of storing the value as
            // decibels, we need logarithmic smoothing
            .with_smoother(SmoothingStyle::Logarithmic(50.0))
            .with_unit(" dB")
            // There are many predefined formatters we can use here. If the gain was stored as
            // decibels instead of as a linear gain value, we could have also used the
            // `.with_step_size(0.1)` function to get internal rounding.
            .with_value_to_string(formatters::v2s_f32_gain_to_db(1))
            .with_string_to_value(formatters::s2v_f32_gain_to_db()),
            global_drive: FloatParam::new(
                "drive",
                util::db_to_gain(0.0),
                FloatRange::Skewed {
                    min: util::db_to_gain(-30.0),
                    max: util::db_to_gain(30.0),
                    // This makes the range appear as if it was linear when displaying the values as
                    // decibels
                    factor: FloatRange::gain_skew_factor(-30.0, 30.0),
                },
            )
            // Because the gain parameter is stored as linear gain instead of storing the value as
            // decibels, we need logarithmic smoothing
            .with_smoother(SmoothingStyle::Logarithmic(50.0))
            .with_unit(" dB")
            // There are many predefined formatters we can use here. If the gain was stored as
            // decibels instead of as a linear gain value, we could have also used the
            // `.with_step_size(0.1)` function to get internal rounding.
            .with_value_to_string(formatters::v2s_f32_gain_to_db(1))
            .with_string_to_value(formatters::s2v_f32_gain_to_db()),
        }
    }
}

/// Contains the high and low tap parameters.
#[derive(Params)]
pub struct DualFilterGuiParams {
    #[nested(id_prefix = "velocity_low", group = "velocity_low")]
    pub velocity_low: Arc<FilterGuiParams>,
    #[nested(id_prefix = "velocity_high", group = "velocity_high")]
    pub velocity_high: Arc<FilterGuiParams>,
}

impl DualFilterGuiParams {
    pub fn new(should_update_filter: Arc<AtomicBool>) -> Self {
        DualFilterGuiParams {
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
                6000.0,                // Default cutoff for velocity_high
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
                    min: 5.0,
                    max: 20_000.0,
                    factor: FloatRange::skew_factor(-2.5),
                },
            )
            .with_value_to_string(formatters::v2s_f32_hz_then_khz(2))
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
    MidiLearn,
}

impl Default for Del2 {
    fn default() -> Self {
        let initial_delay_data: DelayData = DelayData::default();
        let (delay_data_input, delay_data_output) = TripleBuffer::new(&initial_delay_data).split();

        let filter_params = array_init(|_| Arc::new(FilterParams::new()));
        let should_update_filter = Arc::new(AtomicBool::new(false));
        let ladders: [LadderFilter; MAX_NR_TAPS] =
            array_init(|i| LadderFilter::new(filter_params[i].clone()));
        Self {
            params: Arc::new(Del2Params::new(should_update_filter.clone())),
            filter_params,
            ladders,
            delay_buffer: [
                BMRingBuf::<f32>::from_len(TOTAL_DELAY_SAMPLES),
                BMRingBuf::<f32>::from_len(TOTAL_DELAY_SAMPLES),
            ],
            temp_l: vec![0.0; MAX_SOUNDCARD_BUFFER_SIZE],
            temp_r: vec![0.0; MAX_SOUNDCARD_BUFFER_SIZE],

            delay_data: initial_delay_data,
            delay_data_input,
            delay_data_output: Arc::new(Mutex::new(delay_data_output)),
            fade_in_states: [0; MAX_NR_TAPS],
            learned_notes: [0; MAX_NR_TAPS],
            sample_rate: 1.0,
            peak_meter_decay_weight: 1.0,
            input_meter: Arc::new(AtomicF32::new(util::MINUS_INFINITY_DB)),
            output_meter: Arc::new(AtomicF32::new(util::MINUS_INFINITY_DB)),
            delay_write_index: 0,
            fade_samples: 0,
            fade_out_state: 0,
            fade_out_tap: 0,
            learning_index: 0,
            samples_since_last_event: 0,
            timing_last_event: 0,
            min_tap_samples: 0,
            delay_buffer_size: 0,
            counting_state: CountingState::TimeOut,
            should_update_filter,
        }
    }
}

impl Default for DelayData {
    fn default() -> Self {
        Self {
            delay_times: [0; MAX_NR_TAPS],
            velocities: [0.0; MAX_NR_TAPS],
            notes: [0; MAX_NR_TAPS],
            current_tap: 0,
            current_time: 0,
            max_tap_samples: 0,
        }
    }
}

impl Del2Params {
    fn new(should_update_filter: Arc<AtomicBool>) -> Self {
        Self {
            editor_state: editor::default_state(),
            taps: DualFilterGuiParams::new(should_update_filter.clone()),
            global: GlobalParams::new(),
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
        self.fade_samples = (FADE_DURATION_SECONDS * self.sample_rate) as usize;
        self.fade_out_state = self.fade_samples;

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

        true
    }

    fn reset(&mut self) {
        for i in 0..MAX_NR_TAPS {
            self.ladders[i].s = [f32x4::splat(0.); 4];
        }
        // Reset buffers and envelopes here. This can be called from the audio thread and may not
        // allocate. You can remove this function if you do not need it.
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        self.update_timing_params();

        self.process_midi_events(context);
        self.prepare_for_delay(buffer.samples());

        if self.should_update_filter() {
            for tap in 0..self.fade_out_tap {
                self.update_filter(tap);
            }
        }

        self.update_peak_meter(buffer, &self.input_meter);
        self.process_audio_blocks(buffer);
        self.apply_fade_out(buffer);
        self.update_peak_meter(buffer, &self.output_meter);
        ProcessStatus::Normal
    }
}

impl Del2 {
    fn update_timing_params(&mut self) {
        let sample_rate = self.sample_rate as f32;
        self.delay_data.max_tap_samples =
            (sample_rate * self.params.global.timing_params.max_tap_seconds.value()) as u32;
        self.min_tap_samples = (sample_rate
            * self
                .params
                .global
                .timing_params
                .min_tap_milliseconds
                .value()
            * 0.001) as u32;
    }

    fn process_midi_events(&mut self, context: &mut impl ProcessContext<Self>) {
        while let Some(event) = context.next_event() {
            if let NoteEvent::NoteOn {
                timing,
                note,
                velocity,
                ..
            } = event
            {
                let should_record_tap = self.delay_data.current_tap < MAX_NR_TAPS;

                match self.counting_state {
                    CountingState::TimeOut => {
                        // If in TimeOut state, reset and start new counting phase
                        self.fade_out_state = 0;
                        self.delay_data.current_tap = 0;
                        self.timing_last_event = timing;
                        self.counting_state = CountingState::CountingInBuffer;
                    }
                    CountingState::CountingInBuffer => {
                        // Validate and record a new tap within the buffer
                        if timing - self.timing_last_event > self.min_tap_samples
                            && should_record_tap
                        {
                            self.samples_since_last_event = timing - self.timing_last_event;
                            self.timing_last_event = timing;
                        } else {
                            continue; // Debounce or max taps reached, ignore tap
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
                            continue; // Debounce across buffer or max taps, ignore
                        }
                    }
                    CountingState::MidiLearn => {
                        // Store learned note and revert to TimeOut state
                        self.learned_notes[self.learning_index] = note;
                        self.counting_state = CountingState::TimeOut;
                    }
                }

                // Check for timeout condition and reset if necessary
                if self.samples_since_last_event > self.delay_data.max_tap_samples {
                    self.counting_state = CountingState::TimeOut;
                    self.timing_last_event = 0;
                    self.samples_since_last_event = 0;
                } else if should_record_tap
                    && !matches!(
                        self.counting_state,
                        CountingState::TimeOut | CountingState::MidiLearn
                    )
                    && self.samples_since_last_event > 0
                    && velocity > 0.0
                {
                    // Update tap information with timing and velocity
                    let current_tap = self.delay_data.current_tap;
                    if current_tap > 0 {
                        self.delay_data.delay_times[current_tap] = self.samples_since_last_event
                            + self.delay_data.delay_times[current_tap - 1];
                    } else {
                        self.delay_data.delay_times[current_tap] = self.samples_since_last_event;
                    }

                    self.delay_data.velocities[current_tap] = velocity;
                    self.delay_data.notes[current_tap] = note;
                    self.fade_in_states[current_tap] = 0;
                    self.delay_data.current_tap += 1;
                    self.fade_out_tap = self.delay_data.current_tap;

                    // Indicate filter update needed
                    self.should_update_filter.store(true, Ordering::Release);
                }
            }
        }
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
            CountingState::MidiLearn => {}
        }

        if self.samples_since_last_event > self.delay_data.max_tap_samples {
            self.counting_state = CountingState::TimeOut;
            self.timing_last_event = 0;
            self.samples_since_last_event = 0;
            // println!("time out no_more_events");
        };
    }
    fn should_update_filter(&mut self) -> bool {
        self.should_update_filter
            .compare_exchange(
                true,
                false,
                std::sync::atomic::Ordering::Acquire,
                std::sync::atomic::Ordering::Relaxed,
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
            let cutoff = Del2::log_interpolate(
                low_params.cutoff.value(),
                high_params.cutoff.value(),
                velocity,
            );
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
            // Either we resize here, or in the initialization fn
            // If we don't, we are slower.
            // self.resize_temp_buffers(block_len);
            let mut block_channels = block.into_iter();

            let out_l = block_channels.next().expect("Left output channel missing");
            let out_r = block_channels.next().expect("Right output channel missing");

            self.delay_buffer[0].write_latest(out_l, self.delay_write_index as isize);
            self.delay_buffer[1].write_latest(out_r, self.delay_write_index as isize);
            // TODO: no dry signal yet
            out_l.fill(0.0);
            out_r.fill(0.0);

            for tap in 0..self.fade_out_tap {
                self.process_tap(block_len, tap, out_l, out_r);
            }

            self.delay_write_index =
                (self.delay_write_index + block_len) % self.delay_buffer_size as usize;
        }
    }

    fn process_tap(&mut self, block_len: usize, tap: usize, out_l: &mut [f32], out_r: &mut [f32]) {
        let delay_time = self.delay_data.delay_times[tap] as isize;
        // delay_time - 1 because we are processing 2 samples at once in process_audio
        let read_index = self.delay_write_index as isize - (delay_time - 1).max(0);

        self.read_buffers_into_temp(read_index);
        self.process_audio(block_len, tap, out_l, out_r);
    }

    fn read_buffers_into_temp(&mut self, read_index: isize) {
        self.delay_buffer[0].read_into(&mut self.temp_l, read_index);
        self.delay_buffer[1].read_into(&mut self.temp_r, read_index);
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
        let fade_samples = self.fade_samples;
        for i in (0..block_len).step_by(2) {
            // Begin the loop by dealing with mutable borrowing
            let (fade_in_factor1, fade_in_factor2) = {
                let tap_fade_in_state = &mut self.fade_in_states[tap]; // Mutable borrow here

                // Calculate fade-in factors for two consecutive samples
                let fade_in_factor1 = if *tap_fade_in_state < fade_samples {
                    *tap_fade_in_state as f32 / fade_samples as f32
                } else {
                    1.0
                };

                let fade_in_factor2 = if *tap_fade_in_state + 1 < fade_samples {
                    (*tap_fade_in_state + 1) as f32 / fade_samples as f32
                } else {
                    1.0
                };

                // Increment fade progress appropriately for two samples
                if *tap_fade_in_state < fade_samples {
                    *tap_fade_in_state += 1;
                }
                if *tap_fade_in_state < fade_samples {
                    *tap_fade_in_state += 1;
                }

                (fade_in_factor1, fade_in_factor2)
            };

            // Proceed with immutable operations now that mutable borrow scope is closed
            let output_gain1 = self.params.global.gain_params.output_gain.smoothed.next();
            let output_gain2 = self.params.global.gain_params.output_gain.smoothed.next();
            let drive = self.filter_params[tap].clone().drive;

            let pre_filter_gain1 = self.params.global.gain_params.global_drive.smoothed.next();
            let pre_filter_gain2 = self.params.global.gain_params.global_drive.smoothed.next();

            // Calculate post-filter gains, including the fade effect
            let post_filter_gain1 = (output_gain1 / (drive * pre_filter_gain1)) * fade_in_factor1;
            let post_filter_gain2 = (output_gain2 / (drive * pre_filter_gain2)) * fade_in_factor2;

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
    fn apply_fade_out(&mut self, buffer: &mut Buffer) {
        let fade_samples = self.fade_samples;
        let mut fade_out_state = self.fade_out_state;
        if fade_out_state < fade_samples {
            for channel_samples in buffer.iter_samples() {
                for sample in channel_samples {
                    let fade_out_factor = if fade_out_state < fade_samples {
                        1.0 - (fade_out_state as f32 / fade_samples as f32)
                    } else {
                        self.fade_out_tap = 0;
                        // self.fade_out_state = fade_samples;
                        0.0
                    };
                    *sample *= fade_out_factor;
                    if fade_out_state < fade_samples {
                        fade_out_state += 1;
                    }
                }
            }
            self.fade_out_state = fade_out_state;
        }
    }
    // for fn initialize():

    // Either we resize in the audio thread, or in the initialization fn
    // If we don't, we are slower.
    fn resize_temp_buffers(&mut self, max_buffer_size: u32) {
        let max_size = max_buffer_size as usize;
        self.temp_l.resize(max_size, 0.0);
        self.temp_r.resize(max_size, 0.0);
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
        for tap in 0..MAX_NR_TAPS {
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

nih_export_clap!(Del2);
nih_export_vst3!(Del2);
