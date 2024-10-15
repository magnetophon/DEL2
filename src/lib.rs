// #![allow(non_snake_case)]
#![feature(portable_simd)]
#![feature(get_mut_unchecked)]
use array_init::array_init;
use bit_mask_ring_buf::BMRingBuf;
use nih_plug::prelude::*;
use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::ViziaState;
use std::simd::f32x4;
use std::sync::{atomic::AtomicBool, Arc, Mutex};
use synfx_dsp::fh_va::{FilterParams, LadderFilter, LadderMode};
use triple_buffer::TripleBuffer;

mod editor;

// max seconds per tap
const MAX_TAP_SECONDS: usize = 10;
const MAX_DEBOUNCE_MILLISECONDS: f32 = 1000.0;
const MAX_NR_TAPS: usize = 8;
const TOTAL_DELAY_SECONDS: usize = MAX_TAP_SECONDS * MAX_NR_TAPS;
const MAX_SAMPLE_RATE: usize = 192000;
const TOTAL_DELAY_SAMPLES: usize = TOTAL_DELAY_SECONDS * MAX_SAMPLE_RATE;
const VELOCITY_BOTTOM_NAME_PREFIX: &str = "Bottom Velocity";
const VELOCITY_TOP_NAME_PREFIX: &str = "Top Velocity";
// this seems to be the number JUCE is using
// TODO: does this need to be set at runtime?
const MAX_SOUNDCARD_BUFFER_SIZE: usize = 32768;

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
    sample_rate: f32,
    delay_write_index: usize,
    samples_since_last_event: u32,
    timing_last_event: u32,
    debounce_tap_samples: u32,
    delay_buffer_size: u32,
    counting_state: CountingState,
    should_update_filter: Arc<AtomicBool>,
}

// for use in graph
#[derive(Clone)]
pub struct DelayData {
    velocity_array: [f32; MAX_NR_TAPS],
    delay_times_array: [u32; MAX_NR_TAPS],
    current_tap: usize,
    current_time: u32,
    time_out_samples: u32,
}
pub type DelayDataInput = triple_buffer::Input<DelayData>;
pub type DelayDataOutput = triple_buffer::Output<DelayData>;

impl Data for DelayData {
    // #[inline(always)]
    fn same(&self, other: &Self) -> bool {
        self.velocity_array == other.velocity_array
            && self.delay_times_array == other.delay_times_array
            && self.current_tap == other.current_tap
    }
}
/// All the parameters
#[derive(Params)]
struct Del2Params {
    #[persist = "editor-state"]
    editor_state: Arc<ViziaState>,
    #[id = "output_gain"]
    pub output_gain: FloatParam,
    #[id = "time_out_seconds"]
    pub time_out_seconds: FloatParam,
    #[id = "debounce_tap_milliseconds"]
    pub debounce_tap_milliseconds: FloatParam,
    #[nested(group = "taps")]
    pub taps: TapsSetParams,
}

/// Contains the top and bottom tap parameters.
#[derive(Params)]
pub struct TapsSetParams {
    #[nested(id_prefix = "velocity_bottom", group = "velocity_bottom")]
    pub velocity_bottom: Arc<TapGuiParams>,
    #[nested(id_prefix = "velocity_top", group = "velocity_top")]
    pub velocity_top: Arc<TapGuiParams>,
}

impl TapsSetParams {
    pub fn new(should_update_filter: Arc<AtomicBool>) -> Self {
        TapsSetParams {
            velocity_bottom: Arc::new(TapGuiParams::new(
                VELOCITY_BOTTOM_NAME_PREFIX,
                should_update_filter.clone(),
            )),
            velocity_top: Arc::new(TapGuiParams::new(
                VELOCITY_TOP_NAME_PREFIX,
                should_update_filter.clone(),
            )),
        }
    }
}

/// This struct contains the parameters for either the top or bottom tap. The `Params`
/// trait is implemented manually to avoid copy-pasting parameters for both types of compressor.
/// Both versions will have a parameter ID and a parameter name prefix to distinguish them.
#[derive(Params)]
pub struct TapGuiParams {
    #[id = "cutoff"]
    pub cutoff: FloatParam,
    #[id = "res"]
    pub res: FloatParam,
    #[id = "drive"]
    pub drive: FloatParam,
    #[id = "mode"]
    pub mode: EnumParam<MyLadderMode>,
}

impl TapGuiParams {
    /// Create a new [`TapSetParams`] object with a prefix for all parameter names.
    //TODO: Changing any of the threshold, ratio, or knee parameters causes the passed atomics to be updated.
    //TODO: These should be taken from a [`CompressorBank`] so the parameters are linked to it.
    pub fn new(name_prefix: &str, should_update_filter: Arc<AtomicBool>) -> Self {
        TapGuiParams {
            cutoff: FloatParam::new(
                format!("{name_prefix} Cutoff"),
                1000.0,
                FloatRange::Skewed {
                    min: 5.0, // This must never reach 0
                    max: 20_000.0,
                    factor: FloatRange::skew_factor(-2.5),
                },
            )
            .with_unit(" Hz")
            .with_value_to_string(formatters::v2s_f32_rounded(0))
            .with_callback(Arc::new({
                let should_update_filter = should_update_filter.clone();
                move |_| should_update_filter.store(true, std::sync::atomic::Ordering::Release)
            })),
            res: FloatParam::new(
                format!("{name_prefix} Res"),
                0.5,
                FloatRange::Linear { min: 0., max: 1. },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(2))
            .with_callback(Arc::new({
                let should_update_filter = should_update_filter.clone();
                move |_| should_update_filter.store(true, std::sync::atomic::Ordering::Release)
            })),
            drive: FloatParam::new(
                format!("{name_prefix} Drive"),
                1.0,
                FloatRange::Skewed {
                    min: 1.0, // This must never reach 0
                    // max: 15.8490, // 24 dB
                    max: 251.188643, // 48 dB
                    factor: FloatRange::skew_factor(-1.2),
                },
            )
            .with_unit(" dB")
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .with_string_to_value(formatters::s2v_f32_gain_to_db())
            // not strictly needed, but the easy way out.  TODO:  fix
            .with_callback(Arc::new({
                let should_update_filter = should_update_filter.clone();
                move |_| should_update_filter.store(true, std::sync::atomic::Ordering::Release)
            })),

            mode: EnumParam::new(format!("{name_prefix} Mode"), MyLadderMode::lp6())
                // not strictly needed, but the easy way out.  TODO:  fix
                .with_callback(Arc::new({
                    let should_update_filter = should_update_filter.clone();
                    move |_| should_update_filter.store(true, std::sync::atomic::Ordering::Release)
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
            sample_rate: 1.0,
            delay_write_index: 0,
            samples_since_last_event: 0,
            timing_last_event: 0,
            debounce_tap_samples: 0,
            delay_buffer_size: 0,
            counting_state: CountingState::TimeOut,
            should_update_filter,
        }
    }
}

impl Default for DelayData {
    fn default() -> Self {
        Self {
            velocity_array: [0.0; MAX_NR_TAPS],
            delay_times_array: [0; MAX_NR_TAPS],
            current_tap: 0,
            current_time: 0,
            time_out_samples: 0,
        }
    }
}

impl Del2Params {
    fn new(should_update_filter: Arc<AtomicBool>) -> Self {
        Self {
            editor_state: editor::default_state(),
            // This gain is stored as linear gain. NIH-plug comes with useful conversion functions
            // to treat these kinds of parameters as if we were dealing with decibels. Storing this
            // as decibels is easier to work with, but requires a conversion for every sample.
            output_gain: FloatParam::new(
                "output gain",
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
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .with_string_to_value(formatters::s2v_f32_gain_to_db()),
            time_out_seconds: FloatParam::new(
                "max tap time",
                3.0,
                FloatRange::Linear {
                    min: 0.0,
                    max: MAX_TAP_SECONDS as f32,
                },
            )
            .with_step_size(0.01)
            .with_unit(" s"),
            debounce_tap_milliseconds: FloatParam::new(
                "debounce time",
                10.0,
                FloatRange::Skewed {
                    min: 0.0,
                    max: MAX_DEBOUNCE_MILLISECONDS,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_step_size(0.01)
            .with_unit(" ms"),
            taps: TapsSetParams::new(should_update_filter.clone()),
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
        // Resize buffers and perform other potentially expensive initialization operations here.
        // The `reset()` function is always called right after this function. You can remove this
        // function if you do not need it.
        self.sample_rate = buffer_config.sample_rate;
        // Either we resize here, or in the audio thread
        // If we don't, we are slower.
        let max_buffer_size = buffer_config.max_buffer_size as usize;
        self.temp_l.resize(max_buffer_size, 0.0);
        self.temp_r.resize(max_buffer_size, 0.0);

        // TODO: check for correctness in all cases!
        // Calculate delay buffer size based on both min and max buffer sizes
        let calculate_buffer_size = |buffer_size: u32| -> u32 {
            ((TOTAL_DELAY_SAMPLES as f64 / buffer_size as f64).ceil() as u32 * buffer_size)
                .next_power_of_two()
        };

        let delay_buffer_size_min =
            calculate_buffer_size(buffer_config.min_buffer_size.unwrap_or(1));
        let delay_buffer_size_max = calculate_buffer_size(buffer_config.max_buffer_size);

        // Use the larger of the two calculated sizes
        self.delay_buffer_size = u32::max(delay_buffer_size_min, delay_buffer_size_max);

        // Apply the calculated size to both buffers in the array
        self.delay_buffer
            .iter_mut()
            .for_each(|buffer| buffer.clear_set_len(self.delay_buffer_size as usize));

        for tap in 0..MAX_NR_TAPS {
            unsafe {
                let filter_params = Arc::get_mut_unchecked(&mut self.filter_params[tap]);
                filter_params.set_sample_rate(self.sample_rate);
            };
        }

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
        self.delay_data.time_out_samples =
            (self.sample_rate as f64 * self.params.time_out_seconds.value() as f64) as u32;
        self.debounce_tap_samples = (self.sample_rate as f64
            * self.params.debounce_tap_milliseconds.value() as f64
            * 0.001) as u32;
        // process midi
        let mut next_event = context.next_event();
        // process midi events
        while let Some(event) = next_event {
            if let NoteEvent::NoteOn {
                timing, velocity, ..
            } = event
            {
                self.note_on(timing, velocity);
            }
            next_event = context.next_event();
        }

        // the actual delay
        let buffer_samples = buffer.samples();
        self.no_more_events(buffer_samples as u32);

        if self.delay_data.current_tap > 0 {
            self.delay_data.current_time = self.delay_data.delay_times_array
                [self.delay_data.current_tap - 1]
                + self.samples_since_last_event;
        } else {
            self.delay_data.current_time = self.samples_since_last_event;
        };
        self.delay_data_input.write(self.delay_data.clone());

        if self
            .should_update_filter
            .compare_exchange(
                true,
                false,
                std::sync::atomic::Ordering::Acquire,
                std::sync::atomic::Ordering::Relaxed,
            )
            .is_ok()
        {
            for tap in 0..self.delay_data.current_tap {
                let velocity = self.delay_data.velocity_array[tap];
                let velocity_params = &self.params.taps;

                unsafe {
                    let filter_params = Arc::get_mut_unchecked(&mut self.filter_params[tap]);

                    // Cache repeated calculations
                    let bottom_velocity = &velocity_params.velocity_bottom;
                    let top_velocity = &velocity_params.velocity_top;

                    let bottom_res = bottom_velocity.res.value();
                    let top_res = top_velocity.res.value();
                    let res = Del2::lerp(bottom_res, top_res, velocity);

                    let bottom_cutoff = bottom_velocity.cutoff.value();
                    let top_cutoff = top_velocity.cutoff.value();
                    let cutoff = Del2::log_interpolate(bottom_cutoff, top_cutoff, velocity);

                    let bottom_drive_db = util::gain_to_db(bottom_velocity.drive.value());
                    let top_drive_db = util::gain_to_db(top_velocity.drive.value());
                    let drive_db = Del2::lerp(bottom_drive_db, top_drive_db, velocity);
                    let drive = util::db_to_gain(drive_db);

                    let bottom_mode = bottom_velocity.mode.value();
                    let top_mode = top_velocity.mode.value();
                    let mode = MyLadderMode::lerp(bottom_mode, top_mode, velocity);
                    // println!("mode {}: {}", tap, mode);
                    // Updating filter parameters
                    filter_params.set_resonance(res);
                    filter_params.set_frequency(cutoff);
                    filter_params.drive = drive;
                    filter_params.ladder_mode = mode;
                    self.ladders[tap].set_mix(mode);
                }
            }
        };

        for (_, block) in buffer.iter_blocks(buffer_samples) {
            let block_len = block.samples();
            // Either we resize here, or in the initialization fn
            // If we don't, we are slower.
            // self.temp_l.resize(block_len, 0.0);
            // self.temp_r.resize(block_len, 0.0);
            let mut block_channels = block.into_iter();

            let out_l = block_channels.next().unwrap();
            let out_r = block_channels.next().unwrap();

            self.delay_buffer[0].write_latest(out_l, self.delay_write_index as isize);
            self.delay_buffer[1].write_latest(out_r, self.delay_write_index as isize);

            // TODO: no dry signal yet
            // Clear output buffers for accumulated results
            out_l.fill(0.0);
            out_r.fill(0.0);

            for tap in 0..self.delay_data.current_tap {
                let delay_time = self.delay_data.delay_times_array[tap] as isize;
                // - 1 because we process 2 stereo samples at a time
                let read_index = self.delay_write_index as isize - (delay_time - 1).max(0);
                // let velocity = self.delay_data.velocity_array[tap];
                let recip_drive = 1.0 / self.filter_params[tap].clone().drive;

                self.delay_buffer[0].read_into(&mut self.temp_l, read_index);
                self.delay_buffer[1].read_into(&mut self.temp_r, read_index);

                // TODO: in this configuration, the filter does not work fully correctly.
                // You can't process a sample without having processed the sample that came before it, otherwise the filter states won't be correct.
                // The correct sollution, is to process 2 stereo taps at a time.
                // For that we need to feed two different parameter values to the filter, one for each tap.
                // No idea how...

                // Process audio in blocks of 2 samples, using 4 channels at a time
                for i in (0..block_len).step_by(2) {
                    // Prepare the frame with two stereo pairs
                    let frame = f32x4::from_array([
                        self.temp_l[i],
                        self.temp_r[i],
                        self.temp_l.get(i + 1).copied().unwrap_or(0.0),
                        self.temp_r.get(i + 1).copied().unwrap_or(0.0),
                    ]);

                    // Process the frame
                    let processed = self.ladders[tap].tick_newton(frame);
                    let frame_out = *processed.as_array();

                    // Accumulate the processed results
                    out_l[i] += frame_out[0] * recip_drive;
                    out_r[i] += frame_out[1] * recip_drive;
                    if i + 1 < block_len {
                        out_l[i + 1] += frame_out[2] * recip_drive;
                        out_r[i + 1] += frame_out[3] * recip_drive;
                    }
                }
            }

            self.delay_write_index =
                (self.delay_write_index + block_len) % self.delay_buffer_size as usize;
        }

        ProcessStatus::Normal
    }
}

impl Del2 {
    // #[inline(always)]
    fn note_on(&mut self, timing: u32, velocity: f32) {
        match self.counting_state {
            CountingState::TimeOut => {
                self.delay_data
                    .delay_times_array
                    .iter_mut()
                    .for_each(|x| *x = 0);
                self.delay_data
                    .velocity_array
                    .iter_mut()
                    .for_each(|x| *x = 0.0);
                self.delay_data.current_tap = 0;
                self.timing_last_event = timing;
                self.counting_state = CountingState::CountingInBuffer;
            }
            CountingState::CountingInBuffer => {
                if (timing - self.timing_last_event) > self.debounce_tap_samples
                    && self.delay_data.current_tap < MAX_NR_TAPS
                {
                    self.samples_since_last_event = timing - self.timing_last_event;
                    self.timing_last_event = timing;
                } else {
                    // println!("debounce in!");
                    return;
                }
            }
            CountingState::CountingAcrossBuffer => {
                if (self.samples_since_last_event + timing) > self.debounce_tap_samples
                    && self.delay_data.current_tap < MAX_NR_TAPS
                {
                    self.samples_since_last_event += timing;
                    self.timing_last_event = timing;
                    // println!("across to in buffer!");
                    self.counting_state = CountingState::CountingInBuffer;
                } else {
                    // println!("debounce across!");
                    return;
                }
            }
        }
        if self.samples_since_last_event <= self.delay_data.time_out_samples {
            if self.delay_data.current_tap < MAX_NR_TAPS
                && self.counting_state != CountingState::TimeOut
                && self.samples_since_last_event > 0
                && velocity > 0.0
            {
                if self.delay_data.current_tap > 0 {
                    self.delay_data.delay_times_array[self.delay_data.current_tap] = self
                        .samples_since_last_event
                        + self.delay_data.delay_times_array[self.delay_data.current_tap - 1];
                } else {
                    self.delay_data.delay_times_array[self.delay_data.current_tap] =
                        self.samples_since_last_event;
                }
                self.delay_data.velocity_array[self.delay_data.current_tap] = velocity;
                self.delay_data.current_tap += 1;
                // we have a new tap, so we're interpolating new filter parameters
                self.should_update_filter
                    .store(true, std::sync::atomic::Ordering::Release);
            };
        } else {
            self.counting_state = CountingState::TimeOut;
            self.timing_last_event = 0;
            self.samples_since_last_event = 0;
            // println!("time out note on");
        };
    }

    // #[inline(always)]
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

        if self.samples_since_last_event > self.delay_data.time_out_samples {
            self.counting_state = CountingState::TimeOut;
            self.timing_last_event = 0;
            self.samples_since_last_event = 0;
            // println!("time out no_more_events");
        };
    }
    #[inline]
    fn lerp(a: f32, b: f32, x: f32) -> f32 {
        a + (b - a) * x
    }
    #[inline]
    fn log_interpolate(a: f32, b: f32, x: f32) -> f32 {
        a * (b / a).powf(x)
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
