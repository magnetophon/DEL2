#![feature(portable_simd)]
use bit_mask_ring_buf::BMRingBuf;
use nih_plug::prelude::*;
use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::ViziaState;
use std::simd::f32x4;
use std::sync::{atomic::AtomicBool, Arc, Mutex};
use synfx_dsp::fh_va::{FilterParams, LadderFilter, LadderMode, SvfMode};
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

struct Del2 {
    params: Arc<Del2Params>,
    filter_params: [FilterParams; MAX_NR_TAPS],

    ladders: [LadderFilter; MAX_NR_TAPS],

    delay_buffer: [BMRingBuf<f32>; 2],
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
    should_update_filter: Arc<std::sync::atomic::AtomicBool>,
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
    #[id = "gain"]
    pub gain: FloatParam,
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
    pub fn default() -> Self {
        TapsSetParams {
            velocity_bottom: Arc::new(TapGuiParams::new(
                VELOCITY_BOTTOM_NAME_PREFIX,
                Arc::new(AtomicBool::new(true)),
            )),
            velocity_top: Arc::new(TapGuiParams::new(
                VELOCITY_TOP_NAME_PREFIX,
                Arc::new(AtomicBool::new(true)),
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
    #[id = "slope"]
    pub slope: EnumParam<LadderSlope>,
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
            .with_smoother(SmoothingStyle::Logarithmic(20.0))
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
            .with_smoother(SmoothingStyle::Linear(20.0))
            .with_value_to_string(formatters::v2s_f32_rounded(2))
            .with_callback(Arc::new({
                let should_update_filter = should_update_filter;
                move |_| should_update_filter.store(true, std::sync::atomic::Ordering::Release)
            })),
            // TODO: with_value_to_string should actually convert it to db
            drive: FloatParam::new(
                "{name_prefix} Drive",
                1.0,
                FloatRange::Skewed {
                    min: 1.0, // This must never reach 0
                    max: 15.8490,
                    factor: FloatRange::skew_factor(-1.2),
                },
            )
            .with_smoother(SmoothingStyle::Logarithmic(100.0))
            .with_unit(" dB")
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2)),

            slope: EnumParam::new("{name_prefix} Slope", LadderSlope::LP6),
        }
    }
}

#[derive(Enum, Debug, PartialEq, Eq)]
pub enum LadderSlope {
    LP6,
    LP12,
    LP18,
    LP24,
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

        let filter_params = array_init::array_init(|_| FilterParams::new());
        let should_update_filter = Arc::new(std::sync::atomic::AtomicBool::new(true));
        // Create an array of LadderFilter using const generics
        let ladders =
            array_init::array_init(|_| LadderFilter::new(filter_params[0].clone().into()));
        Self {
            params: Arc::new(Del2Params::default()),
            filter_params,
            ladders,
            delay_buffer: [
                BMRingBuf::<f32>::from_len(TOTAL_DELAY_SAMPLES),
                BMRingBuf::<f32>::from_len(TOTAL_DELAY_SAMPLES),
            ],
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

impl Default for Del2Params {
    fn default() -> Self {
        Self {
            editor_state: editor::default_state(),
            // This gain is stored as linear gain. NIH-plug comes with useful conversion functions
            // to treat these kinds of parameters as if we were dealing with decibels. Storing this
            // as decibels is easier to work with, but requires a conversion for every sample.
            gain: FloatParam::new(
                "gain",
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
            taps: TapsSetParams::default(),
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
        self.sample_rate = buffer_config.sample_rate;

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

        // Resize buffers and perform other potentially expensive initialization operations here.
        // The `reset()` function is always called right after this function. You can remove this
        // function if you do not need it.
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
            println!(
                "bottom cutoff: {:?}",
                self.params.taps.velocity_bottom.cutoff.value()
            );
            println!(
                "bottom res: {:?}",
                self.params.taps.velocity_bottom.res.value()
            );
            println!("sample_rate: {:?}", self.sample_rate);
            for i in 0..MAX_NR_TAPS {
                self.filter_params[i].set_sample_rate(self.sample_rate);
                self.filter_params[i].set_resonance(self.params.taps.velocity_bottom.res.value());
                self.filter_params[i]
                    .set_frequency(self.params.taps.velocity_bottom.cutoff.value());
            }
        };
        // TODO: put behind a should_update with a callback?
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

        for (_, block) in buffer.iter_blocks(buffer_samples) {
            let block_len = block.samples();
            let mut block_channels = block.into_iter();

            let out_l = block_channels.next().unwrap();
            let out_r = block_channels.next().unwrap();

            self.delay_buffer[0].write_latest(out_l, self.delay_write_index as isize);
            self.delay_buffer[1].write_latest(out_r, self.delay_write_index as isize);

            // TODO: no dry signal yet
            out_l.fill(0.0);
            out_r.fill(0.0);

            for tap in 0..self.delay_data.current_tap {
                let delay_time = self.delay_data.delay_times_array[tap] as isize;
                let read_index = self.delay_write_index as isize - delay_time;
                let velocity = self.delay_data.velocity_array[tap];
                let velocity_squared = f32::powi(velocity, 2);
                // Temporary buffers to hold the read values for processing
                let mut temp_l = vec![0.0; block_len];
                let mut temp_r = vec![0.0; block_len];
                self.delay_buffer[0].read_into(&mut temp_l, read_index);
                self.delay_buffer[1].read_into(&mut temp_r, read_index);
                // Accumulate the contributions
                for i in 0..block_len {
                    let frame = f32x4::from_array([temp_l[i], temp_r[i], 0.0, 0.0]);
                    // TODO: dc-filter, upsample, downsample
                    let processed = self.ladders[tap].tick_newton(frame);
                    let frame_out = *processed.as_array();
                    out_l[i] += frame_out[0] * velocity_squared;
                    out_r[i] += frame_out[1] * velocity_squared;
                    // out_l[i] += temp_l[i] * velocity_squared;
                    // out_r[i] += temp_r[i] * velocity_squared;
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
                // println!("current_tap: {}", self.delay_data.current_tap);
                // println!("times: {:#?}", self.delay_data.delay_times_array);
                // println!("velocities: {:#?}", self.delay_data.velocity_array);
                self.delay_data.current_tap += 1;
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

nih_export_clap!(Del2);
nih_export_vst3!(Del2);
