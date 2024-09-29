use bit_mask_ring_buf::BMRingBuf;
use nih_plug::prelude::*;
use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::ViziaState;
use std::sync::{Arc, Mutex};
use triple_buffer::TripleBuffer;

mod editor;

// This is a shortened version of the gain example with most comments removed, check out
// https://github.com/robbert-vdh/nih-plug/blob/master/plugins/examples/gain/src/lib.rs to get
// started

// max seconds per tap
const MAX_TAP_SECONDS: usize = 10;
const MAX_DEBOUNCE_MILLISECONDS: f32 = 1000.0;
const MAX_NR_TAPS: usize = 16;
const TOTAL_DELAY_SECONDS: usize = MAX_TAP_SECONDS * MAX_NR_TAPS;
const MAX_SAMPLE_RATE: usize = 192000;
const TOTAL_DELAY_SAMPLES: usize = TOTAL_DELAY_SECONDS * MAX_SAMPLE_RATE;
const DSP_BLOCK_SIZE: usize = 128;

pub type WaveformBuffer = Vec<f32>;
pub type WaveformBufferInput = triple_buffer::Input<WaveformBuffer>;
pub type WaveformBufferOutput = triple_buffer::Output<WaveformBuffer>;

struct Del2 {
    params: Arc<Del2Params>,
    delay_buffer: [BMRingBuf<f32>; 2],
    delay_graph_data: DelayGraphData,
    delay_graph_data_input: DelayGraphDataInput,
    delay_graph_data_output: Arc<Mutex<DelayGraphDataOutput>>,
    // waveform_buffer_input: WaveformBufferInput,
    // waveform_buffer_output: Arc<Mutex<WaveformBufferOutput>>,
    sample_rate: f32,
    delay_write_index: usize,
    samples_since_last_event: u32,
    timing_last_event: u32,
    time_out_tap_samples: u32,
    debounce_tap_samples: u32,
    delay_buffer_size: u32,
    counting_state: CountingState,
}

#[derive(Clone)]
pub struct DelayGraphData {
    velocity_array: [f32; MAX_NR_TAPS],
    delay_times_array: [u32; MAX_NR_TAPS],
    current_tap: usize,
}
pub type DelayGraphDataInput = triple_buffer::Input<DelayGraphData>;
pub type DelayGraphDataOutput = triple_buffer::Output<DelayGraphData>;

impl Data for DelayGraphData {
    fn same(&self, other: &Self) -> bool {
        self.velocity_array == other.velocity_array
            && self.delay_times_array == other.delay_times_array
            && self.current_tap == other.current_tap
    }
}

#[derive(Params)]
struct Del2Params {
    #[persist = "editor-state"]
    editor_state: Arc<ViziaState>,
    #[id = "gain"]
    pub gain: FloatParam,
    #[id = "time_out_tap_seconds"]
    pub time_out_tap_seconds: FloatParam,
    #[id = "debounce_tap_milliseconds"]
    pub debounce_tap_milliseconds: FloatParam,
    #[id = "zoom_mode"]
    zoom_mode: EnumParam<ZoomMode>,
}

#[derive(Enum, Debug, PartialEq)]
enum ZoomMode {
    /// Show the length of the taps relative to each other.
    #[id = "relative"]
    #[name = "relative"]
    Relative,
    /// Show the absolute length of the taps
    #[id = "absolute"]
    #[name = "absolute"]
    Absolute,
}

#[derive(PartialEq)]
enum CountingState {
    TimeOut,
    CountingInBuffer,
    CountingAcrossBuffer,
}

impl Default for Del2 {
    fn default() -> Self {
        // let initial_waveform_buffer: Vec<f32> = Vec::new();
        let initial_delay_data: DelayGraphData = DelayGraphData::default();
        let (delay_graph_data_input, delay_graph_data_output) =
            TripleBuffer::new(&initial_delay_data).split();

        Self {
            params: Arc::new(Del2Params::default()),
            delay_buffer: [
                BMRingBuf::<f32>::from_len(TOTAL_DELAY_SAMPLES),
                BMRingBuf::<f32>::from_len(TOTAL_DELAY_SAMPLES),
            ],
            delay_graph_data: initial_delay_data,
            delay_graph_data_input,
            delay_graph_data_output: Arc::new(Mutex::new(delay_graph_data_output)),
            // waveform_buffer_input,
            // waveform_buffer_output: Arc::new(Mutex::new(waveform_buffer_output)),
            sample_rate: 1.0,
            delay_write_index: 0,
            samples_since_last_event: 0,
            timing_last_event: 0,
            time_out_tap_samples: 0,
            debounce_tap_samples: 0,
            delay_buffer_size: 0,
            counting_state: CountingState::TimeOut,
        }
    }
}

impl Default for DelayGraphData {
    fn default() -> Self {
        Self {
            velocity_array: [0.0; MAX_NR_TAPS],
            delay_times_array: [0; MAX_NR_TAPS],
            current_tap: 0,
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
            time_out_tap_seconds: FloatParam::new(
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
            zoom_mode: EnumParam::new("zoom_mode", ZoomMode::Relative)
                .hide()
                .hide_in_generic_ui(),
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

    // fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
    //     editor::create(
    //         self.params.clone(),
    //         self.delay_graph_data.clone(),
    //         self.waveform_buffer_output.clone(),
    //         self.params.editor_state.clone(),
    //     )
    // }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        editor::create(
            editor::Data {
                params: self.params.clone(),
                delay_data: self.delay_graph_data_output.clone(),
                // waveform_buffer_output: self.waveform_buffer_output.clone(),
            },
            self.params.editor_state.clone(),
        )
    }
    // fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
    // editor::create(
    // editor::Data {
    // buffer_output: self.waveform_buffer_output.clone(),
    // recording_progress: self.recording_progress.clone(),
    // command_sender: self.command_sender.clone(),
    // is_info_visible: false,
    // },
    // self.params.editor_state.clone(),
    // )
    // }

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
        // TODO

        // Reset buffers and envelopes here. This can be called from the audio thread and may not
        // allocate. You can remove this function if you do not need it.
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        // TODO: put behind a should_update with a callback?
        self.time_out_tap_samples =
            (self.sample_rate as f64 * self.params.time_out_tap_seconds.value() as f64) as u32;
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

        // let mut delay_graph_data = self.delay_graph_data_output.lock().unwrap();
        // let mut delay_graph_data = self.delay_graph_data_input.clone();

        let buffer_samples = buffer.samples();
        self.no_more_events(buffer_samples as u32);

        for (_, block) in buffer.iter_blocks(DSP_BLOCK_SIZE) {
            // for (_, block) in buffer.iter_blocks(buffer_samples) {
            let block_len = block.samples();
            let mut block_channels = block.into_iter();

            let out_l = block_channels.next().unwrap();
            let out_r = block_channels.next().unwrap();

            self.delay_buffer[0].write_latest(out_l, self.delay_write_index as isize);
            self.delay_buffer[1].write_latest(out_r, self.delay_write_index as isize);

            // TODO: no dry signal yet
            out_l.fill(0.0);
            out_r.fill(0.0);

            // for tap in 0..self.delay_graph_data_input.current_tap {
            for tap in 0..self.delay_graph_data.current_tap {
                let delay_time = self.delay_graph_data.delay_times_array[tap] as isize;
                let read_index = self.delay_write_index as isize - delay_time;
                let velocity_squared = f32::powi(self.delay_graph_data.velocity_array[tap], 2);
                // Temporary buffers to hold the read values for processing
                let mut temp_l = vec![0.0; block_len];
                let mut temp_r = vec![0.0; block_len];
                self.delay_buffer[0].read_into(&mut temp_l, read_index);
                self.delay_buffer[1].read_into(&mut temp_r, read_index);
                // Accumulate the contributions
                for i in 0..block_len {
                    out_l[i] += temp_l[i] * velocity_squared;
                    out_r[i] += temp_r[i] * velocity_squared;
                }
            }

            self.delay_write_index =
                (self.delay_write_index + block_len) % self.delay_buffer_size as usize;
        }
        ProcessStatus::Normal
    }
}

impl Del2 {
    fn note_on(&mut self, timing: u32, velocity: f32) {
        match self.counting_state {
            CountingState::TimeOut => {
                self.delay_graph_data
                    .delay_times_array
                    .iter_mut()
                    .for_each(|x| *x = 0);
                self.delay_graph_data
                    .velocity_array
                    .iter_mut()
                    .for_each(|x| *x = 0.0);
                self.delay_graph_data.current_tap = 0;
                self.timing_last_event = timing;
                self.counting_state = CountingState::CountingInBuffer;
            }
            CountingState::CountingInBuffer => {
                if (timing - self.timing_last_event) > self.debounce_tap_samples {
                    self.samples_since_last_event = timing - self.timing_last_event;
                    self.timing_last_event = timing;
                } else {
                    // println!("debounce in!");
                    return;
                }
            }
            CountingState::CountingAcrossBuffer => {
                if (self.samples_since_last_event + timing) > self.debounce_tap_samples {
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
        if self.samples_since_last_event <= self.time_out_tap_samples {
            if self.delay_graph_data.current_tap < MAX_NR_TAPS
                && self.counting_state != CountingState::TimeOut
                && self.samples_since_last_event > 0
                && velocity > 0.0
            {
                if self.delay_graph_data.current_tap > 0 {
                    self.delay_graph_data.delay_times_array[self.delay_graph_data.current_tap] =
                        self.samples_since_last_event
                            + self.delay_graph_data.delay_times_array
                                [self.delay_graph_data.current_tap - 1];
                } else {
                    self.delay_graph_data.delay_times_array[self.delay_graph_data.current_tap] =
                        self.samples_since_last_event;
                }
                self.delay_graph_data.velocity_array[self.delay_graph_data.current_tap] = velocity;
                // println!("current_tap: {}", self.delay_graph_data.current_tap);
                // println!("times: {:#?}", self.delay_graph_data.delay_times_array);
                // println!("velocities: {:#?}", self.delay_graph_data.velocity_array);
                self.delay_graph_data.current_tap += 1;
            };
        } else {
            self.counting_state = CountingState::TimeOut;
            self.timing_last_event = 0;
            self.samples_since_last_event = 0;
            // println!("time out note on");
        };

        self.delay_graph_data_input
            .write(self.delay_graph_data.clone());
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

        if self.samples_since_last_event > self.time_out_tap_samples {
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
