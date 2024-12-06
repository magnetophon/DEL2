use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};
use std::time::{SystemTime, UNIX_EPOCH};

use nih_plug::prelude::Editor;
use nih_plug_vizia::{
    assets, create_vizia_editor,
    vizia::{prelude::*, vg},
    widgets::{GenericUi, ParamSlider, ParamSliderExt, ParamSliderStyle, ResizeHandle},
    ViziaState, ViziaTheming,
};

use crate::{
    format_time, util, AtomicBoolArray, AtomicByteArray, AtomicF32, AtomicF32Array,
    AtomicUsizeArray, Del2Params, LastPlayedNotes, CLEAR_TAPS, LEARNING, LOCK_TAPS, MUTE_IN,
    MUTE_OUT, NO_GUI_SMOOTHING, NO_LEARNED_NOTE, NUM_TAPS,
};

#[allow(unused_imports)]
use crate::nih_log;

const GUI_SMOOTHING_DECAY_MS: f64 = 242.0;
const MAX_LEARNING_NANOS: u64 = 10_000_000_000; // 10 seconds

/// The minimum decibel value that the meters display
const MIN_TICK: f32 = -60.0;
/// The maximum decibel value that the meters display
const MAX_TICK: f32 = 0.0;

// warning still there. ¯\_(ツ)_/¯
// seems like there shouldn't be a warning in the first place
#[allow(clippy::expl_impl_clone_on_copy)]
#[derive(Lens, Clone)]
pub struct Data {
    pub params: Arc<Del2Params>,
    pub input_meter: Arc<AtomicF32>,
    pub output_meter: Arc<AtomicF32>,
    pub tap_meters: Arc<AtomicF32Array>,
    pub meter_indexes: Arc<AtomicUsizeArray>,
    pub is_learning: Arc<AtomicBool>,
    pub learning_index: Arc<AtomicUsize>,
    pub learned_notes: Arc<AtomicByteArray>,
    // to temp store the note we had during learning
    // so we can keep abusing the notes above 127 to signify the states
    // LEARNING and NO_LEARNED_NOTE but we can still go back
    // to the last properly learned note before we started learning
    pub last_learned_notes: Arc<AtomicByteArray>,
    pub last_played_notes: Arc<LastPlayedNotes>,
    pub enabled_actions: Arc<AtomicBoolArray>,
}

impl Model for Data {}

// Makes sense to also define this here, makes it a bit easier to keep track of
pub fn default_state() -> Arc<ViziaState> {
    ViziaState::new(|| (1428, 714))
}

pub fn create(editor_data: Data, editor_state: Arc<ViziaState>) -> Option<Box<dyn Editor>> {
    create_vizia_editor(editor_state, ViziaTheming::Custom, move |cx, _| {
        assets::register_noto_sans_light(cx);
        assets::register_noto_sans_thin(cx);

        // Add the stylesheet to the app
        cx.add_stylesheet(include_style!("src/style.css"))
            .expect("Failed to load stylesheet");

        editor_data.clone().build(cx);

        HStack::new(cx, |cx| {
            VStack::new(cx, |cx| {
                Label::new(cx, "global").class("group-title");
                HStack::new(cx, |cx| {
                    HStack::new(cx, |cx| {
                        Label::new(cx, "dry/wet").class("slider-label");
                        ParamSlider::new(cx, Data::params, |params| &params.global.dry_wet)
                            .class("widget");
                    })
                    .class("row");
                    HStack::new(cx, |cx| {
                        Label::new(cx, "listen to").class("slider-label");
                        ParamSlider::new(cx, Data::params, |params| &params.global.channel)
                            .class("widget");
                    })
                    .class("row");
                })
                .class("param-group");
                HStack::new(cx, |cx| {
                    HStack::new(cx, |cx| {
                        Label::new(cx, "wet gain").class("slider-label");
                        ParamSlider::new(cx, Data::params, |params| &params.global.wet_gain)
                            .class("widget");
                    })
                    .class("row");
                    HStack::new(cx, |cx| {
                        Label::new(cx, "drive").class("slider-label");
                        ParamSlider::new(cx, Data::params, |params| &params.global.global_drive)
                            .class("widget");
                    })
                    .class("row");
                })
                .class("param-group");

                HStack::new(cx, |cx| {
                    HStack::new(cx, |cx| {
                        Label::new(cx, "attack").class("slider-label");
                        ParamSlider::new(cx, Data::params, |params| &params.global.attack_ms)
                            .class("widget");
                    })
                    .class("row");
                    HStack::new(cx, |cx| {
                        Label::new(cx, "release").class("slider-label");
                        ParamSlider::new(cx, Data::params, |params| &params.global.release_ms)
                            .class("widget");
                    })
                    .class("row");
                })
                .class("param-group");
                HStack::new(cx, |cx| {
                    HStack::new(cx, |cx| {
                        Label::new(cx, "min tap").class("slider-label");
                        ParamSlider::new(cx, Data::params, |params| {
                            &params.global.min_tap_milliseconds
                        })
                        .class("widget");
                    })
                    .class("row");
                    HStack::new(cx, |cx| {
                        Label::new(cx, "max tap").class("slider-label");
                        ParamSlider::new(cx, Data::params, |params| &params.global.max_tap_ms)
                            .set_style(ParamSliderStyle::FromLeft)
                            .class("widget");
                    })
                    .class("row");
                })
                .class("param-group");
                HStack::new(cx, |cx| {
                    HStack::new(cx, |cx| {
                        Label::new(cx, "mutes").class("slider-label");
                        ParamSlider::new(cx, Data::params, |params| &params.global.mute_is_toggle)
                            .set_style(ParamSliderStyle::CurrentStepLabeled { even: true })
                            .class("widget");
                    })
                    .class("row");
                    HStack::new(cx, |cx| {
                        Label::new(cx, "sync").class("slider-label");
                        ParamSlider::new(cx, Data::params, |params| &params.global.sync)
                            .set_style(ParamSliderStyle::CurrentStepLabeled { even: true })
                            .class("widget");
                    })
                    .class("row");
                })
                .class("param-group");
                Label::new(cx, "triggers").class("group-title");
                HStack::new(cx, |cx| {
                    HStack::new(cx, |cx| {
                        Label::new(cx, "mute in").class("action-name");
                        ActionTrigger::new(
                            cx,
                            Data::params,
                            Data::is_learning,
                            Data::learning_index,
                            Data::learned_notes,
                            Data::last_learned_notes,
                            Data::last_played_notes,
                            Data::enabled_actions,
                            MUTE_IN,
                        );
                    })
                    .class("row");
                    HStack::new(cx, |cx| {
                        Label::new(cx, "clear taps").class("action-name");
                        ActionTrigger::new(
                            cx,
                            Data::params,
                            Data::is_learning,
                            Data::learning_index,
                            Data::learned_notes,
                            Data::last_learned_notes,
                            Data::last_played_notes,
                            Data::enabled_actions,
                            CLEAR_TAPS,
                        );
                    })
                    .class("row");
                })
                .class("param-group");

                HStack::new(cx, |cx| {
                    HStack::new(cx, |cx| {
                        Label::new(cx, "mute out").class("action-name");
                        ActionTrigger::new(
                            cx,
                            Data::params,
                            Data::is_learning,
                            Data::learning_index,
                            Data::learned_notes,
                            Data::last_learned_notes,
                            Data::last_played_notes,
                            Data::enabled_actions,
                            MUTE_OUT,
                        );
                    })
                    .class("row");
                    HStack::new(cx, |cx| {
                        Label::new(cx, "lock taps").class("action-name");
                        ActionTrigger::new(
                            cx,
                            Data::params,
                            Data::is_learning,
                            Data::learning_index,
                            Data::learned_notes,
                            Data::last_learned_notes,
                            Data::last_played_notes,
                            Data::enabled_actions,
                            LOCK_TAPS,
                        );
                    })
                    .class("row");
                })
                .class("param-group");

                Label::new(cx, "panning").class("group-title");

                HStack::new(cx, |cx| {
                    HStack::new(cx, |cx| {
                        Label::new(cx, "center").class("slider-label");
                        ParamSlider::new(cx, Data::params, |params| &params.taps.panning_center)
                            .class("widget");
                    })
                    .class("row");
                    HStack::new(cx, |cx| {
                        Label::new(cx, "note>pan").class("slider-label");
                        ParamSlider::new(cx, Data::params, |params| &params.taps.panning_amount)
                            .class("widget");
                    })
                    .class("row");
                })
                .class("param-group");

                Label::new(cx, "filters").class("group-title");

                HStack::new(cx, |cx| {
                    HStack::new(cx, |cx| {
                        Label::new(cx, "vel>cut").class("slider-label");
                        ParamSlider::new(cx, Data::params, |params| {
                            &params.taps.velocity_to_cutoff_amount
                        })
                        .class("widget");
                    })
                    .class("row");
                    HStack::new(cx, |cx| {
                        Label::new(cx, "note>cut").class("slider-label");
                        ParamSlider::new(cx, Data::params, |params| {
                            &params.taps.note_to_cutoff_amount
                        })
                        .class("widget");
                    })
                    .class("row");
                })
                .class("param-group");

                HStack::new(cx, |cx| {
                    make_column(cx, "low velocity", |cx| {
                        let velocity_low_params = Data::params.map(|p| p.taps.velocity_low.clone());
                        GenericUi::new_custom(cx, velocity_low_params, |cx, param_ptr| {
                            HStack::new(cx, |cx| {
                                Label::new(
                                    cx,
                                    unsafe { param_ptr.name() }
                                        .strip_prefix("low velocity ")
                                        .expect("Expected parameter name prefix, this is a bug"),
                                )
                                .class("label");

                                GenericUi::draw_widget(cx, velocity_low_params, param_ptr);
                            })
                            .class("row");
                        });
                    });

                    make_column(cx, "high velocity", |cx| {
                        let velocity_high_params =
                            Data::params.map(|p| p.taps.velocity_high.clone());
                        GenericUi::new_custom(cx, velocity_high_params, |cx, param_ptr| {
                            HStack::new(cx, |cx| {
                                Label::new(
                                    cx,
                                    unsafe { param_ptr.name() }
                                        .strip_prefix("high velocity ")
                                        .expect("Expected parameter name prefix, this is a bug"),
                                )
                                .class("label");

                                GenericUi::draw_widget(cx, velocity_high_params, param_ptr);
                            })
                            .class("row");
                        });
                    });
                })
                .size(Auto)
                .class("param-group");
            })
            .class("parameters");
            VStack::new(cx, |cx| {
                ZStack::new(cx, |cx| {
                    DelayGraph::new(cx, Data::params, Data::tap_meters, Data::input_meter, Data::output_meter, Data::meter_indexes)
                    // .overflow(Overflow::Hidden)
                        ;
                    Label::new(cx, "DEL2").class("plugin-name");
                });
                ResizeHandle::new(cx);
            });
        });
    })
}

///////////////////////////////////////////////////////////////////////////////
//                             DelayGraph                            //
///////////////////////////////////////////////////////////////////////////////

pub struct DelayGraph {
    params: Arc<Del2Params>,
    tap_meters: Arc<AtomicF32Array>,
    input_meter: Arc<AtomicF32>,
    output_meter: Arc<AtomicF32>,
    meter_indexes: Arc<AtomicUsizeArray>,
}

// TODO: add grid to show bars & beats
impl View for DelayGraph {
    // For CSS:
    fn element(&self) -> Option<&'static str> {
        Some("delay-graph")
    }

    fn draw(&self, draw_context: &mut DrawContext, canvas: &mut Canvas) {
        let params = self.params.clone();

        let tap_counter = params.tap_counter.load(Ordering::SeqCst);
        let input_meter = self.input_meter.clone();
        let output_meter = self.output_meter.clone();
        let bounds = draw_context.bounds();
        let border_color: vg::Color = draw_context.border_color().into();
        let outline_width = draw_context.outline_width();
        let border_width = draw_context.border_width();
        let background_color: vg::Color = draw_context.background_color().into();
        let first_note = params.first_note.load(Ordering::SeqCst);

        // Draw components
        Self::draw_background(canvas, bounds, background_color);

        if first_note != NO_LEARNED_NOTE {
            let outline_color: vg::Color = draw_context.outline_color().into();
            let selection_color: vg::Color = draw_context.selection_color().into();
            let font_color: vg::Color = draw_context.font_color().into();
            let tap_meters = self.tap_meters.clone();
            let meter_indexes = self.meter_indexes.clone();
            let current_time = params.current_time.load(Ordering::SeqCst);
            // Compute the time scaling factor
            let target_time_scaling_factor = Self::compute_time_scaling_factor(&params);

            let gui_decay_weight = Self::calculate_gui_decay_weight(&params);

            let available_width = outline_width.mul_add(-0.5, bounds.w - border_width);

            let time_scaling_factor = (Self::gui_smooth(
                target_time_scaling_factor,
                &params.previous_time_scaling_factor,
                gui_decay_weight,
            ) / available_width)
                .recip();

            if tap_counter > 0 {
                Self::draw_delay_times_as_lines(
                    canvas,
                    &params,
                    bounds,
                    border_color,
                    border_width,
                    time_scaling_factor,
                );
                Self::draw_tap_velocities_and_meters(
                    canvas,
                    &params,
                    &tap_meters,
                    &meter_indexes,
                    bounds,
                    outline_color,
                    border_color,
                    outline_width,
                    time_scaling_factor,
                    border_width,
                );
            }
            Self::draw_tap_notes_and_pans(
                canvas,
                &params,
                bounds,
                selection_color,
                outline_width,
                time_scaling_factor,
                gui_decay_weight,
                border_width,
                font_color,
                border_color,
                background_color,
            );

            if current_time > 0.0 {
                Self::draw_time_line(
                    canvas,
                    &params,
                    bounds,
                    selection_color,
                    outline_width,
                    time_scaling_factor,
                    border_width,
                );
            }
        }

        Self::draw_in_out_meters(
            canvas,
            &input_meter,
            &output_meter,
            bounds,
            border_color,
            outline_width,
        );
        Self::draw_bounding_outline(canvas, bounds, border_color, border_width);
    }
}

impl DelayGraph {
    fn new<ParamsL, TapMetersL, InputMeterL, OutputMeterL, MeterIndexL>(
        cx: &mut Context,
        params: ParamsL,
        tap_meters: TapMetersL,
        input_meter: InputMeterL,
        output_meter: OutputMeterL,
        meter_indexes: MeterIndexL,
    ) -> Handle<Self>
    where
        ParamsL: Lens<Target = Arc<Del2Params>>,
        TapMetersL: Lens<Target = Arc<AtomicF32Array>>,
        InputMeterL: Lens<Target = Arc<AtomicF32>>,
        OutputMeterL: Lens<Target = Arc<AtomicF32>>,
        MeterIndexL: Lens<Target = Arc<AtomicUsizeArray>>,
    {
        Self {
            params: params.get(cx),
            tap_meters: tap_meters.get(cx),
            input_meter: input_meter.get(cx),
            output_meter: output_meter.get(cx),
            meter_indexes: meter_indexes.get(cx),
        }
        .build(cx, |cx| {
            Label::new(
                cx,
                params.map(move |params| {
                    let tap_counter = params.tap_counter.load(Ordering::SeqCst);
                    let current_time = params.current_time.load(Ordering::SeqCst);
                    match tap_counter {
                        0 => {
                            if current_time > 0.0 {
                                String::from("no taps")
                            } else {
                                String::new()
                            }
                        }
                        1 => String::from("1 tap"),
                        tap_nr => format!("{tap_nr} taps"),
                    }
                }),
            )
            .class("tap-nr-label");

            Label::new(cx, params.map(Self::create_tap_length_text)).class("tap-length-label");
        })
    }
    fn create_tap_length_text(params: &Arc<Del2Params>) -> String {
        const TOTAL_DIGITS: usize = 3;

        let tap_counter = params.tap_counter.load(Ordering::SeqCst);
        let current_time = params.current_time.load(Ordering::SeqCst);
        let host_tempo = params.host_tempo.load(Ordering::SeqCst);
        let time_sig_numerator = params.time_sig_numerator.load(Ordering::SeqCst);
        let sync = params.global.sync.value();
        let conversion_factor = if sync {
            params.preset_tempo.load(Ordering::SeqCst) / host_tempo
        } else {
            1.0
        };

        // nih_log!("label conversion_factor: {conversion_factor}");
        let seconds = if current_time > 0.0 {
            current_time
        } else if tap_counter > 0 {
            params.delay_times[tap_counter - 1].load(Ordering::SeqCst) * conversion_factor
        } else {
            0.0 // Default value in case both conditions fail
        };

        if current_time == 0.0 && tap_counter == 0 {
            // return String::from("tap a rhythm!");
            return String::new();
        }

        if sync && host_tempo > 0.0 && time_sig_numerator > 0 {
            let seconds_per_beat = 60.0 / host_tempo;
            let seconds_per_measure = seconds_per_beat * time_sig_numerator as f32;
            let full_bars = (seconds / seconds_per_measure).floor() as i32;
            let remaining_seconds = seconds % seconds_per_measure;
            let additional_beats = (remaining_seconds / seconds_per_beat).floor() as i32;

            if full_bars > 0 {
                let bar_str = if current_time > 0.0 || full_bars > 1 {
                    "bars"
                } else {
                    "bar"
                };
                if current_time > 0.0 || additional_beats > 0 {
                    let beat_str = if current_time > 0.0 || additional_beats != 1 {
                        "beats"
                    } else {
                        "beat"
                    };
                    format!("{full_bars} {bar_str}, {additional_beats} {beat_str}")
                } else {
                    format!("{full_bars} {bar_str}")
                }
            } else {
                let beat_str = if current_time > 0.0 || additional_beats != 1 {
                    "beats"
                } else {
                    "beat"
                };
                format!("{additional_beats} {beat_str}")
            }
        } else {
            format_time(seconds, TOTAL_DIGITS)
        }
    }
    fn compute_time_scaling_factor(params: &Arc<Del2Params>) -> f32 {
        // Load atomic values once
        let tap_counter = params.tap_counter.load(Ordering::SeqCst);
        let current_time = params.current_time.load(Ordering::SeqCst);
        let max_tap_time = params.max_tap_time.load(Ordering::SeqCst);

        // Calculate max delay time if necessary
        let max_delay_time = if tap_counter > 0 {
            params.delay_times[tap_counter - 1].load(Ordering::SeqCst)
        } else {
            0.0
        };

        let zoom_tap_samples = if current_time == 0.0 && tap_counter == 1 {
            // one delay tap, put it in the middle
            max_delay_time
        } else if tap_counter == NUM_TAPS || current_time == 0.0 {
            // time out, zoom in but leave a margin
            0.18 * max_delay_time
        } else {
            max_tap_time
        };

        // Return the total delay
        max_delay_time + zoom_tap_samples
    }

    fn calculate_gui_decay_weight(params: &Arc<Del2Params>) -> f32 {
        // Get current system time in nanoseconds since the UNIX epoch
        let now_nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("System time should be after UNIX epoch")
            .as_nanos() as u64;

        // Load the last frame time
        let last_time_nanos = params.last_frame_time.load(Ordering::SeqCst);

        // Calculate the frame duration
        let frame_duration_nanos = if now_nanos < last_time_nanos {
            // Time went backwards; assume 60 FPS (16.67 ms per frame)
            (1_000_000_000 / 60) as u64 // 16,666,666.67 nanoseconds for 60 FPS
        } else {
            now_nanos - last_time_nanos
        };

        // Store the current time value inside the AtomicU64
        params.last_frame_time.store(now_nanos, Ordering::SeqCst);

        // Calculate and return the GUI smoothing decay weight
        let decay_ms_per_frame =
            (GUI_SMOOTHING_DECAY_MS * 1_000_000.0) / frame_duration_nanos as f64;
        0.25f32.powf(decay_ms_per_frame.recip() as f32)
    }

    /// Smoothly updates the value stored within an `f32` based on a target value.
    /// If the current value is `NO_GUI_SMOOTHING`, it initializes with the target value.
    fn gui_smooth(target_value: f32, atomic_value: &AtomicF32, gui_decay_weight: f32) -> f32 {
        // Define the threshold relative to the target value
        let threshold = 0.001 * target_value.abs();
        // Load the current value once
        let current_value = atomic_value.load(Ordering::SeqCst);

        // Early exit if the current value is very close to the target value
        if (target_value - current_value).abs() < threshold {
            atomic_value.store(target_value, Ordering::SeqCst);
            return target_value;
        }

        // Check if initial condition is met and initialize with the target value if so
        if current_value >= NO_GUI_SMOOTHING - 1.0 {
            atomic_value.store(target_value, Ordering::SeqCst);
            return target_value;
        }

        // Compute the smoothed value using frame-rate independent smoothing
        let smoothed_value = current_value.mul_add(
            gui_decay_weight,
            target_value.mul_add(-gui_decay_weight, target_value),
        );

        // Store the change
        atomic_value.store(smoothed_value, Ordering::SeqCst);

        // Return the smoothed value
        smoothed_value
    }

    fn draw_delay_times_as_lines(
        canvas: &mut Canvas,
        params: &Arc<Del2Params>,
        bounds: BoundingBox,
        border_color: vg::Color,
        border_width: f32,
        time_scaling_factor: f32,
    ) {
        let mut path = vg::Path::new();

        let tap_counter = params.tap_counter.load(Ordering::SeqCst);

        for i in 0..tap_counter {
            let delay_time_value = params.delay_times[i].load(Ordering::SeqCst);
            let x_offset = delay_time_value.mul_add(time_scaling_factor, border_width * 0.5);

            let start_y = border_width.mul_add(-0.5, bounds.y + bounds.h);
            let end_y = border_width.mul_add(0.5, bounds.y);

            path.move_to(bounds.x + x_offset, start_y);
            path.line_to(bounds.x + x_offset, end_y);
        }

        canvas.stroke_path(&path, &vg::Paint::color(border_color).with_line_width(0.7));
    }

    fn draw_background(canvas: &mut Canvas, bounds: BoundingBox, color: vg::Color) {
        let mut path = vg::Path::new();
        // Use the original bounds directly
        path.rect(bounds.x, bounds.y, bounds.w, bounds.h);
        path.close();

        let paint = vg::Paint::color(color);
        canvas.fill_path(&path, &paint);
    }

    fn draw_time_line(
        canvas: &mut Canvas,
        params: &Arc<Del2Params>,
        bounds: BoundingBox,
        color: vg::Color,
        line_width: f32,
        time_scaling_factor: f32,
        border_width: f32,
    ) {
        // Load the values once
        let current_time = params.current_time.load(Ordering::SeqCst);

        // Compute offset using mul_add for precision and performance
        let x_offset = current_time.mul_add(time_scaling_factor, border_width * 0.5);

        // Calculate y_move using mul_add
        let y_move = border_width.mul_add(-0.5, bounds.y + bounds.h);

        let mut path = vg::Path::new();
        path.move_to(bounds.x + x_offset, y_move);
        path.line_to(bounds.x + x_offset, bounds.y);
        path.close();

        canvas.stroke_path(&path, &vg::Paint::color(color).with_line_width(line_width));
    }

    fn draw_in_out_meters(
        canvas: &mut Canvas,
        input_meter: &Arc<AtomicF32>,
        output_meter: &Arc<AtomicF32>,
        bounds: BoundingBox,
        meter_color: vg::Color,
        line_width: f32,
    ) {
        // Calculate and draw input meter
        let input_db = util::gain_to_db(input_meter.load(Ordering::Relaxed));
        let input_height = {
            let tick_fraction = (input_db - MIN_TICK) / (MAX_TICK - MIN_TICK);
            (tick_fraction * bounds.h).max(0.0)
        };
        let mut path = vg::Path::new();
        let x_val = line_width.mul_add(0.75, bounds.x);
        path.move_to(x_val, bounds.y + bounds.h - input_height);
        path.line_to(x_val, bounds.y + bounds.h);
        canvas.stroke_path(
            &path,
            &vg::Paint::color(meter_color).with_line_width(line_width * 1.5),
        );

        // Calculate and draw output meter
        let output_db = util::gain_to_db(output_meter.load(Ordering::Relaxed));
        let output_height = {
            let tick_fraction = (output_db - MIN_TICK) / (MAX_TICK - MIN_TICK);
            (tick_fraction * bounds.h).max(0.0)
        };
        path = vg::Path::new();
        let x_val = line_width.mul_add(-0.75, bounds.x + bounds.w);
        path.move_to(x_val, bounds.y + bounds.h - output_height);
        path.line_to(x_val, bounds.y + bounds.h);
        canvas.stroke_path(
            &path,
            &vg::Paint::color(meter_color).with_line_width(line_width * 1.5),
        );
    }
    fn draw_tap_velocities_and_meters(
        canvas: &mut Canvas,
        params: &Arc<Del2Params>,
        tap_meters: &Arc<AtomicF32Array>,
        meter_indexes: &Arc<AtomicUsizeArray>,
        bounds: BoundingBox,
        velocity_color: vg::Color,
        meter_color: vg::Color,
        line_width: f32,
        time_scaling_factor: f32,
        border_width: f32,
    ) {
        let tap_counter = params.tap_counter.load(Ordering::SeqCst);

        for i in 0..tap_counter {
            let delay_time = params.delay_times[i].load(Ordering::SeqCst);
            let x_offset = delay_time.mul_add(time_scaling_factor, border_width * 0.5);

            let velocity_value = params.velocities[i].load(Ordering::SeqCst);
            let velocity_height = velocity_value.mul_add(bounds.h, -border_width);

            // Draw the velocity path
            let mut path = vg::Path::new();
            let x_val = line_width.mul_add(-0.75, bounds.x + x_offset);
            path.move_to(x_val, bounds.y + bounds.h - velocity_height);
            path.line_to(x_val, bounds.y + bounds.h);

            canvas.stroke_path(
                &path,
                &vg::Paint::color(velocity_color).with_line_width(line_width * 1.5),
            );
        }

        for i in 0..tap_counter {
            let delay_time = params.delay_times[i].load(Ordering::SeqCst);
            let x_offset = delay_time.mul_add(time_scaling_factor, border_width * 0.5);

            let meter_index = meter_indexes[i].load(Ordering::Relaxed);
            let meter_db = util::gain_to_db(tap_meters[meter_index].load(Ordering::Relaxed));
            let meter_height = {
                let tick_fraction = (meter_db - MIN_TICK) / (MAX_TICK - MIN_TICK);
                (tick_fraction * bounds.h).max(0.0)
            };

            // Draw the meter path
            let mut path = vg::Path::new();
            let x_val = line_width.mul_add(0.75, bounds.x + x_offset);
            path.move_to(x_val, bounds.y + bounds.h - meter_height);
            path.line_to(x_val, bounds.y + bounds.h);

            canvas.stroke_path(
                &path,
                &vg::Paint::color(meter_color).with_line_width(line_width * 1.5),
            );
        }
    }

    fn draw_tap_notes_and_pans(
        canvas: &mut Canvas,
        params: &Arc<Del2Params>,
        bounds: BoundingBox,
        color: vg::Color,
        line_width: f32,
        time_scaling_factor: f32,
        gui_decay_weight: f32,
        border_width: f32,
        font_color: vg::Color,
        border_color: vg::Color,
        background_color: vg::Color,
    ) {
        let first_note = params.first_note.load(Ordering::SeqCst);

        let mut note_path = vg::Path::new();
        let mut center_path = vg::Path::new();
        let mut pan_background_path = vg::Path::new();
        let mut pan_foreground_path = vg::Path::new();

        let tap_counter = params.tap_counter.load(Ordering::SeqCst);
        let panning_center_value = params.taps.panning_center.value();
        let panning_amount = params.taps.panning_amount.value();

        let panning_center = if panning_center_value < 0 {
            first_note
        } else {
            panning_center_value as u8
        };

        let (min_note_value, max_note_value) = {
            // Initialize min and max with first_note and panning_center
            let mut min = first_note.min(panning_center);
            let mut max = first_note.max(panning_center);

            // Iterate through notes to find min and max
            for i in 0..tap_counter {
                let loaded_note = params.notes[i].load(Ordering::SeqCst);
                if loaded_note < min {
                    min = loaded_note;
                } else if loaded_note > max {
                    max = loaded_note;
                }
            }

            let zoom_min = if first_note < min + 12 {
                first_note.saturating_sub(12)
            } else {
                min
            };
            let zoom_max = if max < first_note + 12 {
                first_note.saturating_add(12)
            } else {
                max
            };

            let range_too_large = (i16::from(min) - i16::from(max)).abs() > 24;
            let (final_min, final_max) = if range_too_large {
                (min, max)
            } else {
                (zoom_min, zoom_max)
            };

            (f32::from(final_min), f32::from(final_max))
        };

        let note_size = line_width * 2.0; // Width and height of a note
        let margin = 10.0 * line_width;
        let available_height = (-(margin + note_size + border_width)).mul_add(2.0, bounds.h);

        let get_normalized_value = |value: u8, min: f32, max: f32| -> f32 {
            if (max - min).abs() < 0.5 {
                f32::from(value) / 127.0
            } else {
                (f32::from(value) - min) / (max - min)
            }
        };
        // Draw half a note for panning center
        let normalized_panning_center =
            get_normalized_value(panning_center, min_note_value, max_note_value);
        let panning_center_height = Self::gui_smooth(
            1.0 - normalized_panning_center,
            &params.previous_panning_center_height,
            gui_decay_weight,
        )
        .mul_add(available_height, margin + note_size);

        let note_half_size = line_width;

        let panning_center_x = bounds.x;
        let panning_center_y = bounds.y + panning_center_height;

        center_path.move_to(panning_center_x, panning_center_y + note_half_size);
        center_path.line_to(panning_center_x + note_half_size, panning_center_y);
        center_path.line_to(panning_center_x, panning_center_y - note_half_size);
        center_path.close();

        // Draw half a note for the first note at time 0
        let normalized_first_note =
            get_normalized_value(first_note, min_note_value, max_note_value);
        let note_height = Self::gui_smooth(
            1.0 - normalized_first_note,
            &params.previous_first_note_height,
            gui_decay_weight,
        )
        .mul_add(available_height, margin + note_size);

        let first_note_center_x = bounds.x;
        let first_note_center_y = bounds.y + note_height;
        let note_half_size = line_width;

        note_path.move_to(first_note_center_x, first_note_center_y + note_half_size);
        note_path.line_to(first_note_center_x + note_half_size, first_note_center_y);
        note_path.line_to(first_note_center_x, first_note_center_y - note_half_size);
        note_path.close();

        for i in 0..tap_counter {
            let delay_time = params.delay_times[i].load(Ordering::SeqCst);
            let x_offset = delay_time.mul_add(time_scaling_factor, border_width * 0.5);

            let note_value = params.notes[i].load(Ordering::SeqCst);
            let normalized_note = get_normalized_value(note_value, min_note_value, max_note_value);
            let note_height = Self::gui_smooth(
                1.0 - normalized_note,
                &params.previous_note_heights[i],
                gui_decay_weight,
            )
            .mul_add(available_height, margin + note_size);

            let note_center_x = bounds.x + x_offset;
            let note_center_y = bounds.y + note_height;
            let note_half_size = line_width;

            note_path.move_to(note_center_x + note_half_size, note_center_y);
            note_path.line_to(note_center_x, note_center_y + note_half_size);
            note_path.line_to(note_center_x - note_half_size, note_center_y);
            note_path.line_to(note_center_x, note_center_y - note_half_size);
            note_path.close();

            let pan_value = ((f32::from(note_value) - f32::from(panning_center)) * panning_amount)
                .clamp(-1.0, 1.0);

            let line_length = if pan_value.abs() > 1.0 / 50.0 {
                50.0
            } else {
                0.0
            };

            let target_pan_foreground_length = pan_value * line_length;
            let target_pan_background_length = if pan_value < 0.0 {
                -line_length
            } else {
                line_length
            };

            let pan_foreground_length = Self::gui_smooth(
                target_pan_foreground_length,
                &params.previous_pan_foreground_lengths[i],
                gui_decay_weight,
            );
            let pan_background_length = Self::gui_smooth(
                target_pan_background_length,
                &params.previous_pan_background_lengths[i],
                gui_decay_weight,
            );

            pan_background_path.move_to(note_center_x, note_center_y);
            pan_background_path.line_to(
                (note_center_x + pan_background_length)
                    .clamp(bounds.x + 1.0, bounds.x + bounds.w - 1.0),
                note_center_y,
            );

            pan_foreground_path.move_to(note_center_x, note_center_y);
            pan_foreground_path.line_to(
                (note_center_x + pan_foreground_length)
                    .clamp(bounds.x + 1.0, bounds.x + bounds.w - 1.0),
                note_center_y,
            );
        }

        canvas.stroke_path(
            &pan_background_path,
            &vg::Paint::color(border_color).with_line_width(line_width),
        );
        canvas.stroke_path(
            &center_path,
            &vg::Paint::color(font_color).with_line_width(line_width),
        );
        canvas.stroke_path(
            &pan_foreground_path,
            &vg::Paint::color(font_color).with_line_width(line_width),
        );
        canvas.stroke_path(
            &note_path,
            &vg::Paint::color(color).with_line_width(line_width),
        );

        // Fix cover line drawing as needed
        let mut cover_line_path = vg::Path::new();
        let cover_x = border_width.mul_add(0.5, line_width.mul_add(-0.5, bounds.x));
        cover_line_path.move_to(cover_x, bounds.y);
        cover_line_path.line_to(cover_x, bounds.y + bounds.h);

        canvas.stroke_path(
            &cover_line_path,
            &vg::Paint::color(background_color).with_line_width(line_width),
        );
    }

    fn draw_bounding_outline(
        canvas: &mut Canvas,
        bounds: BoundingBox,
        color: vg::Color,
        border_width: f32,
    ) {
        let half_border = border_width / 2.0;

        let mut path = vg::Path::new();
        path.rect(
            bounds.x + half_border,
            bounds.y + half_border,
            bounds.w - border_width,
            bounds.h - border_width,
        );
        path.close();

        canvas.stroke_path(
            &path,
            &vg::Paint::color(color).with_line_width(border_width),
        );
    }
}

fn make_column(cx: &mut Context, title: &str, contents: impl FnOnce(&mut Context)) {
    VStack::new(cx, |cx| {
        Label::new(cx, title).class("column-title");

        contents(cx);
    })
    .class("column");
}

///////////////////////////////////////////////////////////////////////////////
//                               ActionTrigger                               //
///////////////////////////////////////////////////////////////////////////////

pub struct ActionTrigger {
    params: Arc<Del2Params>,
    is_learning: Arc<AtomicBool>,
    learning_index: Arc<AtomicUsize>,
    learned_notes: Arc<AtomicByteArray>,
    last_learned_notes: Arc<AtomicByteArray>,
    last_played_notes: Arc<LastPlayedNotes>,
    enabled_actions: Arc<AtomicBoolArray>,
    own_index: usize,
}
impl ActionTrigger {
    pub fn new<
        ParamsL,
        IsLearningL,
        LearningIndexL,
        LearnedNotesL,
        LastLearnedNotesL,
        LastPlayedNotesL,
        EnabledActionsL,
    >(
        cx: &mut Context,
        params: ParamsL,
        is_learning: IsLearningL,
        learning_index: LearningIndexL,
        learned_notes: LearnedNotesL,
        last_learned_notes: LastLearnedNotesL,
        last_played_notes: LastPlayedNotesL,
        enabled_actions: EnabledActionsL,
        own_index: usize,
    ) -> Handle<Self>
    where
        ParamsL: Lens<Target = Arc<Del2Params>>,
        IsLearningL: Lens<Target = Arc<AtomicBool>>,
        LearningIndexL: Lens<Target = Arc<AtomicUsize>>,
        LearnedNotesL: Lens<Target = Arc<AtomicByteArray>>,
        LastLearnedNotesL: Lens<Target = Arc<AtomicByteArray>>,
        LastPlayedNotesL: Lens<Target = Arc<LastPlayedNotes>>,
        EnabledActionsL: Lens<Target = Arc<AtomicBoolArray>>,
    {
        Self {
            params: params.get(cx),
            is_learning: is_learning.get(cx),
            learning_index: learning_index.get(cx),
            learned_notes: learned_notes.get(cx),
            last_learned_notes: last_learned_notes.get(cx),
            last_played_notes: last_played_notes.get(cx),
            enabled_actions: enabled_actions.get(cx),
            own_index,
        }
        .build(cx, move |cx| {
            Label::new(
                cx,
                learned_notes.map(move |notes| {
                    let note_nr = notes.load(own_index);
                    Self::get_note_name(note_nr)
                }),
            )
            .color(learned_notes.map(move |notes| {
                let note_nr = notes.load(own_index);
                Self::get_note_color(note_nr)
            }))
            .class("action-label");
        })
    }

    pub fn start_learning(&self) {
        let index = self.learning_index.load(Ordering::SeqCst);
        self.learned_notes
            .store(index, self.last_learned_notes.load(index));
        self.is_learning.store(true, Ordering::SeqCst);
        let index = self.own_index;
        self.last_learned_notes
            .store(index, self.learned_notes.load(index));
        self.learned_notes.store(index, LEARNING);
        self.learning_index.store(index, Ordering::SeqCst);

        // Get current system time in nanoseconds since the UNIX epoch
        let now_nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("System time should be after UNIX epoch")
            .as_nanos() as u64;
        self.params
            .learning_start_time
            .store(now_nanos, Ordering::SeqCst);
    }
    pub fn stop_learning(&self) {
        self.is_learning.store(false, Ordering::SeqCst);
        self.learned_notes
            .store(self.own_index, self.last_learned_notes.load(self.own_index));
        self.params.learning_start_time.store(0, Ordering::SeqCst);
    }

    // Checks if learning is active for this trigger
    pub fn is_learning(&self) -> bool {
        self.is_learning.load(Ordering::SeqCst)
            && self.learned_notes.load(self.own_index) == LEARNING
    }

    pub fn is_playing(&self) -> bool {
        self.last_played_notes
            .is_playing(self.learned_notes.load(self.own_index))
    }
    pub fn is_enabled(&self) -> bool {
        self.enabled_actions.load(self.own_index)
    }

    fn get_note_name(note_nr: u8) -> String {
        if note_nr == LEARNING {
            "learning".to_string()
        } else if note_nr == NO_LEARNED_NOTE {
            "click to learn".to_string()
        } else {
            let note_name = util::NOTES[(note_nr % 12) as usize];
            let octave = (note_nr / 12) as i8 - 1; // Calculate the octave, ensuring i8 for potential negative values
            format!("{note_name}{octave}") // Format the note correctly with the octave
        }
    }

    const fn get_note_color(note_nr: u8) -> Color {
        if note_nr == LEARNING {
            // blue
            Color::rgb(88, 121, 175)
            // Color::rgb(13, 25, 49) // dark grey
        } else {
            Color::rgb(224, 206, 145) // yellow
        }
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                       for drawing
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // clippy's suggestion doesn't work, cause we need the early exit
    #[allow(clippy::match_same_arms)]
    fn draw_background(
        &self,
        canvas: &mut Canvas,
        bounds: BoundingBox,
        background_color: vg::Color,
        outline_color: vg::Color,
        selection_color: vg::Color,
        border_color: vg::Color,
        border_width: f32,
    ) {
        // Adjust bounds for borders
        let x = border_width.mul_add(0.5, bounds.x);
        let y = border_width.mul_add(0.5, bounds.y);
        let w = bounds.w - border_width;
        let h = bounds.h - border_width;

        // Drawing the background rectangle
        let mut path = vg::Path::new();
        path.rect(x, y, w, h);
        path.close();

        if self.is_learning() {
            // Load the learning time
            let learning_start_time_nanos = self.params.learning_start_time.load(Ordering::SeqCst);
            // Get current system time in nanoseconds since the UNIX epoch
            let now_nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("System time should be after UNIX epoch")
                .as_nanos() as u64;

            // Calculate the learning time duration
            let learning_duration_nanos = now_nanos.saturating_sub(learning_start_time_nanos);
            if learning_duration_nanos > MAX_LEARNING_NANOS {
                self.stop_learning();
            }
        }

        // Determine the paint color based on the state
        let paint = match (
            self.is_learning(),
            self.params.global.mute_is_toggle.value(),
            self.is_enabled(),
            self.is_playing(),
            self.own_index == CLEAR_TAPS,
        ) {
            (true, _, _, _, _) => vg::Paint::color(border_color),
            (_, _, _, true, true) => vg::Paint::color(outline_color),
            (_, true, true, _, _) => vg::Paint::color(outline_color),
            (_, true, false, _, _) => vg::Paint::color(background_color),
            (_, _, true, _, _) => vg::Paint::color(outline_color),
            (_, _, _, true, _) => vg::Paint::color(selection_color),
            _ => vg::Paint::color(background_color), // Default: paint with background color
        };

        canvas.fill_path(&path, &paint);

        // Drawing the border around the rectangle
        let mut path = vg::Path::new();
        path.rect(x, y, w, h);
        path.close();

        canvas.stroke_path(
            &path,
            &vg::Paint::color(border_color).with_line_width(border_width),
        );
    }
}

impl View for ActionTrigger {
    // For CSS:
    fn element(&self) -> Option<&'static str> {
        Some("action-trigger")
    }

    fn event(&mut self, _cx: &mut EventContext, event: &mut Event) {
        event.map(|window_event, meta| match window_event {
            // We don't need special double and triple click handling
            WindowEvent::MouseDown(MouseButton::Left)
            | WindowEvent::MouseDoubleClick(MouseButton::Left)
            | WindowEvent::MouseTripleClick(MouseButton::Left) => {
                if self.is_learning() {
                    self.stop_learning();
                } else {
                    self.start_learning();
                }
                meta.consume();
            }
            _ => {}
        });
    }
    fn draw(&self, draw_context: &mut DrawContext, canvas: &mut Canvas) {
        let bounds = draw_context.bounds();
        let background_color: vg::Color = draw_context.background_color().into();
        let border_color: vg::Color = draw_context.border_color().into();
        let outline_color: vg::Color = draw_context.outline_color().into();
        let selection_color: vg::Color = draw_context.selection_color().into();
        let border_width = draw_context.border_width();
        // let outline_width = draw_context.outline_width();

        self.draw_background(
            canvas,
            bounds,
            background_color,
            border_color,
            outline_color,
            selection_color,
            border_width,
        );
    }
}
