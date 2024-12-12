use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};
use std::time::{SystemTime, UNIX_EPOCH};

use nih_plug::prelude::Editor;
use vizia_plug::{
    create_vizia_editor,
    vizia::{prelude::*, vg},
    widgets::{ParamSlider, ParamSliderExt, ParamSliderStyle},
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
const GLOW_SIZE: vg::Point3 = vg::Point3::new(0.0, 0.0, 20.0);
const LIGHT_POS: vg::Point3 = vg::Point3::new(0.0, 0.0, 0.0); // light position only matters for the part I'm not drawing
const SPOT_COLOR: Color = Color::rgba(0, 0, 0, 0); // don't draw this
const PINK_GLOW_ALPHA: u8 = 57;
const YELLOW_GLOW_ALPHA: u8 = 50;
const BLUE_GLOW_ALPHA: u8 = 71;

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
    pub show_full_parameters: bool,
}
enum AppEvent {
    ToggleShowView,
}

impl Model for Data {
    fn event(&mut self, _cx: &mut EventContext, event: &mut Event) {
        event.map(|app_event, _| match app_event {
            AppEvent::ToggleShowView => self.show_full_parameters ^= true,
        });
    }
}

// Makes sense to also define this here, makes it a bit easier to keep track of
pub fn default_state() -> Arc<ViziaState> {
    ViziaState::new(|| (1301, 804))
}
pub fn create(editor_data: Data, editor_state: Arc<ViziaState>) -> Option<Box<dyn Editor>> {
    create_vizia_editor(editor_state, ViziaTheming::Custom, move |cx, _| {
        // Add the stylesheet to the app
        cx.add_stylesheet(include_style!("src/style.css"))
            .expect("Failed to load stylesheet");

        editor_data.clone().build(cx);

        VStack::new(cx, |cx| {
            ZStack::new(cx, |cx| {
                DelayGraph::new(cx, Data::params, Data::tap_meters, Data::input_meter, Data::output_meter, Data::meter_indexes)
                // .overflow(Overflow::Hidden)
                    ;
                Label::new(cx, "DEL2").class("plugin-name");
            });
            Binding::new(cx, Data::show_full_parameters, |cx, show| {
                if show.get(cx) {
                    full_parameters(cx);
                } else {
                    minimal_parameters(cx);
                }
            });
        });
    })
}
fn full_parameters(cx: &mut Context) {
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
            ZStack::new(cx, |cx| {
                CollapseButton::new(cx).class("show-full-parameters");
                Label::new(cx, "triggers").class("mid-group-title");
            });
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
        })
        .class("parameters-left");

        VStack::new(cx, |cx| {
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

            Label::new(cx, "filters").class("mid-group-title");

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
                Label::new(cx, "low velocity").class("column-title");
                Label::new(cx, "high velocity").class("column-title");
            })
            .class("column-title-group");
            HStack::new(cx, |cx| {
                HStack::new(cx, |cx| {
                    Label::new(cx, "cutoff").class("slider-label");
                    ParamSlider::new(cx, Data::params, |params| &params.taps.velocity_low.cutoff)
                        .class("widget");
                })
                .class("row");
                HStack::new(cx, |cx| {
                    Label::new(cx, "cutoff").class("slider-label");
                    ParamSlider::new(cx, Data::params, |params| &params.taps.velocity_high.cutoff)
                        .class("widget");
                })
                .class("row");
            })
            .class("param-group");

            HStack::new(cx, |cx| {
                HStack::new(cx, |cx| {
                    Label::new(cx, "res").class("slider-label");
                    ParamSlider::new(cx, Data::params, |params| &params.taps.velocity_low.res)
                        .class("widget");
                })
                .class("row");
                HStack::new(cx, |cx| {
                    Label::new(cx, "res").class("slider-label");
                    ParamSlider::new(cx, Data::params, |params| &params.taps.velocity_high.res)
                        .class("widget");
                })
                .class("row");
            })
            .class("param-group");

            HStack::new(cx, |cx| {
                HStack::new(cx, |cx| {
                    Label::new(cx, "drive").class("slider-label");
                    ParamSlider::new(cx, Data::params, |params| &params.taps.velocity_low.drive)
                        .class("widget");
                })
                .class("row");
                HStack::new(cx, |cx| {
                    Label::new(cx, "drive").class("slider-label");
                    ParamSlider::new(cx, Data::params, |params| &params.taps.velocity_high.drive)
                        .class("widget");
                })
                .class("row");
            })
            .class("param-group");

            HStack::new(cx, |cx| {
                HStack::new(cx, |cx| {
                    Label::new(cx, "mode").class("slider-label");
                    ParamSlider::new(cx, Data::params, |params| &params.taps.velocity_low.mode)
                        .class("widget");
                })
                .class("row");
                HStack::new(cx, |cx| {
                    Label::new(cx, "mode").class("slider-label");
                    ParamSlider::new(cx, Data::params, |params| &params.taps.velocity_high.mode)
                        .class("widget");
                })
                .class("row");
            })
            .class("param-group");
        })
        .class("parameters-right");
    })
    .class("parameters-all");
}

fn minimal_parameters(cx: &mut Context) {
    VStack::new(cx, |cx| {
        ZStack::new(cx, |cx| {
            HStack::new(cx, |cx| {
                Label::new(cx, "triggers").class("mid-group-title");
                Label::new(cx, "low velocity").class("low-velocity-minimal-title");
                Label::new(cx, "high velocity").class("column-title");
            })
            .class("column-title-group-minimal");
            CollapseButton::new(cx).class("show-full-parameters");
        });
        HStack::new(cx, |cx| {
            VStack::new(cx, |cx| {
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
            })
            .class("parameters-left");

            VStack::new(cx, |cx| {
                // Label::new(cx, "filters").class("mid-group-title");

                HStack::new(cx, |cx| {
                    HStack::new(cx, |cx| {
                        Label::new(cx, "cutoff").class("slider-label");
                        ParamSlider::new(cx, Data::params, |params| {
                            &params.taps.velocity_low.cutoff
                        })
                        .class("widget");
                    })
                    .class("row");
                    HStack::new(cx, |cx| {
                        Label::new(cx, "cutoff").class("slider-label");
                        ParamSlider::new(cx, Data::params, |params| {
                            &params.taps.velocity_high.cutoff
                        })
                        .class("widget");
                    })
                    .class("row");
                })
                .class("param-group");

                HStack::new(cx, |cx| {
                    HStack::new(cx, |cx| {
                        Label::new(cx, "drive").class("slider-label");
                        ParamSlider::new(cx, Data::params, |params| {
                            &params.taps.velocity_low.drive
                        })
                        .class("widget");
                    })
                    .class("row");
                    HStack::new(cx, |cx| {
                        Label::new(cx, "drive").class("slider-label");
                        ParamSlider::new(cx, Data::params, |params| {
                            &params.taps.velocity_high.drive
                        })
                        .class("widget");
                    })
                    .class("row");
                })
                .class("param-group");
            })
            .class("parameters-right");
        })
        .class("parameters-minimal");
    })
    .class("parameters-labels-minimal");
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

    fn draw(&self, draw_context: &mut DrawContext, canvas: &Canvas) {
        draw_context.needs_redraw();

        let params = self.params.clone();
        let tap_counter = params.tap_counter.load(Ordering::SeqCst);
        let tap_meters = self.tap_meters.clone();
        let meter_indexes = self.meter_indexes.clone();

        let input_meter = &self.input_meter;
        let output_meter = &self.output_meter;

        let bounds = draw_context.bounds();
        let border_color: vg::Color = draw_context.border_color().into();
        let outline_width = draw_context.outline_width();
        let border_width = draw_context.border_width();
        // let background_color: vg::Color = draw_context.background_color().into();
        let outline_color: vg::Color = draw_context.outline_color().into();
        let selection_color: vg::Color = draw_context.selection_color().into();
        let font_color: vg::Color = draw_context.font_color().into();

        let first_note = params.first_note.load(Ordering::SeqCst);
        let current_time = params.current_time.load(Ordering::SeqCst);

        let target_time_scaling_factor = Self::compute_time_scaling_factor(&params);
        let gui_decay_weight = Self::calculate_gui_decay_weight(&params);
        let available_width = bounds.w - outline_width - border_width * 2.0;
        let time_scaling_factor = (Self::gui_smooth(
            target_time_scaling_factor,
            &params.previous_time_scaling_factor,
            gui_decay_weight,
        ) / available_width)
            .recip();

        // Start drawing

        Self::draw_bounding_outline(canvas, bounds, border_color, border_width);
        // Self::draw_bounding_outline(canvas, bounds, font_color, border_width);

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

        Self::draw_in_out_meters(
            canvas,
            input_meter,
            output_meter,
            bounds,
            border_color,
            outline_width,
            border_width,
        );

        if first_note != NO_LEARNED_NOTE {
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
            0.11 * max_delay_time
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
        canvas: &Canvas,
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
            let x_offset = delay_time_value.mul_add(time_scaling_factor, border_width);

            let start_y = bounds.y + border_width;
            let end_y = bounds.y + bounds.h - border_width;

            path.move_to((bounds.x + x_offset, start_y));
            path.line_to((bounds.x + x_offset, end_y));
        }

        let mut paint = vg::Paint::default();
        paint.set_color(border_color);
        paint.set_anti_alias(true);
        paint.set_stroke_width(0.7);
        paint.set_style(vg::PaintStyle::Stroke);

        canvas.draw_path(&path, &paint);
    }

    fn _draw_background(canvas: &Canvas, bounds: BoundingBox, color: vg::Color) {
        let mut path = vg::Path::new();
        // Use the original bounds directly
        path.add_rect(
            vg::Rect::from_xywh(bounds.x, bounds.y, bounds.w, bounds.h),
            None,
        );

        let mut paint = vg::Paint::default();
        paint.set_color(color);
        paint.set_anti_alias(true);
        paint.set_style(vg::PaintStyle::Fill);

        canvas.draw_path(&path, &paint);
    }
    fn draw_time_line(
        canvas: &Canvas,
        params: &Arc<Del2Params>,
        bounds: BoundingBox,
        color: vg::Color,
        line_width: f32,
        time_scaling_factor: f32,
        border_width: f32,
    ) {
        let current_time = params.current_time.load(Ordering::SeqCst);
        let x_offset = current_time.mul_add(time_scaling_factor, border_width);

        let mut path = vg::Path::new();
        path.add_rect(
            vg::Rect::from_xywh(
                bounds.x + x_offset,
                bounds.y + border_width,
                line_width,
                bounds.h - border_width * 2.0,
            ),
            None,
        );

        let (r, g, b) = (color.r(), color.g(), color.b());
        let ambient_color = Color::rgba(r, g, b, PINK_GLOW_ALPHA);

        canvas.draw_shadow(
            &path,
            GLOW_SIZE,
            LIGHT_POS,
            0.0,
            ambient_color,
            SPOT_COLOR,
            None,
        );

        let mut paint = vg::Paint::default();
        paint.set_color(color);
        paint.set_anti_alias(true);
        canvas.draw_path(&path, &paint);
    }

    fn draw_in_out_meters(
        canvas: &Canvas,
        input_meter: &Arc<AtomicF32>,
        output_meter: &Arc<AtomicF32>,
        bounds: BoundingBox,
        meter_color: vg::Color,
        line_width: f32,
        border_width: f32,
    ) {
        let available_height = bounds.h - border_width * 2.0;
        // Calculate and draw input meter
        let input_db = util::gain_to_db(input_meter.load(Ordering::Relaxed));
        let input_height = {
            let tick_fraction = (input_db - MIN_TICK) / (MAX_TICK - MIN_TICK);
            (tick_fraction * available_height).max(0.0)
        };
        let mut path = vg::Path::new();
        // let x_val = bounds.x + border_width + 0.5 * line_width;
        let x_val = line_width.mul_add(0.5, bounds.x + border_width);
        path.move_to((x_val, bounds.y + bounds.h - border_width));
        path.line_to((x_val, bounds.y + bounds.h - border_width - input_height));

        let mut paint = vg::Paint::default();
        paint.set_color(meter_color);
        paint.set_anti_alias(true);
        paint.set_stroke_width(line_width);
        paint.set_style(vg::PaintStyle::Stroke);

        canvas.draw_path(&path, &paint);

        // Calculate and draw output meter
        let output_db = util::gain_to_db(output_meter.load(Ordering::Relaxed));
        let output_height = {
            let tick_fraction = (output_db - MIN_TICK) / (MAX_TICK - MIN_TICK);
            (tick_fraction * available_height).max(0.0)
        };
        path = vg::Path::new();
        let x_val = bounds.x + bounds.w - border_width - 0.5 * line_width;
        path.move_to((x_val, bounds.y + bounds.h - border_width));
        path.line_to((x_val, bounds.y + bounds.h - border_width - output_height));
        // let x_val = line_width.mul_add(-0.5, bounds.x + bounds.w);
        // path.move_to((x_val, bounds.y + bounds.h - output_height));
        // path.line_to((x_val, bounds.y + bounds.h));

        let mut paint = vg::Paint::default();
        paint.set_color(meter_color);
        paint.set_anti_alias(true);
        paint.set_stroke_width(line_width);
        paint.set_style(vg::PaintStyle::Stroke);

        canvas.draw_path(&path, &paint);
    }
    fn draw_tap_velocities_and_meters(
        canvas: &Canvas,
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
        let available_height = bounds.h - border_width * 2.0;

        let (r, g, b) = (velocity_color.r(), velocity_color.g(), velocity_color.b());
        let ambient_color = Color::rgba(r, g, b, BLUE_GLOW_ALPHA);

        // let flags =
        // vg::utils::shadow_utils::ShadowFlags::TRANSPARENT_OCCLUDER
        // |
        // vg::utils::shadow_utils::ShadowFlags::GEOMETRIC_ONLY
        // |
        // vg::utils::shadow_utils::ShadowFlags::DIRECTIONAL_LIGHT
        // |
        // vg::utils::shadow_utils::ShadowFlags::CONCAVE_BLUR_ONLY
        // ;

        let radius = line_width * 0.25;
        for i in 0..tap_counter {
            let delay_time = params.delay_times[i].load(Ordering::SeqCst);
            let x_offset = delay_time.mul_add(time_scaling_factor, border_width);

            let velocity_value = params.velocities[i].load(Ordering::SeqCst);
            let velocity_height = -1.0 * velocity_value * available_height;

            let mut velocity_path = vg::Path::new();
            velocity_path.add_round_rect(
                vg::Rect::from_xywh(
                    bounds.x + x_offset - line_width,
                    bounds.y + bounds.h - border_width,
                    line_width,
                    velocity_height,
                ),
                (radius, radius),
                None,
            );
            canvas.draw_shadow(
                &velocity_path,
                GLOW_SIZE,
                LIGHT_POS,
                1.0,
                ambient_color,
                SPOT_COLOR,
                // Some(flags),
                None,
            );
            let mut paint = vg::Paint::default();
            paint.set_color(velocity_color);
            paint.set_anti_alias(true);
            canvas.draw_path(&velocity_path, &paint);
        }

        let mut path = vg::Path::new();
        for i in 0..tap_counter {
            let delay_time = params.delay_times[i].load(Ordering::SeqCst);
            let x_offset = delay_time.mul_add(time_scaling_factor, border_width);

            let meter_index = meter_indexes[i].load(Ordering::Relaxed);
            let meter_db = util::gain_to_db(tap_meters[meter_index].load(Ordering::Relaxed));
            let meter_height = {
                let tick_fraction = (meter_db - MIN_TICK) / (MAX_TICK - MIN_TICK);
                (-1.0 * tick_fraction * available_height).min(0.0)
            };

            path.add_rect(
                vg::Rect::from_xywh(
                    bounds.x + x_offset,
                    bounds.y + bounds.h - border_width,
                    line_width,
                    meter_height,
                ),
                None,
            );
        }
        let mut paint = vg::Paint::default();
        paint.set_color(meter_color);
        paint.set_anti_alias(true);
        canvas.draw_path(&path, &paint);
    }

    fn draw_tap_notes_and_pans(
        canvas: &Canvas,
        params: &Arc<Del2Params>,
        bounds: BoundingBox,
        color: vg::Color,
        line_width: f32,
        time_scaling_factor: f32,
        gui_decay_weight: f32,
        border_width: f32,
        font_color: vg::Color,
        border_color: vg::Color,
    ) {
        let first_note = params.first_note.load(Ordering::SeqCst);

        let mut center_path = vg::Path::new();

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

        // let note_size = line_width * 2.8284; // Width and height of a note
        let note_size = line_width * 2.5; // Width and height of a note
        let margin = 57.0;
        let available_height = (-(margin + border_width)).mul_add(2.0, bounds.h);

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
        .mul_add(available_height, margin);

        let note_half_size = note_size * 0.5;

        let panning_center_x = bounds.x + border_width;
        let panning_center_y = bounds.y + panning_center_height;

        center_path.move_to((panning_center_x, panning_center_y + note_half_size));
        center_path.line_to((panning_center_x + note_half_size, panning_center_y));
        center_path.line_to((panning_center_x, panning_center_y - note_half_size));
        center_path.close();

        // Draw half a note for the first note at time 0
        let normalized_first_note =
            get_normalized_value(first_note, min_note_value, max_note_value);
        let first_note_height = Self::gui_smooth(
            1.0 - normalized_first_note,
            &params.previous_first_note_height,
            gui_decay_weight,
        )
        .mul_add(available_height, margin);

        let note_half_size = note_size * 0.5;

        let first_note_x = bounds.x + border_width;
        let first_note_y = bounds.y + first_note_height;

        let mut note_path = vg::Path::new();
        note_path.move_to((first_note_x, first_note_y + note_half_size));
        note_path.line_to((first_note_x + note_half_size, first_note_y));
        note_path.line_to((first_note_x, first_note_y - note_half_size));
        note_path.close();

        let (r, g, b) = (color.r(), color.g(), color.b());
        let ambient_color = Color::rgba(r, g, b, PINK_GLOW_ALPHA);

        canvas.draw_shadow(
            &note_path,
            GLOW_SIZE,
            LIGHT_POS,
            0.0,
            ambient_color,
            SPOT_COLOR,
            None,
        );
        let mut paint = vg::Paint::default();
        paint.set_color(color);
        paint.set_anti_alias(true);
        canvas.draw_path(&note_path, &paint);

        for i in 0..tap_counter {
            let delay_time = params.delay_times[i].load(Ordering::SeqCst);
            let x_offset = delay_time.mul_add(time_scaling_factor, border_width);

            let note_value = params.notes[i].load(Ordering::SeqCst);
            let normalized_note = get_normalized_value(note_value, min_note_value, max_note_value);
            let note_height = Self::gui_smooth(
                1.0 - normalized_note,
                &params.previous_note_heights[i],
                gui_decay_weight,
            )
            .mul_add(available_height, margin);

            let note_center_x = bounds.x + x_offset;
            let note_center_y = bounds.y + note_height;

            let mut note_path = vg::Path::new();
            note_path.move_to((note_center_x + note_half_size, note_center_y));
            note_path.line_to((note_center_x, note_center_y + note_half_size));
            note_path.line_to((note_center_x - note_half_size, note_center_y));
            note_path.line_to((note_center_x, note_center_y - note_half_size));
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

            let smoothed_pan_foreground_length = Self::gui_smooth(
                target_pan_foreground_length,
                &params.previous_pan_foreground_lengths[i],
                gui_decay_weight,
            );
            let pan_foreground_length = smoothed_pan_foreground_length
                .max(bounds.x - note_center_x + border_width)
                .min(bounds.x + bounds.w - note_center_x - border_width);
            let pan_background_length = Self::gui_smooth(
                target_pan_background_length,
                &params.previous_pan_background_lengths[i],
                gui_decay_weight,
            );

            let mut pan_background_path = vg::Path::new();
            pan_background_path.move_to((note_center_x, note_center_y));
            pan_background_path.line_to((
                (note_center_x + pan_background_length)
                    .clamp(bounds.x + 1.0, bounds.x + bounds.w - 1.0),
                note_center_y,
            ));

            let mut paint = vg::Paint::default();
            paint.set_color(border_color);
            paint.set_anti_alias(true);
            paint.set_stroke_width(line_width * 0.7);
            paint.set_style(vg::PaintStyle::Stroke);
            canvas.draw_path(&pan_background_path, &paint);

            let mut pan_foreground_path = vg::Path::new();
            pan_foreground_path.add_rect(
                vg::Rect::from_xywh(
                    note_center_x,
                    note_center_y - line_width * 0.35,
                    pan_foreground_length,
                    line_width * 0.7,
                ),
                None,
            );
            let pan_glow_size = vg::Point3::new(0.0, 0.0, 16.0); // set the glow size
            let (r, g, b) = (font_color.r(), font_color.g(), font_color.b());
            let ambient_color = Color::rgba(r, g, b, YELLOW_GLOW_ALPHA);
            canvas.draw_shadow(
                &pan_foreground_path,
                pan_glow_size,
                LIGHT_POS,
                0.0,
                ambient_color,
                SPOT_COLOR,
                None,
            );
            let mut fg_paint = vg::Paint::default();
            fg_paint.set_color(font_color);
            fg_paint.set_anti_alias(true);
            canvas.draw_path(&pan_foreground_path, &fg_paint);

            let (r, g, b) = (color.r(), color.g(), color.b());
            let ambient_color = Color::rgba(r, g, b, PINK_GLOW_ALPHA);
            canvas.draw_shadow(
                &note_path,
                GLOW_SIZE,
                LIGHT_POS,
                0.0,
                ambient_color,
                SPOT_COLOR,
                None,
            );
            let mut paint = vg::Paint::default();
            paint.set_color(color);
            paint.set_anti_alias(true);
            canvas.draw_path(&note_path, &paint);
        }

        let (r, g, b) = (font_color.r(), font_color.g(), font_color.b());
        let ambient_color = Color::rgba(r, g, b, YELLOW_GLOW_ALPHA);

        if (panning_center_y - first_note_y).abs() > 3.0 {
            canvas.draw_shadow(
                &center_path,
                GLOW_SIZE,
                LIGHT_POS,
                0.0,
                ambient_color,
                SPOT_COLOR,
                None,
            );
            let mut paint = vg::Paint::default();
            paint.set_color(font_color);
            paint.set_anti_alias(true);
            canvas.draw_path(&center_path, &paint);
        }
    }

    fn draw_bounding_outline(
        canvas: &Canvas,
        bounds: BoundingBox,
        color: vg::Color,
        border_width: f32,
    ) {
        let half_border = border_width / 2.0;

        let mut path = vg::Path::new();
        path.add_rect(
            vg::Rect::from_xywh(
                bounds.x + half_border,
                bounds.y + half_border,
                bounds.w - border_width,
                bounds.h - border_width,
            ),
            None,
        );

        let mut paint = vg::Paint::default();
        paint.set_color(color);
        paint.set_anti_alias(true);
        paint.set_stroke_width(border_width);
        paint.set_style(vg::PaintStyle::Stroke);

        canvas.draw_path(&path, &paint);
    }
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
            Element::new(cx).class("trigger-element").hoverable(true);
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
        .hoverable(true)
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
        canvas: &Canvas,
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
        let w = bounds.w;
        let h = bounds.h;

        // Drawing the background rectangle
        let mut path = vg::Path::new();
        path.add_rect(vg::Rect::from_xywh(x, y, w, h), None);

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
        let paint_color = match (
            self.is_learning(),
            self.params.global.mute_is_toggle.value(),
            self.is_enabled(),
            self.is_playing(),
            self.own_index == CLEAR_TAPS,
        ) {
            (true, _, _, _, _) => border_color,
            (_, _, _, true, true) => outline_color,
            (_, true, true, _, _) => outline_color,
            (_, true, false, _, _) => background_color,
            (_, _, true, _, _) => outline_color,
            (_, _, _, true, _) => selection_color,
            _ => background_color, // Default: paint with background color
        };

        let mut paint = vg::Paint::default();
        paint.set_color(paint_color);
        paint.set_anti_alias(true);

        canvas.draw_path(&path, &paint);
    }

    // clippy's suggestion doesn't work, cause we need the early exit
    #[allow(clippy::match_same_arms)]
    fn get_action_class(
        params: &Arc<Del2Params>,
        is_learning: Arc<AtomicBool>,
        learned_notes: Arc<AtomicByteArray>,
        last_played_notes: Arc<LastPlayedNotes>,
        enabled_actions: Arc<AtomicBoolArray>,
        own_index: usize,
    ) -> &str {
        let is_learning =
            is_learning.load(Ordering::SeqCst) && learned_notes.load(own_index) == LEARNING;
        let is_playing = last_played_notes.is_playing(learned_notes.load(own_index));
        let is_enabled = enabled_actions.load(own_index);

        // Determine the paint color based on the state
        match (
            // true,true,true,true,true,
            is_learning,
            params.global.mute_is_toggle.value(),
            is_enabled,
            is_playing,
            own_index == CLEAR_TAPS,
        ) {
            (true, _, _, _, _) => "learning",
            (_, _, _, true, true) => "muted",
            (_, true, true, _, _) => "muted",
            (_, true, false, _, _) => "default",
            (_, _, true, _, _) => "muted",
            (_, _, _, true, _) => "live",
            _ => "default", // Default: paint with background color
        }
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
    fn draw(&self, draw_context: &mut DrawContext, canvas: &Canvas) {
        draw_context.needs_redraw();

        let bounds = draw_context.bounds();
        let background_color: vg::Color = draw_context.background_color().into();
        let border_color: vg::Color = draw_context.border_color().into();
        let outline_color: vg::Color = draw_context.outline_color().into();
        let selection_color: vg::Color = draw_context.selection_color().into();
        let border_width = draw_context.border_width();

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

///////////////////////////////////////////////////////////////////////////////
//                               CollapseButton                               //
///////////////////////////////////////////////////////////////////////////////

pub struct CollapseButton {}
impl CollapseButton {
    pub fn new(cx: &mut Context) -> Handle<Self> {
        Self {}.build(cx, move |cx| {
            Label::new(
                cx,
                Data::show_full_parameters.map(|show_full_parameters| {
                    // ▲ ▼ ◀ ▶
                    if *show_full_parameters {
                        String::from("▴")
                    } else {
                        String::from("▸")
                    }
                }),
            );
        })
    }
}

impl View for CollapseButton {
    fn event(&mut self, cx: &mut EventContext, event: &mut Event) {
        event.map(|window_event, meta| match window_event {
            // We don't need special double and triple click handling
            WindowEvent::MouseDown(MouseButton::Left)
            | WindowEvent::MouseDoubleClick(MouseButton::Left)
            | WindowEvent::MouseTripleClick(MouseButton::Left) => {
                cx.emit(AppEvent::ToggleShowView);
                meta.consume();
            }
            _ => {}
        });
    }
    fn draw(&self, _draw_context: &mut DrawContext, _canvas: &Canvas) {}
}
