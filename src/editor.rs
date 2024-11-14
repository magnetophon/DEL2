use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, Mutex,
};

use nih_plug::prelude::{AtomicF32, Editor};
use nih_plug_vizia::{
    assets, create_vizia_editor,
    vizia::{prelude::*, vg},
    widgets::{GenericUi, ParamSlider, ParamSliderExt, ParamSliderStyle, ResizeHandle},
    ViziaState, ViziaTheming,
};

use crate::{
    util, AtomicBoolArray, AtomicByteArray, Del2Params, LastPlayedNotes, SharedDelayData,
    SharedDelayDataOutput, CLEAR_TAPS, LEARNING, LOCK_TAPS, MUTE_IN, MUTE_OUT, NO_LEARNED_NOTE,
};

mod dual_meter;
use crate::editor::dual_meter::DualMeter;

#[derive(Lens, Clone)]
pub struct Data {
    pub params: Arc<Del2Params>,
    pub delay_data: Arc<Mutex<SharedDelayDataOutput>>,
    pub input_meter: Arc<AtomicF32>,
    pub output_meter: Arc<AtomicF32>,
    pub is_learning: Arc<AtomicBool>,
    pub learning_index: Arc<AtomicUsize>,
    pub learned_notes: Arc<AtomicByteArray>,
    pub last_played_notes: Arc<LastPlayedNotes>,
    pub enabled_actions: Arc<AtomicBoolArray>,
}

impl Model for Data {}

// Makes sense to also define this here, makes it a bit easier to keep track of
pub fn default_state() -> Arc<ViziaState> {
    ViziaState::new(|| (1408, 704))
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
                    // HStack::new(cx, |cx| {
                    HStack::new(cx, |cx| {
                        HStack::new(cx, |cx| {
                            Label::new(cx, "dry/wet").class("slider-label");
                            ParamSlider::new(cx, Data::params, |params| &params.global.dry_wet)
                                .class("widget");
                        })
                        .class("row");
                    })
                    .class("column");
                    HStack::new(cx, |cx| {
                        HStack::new(cx, |cx| {
                            Label::new(cx, "attack").class("slider-label");
                            ParamSlider::new(cx, Data::params, |params| &params.global.attack_ms)
                                .class("widget");
                        })
                        .class("row");
                    })
                    .class("column");
                })
                .class("param-group");
                HStack::new(cx, |cx| {
                    HStack::new(cx, |cx| {
                        HStack::new(cx, |cx| {
                            Label::new(cx, "dry gain").class("slider-label");
                            ParamSlider::new(cx, Data::params, |params| &params.global.output_gain)
                                .class("widget");
                        })
                        .class("row");
                    })
                    .class("column");
                    HStack::new(cx, |cx| {
                        HStack::new(cx, |cx| {
                            Label::new(cx, "release").class("slider-label");
                            ParamSlider::new(cx, Data::params, |params| &params.global.release_ms)
                                .class("widget");
                        })
                        .class("row");
                    })
                    .class("column");
                })
                .class("param-group");

                HStack::new(cx, |cx| {
                    // HStack::new(cx, |cx| {
                    HStack::new(cx, |cx| {
                        HStack::new(cx, |cx| {
                            Label::new(cx, "drive").class("slider-label");
                            ParamSlider::new(cx, Data::params, |params| {
                                &params.global.global_drive
                            })
                            .class("widget");
                        })
                        .class("row");
                    })
                    .class("column");
                    HStack::new(cx, |cx| {
                        HStack::new(cx, |cx| {
                            Label::new(cx, "min tap").class("slider-label");
                            ParamSlider::new(cx, Data::params, |params| {
                                &params.global.min_tap_milliseconds
                            })
                            .class("widget");
                        })
                        .class("row");
                    })
                    .class("column");
                })
                .class("param-group");
                HStack::new(cx, |cx| {
                    HStack::new(cx, |cx| {
                        HStack::new(cx, |cx| {
                            Label::new(cx, "mutes").class("slider-label");
                            ParamSlider::new(cx, Data::params, |params| {
                                &params.global.mute_is_toggle
                            })
                            .set_style(ParamSliderStyle::CurrentStepLabeled { even: true })
                            .class("widget");
                        })
                        .class("row");
                    })
                    .class("column");
                    HStack::new(cx, |cx| {
                        HStack::new(cx, |cx| {
                            Label::new(cx, "max tap").class("slider-label");
                            ParamSlider::new(cx, Data::params, |params| {
                                &params.global.max_tap_seconds
                            })
                            .set_style(ParamSliderStyle::FromLeft)
                            .class("widget");
                        })
                        .class("row");
                    })
                    .class("column");
                })
                .class("param-group");
                Label::new(cx, "triggers").class("group-title");
                HStack::new(cx, |cx| {
                    HStack::new(cx, |cx| {
                        HStack::new(cx, |cx| {
                            Label::new(cx, "mute in").class("action-name");
                            ActionTrigger::new(
                                cx,
                                Data::params,
                                Data::is_learning,
                                Data::learning_index,
                                Data::learned_notes,
                                Data::last_played_notes,
                                Data::enabled_actions,
                                MUTE_IN,
                            );
                        })
                        .class("row");
                    })
                    .class("column");
                    HStack::new(cx, |cx| {
                        HStack::new(cx, |cx| {
                            Label::new(cx, "clear taps").class("action-name");
                            ActionTrigger::new(
                                cx,
                                Data::params,
                                Data::is_learning,
                                Data::learning_index,
                                Data::learned_notes,
                                Data::last_played_notes,
                                Data::enabled_actions,
                                CLEAR_TAPS,
                            );
                        })
                        .class("row");
                    })
                    .class("column");
                })
                .class("param-group");

                HStack::new(cx, |cx| {
                    HStack::new(cx, |cx| {
                        HStack::new(cx, |cx| {
                            Label::new(cx, "mute out").class("action-name");
                            ActionTrigger::new(
                                cx,
                                Data::params,
                                Data::is_learning,
                                Data::learning_index,
                                Data::learned_notes,
                                Data::last_played_notes,
                                Data::enabled_actions,
                                MUTE_OUT,
                            );
                        })
                        .class("row");
                    })
                    .class("column");
                    HStack::new(cx, |cx| {
                        HStack::new(cx, |cx| {
                            Label::new(cx, "lock taps").class("action-name");
                            ActionTrigger::new(
                                cx,
                                Data::params,
                                Data::is_learning,
                                Data::learning_index,
                                Data::learned_notes,
                                Data::last_played_notes,
                                Data::enabled_actions,
                                LOCK_TAPS,
                            );
                        })
                        .class("row");
                    })
                    .class("column");
                })
                .class("param-group");

                Label::new(cx, "panning").class("group-title");

                HStack::new(cx, |cx| {
                    HStack::new(cx, |cx| {
                        HStack::new(cx, |cx| {
                            Label::new(cx, "center").class("slider-label");
                            ParamSlider::new(cx, Data::params, |params| {
                                &params.taps.panning_center
                            })
                            .class("widget");
                        })
                        .class("row");
                    })
                    .class("column");
                    HStack::new(cx, |cx| {
                        HStack::new(cx, |cx| {
                            Label::new(cx, "note>pan").class("slider-label");
                            ParamSlider::new(cx, Data::params, |params| {
                                &params.taps.panning_amount
                            })
                            .class("widget");
                        })
                        .class("row");
                    })
                    .class("column");
                })
                .class("param-group");

                Label::new(cx, "filters").class("group-title");

                HStack::new(cx, |cx| {
                    HStack::new(cx, |cx| {
                        HStack::new(cx, |cx| {
                            Label::new(cx, "vel>cut").class("slider-label");
                            ParamSlider::new(cx, Data::params, |params| {
                                &params.taps.velocity_to_cutoff_amount
                            })
                            .class("widget");
                        })
                        .class("row");
                    })
                    .class("column");
                    HStack::new(cx, |cx| {
                        HStack::new(cx, |cx| {
                            Label::new(cx, "note>cut").class("slider-label");
                            ParamSlider::new(cx, Data::params, |params| {
                                &params.taps.note_to_cutoff_amount
                            })
                            .class("widget");
                        })
                        .class("row");
                    })
                    .class("column");
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
                .size(Auto);
            })
            .class("parameters");
            VStack::new(cx, |cx| {
                ZStack::new(cx, |cx| {
                    DelayGraph::new(cx, Data::delay_data)
                    // .overflow(Overflow::Hidden)
                        ;
                    Label::new(cx, "DEL2").class("plugin-name");
                });
                VStack::new(cx, |cx| {
                    //meters
                    HStack::new(cx, |cx| {
                        VStack::new(cx, |cx| {
                            Label::new(cx, "in").class("peak-meter-label");
                            Label::new(cx, "out").class("peak-meter-label");
                        })
                        .class("peak-meter-label-group");
                        DualMeter::new(
                            cx,
                            Data::input_meter.map(|input_meter| {
                                util::gain_to_db(input_meter.load(Ordering::Relaxed))
                            }),
                            Data::output_meter.map(|output_meter| {
                                util::gain_to_db(output_meter.load(Ordering::Relaxed))
                            }),
                            Some(Duration::from_millis(600)),
                        );
                    })
                    .class("peak-meter-plus-label-group");
                })
                .class("peak-meter-group");
                ResizeHandle::new(cx);
            });
        });
    })
}

///////////////////////////////////////////////////////////////////////////////
//                             DelayGraph                            //
///////////////////////////////////////////////////////////////////////////////

pub struct DelayGraph {
    delay_data: Arc<Mutex<SharedDelayDataOutput>>,
}

// TODO: add grid to show bars & beats
impl View for DelayGraph {
    // For CSS:
    fn element(&self) -> Option<&'static str> {
        Some("delay-graph")
    }

    fn draw(&self, draw_context: &mut DrawContext, canvas: &mut Canvas) {
        let mut locked_delay_data = self.delay_data.lock().unwrap();
        let delay_data = locked_delay_data.read();

        let bounds = draw_context.bounds();

        let background_color: vg::Color = draw_context.background_color().into();
        let border_color: vg::Color = draw_context.border_color().into();
        let outline_color: vg::Color = draw_context.outline_color().into();
        let selection_color: vg::Color = draw_context.selection_color().into();
        let border_width = draw_context.border_width();
        let outline_width = draw_context.outline_width();

        // Compute the time scaling factor
        let time_scaling_factor =
            Self::compute_time_scaling_factor(delay_data, bounds.w, border_width, outline_width);

        // Draw components
        Self::draw_background(canvas, bounds, background_color);
        Self::draw_delay_times_as_lines(
            canvas,
            delay_data,
            bounds,
            border_color,
            border_width,
            time_scaling_factor,
        );
        Self::draw_time_line(
            canvas,
            delay_data,
            bounds,
            selection_color,
            outline_width,
            time_scaling_factor,
            border_width,
        );
        Self::draw_tap_velocities(
            canvas,
            delay_data,
            bounds,
            outline_color,
            outline_width,
            time_scaling_factor,
            border_width,
        );
        Self::draw_tap_notes_and_pans(
            canvas,
            delay_data,
            bounds,
            selection_color,
            outline_width,
            time_scaling_factor,
            border_width,
            border_color,
            background_color,
            true,
        );
        Self::draw_bounding_outline(canvas, bounds, border_color, border_width);
    }
}

impl DelayGraph {
    pub fn new<SharedDelayDataL>(cx: &mut Context, delay_data: SharedDelayDataL) -> Handle<Self>
    where
        SharedDelayDataL: Lens<Target = Arc<Mutex<SharedDelayDataOutput>>>,
    {
        Self {
            delay_data: delay_data.get(cx),
        }
        .build(cx, |cx| {
            Label::new(
                cx,
                delay_data.map(move |data| {
                    let mut locked_delay_data = data.lock().unwrap();
                    let delay_data = locked_delay_data.read();
                    match delay_data.current_tap {
                        0 => String::new(),
                        1 => "1 tap".to_string(),
                        tap_nr => format!("{tap_nr} taps"),
                    }
                }),
            )
            .class("tap-nr-label");
        })
    }

    fn compute_time_scaling_factor(
        delay_data: &SharedDelayData,
        rect_width: f32,
        border_width: f32,
        outline_width: f32,
    ) -> f32 {
        let max_delay_time = if delay_data.current_tap > 0 {
            delay_data.delay_times[delay_data.current_tap - 1]
        } else {
            0
        };
        ((max_delay_time as f32 + delay_data.max_tap_samples as f32)
            / outline_width.mul_add(-0.5, rect_width - border_width))
        .recip()
    }

    fn draw_delay_times_as_lines(
        canvas: &mut Canvas,
        delay_data: &SharedDelayData,
        bounds: BoundingBox,
        border_color: vg::Color,
        border_width: f32,
        time_scaling_factor: f32,
    ) {
        let mut path = vg::Path::new();

        for i in 0..delay_data.current_tap {
            // Combine delay time with time scaling factor for correct horizontal scaling
            let x_offset =
                (delay_data.delay_times[i] as f32).mul_add(time_scaling_factor, border_width * 0.5);

            // Line from bottom to top border considering border thickness
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
        delay_data: &SharedDelayData,
        bounds: BoundingBox,
        color: vg::Color,
        line_width: f32,
        scaling_factor: f32,
        border_width: f32,
    ) {
        let max_delay_time = if delay_data.current_tap > 0 {
            delay_data.delay_times[delay_data.current_tap - 1]
        } else {
            0
        };
        if delay_data.current_time > max_delay_time {
            let x_offset =
                (delay_data.current_time as f32).mul_add(scaling_factor, border_width * 0.5);
            let mut path = vg::Path::new();
            path.move_to(
                bounds.x + x_offset,
                border_width.mul_add(-0.5, bounds.y + bounds.h),
            );
            path.line_to(bounds.x + x_offset, bounds.y);
            path.close();

            canvas.stroke_path(&path, &vg::Paint::color(color).with_line_width(line_width));
        }
    }

    fn draw_tap_velocities(
        canvas: &mut Canvas,
        delay_data: &SharedDelayData,
        bounds: BoundingBox,
        color: vg::Color,
        line_width: f32,
        scaling_factor: f32,
        border_width: f32,
    ) {
        let mut path = vg::Path::new();
        for i in 0..delay_data.current_tap {
            let x_offset =
                (delay_data.delay_times[i] as f32).mul_add(scaling_factor, border_width * 0.5);
            let velocity_height = delay_data.velocities[i].mul_add(
                -border_width.mul_add(-0.5, bounds.h),
                border_width.mul_add(-0.5, bounds.h),
            );

            path.move_to(
                bounds.x + x_offset,
                border_width.mul_add(-0.5, bounds.y + bounds.h),
            );
            path.line_to(bounds.x + x_offset, bounds.y + velocity_height);
        }

        canvas.stroke_path(&path, &vg::Paint::color(color).with_line_width(line_width));
    }

    fn draw_tap_notes_and_pans(
        canvas: &mut Canvas,
        delay_data: &SharedDelayData,
        bounds: BoundingBox,
        color: vg::Color,
        line_width: f32,
        scaling_factor: f32,
        border_width: f32,
        border_color: vg::Color,
        background_color: vg::Color,
        zoomed: bool,
    ) {
        let mut diamond_path = vg::Path::new();
        let mut pan_path = vg::Path::new();

        // Determine min and max note values if zoomed
        let (min_note_value, max_note_value) = if zoomed {
            let mut used_notes = Vec::from(&delay_data.notes[0..delay_data.current_tap]);
            if delay_data.first_note != NO_LEARNED_NOTE {
                used_notes.push(delay_data.first_note);
            }
            let min = used_notes.iter().copied().min().unwrap_or(0);
            let max = used_notes.iter().copied().max().unwrap_or(127);
            (f32::from(min), f32::from(max))
        } else {
            (0.0, 127.0)
        };

        let diamond_size = line_width * 2.0; // Width and height of a diamond

        // Calculate available height with margins
        let margin = 3.0 * line_width;
        let available_height = 2.0f32.mul_add(-(margin + diamond_size + border_width), bounds.h);

        // Draw half a diamond for the first note at time 0
        let first_note = delay_data.first_note;
        if first_note != NO_LEARNED_NOTE {
            let normalized_first_note = if max_note_value != min_note_value {
                (f32::from(first_note) - min_note_value) / (max_note_value - min_note_value)
            } else {
                f32::from(first_note) / 127.0
            };

            let first_note_height =
                (1.0 - normalized_first_note).mul_add(available_height, margin + diamond_size);
            let first_diamond_center_x = bounds.x;
            let first_diamond_center_y = bounds.y + first_note_height;
            let diamond_half_size = line_width;

            // Drawing only the right half of the diamond
            diamond_path.move_to(
                first_diamond_center_x,
                first_diamond_center_y + diamond_half_size,
            );
            diamond_path.line_to(
                first_diamond_center_x + diamond_half_size,
                first_diamond_center_y,
            );
            diamond_path.line_to(
                first_diamond_center_x,
                first_diamond_center_y - diamond_half_size,
            );
            diamond_path.close();
        }

        // Continue with the rest of the drawing process as usual
        for i in 0..delay_data.current_tap {
            let x_offset =
                (delay_data.delay_times[i] as f32).mul_add(scaling_factor, border_width * 0.5);

            let normalized_note = if max_note_value != min_note_value {
                (f32::from(delay_data.notes[i]) - min_note_value)
                    / (max_note_value - min_note_value)
            } else {
                f32::from(delay_data.notes[i]) / 127.0
            };

            let note_height =
                (1.0 - normalized_note).mul_add(available_height, margin + diamond_size);

            let diamond_center_x = bounds.x + x_offset;
            let diamond_center_y = bounds.y + note_height;

            let diamond_half_size = line_width;

            diamond_path.move_to(diamond_center_x + diamond_half_size, diamond_center_y);
            diamond_path.line_to(diamond_center_x, diamond_center_y + diamond_half_size);
            diamond_path.line_to(diamond_center_x - diamond_half_size, diamond_center_y);
            diamond_path.line_to(diamond_center_x, diamond_center_y - diamond_half_size);
            diamond_path.close();

            let pan_value = delay_data.pans[i];
            let line_length = 50.0;
            let pan_offset = pan_value * line_length;

            pan_path.move_to(diamond_center_x, diamond_center_y);
            pan_path.line_to(diamond_center_x + pan_offset, diamond_center_y);
        }

        canvas.stroke_path(
            &pan_path,
            &vg::Paint::color(border_color).with_line_width(line_width),
        );

        canvas.stroke_path(
            &diamond_path,
            &vg::Paint::color(color).with_line_width(line_width),
        );

        // FIXME:
        // cover up, probably needed due to https://github.com/vizia/vizia/issues/401
        if first_note != NO_LEARNED_NOTE {
            // Draw a line over the left half in the background color
            let mut cover_line_path = vg::Path::new();
            let cover_x = border_width.mul_add(0.5, line_width.mul_add(-0.5, bounds.x));
            cover_line_path.move_to(cover_x, bounds.y);
            cover_line_path.line_to(cover_x, bounds.y + bounds.h);

            canvas.stroke_path(
                &cover_line_path,
                // &vg::Paint::color(Color::red().into()).with_line_width(line_width),
                &vg::Paint::color(background_color).with_line_width(line_width),
            );
        }
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
    last_played_notes: Arc<LastPlayedNotes>,
    enabled_actions: Arc<AtomicBoolArray>,
    own_index: usize,
    // to temp store the note we had during learning
    // so we can keep abusing the notes above 127 to signify other things
    last_learned_note: u8,
}
impl ActionTrigger {
    pub fn new<
        ParamsL,
        IsLearningL,
        LearningIndexL,
        LearnedNotesL,
        LastPlayedNotesL,
        EnabledActionsL,
    >(
        cx: &mut Context,
        params: ParamsL,
        is_learning: IsLearningL,
        learning_index: LearningIndexL,
        learned_notes: LearnedNotesL,
        last_played_notes: LastPlayedNotesL,
        enabled_actions: EnabledActionsL,
        own_index: usize,
    ) -> Handle<Self>
    where
        ParamsL: Lens<Target = Arc<Del2Params>>,
        IsLearningL: Lens<Target = Arc<AtomicBool>>,
        LearningIndexL: Lens<Target = Arc<AtomicUsize>>,
        LearnedNotesL: Lens<Target = Arc<AtomicByteArray>>,
        LastPlayedNotesL: Lens<Target = Arc<LastPlayedNotes>>,
        EnabledActionsL: Lens<Target = Arc<AtomicBoolArray>>,
    {
        Self {
            params: params.get(cx),
            is_learning: is_learning.get(cx),
            learning_index: learning_index.get(cx),
            learned_notes: learned_notes.get(cx),
            last_played_notes: last_played_notes.get(cx),
            enabled_actions: enabled_actions.get(cx),
            own_index,
            last_learned_note: NO_LEARNED_NOTE,
        }
        .build(cx, move |cx| {
            Label::new(
                cx,
                learned_notes.map(move |notes| {
                    let note_nr = notes.load(own_index);
                    Self::get_note_name(note_nr)
                }),
            )
            .class("action-label");
        })
    }

    pub fn start_learning(&mut self) {
        self.is_learning.store(true, Ordering::SeqCst);
        let index = self.own_index;
        self.last_learned_note = self.learned_notes.load(index);
        self.learned_notes.store(index, LEARNING);
        self.learning_index.store(index, Ordering::SeqCst);
    }
    pub fn stop_learning(&self) {
        self.is_learning.store(false, Ordering::SeqCst);
        self.learned_notes
            .store(self.own_index, self.last_learned_note);
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

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                       for drawing
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
