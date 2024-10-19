use crate::util;
use nih_plug::prelude::{AtomicF32, Editor};
use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::vizia::vg;
use nih_plug_vizia::widgets::*;
use nih_plug_vizia::{assets, create_vizia_editor, ViziaState, ViziaTheming};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use crate::Del2Params;
use crate::DelayDataOutput;

const COLUMN_WIDTH: Units = Pixels(269.0);

#[derive(Lens, Clone)]
pub(crate) struct Data {
    pub(crate) params: Arc<Del2Params>,
    pub(crate) delay_data: Arc<Mutex<DelayDataOutput>>,
    pub input_meter: Arc<AtomicF32>,
    pub output_meter: Arc<AtomicF32>,
}

impl Model for Data {}

// Makes sense to also define this here, makes it a bit easier to keep track of
pub(crate) fn default_state() -> Arc<ViziaState> {
    ViziaState::new(|| (1200, 400))
}

pub(crate) fn create(editor_data: Data, editor_state: Arc<ViziaState>) -> Option<Box<dyn Editor>> {
    create_vizia_editor(editor_state, ViziaTheming::Custom, move |cx, _| {
        assets::register_noto_sans_light(cx);
        assets::register_noto_sans_thin(cx);

        // Add the stylesheet to the app
        cx.add_stylesheet(include_style!("src/style.css"))
            .expect("Failed to load stylesheet");

        editor_data.clone().build(cx);

        HStack::new(cx, |cx| {
            VStack::new(cx, |cx| {
                Label::new(cx, "global").class("global-title");
                HStack::new(cx, |cx| {
                    make_column(cx, "gain", |cx| {
                        let gain_params = Data::params.map(|p| p.global.gain_params.clone());
                        GenericUi::new_custom(cx, gain_params, |cx, param_ptr| {
                            HStack::new(cx, |cx| {
                                Label::new(cx, unsafe { param_ptr.name() }).class("label");
                                GenericUi::draw_widget(cx, gain_params, param_ptr);
                            })
                            .class("row");
                        });
                    });

                    make_column(cx, "timing", |cx| {
                        let timing_params = Data::params.map(|p| p.global.timing_params.clone());
                        GenericUi::new_custom(cx, timing_params, |cx, param_ptr| {
                            let param_name = unsafe { param_ptr.name() };

                            // Check if the parameter is `max_tap_seconds` and replace with specific logic
                            if param_name == "max tap" {
                                HStack::new(cx, |cx| {
                                    Label::new(cx, param_name).class("label");
                                    ParamSlider::new(cx, Data::params, |params| {
                                        &params.global.timing_params.max_tap_seconds
                                    })
                                    .set_style(ParamSliderStyle::FromLeft)
                                    .class("widget");
                                })
                                .class("row");
                            } else {
                                // Default widget drawing for others
                                HStack::new(cx, |cx| {
                                    Label::new(cx, param_name).class("label");
                                    GenericUi::draw_widget(cx, timing_params, param_ptr);
                                })
                                .class("row");
                            }
                        });
                    });
                });
                Label::new(cx, "filters").class("dsp-title");

                HStack::new(cx, |cx| {
                    make_column(cx, "low velocity", |cx| {
                        // We don't want to show the 'Upwards' prefix here, but it should still be in
                        // the parameter name so the parameter list makes sense
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
            // .row_between(Pixels(9.0))
            // .child_left(Stretch(1.0))
            // .child_right(Stretch(1.0))
            // .child_left(Pixels(9.0))
            // .child_right(Pixels(9.0));
            VStack::new(cx, |cx| {
                ZStack::new(cx, |cx| {
                    Label::new(cx, "DEL2").class("plugin-name");
                    DelayGraph::new(cx, Data::delay_data);
                });
                VStack::new(cx, |cx| {
                    //meters
                    HStack::new(cx, |cx| {
                        PeakMeter::new(
                            cx,
                            Data::input_meter.map(|input_meter| {
                                util::gain_to_db(input_meter.load(Ordering::Relaxed))
                            }),
                            Some(Duration::from_millis(600)),
                        );
                        Label::new(cx, "input").class("meter-label");
                    });
                    HStack::new(cx, |cx| {
                        PeakMeter::new(
                            cx,
                            Data::output_meter.map(|output_meter| {
                                util::gain_to_db(output_meter.load(Ordering::Relaxed))
                            }),
                            Some(Duration::from_millis(600)),
                        );
                        Label::new(cx, "output").class("meter-label");
                    });
                })
                .class("meters_and_name");
                ResizeHandle::new(cx);
            });
        });
    })
}

///////////////////////////////////////////////////////////////////////////////
//                             DelayGraph                            //
///////////////////////////////////////////////////////////////////////////////

pub struct DelayGraph {
    delay_data: Arc<Mutex<DelayDataOutput>>,
}

impl DelayGraph {
    pub fn new<DelayDataL>(cx: &mut Context, delay_data: DelayDataL) -> Handle<Self>
    where
        DelayDataL: Lens<Target = Arc<Mutex<DelayDataOutput>>>,
    {
        Self {
            delay_data: delay_data.get(cx),
        }
        .build(cx, |_cx| ())
    }
}

// TODO: add grid to show bars & beats
impl View for DelayGraph {
    // for css:
    fn element(&self) -> Option<&'static str> {
        Some("delay-graph")
    }

    fn draw(&self, cx: &mut DrawContext, canvas: &mut Canvas) {
        let mut delay_data = self.delay_data.lock().unwrap();
        let delay_data = delay_data.read();
        // Get the bounding box of the current view.
        let bounds = cx.bounds();

        let border_color = cx.border_color();
        let outline_color = cx.outline_color();
        let selection_color = cx.selection_color();
        let opacity = cx.opacity();
        let mut background_color: vg::Color = cx.background_color().into();
        background_color.set_alphaf(background_color.a * opacity);
        let mut border_color: vg::Color = border_color.into();
        border_color.set_alphaf(border_color.a * opacity);
        let border_width = cx.border_width();
        // let line_width = cx.scale_factor();
        let line_width = cx.outline_width();

        let x = bounds.x + border_width * 0.5;
        let y = bounds.y;
        let w = bounds.w - border_width;
        let h = bounds.h - border_width * 0.5;

        // Create a new `Path` from the `vg` module.
        let mut path = vg::Path::new();
        {
            path.rect(x, y, w, h);
            // path.move_to(x, y);
            // path.line_to(x, y + h);
            // path.line_to(x + w, y + h);
            // path.line_to(x + w, y);
            // path.line_to(x, y);
            path.close();
        }
        // Fill with background color
        let paint = vg::Paint::color(background_color);
        canvas.fill_path(&path, &paint);
        let mut max_delay = 0;
        if delay_data.current_tap > 0 {
            max_delay = delay_data.delay_times[delay_data.current_tap - 1];
        }
        let x_factor = ((max_delay as f32 + delay_data.max_tap_samples as f32)
            / (w - border_width - line_width * 0.5))
            .recip();

        // draw current time
        if delay_data.current_time > max_delay {
            canvas.stroke_path(
                &{
                    let mut path = vg::Path::new();
                    let x_offset = delay_data.current_time as f32 * x_factor + border_width * 0.5;
                    path.move_to(x + x_offset, y + h);
                    path.line_to(x + x_offset, y);
                    path
                },
                &vg::Paint::color(selection_color.into()).with_line_width(line_width),
            );
        };
        // draw delay tap velocities
        canvas.stroke_path(
            &{
                let mut path = vg::Path::new();
                for i in 0..delay_data.current_tap {
                    let x_offset = delay_data.delay_times[i] as f32 * x_factor + border_width * 0.5;
                    let y_offset = (h - border_width * 0.5)
                        - (delay_data.velocities[i] * (h - border_width * 0.5));
                    path.move_to(x + x_offset, y + h - (border_width * 0.5));
                    path.line_to(x + x_offset, y + y_offset);
                }
                path
            },
            &vg::Paint::color(outline_color.into()).with_line_width(line_width),
        );

        // Draw delay tap notes as diamonds
        canvas.stroke_path(
            &{
                let mut path = vg::Path::new();
                for i in 0..delay_data.current_tap {
                    let x_offset = delay_data.delay_times[i] as f32 * x_factor + border_width * 0.5;
                    let y_offset = (h - border_width * 0.5)
                        - ((delay_data.notes[i] as f32 / 127.0) * (h - border_width * 0.5));

                    let center_x = x + x_offset;
                    let center_y = y + y_offset;
                    let half_size = line_width; // You can adjust this to change the size of the diamond

                    // Draw diamond shape
                    path.move_to(center_x + half_size, center_y); // Top-Right
                    path.line_to(center_x, center_y + half_size); // Bottom-Right
                    path.line_to(center_x - half_size, center_y); // Bottom-Left
                    path.line_to(center_x, center_y - half_size); // Top-Left
                    path.close();
                }
                path
            },
            &vg::Paint::color(selection_color.into()).with_line_width(line_width),
        );

        // add outline
        canvas.stroke_path(
            &{
                let mut path = vg::Path::new();
                path.rect(x, y, w, h);
                // path.move_to(x, y);
                // path.line_to(x, y + h);
                path
            },
            &vg::Paint::color(border_color).with_line_width(border_width),
        );
    }
}

fn make_column(cx: &mut Context, title: &str, contents: impl FnOnce(&mut Context)) {
    VStack::new(cx, |cx| {
        Label::new(cx, title).class("column-title");

        contents(cx);
    })
    // .class("column");
    .width(COLUMN_WIDTH)
    .height(Auto);
}
