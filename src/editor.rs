use nih_plug::prelude::Editor;
use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::vizia::vg;
use nih_plug_vizia::widgets::*;
use nih_plug_vizia::{assets, create_vizia_editor, ViziaState, ViziaTheming};
use std::sync::{Arc, Mutex};

use crate::Del2Params;
// use crate::ZoomMode;
use crate::DelayDataOutput;

#[derive(Lens, Clone)]
pub(crate) struct Data {
    pub(crate) params: Arc<Del2Params>,
    pub(crate) delay_data: Arc<Mutex<DelayDataOutput>>,
}

impl Model for Data {}

// Makes sense to also define this here, makes it a bit easier to keep track of
pub(crate) fn default_state() -> Arc<ViziaState> {
    ViziaState::new(|| (600, 220))
}

pub(crate) fn create(editor_data: Data, editor_state: Arc<ViziaState>) -> Option<Box<dyn Editor>> {
    create_vizia_editor(editor_state, ViziaTheming::Custom, move |cx, _| {
        assets::register_noto_sans_light(cx);
        assets::register_noto_sans_thin(cx);

        editor_data.clone().build(cx);

        HStack::new(cx, |cx| {
            VStack::new(cx, |cx| {
                Label::new(cx, "DEL2")
                    .font_family(vec![FamilyOwned::Name(String::from(assets::NOTO_SANS))])
                    .font_weight(FontWeightKeyword::Thin)
                    .font_size(30.0)
                    .height(Pixels(50.0))
                    .child_top(Stretch(1.0))
                    .child_bottom(Pixels(0.0));

                Label::new(cx, "gain");
                ParamSlider::new(cx, Data::params, |params| &params.gain);
                Label::new(cx, "time out");
                ParamSlider::new(cx, Data::params, |params| &params.time_out_seconds);
                Label::new(cx, "debounce");
                ParamSlider::new(cx, Data::params, |params| &params.debounce_tap_milliseconds);
            })
            // .row_between(Pixels(0.0))
            .child_left(Stretch(1.0))
            .child_right(Stretch(1.0));
            DelayGraph::new(cx, Data::delay_data)
            // .background_color(Color::green())
                .border_color(Color::grey())
                .outline_color(Color::black())
                .border_width(Pixels(1.0))
            // .child_right(Stretch(1.0))
                .width(Pixels(400.0))
            // .height(Pixels(200.0))
                ;
            ResizeHandle::new(cx)
            // .height(Pixels(10.0))
                ;
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
        // let zoom_mode =
        // self.delay_data.get(cx).zoom_mode.value();

        let border_color = cx.border_color();
        let outline_color = cx.outline_color();
        let opacity = cx.opacity();
        let mut background_color: vg::Color = cx.background_color().into();
        background_color.set_alphaf(background_color.a * opacity);
        let mut border_color: vg::Color = border_color.into();
        border_color.set_alphaf(border_color.a * opacity);
        let border_width = cx.border_width();

        let x = bounds.x + border_width / 2.0;
        let y = bounds.y + border_width / 2.0;
        let w = bounds.w - border_width;
        let h = bounds.h - border_width;

        // Create a new `Path` from the `vg` module.
        let mut path = vg::Path::new();
        {
            path.move_to(x, y);
            path.line_to(x, y + h);
            path.line_to(x + w, y + h);
            path.line_to(x + w, y);
            path.line_to(x, y);
            path.close();
        }
        // Fill with background color
        let paint = vg::Paint::color(background_color);
        canvas.fill_path(&path, &paint);
        let mut max_delay = 0;
        if delay_data.current_tap > 0 {
            max_delay = delay_data.delay_times_array[delay_data.current_tap - 1];
        }
        let x_factor = (max_delay as f32 + delay_data.time_out_samples as f32) / w;

        // draw delay taps
        canvas.stroke_path(
            &{
                let mut path = vg::Path::new();
                for i in 0..delay_data.current_tap {
                    let x_offset = delay_data.delay_times_array[i] as f32 / x_factor;
                    let y_offset = h - (delay_data.velocity_array[i] * h);
                    path.move_to(x + x_offset, y + h);
                    path.line_to(x + x_offset, y + y_offset);
                }
                path
            },
            &vg::Paint::color(outline_color.into()).with_line_width(border_width),
        );

        // draw current time
        if delay_data.current_time > max_delay {
            canvas.stroke_path(
                &{
                    let mut path = vg::Path::new();
                    let x_offset = delay_data.current_time as f32 / x_factor;
                    path.move_to(x + x_offset, y + h);
                    path.line_to(x + x_offset, y);
                    path
                },
                &vg::Paint::color(Color::red().into()).with_line_width(border_width),
            );
        };
        // add outline
        canvas.stroke_path(
            &{
                let mut path = vg::Path::new();
                path.rect(x, y, w, h);
                path
            },
            &vg::Paint::color(border_color).with_line_width(border_width),
        );
    }
}
