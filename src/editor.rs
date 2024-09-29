use nih_plug::prelude::Editor;
use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::vizia::vg;
use nih_plug_vizia::widgets::*;
use nih_plug_vizia::{assets, create_vizia_editor, ViziaState, ViziaTheming};
use std::sync::{Arc, Mutex};

use crate::Del2Params;
// use crate::DelayGraphData;
// use crate::ZoomMode;
use crate::MAX_NR_TAPS;
use crate::TOTAL_DELAY_SAMPLES;
// use crate::WaveformBufferOutput;
use crate::DelayGraphDataOutput;

#[derive(Lens, Clone)]
pub(crate) struct Data {
    pub(crate) params: Arc<Del2Params>,
    pub(crate) delay_data: Arc<Mutex<DelayGraphDataOutput>>,
    // pub waveform_buffer_output: Arc<Mutex<WaveformBufferOutput>>,
}

impl Model for Data {}

// Makes sense to also define this here, makes it a bit easier to keep track of
pub(crate) fn default_state() -> Arc<ViziaState> {
    ViziaState::new(|| (600, 220))
}

pub(crate) fn create(
    // params: Arc<Del2Params>,
    editor_data: Data,
    // buffer_output: Arc<Mutex<WaveformBufferOutput>>,
    editor_state: Arc<ViziaState>,
) -> Option<Box<dyn Editor>> {
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
                ParamSlider::new(cx, Data::params, |params| &params.time_out_tap_seconds);
                Label::new(cx, "debounce");
                ParamSlider::new(cx, Data::params, |params| &params.debounce_tap_milliseconds);
            })
            // .row_between(Pixels(0.0))
            .child_left(Stretch(1.0))
            .child_right(Stretch(1.0));
            DelayGraph::new(cx, Data::delay_data)
            // .background_color(Color::green())
                .border_color(Color::grey())
                    .outline_color(Color::red())
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
    delay_data: Arc<Mutex<DelayGraphDataOutput>>,
}

// impl<DelayDataL: Lens<Target = DelayGraphData>> DelayGraph<DelayDataL> {
//     pub fn new(cx: &mut Context, delay_data: DelayDataL) -> Handle<Self> {
//         Self { delay_data }
//         .build(cx, |_cx| {
//             // If we want the view to contain other views we can build those here.
//         })
//         // Redraw when lensed data changes
//         .bind(delay_data, |mut handle, _| handle.needs_redraw())
//     }
// }

impl DelayGraph {
    pub fn new<DelayDataL>(cx: &mut Context, delay_data: DelayDataL) -> Handle<Self>
    where
        // DelayDataL: Lens<Target = DelayGraphData>,
        DelayDataL: Lens<Target = Arc<Mutex<DelayGraphDataOutput>>>,
        // LRecordingProgress: Lens<Target = Arc<Mutex<f32>>>,
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
        let attack = 9.0;
        // self.delay_data.get(cx).attack.value();
        let attack_shape = 9.0;
        // self.delay_data.get(cx).attack_shape.value();
        let release = 9.0;
        // self.delay_data.get(cx).release.value();
        let release_shape = 9.0;
        // self.delay_data.get(cx).release_shape.value();
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

        // add the attack / release curve
        canvas.stroke_path(
            &{
                // let x = bounds.x + border_width * 1.0;
                // let y = bounds.y + border_width * 1.5;
                // let w = bounds.w - border_width * 2.0;
                // let h = bounds.h - border_width * 3.0;
                // let max_delay_samples =
                // match zoom_mode {
                // ZoomMode::Relative => 192000,
                // ZoomMode::Absolute => 192000*2,
                // };

                // let mut start = 0.0;
                // let mut end = w;
                // if zoom_mode == ZoomMode::Absolute {
                // start = ((max_attack - attack) / max_attack) * center;
                // end = center + ((release / max_release) * (w - center));
                // }

                let mut path = vg::Path::new();
                for i in 0..MAX_NR_TAPS {
                    let time =
                        (delay_data.delay_times_array[i] as f32 / TOTAL_DELAY_SAMPLES as f32) * w;
                    if time > 0.0 {
                        println!("time: {}", time);
                        path.move_to(x + time, y + h);
                        path.line_to(x + time, y);
                        // path.move_to(x+w, y+h);
                        // path.line_to(x, y);
                    };
                }
                // path.line_to(x + end, y);
                // path.line_to(x + w, y);
                path
            },
            &vg::Paint::color(outline_color.into()).with_line_width(border_width),
        );
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
