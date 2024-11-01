//! A super simple dual meter widget.

use nih_plug::prelude::util;
use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::vizia::vg;
use std::cell::Cell;
use std::time::Duration;
use std::time::Instant;

/// The decibel value corresponding to the very left of the bar.
const MIN_TICK: f32 = -90.0;
/// The decibel value corresponding to the very right of the bar.
const MAX_TICK: f32 = 20.0;
/// The ticks that will be shown beneath the dual meter's bar. The first value is shown as
/// -infinity, and at the last position we'll draw the `dBFS` string.
const TEXT_TICKS: [i32; 6] = [-80, -60, -40, -20, 0, 12];

/// A simple horizontal dual meter.
///
/// TODO: There are currently no styling options at all
/// TODO: Vertical dual meter, this is just a proof of concept to fit the gain GUI example.
pub struct DualMeter;

/// The bar bit for the dual meter, manually drawn using vertical lines.
struct PeakMeterBar<L, P>
where
    L: Lens<Target = f32>,
    P: Lens<Target = f32>,
{
    level_dbfs: L,
    peak_dbfs: P,
}

impl DualMeter {
    /// Creates a new [`DualMeter`] for the given value in decibel, optionally holding the peak
    /// value for a certain amount of time.
    pub fn new<L1, L2>(
        cx: &mut Context,
        level_dbfs_1: L1,
        level_dbfs_2: L2,
        hold_time: Option<Duration>,
    ) -> Handle<Self>
    where
        L1: Lens<Target = f32>,
        L2: Lens<Target = f32>,
    {
        Self.build(cx, |cx| {
            let held_peak_value_db1 = Cell::new(f32::MIN);
            let last_held_peak_value1: Cell<Option<Instant>> = Cell::new(None);

            let held_peak_value_db2 = Cell::new(f32::MIN);
            let last_held_peak_value2: Cell<Option<Instant>> = Cell::new(None);

            let peak_dbfs_1 = level_dbfs_1.map(move |level| -> f32 {
                match hold_time {
                    Some(hold_time) => {
                        let mut peak_level = held_peak_value_db1.get();
                        let peak_time = last_held_peak_value1.get();

                        let now = Instant::now();
                        if *level >= peak_level
                            || peak_time.is_none()
                            || now > peak_time.unwrap() + hold_time
                        {
                            peak_level = *level;
                            held_peak_value_db1.set(peak_level);
                            last_held_peak_value1.set(Some(now));
                        }

                        peak_level
                    }
                    None => util::MINUS_INFINITY_DB,
                }
            });

            let peak_dbfs_2 = level_dbfs_2.map(move |level| -> f32 {
                match hold_time {
                    Some(hold_time) => {
                        let mut peak_level = held_peak_value_db2.get();
                        let peak_time = last_held_peak_value2.get();

                        let now = Instant::now();
                        if *level >= peak_level
                            || peak_time.is_none()
                            || now > peak_time.unwrap() + hold_time
                        {
                            peak_level = *level;
                            held_peak_value_db2.set(peak_level);
                            last_held_peak_value2.set(Some(now));
                        }

                        peak_level
                    }
                    None => util::MINUS_INFINITY_DB,
                }
            });

            PeakMeterBar {
                level_dbfs: level_dbfs_1,
                peak_dbfs: peak_dbfs_1,
            }
            .build(cx, |_| {})
            .class("bar");

            PeakMeterBar {
                level_dbfs: level_dbfs_2,
                peak_dbfs: peak_dbfs_2,
            }
            .build(cx, |_| {})
            .class("bar");

            ZStack::new(cx, |cx| {
                const WIDTH_PCT: f32 = 50.0;
                for tick_db in TEXT_TICKS {
                    let tick_fraction = (tick_db as f32 - MIN_TICK) / (MAX_TICK - MIN_TICK);
                    let tick_pct = tick_fraction * 100.0;
                    // We'll shift negative numbers slightly to the left so they look more centered
                    let needs_minus_offset = tick_db < 0;

                    ZStack::new(cx, |cx| {
                        let first_tick = tick_db == TEXT_TICKS[0];
                        let last_tick = tick_db == TEXT_TICKS[TEXT_TICKS.len() - 1];

                        if !last_tick {
                            // FIXME: This is not aligned to the pixel grid and some ticks will look
                            //        blurry, is there a way to fix this?
                            Element::new(cx).class("ticks__tick");
                        }

                        let font_size = {
                            let event_cx = EventContext::new(cx);
                            event_cx.font_size() * event_cx.scale_factor()
                        };
                        let label = if first_tick {
                            Label::new(cx, "-inf")
                                .class("ticks__label")
                                .class("ticks__label--inf")
                        } else if last_tick {
                            // This is only included in the array to make positioning this easier
                            Label::new(cx, "dBFS")
                                .class("ticks__label")
                                .class("ticks__label--dbfs")
                        } else {
                            Label::new(cx, &tick_db.to_string()).class("ticks__label")
                        }
                        .overflow(Overflow::Visible);

                        if needs_minus_offset {
                            label.child_right(Pixels(font_size * 0.15));
                        }
                    })
                    .height(Stretch(1.0))
                    .left(Percentage(tick_pct - (WIDTH_PCT / 2.0)))
                    .width(Percentage(WIDTH_PCT))
                    .child_left(Stretch(1.0))
                    .child_right(Stretch(1.0))
                    .overflow(Overflow::Visible);
                }
            })
            .class("ticks")
            .overflow(Overflow::Visible);
        })
        .overflow(Overflow::Visible)
    }
}

impl View for DualMeter {
    fn element(&self) -> Option<&'static str> {
        Some("dual-meter")
    }
}

impl<L, P> View for PeakMeterBar<L, P>
where
    L: Lens<Target = f32>,
    P: Lens<Target = f32>,
{
    fn draw(&self, cx: &mut DrawContext, canvas: &mut Canvas) {
        let level_dbfs = self.level_dbfs.get(cx);
        let peak_dbfs = self.peak_dbfs.get(cx);

        // Basic setup
        let bounds = cx.bounds();
        if bounds.w == 0.0 || bounds.h == 0.0 {
            return;
        }

        let background_color = cx.background_color().into();
        let border_color = cx.border_color().into();
        let font_color = cx.font_color().into();
        let border_width = cx.border_width();
        let outline_width = cx.outline_width();

        let mut path = vg::Path::new();
        {
            let x = bounds.x + border_width / 2.0;
            let y = bounds.y + border_width / 2.0;
            let w = bounds.w - border_width;
            let h = bounds.h - border_width;
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

        // NOTE: We'll scale this with the nearest integer DPI ratio. That way it will still look
        //       good at 2x scaling, and it won't look blurry at 1.x times scaling.
        let dpi_scale = cx.logical_to_physical(1.0).floor().max(1.0);

        // Draw solid bar up to current level_dbfs
        let bar_bounds = bounds.shrink(border_width / 2.0);
        let db_to_x_coord = |db: f32| {
            let tick_fraction = (db - MIN_TICK) / (MAX_TICK - MIN_TICK);
            bar_bounds.left() + (bar_bounds.width() * tick_fraction).round()
        };

        let level_x = db_to_x_coord(level_dbfs);
        if level_dbfs > MIN_TICK {
            let mut filled_path = vg::Path::new();
            filled_path.move_to(bar_bounds.left(), bar_bounds.top());
            filled_path.line_to(level_x, bar_bounds.top());
            filled_path.line_to(level_x, bar_bounds.bottom());
            filled_path.line_to(bar_bounds.left(), bar_bounds.bottom());
            filled_path.close();

            let fill_paint = vg::Paint::color(border_color);
            canvas.fill_path(&filled_path, &fill_paint);
        }

        // Draw the hold peak value if the hold time option has been set
        if (MIN_TICK..MAX_TICK).contains(&peak_dbfs) {
            // femtovg draws paths centered on these coordinates, so in order to be pixel perfect we
            // need to account for that. Otherwise the ticks will be 2px wide instead of 1px.
            let peak_x = db_to_x_coord(peak_dbfs);
            let mut path = vg::Path::new();
            path.move_to(peak_x + (dpi_scale / 2.0), bar_bounds.top());
            path.line_to(peak_x + (dpi_scale / 2.0), bar_bounds.bottom());

            let mut paint = vg::Paint::color(font_color);
            paint.set_line_width(outline_width);
            canvas.stroke_path(&path, &paint);
        }

        // Draw border last
        let mut paint = vg::Paint::color(border_color);
        paint.set_line_width(border_width);
        canvas.stroke_path(&path, &paint);
    }
}
