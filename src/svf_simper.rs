// slightly adapted from https://github.com/wrl/baseplug/blob/trunk/examples/svf/svf_simper.rs
// the original only outputs the lowpass, I need both the low and the high-pass
// Thanks, Will!

/*
copyright William Light 2024

Permission is hereby granted, free of charge, to any
person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the
Software without restriction, including without
limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

*/

// implemented from https://cytomic.com/files/dsp/SvfLinearTrapOptimised2.pdf
// thanks, andy!

use std::f32::consts;

use std::simd::f32x4;
/// for early exit of shelving filters
const ALMOST_UNITY_GAIN: f32x4 = f32x4::from_array([0.999; 4]);

#[derive(Debug, Clone)]
pub struct SVFSimper {
    pub a1: f32x4,
    pub a2: f32x4,
    pub a3: f32x4,

    pub ic1eq: f32x4,
    pub ic2eq: f32x4,
}

impl SVFSimper {
    pub fn new(cutoff: f32, resonance: f32, sample_rate: f32) -> Self {
        let g = (consts::PI * (cutoff / sample_rate)).tan();
        // let k = 2f32 - (1.9f32 * resonance.min(1f32).max(0f32));
        let k = 2.0f32.mul_add(-resonance.clamp(0.0, 1.0), 2f32);

        let a1 = 1.0 / g.mul_add(g + k, 1.0);
        let a2 = g * a1;
        let a3 = g * a2;

        Self {
            a1: f32x4::splat(a1),
            a2: f32x4::splat(a2),
            a3: f32x4::splat(a3),

            ic1eq: f32x4::splat(0.0),
            ic2eq: f32x4::splat(0.0),
        }
    }

    pub fn set(&mut self, cutoff: f32, resonance: f32, sample_rate: f32) {
        let new = Self::new(cutoff, resonance, sample_rate);

        self.a1 = new.a1;
        self.a2 = new.a2;
        self.a3 = new.a3;
    }
    #[allow(dead_code)]
    #[inline]
    pub fn lowpass(&mut self, v0: f32x4) -> f32x4 {
        let v3 = v0 - self.ic2eq;
        let v1 = (self.a1 * self.ic1eq) + (self.a2 * v3);
        let v2 = self.ic2eq + (self.a2 * self.ic1eq) + (self.a3 * v3);

        self.ic1eq = (f32x4::splat(2.0) * v1) - self.ic1eq;
        self.ic2eq = (f32x4::splat(2.0) * v2) - self.ic2eq;

        v2
    }
    #[inline]
    pub fn highpass(&mut self, v0: f32x4) -> f32x4 {
        let v3 = v0 - self.ic2eq;
        let v1 = (self.a1 * self.ic1eq) + (self.a2 * v3);
        let v2 = self.ic2eq + (self.a2 * self.ic1eq) + (self.a3 * v3);

        self.ic1eq = (f32x4::splat(2.0) * v1) - self.ic1eq;
        self.ic2eq = (f32x4::splat(2.0) * v2) - self.ic2eq;

        // should be this:
        // v0 - k * v1 - v2
        // but we don't have k
        v0 - v2
    }

    // TODO:
    // last 0.1dB: fade to input,
    // at 0.0dB: early exit
    #[inline]
    pub fn highshelf(&mut self, v0: f32x4, lin_gain: f32x4) -> f32x4 {
        // Fast path for unity gain
        // we do not use this EQ for boosts
        if lin_gain > ALMOST_UNITY_GAIN {
            return v0;
        }
        let v3 = v0 - self.ic2eq;
        let v1 = (self.a1 * self.ic1eq) + (self.a2 * v3);
        let v2 = self.ic2eq + (self.a2 * self.ic1eq) + (self.a3 * v3);

        self.ic1eq = (f32x4::splat(2.0) * v1) - self.ic1eq;
        self.ic2eq = (f32x4::splat(2.0) * v2) - self.ic2eq;

        v2 + (lin_gain * (v0 - v2))
    }
}
