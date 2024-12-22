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

#[derive(Debug, Clone)]
pub struct SVFSimper {
    pub a1: f32x4,
    pub a2: f32x4,
    pub a3: f32x4,

    pub ic1eq: f32x4,
    pub ic2eq: f32x4,
    sr_recip_pi: f32x4,
}

impl SVFSimper {
    pub fn new(cutoff: f32, resonance: f32, sample_rate: f32) -> Self {
        let sr_recip_pi = consts::PI * sample_rate.recip();
        let (a1, a2, a3) = Self::compute_parameters(cutoff, resonance, sr_recip_pi);

        Self {
            a1: f32x4::splat(a1),
            a2: f32x4::splat(a2),
            a3: f32x4::splat(a3),

            ic1eq: f32x4::splat(0.0),
            ic2eq: f32x4::splat(0.0),
            sr_recip_pi: f32x4::splat(sr_recip_pi),
        }
    }

    pub fn reset(&mut self, cutoff: f32, resonance: f32, sample_rate: f32) {
        let sr_recip_pi = consts::PI * sample_rate.recip();
        self.sr_recip_pi = f32x4::splat(sr_recip_pi);
        self.ic1eq = f32x4::splat(0.0);
        self.ic2eq = f32x4::splat(0.0);

        self.set(cutoff, resonance);
    }

    #[inline]
    pub fn set(&mut self, cutoff: f32, resonance: f32) {
        let sr_recip_pi = self.sr_recip_pi[0]; // Use the precomputed value
        let (a1, a2, a3) = Self::compute_parameters(cutoff, resonance, sr_recip_pi);

        self.a1 = f32x4::splat(a1);
        self.a2 = f32x4::splat(a2);
        self.a3 = f32x4::splat(a3);
    }
    #[inline]
    fn compute_parameters(cutoff: f32, resonance: f32, sr_recip_pi: f32) -> (f32, f32, f32) {
        let g = (cutoff * sr_recip_pi).tan();
        let k = 2f32 * (1.0 - resonance.clamp(0.0, 1.0));

        let a1 = 1.0 / (g * (g + k) + 1.0);
        let a2 = g * a1;
        let a3 = g * a2;

        (a1, a2, a3)
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
        let v3 = v0 - self.ic2eq;
        let v1 = (self.a1 * self.ic1eq) + (self.a2 * v3);
        let v2 = self.ic2eq + (self.a2 * self.ic1eq) + (self.a3 * v3);

        self.ic1eq = (f32x4::splat(2.0) * v1) - self.ic1eq;
        self.ic2eq = (f32x4::splat(2.0) * v2) - self.ic2eq;

        v2 + (lin_gain * (v0 - v2))
    }
}
/*

make separate:
reset(&mut self, cutoff: f32, resonance: f32, sample_rate: f32)
set(&mut self, cutoff: f32, resonance: f32)

precompute sr_recip_pi

implement all filter types

make nonlin variants

make set_x4

make wider variants
iiuc, my cpu can do f32x8 and M1 macs can do f32x16

use wider

 */
