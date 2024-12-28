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
use std::simd::num::SimdFloat;
use std::simd::{LaneCount, Simd, StdFloat, SupportedLaneCount};

#[derive(Debug, Clone)]
pub struct SVFSimper<const LANES: usize>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    pub k: Simd<f32, LANES>,
    pub a1: Simd<f32, LANES>,
    pub a2: Simd<f32, LANES>,
    pub a3: Simd<f32, LANES>,
    pub ic1eq: Simd<f32, LANES>,
    pub ic2eq: Simd<f32, LANES>,
    pi_over_sr: Simd<f32, LANES>,
}

impl<const LANES: usize> SVFSimper<LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    pub fn new(cutoff: f32, resonance: f32, sample_rate: f32) -> Self {
        let pi_over_sr = consts::PI / sample_rate;
        let (k, a1, a2, a3) = Self::compute_parameters(cutoff, resonance, pi_over_sr);

        Self {
            k: Simd::splat(k),
            a1: Simd::splat(a1),
            a2: Simd::splat(a2),
            a3: Simd::splat(a3),
            ic1eq: Simd::splat(0.0),
            ic2eq: Simd::splat(0.0),
            pi_over_sr: Simd::splat(pi_over_sr),
        }
    }

    pub fn reset(&mut self, cutoff: f32, resonance: f32, sample_rate: f32) {
        let pi_over_sr = consts::PI / sample_rate;
        self.pi_over_sr = Simd::splat(pi_over_sr);
        self.ic1eq = Simd::splat(0.0);
        self.ic2eq = Simd::splat(0.0);

        self.set(cutoff, resonance);
    }

    #[inline]
    pub fn set(&mut self, cutoff: f32, resonance: f32) {
        let pi_over_sr = self.pi_over_sr[0]; // Use the precomputed value
        let (k, a1, a2, a3) = Self::compute_parameters(cutoff, resonance, pi_over_sr);

        self.k = Simd::splat(k);
        self.a1 = Simd::splat(a1);
        self.a2 = Simd::splat(a2);
        self.a3 = Simd::splat(a3);
    }

    #[inline]
    fn compute_parameters(cutoff: f32, resonance: f32, pi_over_sr: f32) -> (f32, f32, f32, f32) {
        let g = (cutoff * pi_over_sr).tan();
        // let k = 2.0 * (1.0 - resonance.clamp(0.0, 1.0));
        // let k = (2.0 * (1.0 - resonance)).min(2.0);
        let k = 2.0 * (1.0 - resonance);
        let a1 = g.mul_add(g + k, 1.0).recip();
        let a2 = g * a1;
        let a3 = g * a2;

        (k, a1, a2, a3)
    }
    #[inline]
    pub fn set_simd(&mut self, cutoff: Simd<f32, LANES>, resonance: Simd<f32, LANES>) {
        let (k, a1, a2, a3) = Self::compute_parameters_simd(cutoff, resonance, self.pi_over_sr);

        self.k = k;
        self.a1 = a1;
        self.a2 = a2;
        self.a3 = a3;
    }

    #[inline]
    fn compute_parameters_simd(
        cutoff: Simd<f32, LANES>,
        resonance: Simd<f32, LANES>,
        pi_over_sr: Simd<f32, LANES>,
    ) -> (
        Simd<f32, LANES>,
        Simd<f32, LANES>,
        Simd<f32, LANES>,
        Simd<f32, LANES>,
    ) {
        let g = Self::fast_tan(cutoff * pi_over_sr);

        // let k = (Simd::splat(2.0) * (Simd::splat(1.0) - resonance.clamp(Simd::splat(0.0), Simd::splat(1.0))));
        let k = Simd::splat(2.0) * (Simd::splat(1.0) - resonance);

        let a1 = g.mul_add(g + k, Simd::splat(1.0)).recip();
        let a2 = g * a1;
        let a3 = g * a2;

        (k, a1, a2, a3)
    }
    #[inline]
    fn process(&mut self, v0: Simd<f32, LANES>) -> (Simd<f32, LANES>, Simd<f32, LANES>) {
        let v3 = v0 - self.ic2eq;
        let v1 = (self.a1 * self.ic1eq) + (self.a2 * v3);
        let v2 = self.ic2eq + (self.a2 * self.ic1eq) + (self.a3 * v3);

        // self.ic1eq = (Simd::splat(2.0) * v1) - self.ic1eq;
        self.ic1eq = Self::fast_tanh((Simd::splat(2.0) * v1) - self.ic1eq);
        // self.ic2eq = (Simd::splat(2.0) * v2) - self.ic2eq;
        self.ic2eq = Self::fast_tanh((Simd::splat(2.0) * v2) - self.ic2eq);

        (v1, v2)
    }

    #[inline]
    pub fn lowpass(&mut self, v0: Simd<f32, LANES>) -> Simd<f32, LANES> {
        let (_, v2) = self.process(v0);
        v2
    }

    #[inline]
    pub fn bandpass(&mut self, v0: Simd<f32, LANES>) -> Simd<f32, LANES> {
        let (v1, _) = self.process(v0);
        v1
    }
    #[inline]
    pub fn highpass(&mut self, v0: Simd<f32, LANES>) -> Simd<f32, LANES> {
        let (v1, v2) = self.process(v0);
        v0 - self.k * v1 - v2
    }
    #[inline]
    pub fn highpass_cheap(&mut self, v0: Simd<f32, LANES>) -> Simd<f32, LANES> {
        let (_, v2) = self.process(v0);
        v0 - v2
    }

    #[inline]
    pub fn notch(&mut self, v0: Simd<f32, LANES>) -> Simd<f32, LANES> {
        let (v1, _) = self.process(v0);
        v0 - self.k * v1
    }
    #[inline]
    pub fn peak(&mut self, v0: Simd<f32, LANES>) -> Simd<f32, LANES> {
        let (v1, v2) = self.process(v0);
        let two = Simd::splat(2.0);
        v0 - self.k * v1 - two * v2
    }
    #[inline]
    pub fn allpass(&mut self, v0: Simd<f32, LANES>) -> Simd<f32, LANES> {
        let (v1, _) = self.process(v0);
        let two = Simd::splat(2.0);
        v0 - two * self.k * v1
    }

    #[inline]
    pub fn lowshelf(
        &mut self,
        v0: Simd<f32, LANES>,
        lin_gain: Simd<f32, LANES>,
    ) -> Simd<f32, LANES> {
        let (v1, v2) = self.process(v0);
        // lin_gain.mul_add(v2, v0 - self.k * v1 - v2) - self.k * v1
        (lin_gain - Simd::splat(1.0)).mul_add(v2, v0) - (Simd::splat(2.0) * self.k * v1)
    }
    pub fn lowshelf_cheap(
        &mut self,
        v0: Simd<f32, LANES>,
        lin_gain: Simd<f32, LANES>,
    ) -> Simd<f32, LANES> {
        let (_, v2) = self.process(v0);
        lin_gain.mul_add(v2, v0 - v2)
    }
    #[inline]
    pub fn highshelf(
        &mut self,
        v0: Simd<f32, LANES>,
        lin_gain: Simd<f32, LANES>,
    ) -> Simd<f32, LANES> {
        let (v1, v2) = self.process(v0);
        lin_gain.mul_add(v0 - self.k * v1 - v2, v2) - self.k * v1
    }
    #[inline]
    pub fn highshelf_cheap(
        &mut self,
        v0: Simd<f32, LANES>,
        lin_gain: Simd<f32, LANES>,
    ) -> Simd<f32, LANES> {
        let (_, v2) = self.process(v0);
        lin_gain.mul_add(v0 - v2, v2)
    }

    // https://www.desmos.com/calculator/xj0nabg0we
    // x * (25.95+x * x) / (26.396+8.78 * x * x)
    // https://www.kvraudio.com/forum/viewtopic.php?p=7310333&sid=9308335d2247a9e996b48ab71d47c2bc#p7310333
    #[inline]
    pub fn fast_tanh(v: Simd<f32, LANES>) -> Simd<f32, LANES> {
        let square = v * v; // Element-wise squaring
        (v * (Simd::splat(25.95) + square) / (Simd::splat(26.396) + Simd::splat(8.78) * square))
            .simd_clamp(Simd::splat(-1.0), Simd::splat(1.0))
    }

    // quickerTanh / quickerTanh64 credits to mopo synthesis library:
    // Under GPLv3 or any later.
    // Little IO <littleioaudio@gmail.com>
    // Matt Tytel <matthewtytel@gmail.com>
    #[inline]
    pub fn quicker_tanh(v: Simd<f32, LANES>) -> Simd<f32, LANES> {
        let square = v * v; // Element-wise squaring
        v / (Simd::splat(1.0) + square / (Simd::splat(3.0) + square / Simd::splat(5.0)))
    }
    // fast tan(x) approximation
    // adapted from : https://github.com/AquaEBM/simd_util/blob/7b6b3aff8b828e79fe9c4c63a645efb5af327aea/src/math.rs#L13
    #[inline]
    pub fn fast_tan(x: Simd<f32, LANES>) -> Simd<f32, LANES> {
        // optimized into constants, hopefully
        let na = Simd::splat(1. / 15120.);
        let nb = Simd::splat(-1. / 36.);
        let nc = Simd::splat(1.);
        let da = Simd::splat(1. / 504.);
        let db = Simd::splat(-2. / 9.);
        let dc = Simd::splat(1.);

        let x2 = x * x;
        let num = x.mul_add(x2.mul_add(x2.mul_add(na, nb), nc), Simd::splat(0.));
        let den = x2.mul_add(x2.mul_add(da, db), dc);

        num / den
    }
}

/*


implement all filter types
see: git@github.com:AquaEBM/filte.rs.git
benchmark against it

make nonlin variants
https://discord.com/channels/590254806208217089/590657587939115048/1290733076972044461
that is exactly the thing you're trying to do: applying saturation to the bandpass output before it goes into the integrator that turns it into the lowpass output

make set_x4

use wider
iiuc, my cpu can do f32x8 and M1 macs can do f32x16
rust supports up to f32x64


make lerp that crossfades

look at olegs impl in faust:
mix is a dot mult
A = pow(10.0, G/40.0);

for matching taps to simd:
good enough?
- at startup, get nr_lanes supported by HW
- make enum[nr_lanes] containing simd_blocks for audio and params
- nr_iterations =  ((nr_taps*2)/nr_lanes).ceil()
- run filters
for i 0..nr_iterations {
  if is_smoothing {set(cf,res)}
  highpas
  lowpass
  high_shelf
}

optimal:
- binary count for number of taps, similar to faust slidingMin

 */
