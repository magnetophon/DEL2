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
use std::marker::PhantomData;
use std::simd::num::SimdFloat;
use std::simd::{LaneCount, Simd, StdFloat, SupportedLaneCount};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// pub trait SimdOps<const LANES: usize> {
//     type Vector;

//     unsafe fn load(ptr: *const f32) -> Self::Vector;
//     unsafe fn store(ptr: *mut f32, a: Self::Vector);
//     unsafe fn set1(val: f32) -> Self::Vector;
//     unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector;
//     unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector;
// }

// Add SIMD implementation enum
#[derive(Clone, Copy, Debug)]
enum SimdImpl {
    Avx512,
    Avx2,
    Sse2,
    Neon,
    Fallback,
}

impl SimdImpl {
    fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return SimdImpl::Avx512;
            }
            if is_x86_feature_detected!("avx2") {
                return SimdImpl::Avx2;
            }
            if is_x86_feature_detected!("sse2") {
                return SimdImpl::Sse2;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if is_aarch64_feature_detected!("neon") {
                return SimdImpl::Neon;
            }
        }
        SimdImpl::Fallback
    }
}

pub trait SimdOps {
    type Vector;

    unsafe fn load(ptr: *const f32) -> Self::Vector;
    unsafe fn store(ptr: *mut f32, a: Self::Vector);
    unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector;
    unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector;
    unsafe fn mul_add(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector;
    unsafe fn splat(val: f32) -> Self::Vector;
}

#[cfg(target_arch = "x86_64")]
mod avx512_ops {
    use super::*;
    pub struct Avx512;

    impl SimdOps for Avx512 {
        type Vector = __m512;

        #[inline(always)]
        unsafe fn load(ptr: *const f32) -> Self::Vector {
            _mm512_loadu_ps(ptr)
        }

        #[inline(always)]
        unsafe fn store(ptr: *mut f32, a: Self::Vector) {
            _mm512_storeu_ps(ptr, a)
        }

        #[inline(always)]
        unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
            _mm512_add_ps(a, b)
        }

        #[inline(always)]
        unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
            _mm512_mul_ps(a, b)
        }

        #[inline(always)]
        unsafe fn mul_add(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
            _mm512_fmadd_ps(a, b, c)
        }

        #[inline(always)]
        unsafe fn splat(val: f32) -> Self::Vector {
            _mm512_set1_ps(val)
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2_ops {
    use super::*;
    pub struct Avx2;

    impl SimdOps for Avx2 {
        type Vector = __m256;

        #[inline(always)]
        unsafe fn load(ptr: *const f32) -> Self::Vector {
            _mm256_loadu_ps(ptr)
        }

        #[inline(always)]
        unsafe fn store(ptr: *mut f32, a: Self::Vector) {
            _mm256_storeu_ps(ptr, a)
        }

        #[inline(always)]
        unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
            _mm256_add_ps(a, b)
        }

        #[inline(always)]
        unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
            _mm256_mul_ps(a, b)
        }

        #[inline(always)]
        unsafe fn mul_add(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
            _mm256_fmadd_ps(a, b, c)
        }

        #[inline(always)]
        unsafe fn splat(val: f32) -> Self::Vector {
            _mm256_set1_ps(val)
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod sse2_ops {
    use super::*;
    pub struct Sse2;

    impl SimdOps for Sse2 {
        type Vector = __m128;

        #[inline(always)]
        unsafe fn load(ptr: *const f32) -> Self::Vector {
            _mm_loadu_ps(ptr)
        }

        #[inline(always)]
        unsafe fn store(ptr: *mut f32, a: Self::Vector) {
            _mm_storeu_ps(ptr, a)
        }

        #[inline(always)]
        unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
            _mm_add_ps(a, b)
        }

        #[inline(always)]
        unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
            _mm_mul_ps(a, b)
        }

        #[inline(always)]
        unsafe fn mul_add(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
            // SSE2 doesn't have FMA, so we do it manually
            _mm_add_ps(_mm_mul_ps(a, b), c)
        }

        #[inline(always)]
        unsafe fn splat(val: f32) -> Self::Vector {
            _mm_set1_ps(val)
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon_ops {
    use super::*;
    pub struct Neon;

    impl SimdOps for Neon {
        type Vector = float32x4_t;

        #[inline(always)]
        unsafe fn load(ptr: *const f32) -> Self::Vector {
            vld1q_f32(ptr)
        }

        #[inline(always)]
        unsafe fn store(ptr: *mut f32, a: Self::Vector) {
            vst1q_f32(ptr, a)
        }

        #[inline(always)]
        unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
            vaddq_f32(a, b)
        }

        #[inline(always)]
        unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
            vmulq_f32(a, b)
        }

        #[inline(always)]
        unsafe fn mul_add(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
            vfmaq_f32(c, a, b)
        }

        #[inline(always)]
        unsafe fn splat(val: f32) -> Self::Vector {
            vdupq_n_f32(val)
        }
    }
}

// First, define a trait for the filter behavior
pub trait FilterBehavior {
    fn process_state<const LANES: usize>(
        v1: Simd<f32, LANES>,
        v2: Simd<f32, LANES>,
    ) -> (Simd<f32, LANES>, Simd<f32, LANES>)
    where
        LaneCount<LANES>: SupportedLaneCount;
}

// Implement linear behavior
pub struct Linear;
impl FilterBehavior for Linear {
    #[inline]
    fn process_state<const LANES: usize>(
        v1: Simd<f32, LANES>,
        v2: Simd<f32, LANES>,
    ) -> (Simd<f32, LANES>, Simd<f32, LANES>)
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        ((Simd::splat(2.0) * v1), (Simd::splat(2.0) * v2))
    }
}

// Implement nonlinear behavior
pub struct NonLinear;
impl FilterBehavior for NonLinear {
    #[inline]
    fn process_state<const LANES: usize>(
        v1: Simd<f32, LANES>,
        v2: Simd<f32, LANES>,
    ) -> (Simd<f32, LANES>, Simd<f32, LANES>)
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        (
            SVFSimper::<LANES>::fast_tanh(Simd::splat(2.0) * v1),
            SVFSimper::<LANES>::fast_tanh(Simd::splat(2.0) * v2),
        )
    }
}

// Modify SVFSimper to be generic over the behavior
#[derive(Debug, Clone)]
pub struct SVFSimper<const LANES: usize, B: FilterBehavior = Linear>
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
    _behavior: PhantomData<B>,
}

impl<const LANES: usize, B: FilterBehavior> SVFSimper<LANES, B>
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
            _behavior: PhantomData,
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
        let pi_over_sr = self.pi_over_sr[0];
        let (k, a1, a2, a3) = Self::compute_parameters(cutoff, resonance, pi_over_sr);

        self.k = Simd::splat(k);
        self.a1 = Simd::splat(a1);
        self.a2 = Simd::splat(a2);
        self.a3 = Simd::splat(a3);
    }

    #[inline]
    pub fn set_simd(&mut self, cutoff: Simd<f32, LANES>, resonance: Simd<f32, LANES>) {
        let (k, a1, a2, a3) = Self::compute_parameters_simd(cutoff, resonance, self.pi_over_sr);

        self.k = k;
        self.a1 = a1;
        self.a2 = a2;
        self.a3 = a3;
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn filter_avx512(&mut self, input: __m512) -> __m512 {
        let v0 = input;
        let v3 = _mm512_loadu_ps(self.ic2eq.to_array().as_ptr());

        let ic1 = _mm512_loadu_ps(self.ic1eq.to_array().as_ptr());
        let a1_ic1 = _mm512_mul_ps(_mm512_loadu_ps(self.a1.to_array().as_ptr()), ic1);
        let a2_v3 = _mm512_mul_ps(_mm512_loadu_ps(self.a2.to_array().as_ptr()), v3);
        let sum = _mm512_add_ps(a1_ic1, a2_v3);
        let v1 = _mm512_fmadd_ps(sum, _mm512_loadu_ps(self.k.to_array().as_ptr()), v0);

        let v1k = _mm512_mul_ps(v1, _mm512_loadu_ps(self.k.to_array().as_ptr()));
        let v2 = _mm512_fmadd_ps(ic1, _mm512_loadu_ps(self.a2.to_array().as_ptr()), v1k);

        let two = _mm512_set1_ps(2.0);
        self.ic1eq = Simd::from_array({
            let mut arr = [0.0; LANES];
            _mm512_storeu_ps(arr.as_mut_ptr(), _mm512_sub_ps(_mm512_mul_ps(v1, two), ic1));
            arr
        });
        self.ic2eq = Simd::from_array({
            let mut arr = [0.0; LANES];
            _mm512_storeu_ps(arr.as_mut_ptr(), _mm512_sub_ps(_mm512_mul_ps(v2, two), v3));
            arr
        });

        v2
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn filter_avx2(&mut self, input: __m256) -> __m256 {
        let v0 = input;
        let v3 = _mm256_loadu_ps(self.ic2eq.to_array().as_ptr());

        let ic1 = _mm256_loadu_ps(self.ic1eq.to_array().as_ptr());
        let a1_ic1 = _mm256_mul_ps(_mm256_loadu_ps(self.a1.to_array().as_ptr()), ic1);
        let a2_v3 = _mm256_mul_ps(_mm256_loadu_ps(self.a2.to_array().as_ptr()), v3);
        let sum = _mm256_add_ps(a1_ic1, a2_v3);
        let v1 = _mm256_fmadd_ps(sum, _mm256_loadu_ps(self.k.to_array().as_ptr()), v0);

        let v1k = _mm256_mul_ps(v1, _mm256_loadu_ps(self.k.to_array().as_ptr()));
        let v2 = _mm256_fmadd_ps(ic1, _mm256_loadu_ps(self.a2.to_array().as_ptr()), v1k);

        let two = _mm256_set1_ps(2.0);
        self.ic1eq = Simd::from_array({
            let mut arr = [0.0; LANES];
            _mm256_storeu_ps(arr.as_mut_ptr(), _mm256_sub_ps(_mm256_mul_ps(v1, two), ic1));
            arr
        });
        self.ic2eq = Simd::from_array({
            let mut arr = [0.0; LANES];
            _mm256_storeu_ps(arr.as_mut_ptr(), _mm256_sub_ps(_mm256_mul_ps(v2, two), v3));
            arr
        });

        v2
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse2")]
    unsafe fn filter_sse2(&mut self, input: __m128) -> __m128 {
        let v0 = input;
        let v3 = _mm_loadu_ps(self.ic2eq.to_array().as_ptr());

        let ic1 = _mm_loadu_ps(self.ic1eq.to_array().as_ptr());
        let a1_ic1 = _mm_mul_ps(_mm_loadu_ps(self.a1.to_array().as_ptr()), ic1);
        let a2_v3 = _mm_mul_ps(_mm_loadu_ps(self.a2.to_array().as_ptr()), v3);
        let sum = _mm_add_ps(a1_ic1, a2_v3);
        let k = _mm_loadu_ps(self.k.to_array().as_ptr());
        let v1 = _mm_add_ps(_mm_mul_ps(sum, k), v0);

        let v1k = _mm_mul_ps(v1, k);
        let v2 = _mm_add_ps(
            _mm_mul_ps(ic1, _mm_loadu_ps(self.a2.to_array().as_ptr())),
            v1k,
        );

        let two = _mm_set1_ps(2.0);
        self.ic1eq = Simd::from_array({
            let mut arr = [0.0; LANES];
            _mm_storeu_ps(arr.as_mut_ptr(), _mm_sub_ps(_mm_mul_ps(v1, two), ic1));
            arr
        });
        self.ic2eq = Simd::from_array({
            let mut arr = [0.0; LANES];
            _mm_storeu_ps(arr.as_mut_ptr(), _mm_sub_ps(_mm_mul_ps(v2, two), v3));
            arr
        });

        v2
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn filter_neon(&mut self, input: float32x4_t) -> float32x4_t {
        let v0 = input;
        let v3 = vld1q_f32(self.ic2eq.to_array().as_ptr());

        let ic1 = vld1q_f32(self.ic1eq.to_array().as_ptr());
        let a1_ic1 = vmulq_f32(vld1q_f32(self.a1.to_array().as_ptr()), ic1);
        let a2_v3 = vmulq_f32(vld1q_f32(self.a2.to_array().as_ptr()), v3);
        let sum = vaddq_f32(a1_ic1, a2_v3);
        let k = vld1q_f32(self.k.to_array().as_ptr());
        let v1 = vfmaq_f32(v0, sum, k);

        let v1k = vmulq_f32(v1, k);
        let v2 = vfmaq_f32(v1k, ic1, vld1q_f32(self.a2.to_array().as_ptr()));

        let two = vdupq_n_f32(2.0);
        self.ic1eq = Simd::from_array({
            let mut arr = [0.0; LANES];
            vst1q_f32(arr.as_mut_ptr(), vsubq_f32(vmulq_f32(v1, two), ic1));
            arr
        });
        self.ic2eq = Simd::from_array({
            let mut arr = [0.0; LANES];
            vst1q_f32(arr.as_mut_ptr(), vsubq_f32(vmulq_f32(v2, two), v3));
            arr
        });

        v2
    }

    #[inline]
    fn compute_parameters(cutoff: f32, resonance: f32, pi_over_sr: f32) -> (f32, f32, f32, f32) {
        let g = (cutoff * pi_over_sr).tan();
        let k = 2.0 * (1.0 - resonance);
        let a1 = g.mul_add(g + k, 1.0).recip();
        let a2 = g * a1;
        let a3 = g * a2;

        (k, a1, a2, a3)
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

        let (new_ic1, new_ic2) = B::process_state(v1, v2);
        self.ic1eq = new_ic1 - self.ic1eq;
        self.ic2eq = new_ic2 - self.ic2eq;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svf_simper() {
        let sample_rate = 48000.0;
        let test_freqs = [110.0, 220.0, 440.0, 880.0, 1760.0, 3520.0];
        let resonances = [0.0, 0.5, 0.7, 0.9];
        let cutoff = 1000.0;

        for &resonance in &resonances {
            let mut filter = SVFSimper::<4>::new(cutoff, resonance, sample_rate);

            // Verify k coefficient matches expected value
            let expected_k = 2.0 * (1.0 - resonance);
            assert!(
                (filter.k[0] - expected_k).abs() < 1e-6,
                "k coefficient incorrect for resonance {}: expected {}, got {}",
                resonance,
                expected_k,
                filter.k[0]
            );

            // Test frequency response
            for &freq in &test_freqs {
                let num_samples = (sample_rate / freq * 10.0) as usize;
                let mut power_sum_in = 0.0;
                let mut power_sum_out = 0.0;

                // Warm up filter
                for i in 0..(2.0 * sample_rate / freq) as usize {
                    let t = i as f32 / sample_rate;
                    let input = (2.0 * std::f32::consts::PI * freq * t).sin();
                    filter.process(Simd::splat(input));
                }

                // Measure response
                for i in 0..num_samples {
                    let t = i as f32 / sample_rate;
                    let input = (2.0 * std::f32::consts::PI * freq * t).sin();
                    let (_, lp) = filter.process(Simd::splat(input));
                    let output = lp.to_array()[0];

                    power_sum_in += input * input;
                    power_sum_out += output * output;
                }

                let rms_in = (power_sum_in / num_samples as f32).sqrt();
                let rms_out = (power_sum_out / num_samples as f32).sqrt();
                let ratio = rms_out / rms_in;
                let db = 20.0 * ratio.log10();

                // Verify filter behavior based on resonance and frequency
                if freq < cutoff / 2.0 {
                    // Check passband (should have minimal attenuation)
                    assert!(
                        db > -3.0,
                        "Too much attenuation in passband: {} dB at {} Hz (resonance {})",
                        db,
                        freq,
                        resonance
                    );
                } else if freq > cutoff * 2.0 {
                    // Check stopband (should have significant attenuation)
                    assert!(
                        db < -12.0,
                        "Insufficient attenuation in stopband: {} dB at {} Hz (resonance {})",
                        db,
                        freq,
                        resonance
                    );
                }

                // For high resonance, verify peak at cutoff
                if resonance > 0.7 && (freq as f32 - cutoff).abs() < cutoff * 0.1 {
                    assert!(
                        db > 0.0,
                        "Expected resonant peak near cutoff for resonance {}",
                        resonance
                    );
                }
            }
        }
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
