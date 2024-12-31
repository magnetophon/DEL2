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
    dispatch: FilterDispatch<LANES>,
}
// Separate enum to hold the runtime-selected implementation
#[derive(Debug, Clone)]
enum FilterDispatch<const LANES: usize> {
    #[cfg(target_arch = "x86_64")]
    Avx512,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Sse2,
    #[cfg(target_arch = "aarch64")]
    Neon,
    Generic,
}

impl<const LANES: usize, B: FilterBehavior> SVFSimper<LANES, B>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    pub fn new(cutoff: f32, resonance: f32, sample_rate: f32) -> Self {
        let pi_over_sr = consts::PI / sample_rate;
        let (k, a1, a2, a3) = Self::compute_parameters(cutoff, resonance, pi_over_sr);

        // Select the best available implementation at initialization
        let dispatch = {
            #[cfg(target_arch = "x86_64")]
            {
                if LANES == 16 && is_x86_feature_detected!("avx512f") {
                    FilterDispatch::Avx512
                } else if LANES == 8 && is_x86_feature_detected!("avx2") {
                    FilterDispatch::Avx2
                } else if LANES == 4 && is_x86_feature_detected!("sse2") {
                    FilterDispatch::Sse2
                } else {
                    FilterDispatch::Generic
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                if LANES == 4 && is_aarch64_feature_detected!("neon") {
                    FilterDispatch::Neon
                } else {
                    FilterDispatch::Generic
                }
            }
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                FilterDispatch::Generic
            }
        };

        Self {
            k: Simd::splat(k),
            a1: Simd::splat(a1),
            a2: Simd::splat(a2),
            a3: Simd::splat(a3),
            ic1eq: Simd::splat(0.0),
            ic2eq: Simd::splat(0.0),
            pi_over_sr: Simd::splat(pi_over_sr),
            dispatch,
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
    #[inline(always)]
    unsafe fn filter_sse2(&self, v0: __m128) -> __m128 {
        use std::arch::x86_64::*;

        // Load constants once (they're the same for all lanes)
        let k = _mm_set1_ps(self.k.as_array()[0]);
        let a1 = _mm_set1_ps(self.a1.as_array()[0]);
        let a2 = _mm_set1_ps(self.a2.as_array()[0]);
        let a3 = _mm_set1_ps(self.a3.as_array()[0]);

        // Load state
        let ic1eq = _mm_loadu_ps(self.ic1eq.as_array().as_ptr());
        let ic2eq = _mm_loadu_ps(self.ic2eq.as_array().as_ptr());

        // v1 = ic1eq + k * (v0 - ic2eq)
        let v1 = _mm_add_ps(ic1eq, _mm_mul_ps(k, _mm_sub_ps(v0, ic2eq)));

        // v2 = ic2eq + k * v1
        let v2 = _mm_add_ps(ic2eq, _mm_mul_ps(k, v1));

        // Update state variables
        // ic1eq = a1 * v1 + a2 * v0
        _mm_storeu_ps(
            self.ic1eq.as_array().as_ptr() as *mut f32,
            _mm_add_ps(_mm_mul_ps(a1, v1), _mm_mul_ps(a2, v0)),
        );

        // ic2eq = a2 * v1 + a3 * v0
        _mm_storeu_ps(
            self.ic2eq.as_array().as_ptr() as *mut f32,
            _mm_add_ps(_mm_mul_ps(a2, v1), _mm_mul_ps(a3, v0)),
        );

        v2
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn filter_avx2(&self, v0: __m256) -> __m256 {
        use std::arch::x86_64::*;

        // Load constants once
        let k = _mm256_set1_ps(self.k.as_array()[0]);
        let a1 = _mm256_set1_ps(self.a1.as_array()[0]);
        let a2 = _mm256_set1_ps(self.a2.as_array()[0]);
        let a3 = _mm256_set1_ps(self.a3.as_array()[0]);

        // Load state
        let ic1eq = _mm256_loadu_ps(self.ic1eq.as_array().as_ptr());
        let ic2eq = _mm256_loadu_ps(self.ic2eq.as_array().as_ptr());

        // v1 = ic1eq + k * (v0 - ic2eq)
        let v1 = _mm256_add_ps(ic1eq, _mm256_mul_ps(k, _mm256_sub_ps(v0, ic2eq)));

        // v2 = ic2eq + k * v1
        let v2 = _mm256_add_ps(ic2eq, _mm256_mul_ps(k, v1));

        // Update state variables
        // ic1eq = a1 * v1 + a2 * v0
        _mm256_storeu_ps(
            self.ic1eq.as_array().as_ptr() as *mut f32,
            _mm256_add_ps(_mm256_mul_ps(a1, v1), _mm256_mul_ps(a2, v0)),
        );

        // ic2eq = a2 * v1 + a3 * v0
        _mm256_storeu_ps(
            self.ic2eq.as_array().as_ptr() as *mut f32,
            _mm256_add_ps(_mm256_mul_ps(a2, v1), _mm256_mul_ps(a3, v0)),
        );

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
    fn process_generic(&mut self, v0: Simd<f32, LANES>) -> (Simd<f32, LANES>, Simd<f32, LANES>) {
        let v3 = v0 - self.ic2eq;
        let v1 = (self.a1 * self.ic1eq) + (self.a2 * v3);
        let v2 = self.ic2eq + (self.a2 * self.ic1eq) + (self.a3 * v3);

        let (new_ic1, new_ic2) = B::process_state(v1, v2);
        self.ic1eq = new_ic1 - self.ic1eq;
        self.ic2eq = new_ic2 - self.ic2eq;

        (v1, v2)
    }
    #[inline(always)]
    pub fn process(&mut self, v0: Simd<f32, LANES>) -> (Simd<f32, LANES>, Simd<f32, LANES>) {
        match self.dispatch {
            #[cfg(target_arch = "x86_64")]
            FilterDispatch::Avx512 => unsafe {
                let result = self.filter_avx512(_mm512_loadu_ps(v0.as_array().as_ptr()));
                let mut output_array = [0.0; LANES];
                _mm512_storeu_ps(output_array.as_mut_ptr(), result);
                let output = Simd::from_array(output_array);
                (output, output)
            },
            #[cfg(target_arch = "x86_64")]
            FilterDispatch::Avx2 => unsafe {
                let result = self.filter_avx2(_mm256_loadu_ps(v0.as_array().as_ptr()));
                let mut output_array = [0.0; LANES];
                _mm256_storeu_ps(output_array.as_mut_ptr(), result);
                let output = Simd::from_array(output_array);
                (output, output)
            },
            #[cfg(target_arch = "x86_64")]
            FilterDispatch::Sse2 => unsafe {
                let result = self.filter_sse2(_mm_loadu_ps(v0.as_array().as_ptr()));
                let mut output_array = [0.0; LANES];
                _mm_storeu_ps(output_array.as_mut_ptr(), result);
                let output = Simd::from_array(output_array);
                (output, output)
            },
            #[cfg(target_arch = "aarch64")]
            FilterDispatch::Neon => unsafe {
                let result = self.filter_neon(vld1q_f32(v0.as_array().as_ptr()));
                let mut output_array = [0.0; LANES];
                vst1q_f32(output_array.as_mut_ptr(), result);
                let output = Simd::from_array(output_array);
                (output, output)
            },
            FilterDispatch::Generic => self.process_generic(v0),
        }
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
    use std::hint::black_box;
    use std::time::Instant;

    fn get_cpu_info() -> String {
        #[cfg(target_os = "linux")]
        {
            if let Ok(contents) = std::fs::read_to_string("/proc/cpuinfo") {
                if let Some(model_line) = contents.lines().find(|l| l.starts_with("model name")) {
                    return model_line
                        .split(":")
                        .nth(1)
                        .unwrap_or("Unknown")
                        .trim()
                        .to_string();
                }
            }
        }
        "Unknown CPU".to_string()
    }

    fn print_cpu_features() {
        #[cfg(target_arch = "x86_64")]
        {
            println!("\nCPU Features:");
            println!("  AVX512: {}", is_x86_feature_detected!("avx512f"));
            println!("  AVX2:   {}", is_x86_feature_detected!("avx2"));
            println!("  SSE2:   {}", is_x86_feature_detected!("sse2"));
        }
    }

    fn pin_to_performance_core() {
        #[cfg(target_os = "linux")]
        {
            use core_affinity::CoreId;
            if let Some(core_ids) = core_affinity::get_core_ids() {
                println!("\nAvailable cores: {}", core_ids.len());
                if let Some(core_id) = core_ids.first() {
                    core_affinity::set_for_current(*core_id);
                    println!("Pinned to core {}", core_id.id);
                }
            }
        }
    }

    #[test]
    fn benchmark_filter() {
        println!("\nBenchmarking on: {}", get_cpu_info());
        println!("Current Date and Time (UTC): {}", chrono::Utc::now());
        if let Ok(user) = std::env::var("USER") {
            println!("Current User's Login: {}", user);
        }

        print_cpu_features();
        pin_to_performance_core();

        let sample_rate = 48000.0;
        let cutoff = 1000.0;
        let resonance = 0.5;

        // Test each lane width with both generic and SIMD implementations
        println!("\n=== LANES=4 ===");
        benchmark_comparison::<4>(sample_rate, cutoff, resonance);

        println!("\n=== LANES=8 ===");
        benchmark_comparison::<8>(sample_rate, cutoff, resonance);

        println!("\n=== LANES=16 ===");
        benchmark_comparison::<16>(sample_rate, cutoff, resonance);
    }

    fn benchmark_comparison<const LANES: usize>(sample_rate: f32, cutoff: f32, resonance: f32)
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        // Create two filters - one forced to generic, one using selected SIMD
        let mut filter_generic = SVFSimper::<LANES>::new(cutoff, resonance, sample_rate);
        let mut filter_simd = SVFSimper::<LANES>::new(cutoff, resonance, sample_rate);

        // Force generic implementation for comparison
        filter_generic.dispatch = FilterDispatch::Generic;

        let input = Simd::from_array([0.1; LANES]);

        println!("Generic implementation:");
        let generic_stats = run_benchmark(&mut filter_generic, input);

        println!("\nSIMD implementation ({:?}):", filter_simd.dispatch);
        let simd_stats = run_benchmark(&mut filter_simd, input);

        println!("\nComparison:");
        println!(
            "  Generic: {:.2}ns per sample",
            generic_stats.median / LANES as f64
        );
        println!(
            "  SIMD:    {:.2}ns per sample",
            simd_stats.median / LANES as f64
        );
        println!(
            "  Speedup: {:.2}x",
            generic_stats.median / simd_stats.median
        );
    }

    struct BenchmarkStats {
        median: f64,
        mean: f64,
        min: f64,
        max: f64,
        std_dev: f64,
    }

    fn run_benchmark<const LANES: usize>(
        filter: &mut SVFSimper<LANES>,
        input: Simd<f32, LANES>,
    ) -> BenchmarkStats
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        let num_iterations = 1_000_000;
        let num_runs = 10;
        let mut durations = Vec::with_capacity(num_runs);

        // Warm up
        for _ in 0..num_iterations {
            black_box(filter.process(black_box(input)));
        }

        // Actual benchmark runs
        for run in 1..=num_runs {
            let start = Instant::now();
            for _ in 0..num_iterations {
                black_box(filter.process(black_box(input)));
            }
            let duration = start.elapsed().as_nanos() as f64 / num_iterations as f64;
            durations.push(duration);
            println!("Run {}: {:.2}ns per {} samples", run, duration, LANES);
        }

        durations.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if num_runs % 2 == 0 {
            (durations[num_runs / 2 - 1] + durations[num_runs / 2]) / 2.0
        } else {
            durations[num_runs / 2]
        };

        let mean = durations.iter().sum::<f64>() / num_runs as f64;
        let variance = durations
            .iter()
            .map(|x| (x - mean) * (x - mean))
            .sum::<f64>()
            / num_runs as f64;
        let std_dev = variance.sqrt();

        println!("\nSummary:");
        println!(
            "  Median: {:.2}ns per {} samples ({:.2}ns per sample)",
            median,
            LANES,
            median / LANES as f64
        );
        println!(
            "  Mean:   {:.2}ns per {} samples ({:.2}ns per sample)",
            mean,
            LANES,
            mean / LANES as f64
        );
        println!(
            "  Min:    {:.2}ns per {} samples ({:.2}ns per sample)",
            durations[0],
            LANES,
            durations[0] / LANES as f64
        );
        println!(
            "  Max:    {:.2}ns per {} samples ({:.2}ns per sample)",
            durations[num_runs - 1],
            LANES,
            durations[num_runs - 1] / LANES as f64
        );
        println!("  Std Dev:{:.2}ns", std_dev);
        println!(
            "  Throughput: {:.2}M samples/second",
            1000.0 / (median / LANES as f64) / 1000.0
        );

        BenchmarkStats {
            median,
            mean,
            min: durations[0],
            max: durations[num_runs - 1],
            std_dev,
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
