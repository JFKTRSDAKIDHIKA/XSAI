// generated from xarch_softmax.h by LMUL substitution
#pragma once
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>



static inline float softmax_row_m1(float *row, size_t visible, size_t padded_keys, float scale) {
  const size_t vlmax = __riscv_vsetvlmax_e32m1();
  vfloat32m1_t lane_max = __riscv_vfmv_v_f_f32m1(-INFINITY, vlmax);
  for (size_t j = 0; j < visible;) {
    size_t vl = __riscv_vsetvl_e32m1(visible - j);
    lane_max = __riscv_vfmax_vv_f32m1_tu(lane_max, lane_max, __riscv_vle32_v_f32m1(row + j, vl), vl);
    j += vl;
  }
  const float maximum = __riscv_vfmv_f_s_f32m1_f32(
      __riscv_vfredmax_vs_f32m1_f32m1(lane_max, __riscv_vfmv_v_f_f32m1(-INFINITY, 1), vlmax));
  const float c = scale * XARCH_LOG2E;
  const vfloat32m1_t max_c = __riscv_vfmv_v_f_f32m1(maximum * c, vlmax);
  const vfloat32m1_t c5 = __riscv_vfmv_v_f_f32m1(1.339887440266574e-3f, vlmax);
  const vfloat32m1_t c4 = __riscv_vfmv_v_f_f32m1(9.618437357674640e-3f, vlmax);
  const vfloat32m1_t c3 = __riscv_vfmv_v_f_f32m1(5.550332471162809e-2f, vlmax);
  const vfloat32m1_t c2 = __riscv_vfmv_v_f_f32m1(2.402264791363012e-1f, vlmax);
  const vfloat32m1_t c1 = __riscv_vfmv_v_f_f32m1(6.931472028550421e-1f, vlmax);
  const vfloat32m1_t one = __riscv_vfmv_v_f_f32m1(1.0f, vlmax);
  vfloat32m1_t lane_sum = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
  for (size_t j = 0; j < visible;) {
    size_t vl = __riscv_vsetvl_e32m1(visible - j);
    vfloat32m1_t t = __riscv_vfmsub_vf_f32m1(__riscv_vle32_v_f32m1(row + j, vl), c, max_c, vl);
    t = __riscv_vfmax_vf_f32m1(t, -125.0f, vl);
    vint32m1_t n = __riscv_vfcvt_x_f_v_i32m1(t, vl);
    vfloat32m1_t f = __riscv_vfsub_vv_f32m1(t, __riscv_vfcvt_f_x_v_f32m1(n, vl), vl);
    vfloat32m1_t p = __riscv_vfmacc_vf_f32m1(c5, 1.535336188319500e-4f, f, vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, c4, vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, c3, vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, c2, vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, c1, vl);
    p = __riscv_vfmadd_vv_f32m1(p, f, one, vl);
    p = __riscv_vreinterpret_v_i32m1_f32m1(
        __riscv_vadd_vv_i32m1(__riscv_vreinterpret_v_f32m1_i32m1(p), __riscv_vsll_vx_i32m1(n, 23, vl), vl));
    __riscv_vse32_v_f32m1(row + j, p, vl);
    lane_sum = __riscv_vfadd_vv_f32m1_tu(lane_sum, lane_sum, p, vl);
    j += vl;
  }
  const vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
  for (size_t j = visible; j < padded_keys;) {
    size_t vl = __riscv_vsetvl_e32m1(padded_keys - j);
    __riscv_vse32_v_f32m1(row + j, zero, vl);
    j += vl;
  }
  return __riscv_vfmv_f_s_f32m1_f32(
      __riscv_vfredusum_vs_f32m1_f32m1(lane_sum, __riscv_vfmv_v_f_f32m1(0.0f, 1), vlmax));
}



static inline float softmax_row_m4(float *row, size_t visible, size_t padded_keys, float scale) {
  const size_t vlmax = __riscv_vsetvlmax_e32m4();
  vfloat32m4_t lane_max = __riscv_vfmv_v_f_f32m4(-INFINITY, vlmax);
  for (size_t j = 0; j < visible;) {
    size_t vl = __riscv_vsetvl_e32m4(visible - j);
    lane_max = __riscv_vfmax_vv_f32m4_tu(lane_max, lane_max, __riscv_vle32_v_f32m4(row + j, vl), vl);
    j += vl;
  }
  const float maximum = __riscv_vfmv_f_s_f32m1_f32(
      __riscv_vfredmax_vs_f32m4_f32m1(lane_max, __riscv_vfmv_v_f_f32m1(-INFINITY, 1), vlmax));
  const float c = scale * XARCH_LOG2E;
  const vfloat32m4_t max_c = __riscv_vfmv_v_f_f32m4(maximum * c, vlmax);
  const vfloat32m4_t c5 = __riscv_vfmv_v_f_f32m4(1.339887440266574e-3f, vlmax);
  const vfloat32m4_t c4 = __riscv_vfmv_v_f_f32m4(9.618437357674640e-3f, vlmax);
  const vfloat32m4_t c3 = __riscv_vfmv_v_f_f32m4(5.550332471162809e-2f, vlmax);
  const vfloat32m4_t c2 = __riscv_vfmv_v_f_f32m4(2.402264791363012e-1f, vlmax);
  const vfloat32m4_t c1 = __riscv_vfmv_v_f_f32m4(6.931472028550421e-1f, vlmax);
  const vfloat32m4_t one = __riscv_vfmv_v_f_f32m4(1.0f, vlmax);
  vfloat32m4_t lane_sum = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
  for (size_t j = 0; j < visible;) {
    size_t vl = __riscv_vsetvl_e32m4(visible - j);
    vfloat32m4_t t = __riscv_vfmsub_vf_f32m4(__riscv_vle32_v_f32m4(row + j, vl), c, max_c, vl);
    t = __riscv_vfmax_vf_f32m4(t, -125.0f, vl);
    vint32m4_t n = __riscv_vfcvt_x_f_v_i32m4(t, vl);
    vfloat32m4_t f = __riscv_vfsub_vv_f32m4(t, __riscv_vfcvt_f_x_v_f32m4(n, vl), vl);
    vfloat32m4_t p = __riscv_vfmacc_vf_f32m4(c5, 1.535336188319500e-4f, f, vl);
    p = __riscv_vfmadd_vv_f32m4(p, f, c4, vl);
    p = __riscv_vfmadd_vv_f32m4(p, f, c3, vl);
    p = __riscv_vfmadd_vv_f32m4(p, f, c2, vl);
    p = __riscv_vfmadd_vv_f32m4(p, f, c1, vl);
    p = __riscv_vfmadd_vv_f32m4(p, f, one, vl);
    p = __riscv_vreinterpret_v_i32m4_f32m4(
        __riscv_vadd_vv_i32m4(__riscv_vreinterpret_v_f32m4_i32m4(p), __riscv_vsll_vx_i32m4(n, 23, vl), vl));
    __riscv_vse32_v_f32m4(row + j, p, vl);
    lane_sum = __riscv_vfadd_vv_f32m4_tu(lane_sum, lane_sum, p, vl);
    j += vl;
  }
  const vfloat32m4_t zero = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
  for (size_t j = visible; j < padded_keys;) {
    size_t vl = __riscv_vsetvl_e32m4(padded_keys - j);
    __riscv_vse32_v_f32m4(row + j, zero, vl);
    j += vl;
  }
  return __riscv_vfmv_f_s_f32m1_f32(
      __riscv_vfredusum_vs_f32m4_f32m1(lane_sum, __riscv_vfmv_v_f_f32m1(0.0f, 1), vlmax));
}
