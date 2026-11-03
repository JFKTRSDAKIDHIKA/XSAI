// Softmax microbenchmark on the XSAI emu: cycles per 512-key row for each
// row-softmax variant, rows warm in L1 and rows only in the L2.
#include <math.h>
#include <riscv_vector.h>
#include "xarch_attention.h"
#include "xarch_softmax.h"
#include "softmax_variants.h"

#define KEYS 512
#define ROWS 16
static float rows[ROWS * KEYS] __attribute__((aligned(4096)));
static float evict[32768] __attribute__((aligned(4096)));   // 128 KiB > L1D 64 KiB
static volatile float sink;

static inline vfloat32m8_t exp_of_scaled(vfloat32m8_t x, float scale, size_t vl) {
  x = __riscv_vfmax_vf_f32m8(__riscv_vfmul_vf_f32m8(x, scale, vl), -87.0f, vl);
  vfloat32m8_t n = __riscv_vfcvt_f_x_v_f32m8(
      __riscv_vfcvt_x_f_v_i32m8(__riscv_vfmul_vf_f32m8(x, 1.44269504088896341f, vl), vl), vl);
  vfloat32m8_t r = __riscv_vfnmsac_vf_f32m8(x, 0.693359375f, n, vl);
  r = __riscv_vfnmsac_vf_f32m8(r, -2.12194440e-4f, n, vl);
  vfloat32m8_t p = __riscv_vfmv_v_f_f32m8(1.0f / 720.0f, vl);
  p = __riscv_vfmadd_vv_f32m8(p, r, __riscv_vfmv_v_f_f32m8(1.0f / 120.0f, vl), vl);
  p = __riscv_vfmadd_vv_f32m8(p, r, __riscv_vfmv_v_f_f32m8(1.0f / 24.0f, vl), vl);
  p = __riscv_vfmadd_vv_f32m8(p, r, __riscv_vfmv_v_f_f32m8(1.0f / 6.0f, vl), vl);
  p = __riscv_vfmadd_vv_f32m8(p, r, __riscv_vfmv_v_f_f32m8(0.5f, vl), vl);
  p = __riscv_vfmadd_vv_f32m8(p, r, __riscv_vfmv_v_f_f32m8(1.0f, vl), vl);
  p = __riscv_vfmadd_vv_f32m8(p, r, __riscv_vfmv_v_f_f32m8(1.0f, vl), vl);
  vint32m8_t exponent = __riscv_vsll_vx_i32m8(
      __riscv_vadd_vx_i32m8(__riscv_vfcvt_x_f_v_i32m8(n, vl), 127, vl), 23, vl);
  return __riscv_vfmul_vv_f32m8(p, __riscv_vreinterpret_v_i32m8_f32m8(exponent), vl);
}
static float softmax_row_m8_old(float *row, size_t visible, size_t padded, float scale) {
  vfloat32m1_t running = __riscv_vfmv_v_f_f32m1(-INFINITY, 1);
  for (size_t j = 0; j < visible;) {
    size_t vl = __riscv_vsetvl_e32m8(visible - j);
    running = __riscv_vfredmax_vs_f32m8_f32m1(__riscv_vle32_v_f32m8(row + j, vl), running, vl);
    j += vl;
  }
  float maximum = __riscv_vfmv_f_s_f32m1_f32(running);
  vfloat32m1_t sum = __riscv_vfmv_v_f_f32m1(0.0f, 1);
  for (size_t j = 0; j < visible;) {
    size_t vl = __riscv_vsetvl_e32m8(visible - j);
    vfloat32m8_t p = exp_of_scaled(__riscv_vfsub_vf_f32m8(__riscv_vle32_v_f32m8(row + j, vl), maximum, vl), scale, vl);
    __riscv_vse32_v_f32m8(row + j, p, vl);
    sum = __riscv_vfredusum_vs_f32m8_f32m1(p, sum, vl);
    j += vl;
  }
  (void)padded;
  return __riscv_vfmv_f_s_f32m1_f32(sum);
}
// The max pass alone, LMUL 2 lane-wise.
static float max_only_m2(float *row, size_t visible, size_t padded, float scale) {
  (void)padded; (void)scale;
  size_t vlmax = __riscv_vsetvlmax_e32m2();
  vfloat32m2_t m = __riscv_vfmv_v_f_f32m2(-INFINITY, vlmax);
  for (size_t j = 0; j < visible;) {
    size_t vl = __riscv_vsetvl_e32m2(visible - j);
    m = __riscv_vfmax_vv_f32m2_tu(m, m, __riscv_vle32_v_f32m2(row + j, vl), vl);
    j += vl;
  }
  return __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m2_f32m1(m, __riscv_vfmv_v_f_f32m1(-INFINITY, 1), vlmax));
}

typedef float (*softmax_fn)(float *, size_t, size_t, float);
static const struct { const char *name; softmax_fn fn; } variants[] = {
  {"m8_old", softmax_row_m8_old}, {"m2_tuned", xarch_softmax_row}, {"m1_tuned", softmax_row_m1},
  {"m4_tuned", softmax_row_m4}, {"m2_maxonly", max_only_m2},
};

static void put_str(const char *s) { while (*s) xarch_putc(*s++); }
static void put_dec(uint64_t v) { char d[24]; int n = 0; do { d[n++] = '0' + v % 10; v /= 10; } while (v); while (n) xarch_putc(d[--n]); }

static void fill(void) {
  uint32_t x = 12345;
  for (int i = 0; i < ROWS * KEYS; ++i) { x = x * 1664525u + 1013904223u; rows[i] = ((int)(x >> 9) - (1 << 22)) * (1.0f / (1 << 20)); }
}
static void evict_l1(void) {
  float s = 0;
  for (int i = 0; i < 32768; i += 16) s += evict[i];
  sink = s;
}

int main(void) {
  for (unsigned v = 0; v < sizeof variants / sizeof variants[0]; ++v) {
    fill();
    float acc = 0;
    acc += variants[v].fn(rows, KEYS, KEYS, 0.125f);  // warm row 0 in L1
    uint64_t t0 = xarch_cycles();
    for (int r = 0; r < 4; ++r) acc += variants[v].fn(rows, KEYS, KEYS, 0.125f);
    uint64_t warm = (xarch_cycles() - t0) / 4;
    evict_l1();
    asm volatile("fence rw, rw" ::: "memory");
    t0 = xarch_cycles();
    for (int r = 0; r < ROWS; ++r) acc += variants[v].fn(rows + r * KEYS, KEYS, KEYS, 0.125f);
    uint64_t l2 = (xarch_cycles() - t0) / ROWS;
    sink = acc;
    put_str("MB "); put_str(variants[v].name); put_str(" l1_row="); put_dec(warm); put_str(" l2_row="); put_dec(l2); put_str("\n");
  }
  return 0;
}
