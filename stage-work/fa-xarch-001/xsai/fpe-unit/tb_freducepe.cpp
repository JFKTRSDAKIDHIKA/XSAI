// Bit-exact check of a standalone CUTE FReducePE (ReduceWidth 256) against
// difftest's CUTE MMA model, for FP32 (new), TF32, BF16 and INT8.
// Also reports the FP32 error against an exact (long double) dot product.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <random>
#include <string>

#include "VFReducePE.h"
#include "verilated.h"
#include "mma_backend_cute_model.h"

namespace {

constexpr int kReduceBytes = 32;
enum Opcode : uint8_t { I8I8I32 = 0, BF16 = 2, TF32 = 3, FP32 = 13 };

struct Case {
  uint8_t opcode;
  uint8_t a[kReduceBytes], b[kReduceBytes];
  uint32_t c;
  uint32_t expected;
  long double exact;     // FP32 only
  int max_exponent;      // FP32 only: exponent of the largest term
};

std::mt19937_64 rng(20261020);

uint32_t bits_of(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
float float_of(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

uint32_t random_fp32(int mode) {
  std::uniform_real_distribution<float> uniform(-0.5f, 0.5f);
  switch (mode) {
    case 0: return bits_of(uniform(rng));                                      // attention inputs
    case 1: {                                                                  // moderate exponents
      uint32_t sign = rng() & 1, exp = 127 - 30 + rng() % 61, frac = rng() & 0x7fffff;
      return sign << 31 | exp << 23 | frac;
    }
    case 2: {                                                                  // any normal exponent
      uint32_t sign = rng() & 1, exp = 1 + rng() % 254, frac = rng() & 0x7fffff;
      return sign << 31 | exp << 23 | frac;
    }
    case 3: return (rng() & 1) << 31;                                          // signed zero
    case 4: return (rng() & 1) << 31 | (rng() & 0x7fffff);                     // subnormal
    case 5: return (rng() & 1) << 31 | 0x7f800000u;                            // infinity
    case 6: return 0x7f800000u | (1 + (rng() & 0x1fff));                       // NaN with low bits only
    default: return bits_of(uniform(rng) * 64.0f);
  }
}

int pick_mode(bool special) {
  int r = rng() % 100;
  if (!special) return r < 60 ? 0 : (r < 90 ? 1 : 7);
  if (r < 40) return 0;
  if (r < 65) return 1;
  if (r < 75) return 2;
  if (r < 85) return 3;
  if (r < 95) return 4;
  return r < 97 ? 5 : 6;
}

Case make_case(uint8_t opcode, bool special) {
  Case k{};
  k.opcode = opcode;
  if (opcode == I8I8I32) {
    for (int i = 0; i < kReduceBytes; ++i) { k.a[i] = rng(); k.b[i] = rng(); }
    k.c = rng();
    k.expected = cute_mma_model::reduce_int32_chunk(true, true, k.a, k.b, kReduceBytes, kReduceBytes, k.c);
    return k;
  }
  int lanes = opcode == BF16 ? 16 : 8;
  for (int i = 0; i < lanes; ++i) {
    uint32_t a = random_fp32(pick_mode(special)), b = random_fp32(pick_mode(special));
    if (opcode == BF16) {
      uint16_t a16 = a >> 16, b16 = b >> 16;
      memcpy(k.a + 2 * i, &a16, 2);
      memcpy(k.b + 2 * i, &b16, 2);
    } else {
      memcpy(k.a + 4 * i, &a, 4);
      memcpy(k.b + 4 * i, &b, 4);
    }
  }
  k.c = random_fp32(pick_mode(special) == 3 ? 3 : pick_mode(false));
  auto format = opcode == FP32 ? cute_mma_model::FloatFormat::Fp32
              : opcode == TF32 ? cute_mma_model::FloatFormat::Tf32
                               : cute_mma_model::FloatFormat::Bf16;
  k.expected = cute_mma_model::reduce_fp32_chunk(format, k.a, k.b, lanes, kReduceBytes, k.c);
  if (opcode == FP32) {
    long double sum = float_of(k.c);
    int max_exponent = std::ilogb(float_of(k.c));
    for (int i = 0; i < lanes; ++i) {
      uint32_t a, b;
      memcpy(&a, k.a + 4 * i, 4);
      memcpy(&b, k.b + 4 * i, 4);
      long double term = (long double)float_of(a) * float_of(b);
      sum += term;
      if (term != 0) max_exponent = std::max(max_exponent, std::ilogb((double)term));
    }
    k.exact = sum;
    k.max_exponent = max_exponent;
  }
  return k;
}

void set_wide(uint32_t *dst, const uint8_t *src) { memcpy(dst, src, kReduceBytes); }

}  // namespace

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  int cases_per_kind = argc > 1 ? atoi(argv[1]) : 20000;
  auto *dut = new VFReducePE;
  std::deque<Case> stimulus, inflight;
  const uint8_t opcodes[] = {FP32, TF32, BF16, I8I8I32};
  for (int round = 0; round < cases_per_kind; ++round)
    for (uint8_t op : opcodes) stimulus.push_back(make_case(op, round % 4 == 3));

  dut->reset = 1;
  for (int i = 0; i < 4; ++i) { dut->clock = 0; dut->eval(); dut->clock = 1; dut->eval(); }
  dut->reset = 0;

  uint64_t checked[16] = {}, failed[16] = {};
  double fp32_worst_grid_error = 0, fp32_worst_relative = 0;
  uint64_t fp32_finite = 0;
  int cycle = 0, printed = 0;
  while ((!stimulus.empty() || !inflight.empty()) && cycle < 100000000) {
    dut->clock = 0;
    bool offer = !stimulus.empty() && (rng() % 8 != 0);  // occasional bubbles
    dut->io_AVector_valid = offer;
    dut->io_BVector_valid = offer;
    dut->io_CAdd_valid = offer;
    dut->io_DResult_ready = rng() % 16 != 0;              // occasional back-pressure
    if (offer) {
      const Case &k = stimulus.front();
      set_wide(dut->io_AVector_bits.data(), k.a);
      set_wide(dut->io_BVector_bits.data(), k.b);
      dut->io_CAdd_bits = k.c;
      dut->io_opcode = k.opcode;
    }
    dut->eval();
    bool accepted = offer && dut->io_AVector_ready;
    bool returned = dut->io_DResult_valid && dut->io_DResult_ready;
    uint32_t result = dut->io_DResult_bits;
    dut->clock = 1;
    dut->eval();
    if (returned) {
      if (inflight.empty()) { printf("FAIL: result without input\n"); return 1; }
      Case k = inflight.front();
      inflight.pop_front();
      ++checked[k.opcode];
      if (result != k.expected) {
        ++failed[k.opcode];
        if (printed++ < 10) printf("mismatch op=%u got=%08x expected=%08x c=%08x\n", k.opcode, result, k.expected, k.c);
      }
      if (k.opcode == FP32 && std::isfinite((double)k.exact) && std::isfinite(float_of(result)) &&
          k.max_exponent > -100) {
        ++fp32_finite;
        double error = fabsl((long double)float_of(result) - k.exact);
        double grid = std::ldexp(1.0, k.max_exponent - 23);
        fp32_worst_grid_error = std::max(fp32_worst_grid_error, error / grid);
        if (k.exact != 0) fp32_worst_relative = std::max(fp32_worst_relative, double(error / fabsl(k.exact)));
      }
    }
    if (accepted) { inflight.push_back(stimulus.front()); stimulus.pop_front(); }
    ++cycle;
  }
  bool ok = stimulus.empty() && inflight.empty();
  const char *names[16] = {"INT8", "", "BF16", "TF32", "", "", "", "", "", "", "", "", "", "FP32"};
  for (uint8_t op : opcodes) {
    printf("%s checked=%llu mismatches=%llu\n", names[op], (unsigned long long)checked[op],
           (unsigned long long)failed[op]);
    ok = ok && checked[op] == (uint64_t)cases_per_kind && failed[op] == 0;
  }
  printf("FP32 vs exact: finite=%llu worst_error_in_units_of_2^(maxexp-23)=%.3f worst_relative=%.3e\n",
         (unsigned long long)fp32_finite, fp32_worst_grid_error, fp32_worst_relative);
  printf("cycles=%d %s\n", cycle, ok ? "PASS" : "FAIL");
  delete dut;
  return ok ? 0 : 1;
}
