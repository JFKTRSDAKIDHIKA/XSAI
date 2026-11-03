package cute

case class CuteFpeConfig(
  enableTf32Fp32: Boolean = false,
  // FP32 x FP32 -> FP32 on the TF32 lanes: full 24-bit significand products
  // built from four 12x12 partial products on the multipliers TF32 leaves idle.
  enableFp32Fp32: Boolean = false,
  enableFp16Fp32: Boolean = false,
  enableMxfp4Fp32: Boolean = false,
  enableMxfp8Fp32: Boolean = false,
  enableNvfp4Fp32: Boolean = false
) {
  def enableScalingFactor: Boolean =
    enableNvfp4Fp32 || enableMxfp4Fp32 || enableMxfp8Fp32

  def enableFp4withsf: Boolean =
    enableNvfp4Fp32 || enableMxfp4Fp32
}
