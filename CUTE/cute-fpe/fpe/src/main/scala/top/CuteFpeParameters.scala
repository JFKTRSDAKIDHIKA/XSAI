package cute

import chisel3._
import org.chipsalliance.cde.config._

trait CuteFpeParameters {
  def cuteFpeConfig: CuteFpeConfig

  def ReduceWidth: Int
  def ResultWidth: Int
  def MinGroupSize: Int
  def MinDataTypeWidth: Int
  def ScaleElementWidth: Int
  def cmptreelayers: Int
  def fp8cmptreelayers: Int

  def P3AddNum: Int
  def P2AddNum: Int
  def FP4P0AddNum: Int
  def FP4P1AddNum: Int

  def DEBUG_FP8: Boolean
  def DEBUG_FP4: Boolean
}

object DefaultCuteFpeParameters extends CuteFpeParameters {
  def cuteFpeConfig: CuteFpeConfig = CuteFpeConfig()

  def ReduceWidth: Int = 512
  def ResultWidth: Int = 32
  def MinGroupSize: Int = 16
  def MinDataTypeWidth: Int = 4
  def ScaleElementWidth: Int = 8
  def cmptreelayers: Int = 4
  def fp8cmptreelayers: Int = 4

  def P3AddNum: Int = ReduceWidth / 4 / MinGroupSize
  def P2AddNum: Int = ReduceWidth / (P3AddNum * 16)

  def FP4P0AddNum: Int = 2
  def FP4P1AddNum: Int = 16 / FP4P0AddNum

  def DEBUG_FP8: Boolean = true
  def DEBUG_FP4: Boolean = false
}

case object CuteFpeParamsKey extends Field[CuteFpeParameters](DefaultCuteFpeParameters)

trait HasCuteFpeParameters extends CuteFpeParameters {
  implicit val p: Parameters

  private def fpeParams: CuteFpeParameters = p(CuteFpeParamsKey)

  def cuteFpeConfig: CuteFpeConfig = fpeParams.cuteFpeConfig

  def ReduceWidth: Int = fpeParams.ReduceWidth
  def ResultWidth: Int = fpeParams.ResultWidth
  def MinGroupSize: Int = fpeParams.MinGroupSize
  def MinDataTypeWidth: Int = fpeParams.MinDataTypeWidth
  def ScaleElementWidth: Int = fpeParams.ScaleElementWidth
  def cmptreelayers: Int = fpeParams.cmptreelayers
  def fp8cmptreelayers: Int = fpeParams.fp8cmptreelayers

  // FP32 partial products are 12x12 unsigned (24 bits), so the multipliers take
  // 13-bit signed operands and the product registers hold 25 bits.
  def MultiplierOperandWidth: Int = if (cuteFpeConfig.enableFp32Fp32) 13 else 12
  def MultiplierProductWidth: Int = if (cuteFpeConfig.enableFp32Fp32) 25 else 23

  def P3AddNum: Int = fpeParams.P3AddNum
  def P2AddNum: Int = fpeParams.P2AddNum
  def FP4P0AddNum: Int = fpeParams.FP4P0AddNum
  def FP4P1AddNum: Int = fpeParams.FP4P1AddNum

  def DEBUG_FP8: Boolean = fpeParams.DEBUG_FP8
  def DEBUG_FP4: Boolean = fpeParams.DEBUG_FP4
}

class CuteFpeModule(implicit val p: Parameters) extends Module with HasCuteFpeParameters
class CuteFpeBundle(implicit val p: Parameters) extends Bundle with HasCuteFpeParameters
