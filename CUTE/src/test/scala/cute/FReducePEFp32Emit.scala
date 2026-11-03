package cute

import chisel3._
import circt.stage.ChiselStage
import org.chipsalliance.cde.config.Parameters

/**
  * Emits a standalone FReducePE with the reduce width of CUTE_8Tops_128SCP
  * (DefaultMatrixConfig) and the FP32 and TF32 paths enabled, for the
  * bit-exact Verilator check against difftest's CUTE MMA model
  * (cute_mma_model::reduce_fp32_chunk / reduce_int32_chunk).
  *
  * mill -i CUTE.test.runMain cute.FReducePEFp32Emit <target-dir>
  */
object FReducePEFp32Emit extends App {
  private val baseParams = CuteParams.CUTE_8Tops_128SCP
  private val cuteParams = baseParams.copy(
    MatrixExtension = baseParams.MatrixExtension.copy(
      enableTf32Fp32 = true,
      enableFp32Fp32 = true
    )
  )
  private val fpeParameters = Parameters.empty.alterPartial({
    case CuteFpeParamsKey => cuteParams.cuteFpeParams
  })
  ChiselStage.emitSystemVerilogFile(
    new FReducePE()(fpeParameters),
    args = Array("--target-dir", args(0)),
    firtoolOpts = Array("-disable-all-randomization", "-strip-debug-info")
  )
}
