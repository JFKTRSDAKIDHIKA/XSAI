package cute

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config._

object FReduceComputeType {
    val ComputeTypeBitWidth = 4
    val ComputeTypeUndef = 15.U(ComputeTypeBitWidth.W)
    val I8I8I32 = 0.U(ComputeTypeBitWidth.W)
    val F16F16F32 = 1.U(ComputeTypeBitWidth.W)
    val BF16BF16F32 = 2.U(ComputeTypeBitWidth.W)
    val TF32TF32F32 = 3.U(ComputeTypeBitWidth.W)
    val I8U8I32 = 4.U(ComputeTypeBitWidth.W)
    val U8I8I32 = 5.U(ComputeTypeBitWidth.W)
    val U8U8I32 = 6.U(ComputeTypeBitWidth.W)
    val Mxfp8e4m3F32 = 7.U(ComputeTypeBitWidth.W)
    val Mxfp8e5m2F32 = 8.U(ComputeTypeBitWidth.W)
    val Nvfp4F32 = 9.U(ComputeTypeBitWidth.W)
    val Mxfp4F32 = 10.U(ComputeTypeBitWidth.W)
    val Fp8e4m3F32 = 11.U(ComputeTypeBitWidth.W)
    val Fp8e5m2F32 = 12.U(ComputeTypeBitWidth.W)
    val Fp32F32 = 13.U(ComputeTypeBitWidth.W)
}

class RawFloatException extends Bundle {
    val is_nan = Bool()
    val is_inf = Bool()
    val is_zero = Bool()
  }

  class RawFloat(val expWidth: Int, val precision: Int) extends Bundle {
    val sign = Bool()
    val exp = SInt(expWidth.W)
    val signed_sig = SInt((precision + 1).W)
    val exception = new RawFloatException
  }

  object RawFloat {
    // 不能用于e2m1和e4m3！这两个的异常值定义不同，e2m1没有异常值，e4m3只在0bx1111111时为NAN，指数全一为最大指数，但在这里会被处理成负数
    def fromUInt(x: UInt, expWidth: Int, pc: Int, E4M3: Boolean = false, E2M1: Boolean = false): RawFloat = {
      val fp = Wire(new RawFloat(expWidth, pc))
      val nz =  x(expWidth + pc - 2, pc - 1).orR
      val sign = x(expWidth + pc - 1)
      fp.sign := sign
      val bias = Wire(UInt(expWidth.W))
      bias := (1.U << (expWidth - 1)) - 1.U //
      fp.exp := ((x(expWidth + pc - 2, pc - 1) | !nz).asUInt - bias).asSInt
      val sig = Cat(0.U(1.W), nz, x(pc - 2, 0))
    //   fp.signed_sig := Mux(sign, -sig.asSInt.pad(pc + 1), sig.asSInt.pad(pc + 1))
      fp.signed_sig := sig.asSInt
      // 不能用e4m3
    //   if (E4M3) {
    //     fp.exception.is_nan := x(expWidth + pc - 2, 0).andR 
    //     fp.exception.is_inf := false.B
    //   }
    //   else 
      if (E2M1) {
        fp.exception.is_nan := false.B
        fp.exception.is_inf := false.B
      }
      else {
        fp.exception.is_nan := x(expWidth + pc - 2, pc - 1).andR && x(pc - 2, 0).orR
        fp.exception.is_inf := x(expWidth + pc - 2, pc - 1).andR && !x(pc - 2, 0).orR
      }
      fp.exception.is_zero := !x(expWidth + pc - 2, 0).orR
      fp
    }

    // e4m3单独处理，在计算时用5位表示exp
    def frome4m3(x: UInt): RawFloat = {
        val fp = Wire(new RawFloat(5, 4))
        val nz =  x(4 + 4 - 2, 4 - 1).orR
        val sign = x(4 + 4 - 1)
        fp.sign := sign
        val bias = Wire(UInt(5.W))
        bias := (1.U << (4 - 1)) - 1.U //
        fp.exp := ((x(4 + 4 - 2, 4 - 1) | !nz).asUInt.pad(5) - bias).asSInt
        val sig = Cat(0.U(1.W), nz, x(4 - 2, 0))
        //   fp.signed_sig := Mux(sign, -sig.asSInt.pad(pc + 1), sig.asSInt.pad(pc + 1))
        fp.signed_sig := sig.asSInt

        fp.exception.is_nan := x(4 + 4 - 2, 0).andR 
        fp.exception.is_inf := false.B
        fp.exception.is_zero := !x(4 + 4 - 2, 0).orR
        fp
    }
}

// Transform FP4 to fixed point for accumulation
class FP4toint(implicit p: Parameters) extends CuteFpeModule {
    // Output format is xxxx.x
    // For example, 00001 represents 0.5
    val io = IO(new Bundle{
        val in = Input(UInt(4.W))
        val out = Output(SInt(5.W))
    })

    val sign = io.in(3)
    val exp = io.in(2, 1)
    val mantissa = io.in(0)
    val nz = exp.orR
    val sig = Cat(0.U(3.W), nz, mantissa).asSInt // 5位无符号数
    val unsignedSig = Wire(SInt(5.W))

    when(exp === 2.U){
        unsignedSig := sig << 1.U
    }.elsewhen(exp === 3.U){
        unsignedSig := sig << 2.U
    }.otherwise{
        unsignedSig := sig
    }

    io.out := Mux(sign, -unsignedSig.asSInt, unsignedSig.asSInt)
}

class CLZ(len: Int)(implicit p: Parameters) extends CuteFpeModule {

  val inWidth = len
  val outWidth = (inWidth - 1).U.getWidth

  val io = IO(new Bundle() {
    val in = Input(UInt(inWidth.W))
    val out = Output(UInt(outWidth.W))
  })

  io.out := PriorityEncoder(io.in.asBools.reverse)
}
