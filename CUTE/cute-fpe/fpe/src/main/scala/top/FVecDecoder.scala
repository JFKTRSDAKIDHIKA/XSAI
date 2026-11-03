package cute

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config._

class FDecodeResult(implicit p: Parameters) extends CuteFpeBundle{
    // val Int8Vec = Vec(ReduceWidth/8, UInt(9.W))
    val TF32Vec = Vec(ReduceWidth/8, new RawFloat(10, 11))
    // FP32 operands keep their upper 11 significand bits in TF32Vec and the
    // remaining 13 fraction bits here.
    val fp32LowFractionBits = Option.when(cuteFpeConfig.enableFp32Fp32)(
        Vec(ReduceWidth/32, UInt(13.W))
    )
}

class FPScaleDecoder(implicit p: Parameters) extends CuteFpeModule {
    val io = IO(new Bundle{
        val in = Input(Vec(ReduceWidth/4/MinGroupSize, UInt(8.W)))
        val out = Output(Vec(ReduceWidth/4/MinGroupSize, new RawFloat(8, 11)))
    })

    for(i <- 0 until ReduceWidth/4/MinGroupSize){
        val Bits8  = io.in(i)
        val DecodeE4M3 = RawFloat.frome4m3(Bits8)
        val E4M3toTF32 = Wire(new RawFloat(8, 11))
        E4M3toTF32.exp := DecodeE4M3.exp.pad(8)
        E4M3toTF32.signed_sig := DecodeE4M3.signed_sig
        E4M3toTF32.exception := DecodeE4M3.exception
        E4M3toTF32.sign := DecodeE4M3.sign
        io.out(i) := E4M3toTF32
    }
}

class FVecDecoder(implicit p: Parameters) extends CuteFpeModule {
    val io = IO(new Bundle{
        val in = Input(UInt(ReduceWidth.W))
        val opcode = Input(UInt(FReduceComputeType.ComputeTypeBitWidth.W))
        val out = Output(new FDecodeResult)
    })

    val TF32Zero = RawFloat.fromUInt(0.U(21.W), 10, 11)

    val TF32i8 = Wire(Vec(ReduceWidth/8, new RawFloat(10, 11)))
    for(i <- 0 until ReduceWidth/8){
        TF32i8(i) := RawFloat.fromUInt(0.U(21.W), 10, 11)
        TF32i8(i).signed_sig := Mux(io.opcode === FReduceComputeType.I8I8I32,
            io.in(8 * i + 7, 8 * i).asSInt.pad(12), io.in(8 * i + 7, 8 * i).pad(12).asSInt
        )
    }

    val isF16 = cuteFpeConfig.enableFp16Fp32.B && io.opcode === FReduceComputeType.F16F16F32
    val isBF16 = io.opcode === FReduceComputeType.BF16BF16F32
    val isTF32 = cuteFpeConfig.enableTf32Fp32.B && io.opcode === FReduceComputeType.TF32TF32F32
    val isFP32 = cuteFpeConfig.enableFp32Fp32.B && io.opcode === FReduceComputeType.Fp32F32
    val isE4M3 = io.opcode === FReduceComputeType.Mxfp8e4m3F32 || io.opcode === FReduceComputeType.Fp8e4m3F32
    val isE5M2 = io.opcode === FReduceComputeType.Mxfp8e5m2F32 || io.opcode === FReduceComputeType.Fp8e5m2F32
    val isDefaultTF32Vec = !(isF16 || isBF16 || isTF32 || isFP32 || isE4M3 || isE5M2)
    val isDefaultTF32VecHigh = !(isE4M3 || isE5M2)

    for(i <- 0 until ReduceWidth/32){
        val Bits8  = io.in(8 * i + 7, 8 * i)
        val Bits16 = io.in(16 * i + 15, 16 * i)
        val Bits32 = io.in(32 * i + 31, 32 * i)
        val DecodeBF16 = RawFloat.fromUInt(Bits16, 8, 8)
        val DecodeE4M3 = RawFloat.frome4m3(Bits8)
        val DecodeE5M2 = RawFloat.fromUInt(Bits8, 5, 3)
        val BF16toTF32 = Wire(new RawFloat(10, 11))
        val E4M3toTF32 = Wire(new RawFloat(10, 11))
        val E5M2toTF32 = Wire(new RawFloat(10, 11))
        BF16toTF32.exp := DecodeBF16.exp
        BF16toTF32.signed_sig := Cat(DecodeBF16.signed_sig, 0.U(3.W)).asSInt
        BF16toTF32.exception := DecodeBF16.exception
        BF16toTF32.sign := DecodeBF16.sign
        E4M3toTF32.exp := DecodeE4M3.exp.pad(10)
        E4M3toTF32.signed_sig := Cat(DecodeE4M3.signed_sig, 0.U(7.W)).asSInt
        E4M3toTF32.exception := DecodeE4M3.exception
        E4M3toTF32.sign := DecodeE4M3.sign
        E5M2toTF32.exp := DecodeE5M2.exp.pad(10)
        E5M2toTF32.signed_sig := Cat(DecodeE5M2.signed_sig, 0.U(8.W)).asSInt
        E5M2toTF32.exception := DecodeE5M2.exception
        E5M2toTF32.sign := DecodeE5M2.sign
        var decodeCases = Seq(
            isBF16 -> BF16toTF32,
            isE4M3 -> E4M3toTF32,
            isE5M2 -> E5M2toTF32,
            isDefaultTF32Vec -> TF32i8(i)
        )
        if (cuteFpeConfig.enableFp16Fp32) {
            val DecodeFP16 = RawFloat.fromUInt(Bits16, 5, 11)
            val FP16toTF32 = Wire(new RawFloat(10, 11))
            FP16toTF32.exp := DecodeFP16.exp.pad(10)
            FP16toTF32.signed_sig := DecodeFP16.signed_sig
            FP16toTF32.exception := DecodeFP16.exception
            FP16toTF32.sign := DecodeFP16.sign
            decodeCases = decodeCases :+ (isF16 -> FP16toTF32)
        }
        if (cuteFpeConfig.enableTf32Fp32) {
            val DecodeTF32 = RawFloat.fromUInt(Bits32(31, 13), 8, 11)
            val TF32conv = Wire(new RawFloat(10, 11))
            TF32conv.exp := DecodeTF32.exp.pad(10)
            TF32conv.signed_sig := DecodeTF32.signed_sig
            TF32conv.exception := DecodeTF32.exception
            TF32conv.sign := DecodeTF32.sign
            decodeCases = decodeCases :+ (isTF32 -> TF32conv)
        }
        if (cuteFpeConfig.enableFp32Fp32) {
            // Exponent, sign and exception flags come from all 32 bits, so a
            // NaN or subnormal whose set bits are all below bit 13 is classified
            // as FP32 does.
            val DecodeFP32 = RawFloat.fromUInt(Bits32, 8, 24)
            val FP32UpperSignificand = Wire(new RawFloat(10, 11))
            FP32UpperSignificand.exp := DecodeFP32.exp.pad(10)
            FP32UpperSignificand.signed_sig := DecodeFP32.signed_sig(24, 13).asSInt
            FP32UpperSignificand.exception := DecodeFP32.exception
            FP32UpperSignificand.sign := DecodeFP32.sign
            decodeCases = decodeCases :+ (isFP32 -> FP32UpperSignificand)
            io.out.fp32LowFractionBits.get(i) := Bits32(12, 0)
        }
        io.out.TF32Vec(i) := Mux1H(decodeCases)
    }

    for(i <- ReduceWidth/32 until ReduceWidth/16){
        val Bits8  = io.in(8 * i + 7, 8 * i)
        val Bits16 = io.in(16 * i + 15, 16 * i)
        val DecodeBF16 = RawFloat.fromUInt(Bits16, 8, 8)
        val DecodeE4M3 = RawFloat.frome4m3(Bits8)
        val DecodeE5M2 = RawFloat.fromUInt(Bits8, 5, 3)
        val BF16toTF32 = Wire(new RawFloat(10, 11))
        val E4M3toTF32 = Wire(new RawFloat(10, 11))
        val E5M2toTF32 = Wire(new RawFloat(10, 11))
        BF16toTF32.exp := DecodeBF16.exp
        BF16toTF32.signed_sig := Cat(DecodeBF16.signed_sig, 0.U(3.W)).asSInt// 有符号数，所以12位
        BF16toTF32.exception := DecodeBF16.exception
        BF16toTF32.sign := DecodeBF16.sign
        E4M3toTF32.exp := DecodeE4M3.exp.pad(10)
        E4M3toTF32.signed_sig := Cat(DecodeE4M3.signed_sig, 0.U(7.W)).asSInt
        E4M3toTF32.exception := DecodeE4M3.exception
        E4M3toTF32.sign := DecodeE4M3.sign
        E5M2toTF32.exp := DecodeE5M2.exp.pad(10)
        E5M2toTF32.signed_sig := Cat(DecodeE5M2.signed_sig, 0.U(8.W)).asSInt
        E5M2toTF32.exception := DecodeE5M2.exception
        E5M2toTF32.sign := DecodeE5M2.sign
        var decodeCases = Seq(
            isBF16 -> BF16toTF32,
            isE4M3 -> E4M3toTF32,
            isE5M2 -> E5M2toTF32,
            isDefaultTF32Vec -> TF32i8(i)
        )
        if (cuteFpeConfig.enableFp16Fp32) {
            val DecodeFP16 = RawFloat.fromUInt(Bits16, 5, 11)
            val FP16toTF32 = Wire(new RawFloat(10, 11))
            FP16toTF32.exp := DecodeFP16.exp.pad(10)
            FP16toTF32.signed_sig := DecodeFP16.signed_sig
            FP16toTF32.exception := DecodeFP16.exception
            FP16toTF32.sign := DecodeFP16.sign
            decodeCases = decodeCases :+ (isF16 -> FP16toTF32)
        }
        if (cuteFpeConfig.enableTf32Fp32) {
            decodeCases = decodeCases :+ (isTF32 -> TF32Zero)
        }
        if (cuteFpeConfig.enableFp32Fp32) {
            decodeCases = decodeCases :+ (isFP32 -> TF32Zero)
        }
        io.out.TF32Vec(i) := Mux1H(decodeCases)
    }

    for(i <- ReduceWidth/16 until ReduceWidth/8){
        val Bits8  = io.in(8 * i + 7, 8 * i)
        val DecodeE4M3 = RawFloat.frome4m3(Bits8)
        val DecodeE5M2 = RawFloat.fromUInt(Bits8, 5, 3)
        val E4M3toTF32 = Wire(new RawFloat(10, 11))
        val E5M2toTF32 = Wire(new RawFloat(10, 11))
        E4M3toTF32.exp := DecodeE4M3.exp.pad(10)
        E4M3toTF32.signed_sig := Cat(DecodeE4M3.signed_sig, 0.U(7.W)).asSInt
        E4M3toTF32.exception := DecodeE4M3.exception
        E4M3toTF32.sign := DecodeE4M3.sign
        E5M2toTF32.exp := DecodeE5M2.exp.pad(10)
        E5M2toTF32.signed_sig := Cat(DecodeE5M2.signed_sig, 0.U(8.W)).asSInt
        E5M2toTF32.exception := DecodeE5M2.exception
        E5M2toTF32.sign := DecodeE5M2.sign
        io.out.TF32Vec(i) := Mux1H(Seq(
            isE4M3 -> E4M3toTF32,
            isE5M2 -> E5M2toTF32,
            isDefaultTF32VecHigh -> TF32i8(i)
        ))
    }
}
