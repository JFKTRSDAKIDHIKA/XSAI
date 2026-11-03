package cute

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config._

class FPipe0Result(implicit p: Parameters) extends CuteFpeBundle{
    val Product0 = Vec(ReduceWidth/16, SInt(MultiplierProductWidth.W))
    val Product1 = Vec(ReduceWidth/16, SInt(MultiplierProductWidth.W))
    val FP4ReduceRes = Option.when(cuteFpeConfig.enableFp4withsf)(
        Vec(ReduceWidth/4/FP4P0AddNum, SInt((10 + log2Ceil(FP4P0AddNum)).W))
    )
    val CMantissa = SInt(32.W)
    val ExceptionVec = Vec(ReduceWidth/8 + 1, new RawFloatException) //每个向量的异常标志位
    val MulExpVec = Vec(ReduceWidth/8 + 1, UInt(10.W))
    val CmpTreefp8P0Result = new CmpTreeP0Res(fp8cmptreelayers, ReduceWidth/8 + 1, 10)
    val SignVec = Vec(ReduceWidth/8 + 1, Bool()) //每个向量的符号位
    val opcode = UInt(FReduceComputeType.ComputeTypeBitWidth.W)
    val scaleFP = Option.when(cuteFpeConfig.enableFp4withsf)(
        Vec(ReduceWidth/4/MinGroupSize + 1, new RawFloat(9, 11))
    )
}

class FPipe1Result(implicit p: Parameters) extends CuteFpeBundle{
    val Product0 = Vec(ReduceWidth/16, SInt(MultiplierProductWidth.W))
    val Product1 = Vec(ReduceWidth/16, SInt(MultiplierProductWidth.W))
    // Exact FP32 significand product (2.46 fixed point) truncated to the
    // 2.23 grid that TF32 products reach after their 3-bit left shift.
    val fp32ProductOnTf32Grid = Option.when(cuteFpeConfig.enableFp32Fp32)(
      Vec(ReduceWidth/32, UInt(25.W))
    )
    val CMantissa = SInt(32.W)
    val RightShiftVec = Vec((ReduceWidth/8) + 1, UInt(10.W))
    val FP4ReduceRes = Option.when(cuteFpeConfig.enableFp4withsf)(
      Vec(ReduceWidth/4/16, SInt((10 + log2Ceil(16)).W))
    )
    val MaxExp = UInt(10.W)
    val CmpTreefp4P0Result = Option.when(cuteFpeConfig.enableFp4withsf)(
      new CmpTreeP0Res(8, ReduceWidth/4/MinGroupSize + 1, 9)
    )
    val opcode = UInt(FReduceComputeType.ComputeTypeBitWidth.W)
    val SumException = UInt(4.W) //归约计算结果的异常标志位
    val SignVec = Vec(ReduceWidth/8 + 1, Bool()) //每个向量的符号位
    val scaleFP = Option.when(cuteFpeConfig.enableFp4withsf)(
      Vec(ReduceWidth/4/MinGroupSize + 1, new RawFloat(9, 11))
    )
}

class FPipe2Result(implicit p: Parameters) extends CuteFpeBundle{
    val ReduceRes0 = Vec(P3AddNum, SInt((26 + log2Ceil(P2AddNum)).W))
    val ReduceRes1 = Vec(P3AddNum, SInt((26 + log2Ceil(P2AddNum)).W)) // ReduceRes0是尾数右移后的结果，ReduceRes1是Int8的结果
    val CMantissa = SInt(32.W) // C的尾数
    val MaxExp = UInt(10.W)
    val opcode = UInt(FReduceComputeType.ComputeTypeBitWidth.W)
    val SumException = UInt(4.W) //归约计算结果的异常标志位
    val FP4ABShift = Option.when(cuteFpeConfig.enableFp4withsf)(
      Vec(ReduceWidth/4/MinGroupSize + 1, SInt((32 + 3).W))
    )
}

class FPipe3Result(implicit p: Parameters) extends CuteFpeBundle{
    // val ReduceRes = SInt(32.W) //归约计算结果
    val ReduceRes0 = SInt((32).W)
    val ReduceRes1 = SInt((32).W) // ReduceRes0是尾数右移后的结果，ReduceRes1是Int8的结果
    // val ReduceResFP4 = SInt(32.W)
    val CMantissa = SInt(32.W) // C的尾数
    val MaxExp = UInt(10.W)
    val opcode = UInt(FReduceComputeType.ComputeTypeBitWidth.W)
    val SumException = UInt(4.W) //归约计算结果的异常标志位
}
