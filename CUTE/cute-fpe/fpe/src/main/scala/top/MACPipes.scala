package cute

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config._

// Stage 0:
// - Compute the product of the mantissas and the sum of the exponents
// - Find the maximum exponent and calculate the right shift amount
class FReduceMACPipe0(implicit p: Parameters) extends CuteFpeModule {
    val io = IO(new Bundle{
        val inA = Input(new FDecodeResult)
        val inB = Input(new FDecodeResult)
        val inAFP4Vec = Option.when(cuteFpeConfig.enableFp4withsf)(
            Input(Vec(ReduceWidth/4, SInt(5.W)))
        )
        val inBFP4Vec = Option.when(cuteFpeConfig.enableFp4withsf)(
            Input(Vec(ReduceWidth/4, SInt(5.W)))
        )
        val inAscale = Option.when(cuteFpeConfig.enableScalingFactor)(
            Input(Vec(ReduceWidth/4/MinGroupSize, UInt(8.W)))
        )
        val inBscale = Option.when(cuteFpeConfig.enableScalingFactor)(
            Input(Vec(ReduceWidth/4/MinGroupSize, UInt(8.W)))
        )
        val inC = Input(UInt(32.W))
        val opcode = Input(UInt(FReduceComputeType.ComputeTypeBitWidth.W))
        val out = Output(new FPipe0Result)
    })

    val CDecode = RawFloat.fromUInt(io.inC, 8, 24)

    // 1. Mantissa product
    if (DEBUG_FP8) {
        for (i <- 0 until ReduceWidth/8) {
            printf("inAexp[%d]: %x\n", i.U,io.inA.TF32Vec(i).exp)
            printf("inBexp[%d]: %x\n", i.U,io.inB.TF32Vec(i).exp)
        }
    }

    // Product0
    // The width of the multiplier needs to support the multiplication of 12-bit signed integers
    // FP32: element j has significand {upper 11 bits in TF32Vec(j), 13 low
    // fraction bits}, split into 12-bit halves. Its four partial products
    // high*high, high*low, low*high and low*low use Product0(2j), Product0(2j+1),
    // Product1(2j) and Product1(2j+1).
    val isFp32Op = cuteFpeConfig.enableFp32Fp32.B && io.opcode === FReduceComputeType.Fp32F32
    def fp32SignificandHalves(in: FDecodeResult, element: Int): (UInt, UInt) = {
        val significand = Cat(in.TF32Vec(element).signed_sig(10, 0).asUInt, in.fp32LowFractionBits.get(element))
        (significand(23, 12), significand(11, 0))
    }
    def fp32PartialProductOperand(in: FDecodeResult, product: Int, takesLowHalf: Boolean): SInt = {
        val (highHalf, lowHalf) = fp32SignificandHalves(in, product / 2)
        Cat(0.U(1.W), if (takesLowHalf) lowHalf else highHalf).asSInt
    }

    for(i <- 0 until ReduceWidth/16){
        val Product0A = Wire(SInt(MultiplierOperandWidth.W))
        val Product0B = Wire(SInt(MultiplierOperandWidth.W))

        Product0A := io.inA.TF32Vec(i).signed_sig
        Product0B := io.inB.TF32Vec(i).signed_sig
        if (cuteFpeConfig.enableFp32Fp32) {
            when(isFp32Op) {
                Product0A := fp32PartialProductOperand(io.inA, i, takesLowHalf = false)
                Product0B := fp32PartialProductOperand(io.inB, i, takesLowHalf = i % 2 == 1)
            }
        }

        io.out.Product0(i) := Product0A * Product0B
        if (DEBUG_FP8)
        {printf("Product0A[%d]: %x\n", i.U,Product0A)
        printf("Product0B[%d]: %x\n", i.U,Product0B)
        printf("product0[%d]: %x\n", i.U,io.out.Product0(i))}
    }

    // Product1
    // The width of the multiplier needs to support the multiplication of 9-bit signed integers
    for(i <- ReduceWidth/16 until ReduceWidth/8){
        val Product1A = Wire(SInt(MultiplierOperandWidth.W))
        val Product1B = Wire(SInt(MultiplierOperandWidth.W))

        Product1A := io.inA.TF32Vec(i).signed_sig.asSInt
        Product1B := io.inB.TF32Vec(i).signed_sig.asSInt
        if (cuteFpeConfig.enableFp32Fp32) {
            when(isFp32Op) {
                Product1A := fp32PartialProductOperand(io.inA, i - ReduceWidth/16, takesLowHalf = true)
                Product1B := fp32PartialProductOperand(io.inB, i - ReduceWidth/16, takesLowHalf = i % 2 == 1)
            }
        }

        io.out.Product1(i - ReduceWidth/16) := Product1A * Product1B
    }

    // 2. Exponent calculation
    val MulExpVec = Wire(Vec((ReduceWidth/8) + 1, UInt(10.W)))
    val MulExpVecSigned = Wire(Vec((ReduceWidth/8) + 1, SInt(9.W))) //for debug
         
    for(i <- 0 until ReduceWidth/8){

        // printf("inA exp[%d]: %x\n", i.U,io.inA.TF32Vec(i).exp(5, 0))
        // printf("inB exp[%d]: %x\n", i.U,io.inB.TF32Vec(i).exp(5, 0))
        // printf("fp8MulExpVec[%d]: %x\n", i.U,MulExpVec(i))
        MulExpVecSigned(i) := io.inA.TF32Vec(i).exp.pad(9) + io.inB.TF32Vec(i).exp.pad(9)
        MulExpVec(i) := Mux(
            io.inA.TF32Vec(i).exception.is_zero | io.inB.TF32Vec(i).exception.is_zero,
            0.U, 
            (io.inA.TF32Vec(i).exp + io.inB.TF32Vec(i).exp).asUInt + 511.U(10.W))
        if (DEBUG_FP8) {
          printf("MulExpVec[%d]: %x\n", i.U,MulExpVec(i))
        }
    }

    MulExpVec(ReduceWidth/8) := CDecode.exp.pad(10).asUInt + 511.U(10.W) // C的阶码加上偏移量

    if (DEBUG_FP8) {
        printf("io.inC: %x\n", io.inC)
        printf("CDecode exp: %x\n", CDecode.exp.pad(10))
        printf("C exp: %x\n", MulExpVec(ReduceWidth/8))
    }

    // Each product exponent's difference with the maximum exponent is the right shift amount
    for(i <- 0 until ReduceWidth/8 + 1){
        io.out.MulExpVec(i) := MulExpVec(i)
    }

    // 3. Select max exponent
    val cmptreefp8p0 = Module(new CmpTreeP0(fp8cmptreelayers, ReduceWidth/8 + 1, 10))
    
    for (i <- 0 until ReduceWidth/16){
        cmptreefp8p0.io.in(i) := MulExpVec(i)
        val useHighHalfMulExp = Seq(
            Option.when(cuteFpeConfig.enableMxfp8Fp32)(io.opcode === FReduceComputeType.Mxfp8e4m3F32),
            Option.when(cuteFpeConfig.enableMxfp8Fp32)(io.opcode === FReduceComputeType.Mxfp8e5m2F32),
            Some(io.opcode === FReduceComputeType.Fp8e4m3F32),
            Some(io.opcode === FReduceComputeType.Fp8e5m2F32)
        ).flatten.reduce(_ || _)
        cmptreefp8p0.io.in(i + ReduceWidth/16) := Mux(
            useHighHalfMulExp,
            MulExpVec(i + ReduceWidth/16), 0.U
        )
    }

    cmptreefp8p0.io.in(ReduceWidth/8) := MulExpVec(ReduceWidth/8) //最后一个C的阶码

    // Output opcode and max exponent for later use
    io.out.opcode := io.opcode
    io.out.CmpTreefp8P0Result := cmptreefp8p0.io.out

    // FP4 path
    val FP4AScaleDecoder = Option.when(cuteFpeConfig.enableFp4withsf)(Module(new FPScaleDecoder()))
    val FP4BScaleDecoder = Option.when(cuteFpeConfig.enableFp4withsf)(Module(new FPScaleDecoder()))
    val mxfp4ScaleSum = Option.when(cuteFpeConfig.enableFp4withsf)(
        Wire(Vec(ReduceWidth/4/32, UInt(10.W)))
    )
    val mxfp4ScaleSat = Option.when(cuteFpeConfig.enableFp4withsf)(
        Wire(Vec(ReduceWidth/4/32, UInt(9.W)))
    )
    val FP4AScaleExceptionVec = Option.when(cuteFpeConfig.enableFp4withsf)(Wire(Vec(ReduceWidth/4/MinGroupSize, new RawFloatException)))
    val FP4BScaleExceptionVec = Option.when(cuteFpeConfig.enableFp4withsf)(Wire(Vec(ReduceWidth/4/MinGroupSize, new RawFloatException)))

    if (cuteFpeConfig.enableFp4withsf) {
        // 1. FP4 scale product
        FP4AScaleDecoder.zip(io.inAscale).foreach{
            case (decoder, scale) =>
                decoder.io.in := scale
        }
        FP4BScaleDecoder.zip(io.inBscale).foreach{
            case (decoder, scale) =>
                decoder.io.in := scale
        }

        // Results exceeding 127 will be treated as infinity.
        // So we can directly truncate to 9 bits, and results exceeding 9 bits will also be treated as infinity
        for (i <- 0 until ReduceWidth/4/32){
            mxfp4ScaleSum.get(i) := io.inAscale.get(i).asUInt.pad(10) + io.inBscale.get(i).asUInt.pad(10) + (1 + 6).U
            mxfp4ScaleSat.get(i) := Mux(mxfp4ScaleSum.get(i)(9) === 1.U, 511.U, mxfp4ScaleSum.get(i)(8, 0))
        }

        // 2. Generate exception flags for FP4 scale
        // Case Mxfp4
        val mxfp4AScaleExceptionVec = Option.when(cuteFpeConfig.enableMxfp4Fp32)(Wire(Vec(ReduceWidth/4/32, new RawFloatException)))
        val mxfp4BScaleExceptionVec = Option.when(cuteFpeConfig.enableMxfp4Fp32)(Wire(Vec(ReduceWidth/4/32, new RawFloatException)))
        mxfp4AScaleExceptionVec.foreach(_ := 0.U.asTypeOf(Vec(ReduceWidth/4/32, new RawFloatException)))
        mxfp4BScaleExceptionVec.foreach(_ := 0.U.asTypeOf(Vec(ReduceWidth/4/32, new RawFloatException)))
        if (cuteFpeConfig.enableMxfp4Fp32) {
            // The format of scaling factor in mxfp4 is UE8M0
            // - 255 represents nan
            for (i <- 0 until ReduceWidth/4/32){
                mxfp4AScaleExceptionVec.get(i).is_nan := io.inAscale.get(i) === 255.U
                mxfp4BScaleExceptionVec.get(i).is_nan := io.inBscale.get(i) === 255.U
            }
        }

        // Number of blocks in mxfp4 is half of that in nvfp4
        for (i <- 0 until ReduceWidth/4/MinGroupSize){
            FP4AScaleExceptionVec.get(i) := Mux(
                io.opcode === FReduceComputeType.Mxfp4F32,
                mxfp4AScaleExceptionVec.get(i / 2),
                FP4AScaleDecoder.get.io.out(i).exception
            )
            FP4BScaleExceptionVec.get(i) := Mux(
                io.opcode === FReduceComputeType.Mxfp4F32,
                mxfp4BScaleExceptionVec.get(i / 2),
                FP4BScaleDecoder.get.io.out(i).exception
            )
        }

        // 3. Generate the scale factor in FP
        io.out.scaleFP.get(ReduceWidth/4/MinGroupSize).sign := CDecode.sign
        io.out.scaleFP.get(ReduceWidth/4/MinGroupSize).exp := CDecode.exp.pad(9) + 255.S
        if (DEBUG_FP4) {
            printf("CDecode.exp:%x\n", CDecode.exp)
            printf("io.out.scaleFP(ReduceWidth/4/MinGroupSize).exp:%x\n", io.out.scaleFP.get(ReduceWidth/4/MinGroupSize).exp)
        }
        io.out.scaleFP.get(ReduceWidth/4/MinGroupSize).signed_sig := CDecode.signed_sig
        io.out.scaleFP.get(ReduceWidth/4/MinGroupSize).exception := CDecode.exception

        for (i <- 0 until ReduceWidth/4/MinGroupSize) {
            if (DEBUG_FP4) {
                printf("io.inAscale[%d]: %x\n", i.U, io.inAscale.get(i))
                printf("io.inBscale[%d]: %x\n", i.U, io.inBscale.get(i))
                printf("FP4AScaleDecoder.io.out[%d].sign: %x\n", i.U, FP4AScaleDecoder.get.io.out(i).sign)
                printf("FP4BScaleDecoder.io.out[%d].sign: %x\n", i.U, FP4BScaleDecoder.get.io.out(i).sign)
                printf("FP4AScaleDecoder.io.out[%d].exp: %x\n", i.U, FP4AScaleDecoder.get.io.out(i).exp)
                printf("FP4BScaleDecoder.io.out[%d].exp: %x\n", i.U, FP4BScaleDecoder.get.io.out(i).exp)
            }
            io.out.scaleFP.get(i).sign := Mux(
                io.opcode === FReduceComputeType.Nvfp4F32,
                FP4AScaleDecoder.get.io.out(i).sign ^ FP4BScaleDecoder.get.io.out(i).sign,
                0.B
            )
            io.out.scaleFP.get(i).exp := Mux(
                io.out.ExceptionVec(i).is_zero || (io.opcode =/= FReduceComputeType.Nvfp4F32 && io.opcode =/= FReduceComputeType.Mxfp4F32),
                0.S, /*这里的9对应于之后对齐前的左移6，两者合起来15与FP32的尾数对齐，之所以避让出9位空位是因为FP4直接转换成整数计算中间结果位宽很大*/
                Mux(io.opcode === FReduceComputeType.Mxfp4F32,
                    mxfp4ScaleSat.get(i / 2).asSInt,
                    FP4AScaleDecoder.get.io.out(i).exp.pad(9) + FP4BScaleDecoder.get.io.out(i).exp.pad(9) + (255 + 9).S
                )
            )

            if (DEBUG_FP8 || DEBUG_FP4) {
                printf("io.out.scaleFP(%d).sign:%x\n", i.U,io.out.scaleFP.get(i).sign)
                printf("io.out.scaleFP(%d).exp:%x\n", i.U,io.out.scaleFP.get(i).exp)
            }

            val sig_mul = Wire(SInt(10.W))
            sig_mul := Mux(
                io.opcode === FReduceComputeType.Nvfp4F32,
                (FP4AScaleDecoder.get.io.out(i).signed_sig(4, 0).asSInt * FP4BScaleDecoder.get.io.out(i).signed_sig(4, 0).asSInt).asSInt,
                Mux(io.opcode === FReduceComputeType.Mxfp4F32, 1.S, 0.S)
            )
            io.out.scaleFP.get(i).signed_sig := Mux(io.out.scaleFP.get(i).sign, 
                -sig_mul.asSInt, sig_mul.asSInt
            ) 
            if (DEBUG_FP4 && cuteFpeConfig.enableFp4withsf) {
                printf("io.out.scaleFP[%d].sign: %x\n", i.U, io.out.scaleFP.get(i).sign)
                printf("sig_mul[%d]: %x\n", i.U, sig_mul)
                printf("io.out.scaleFP[%d].signed_sig: %x\n", i.U, io.out.scaleFP.get(i).signed_sig)
            }

            io.out.scaleFP.get(i).exception := 0.U.asTypeOf(new RawFloatException)
        }

        // 4. FP4Product
        val FP4Product = Wire(Vec(ReduceWidth/4, SInt(10.W)))
        for (i <- 0 until ReduceWidth/4){
            FP4Product(i) := io.inAFP4Vec.get(i) * io.inBFP4Vec.get(i)
        }

        // 5. First level reduction
        for(i <- 0 until ReduceWidth/4/FP4P0AddNum){ 
            io.out.FP4ReduceRes.get(i) := FP4Product.slice(i * FP4P0AddNum, (i + 1) * FP4P0AddNum).reduce(_ + _)
        }
    }

    val isFp4WithScaleOp = io.opcode === FReduceComputeType.Nvfp4F32 || io.opcode === FReduceComputeType.Mxfp4F32
    for(i <- 0 until ReduceWidth/8){
        val AException = Wire(new RawFloatException)
        val BException = Wire(new RawFloatException)
        AException := io.inA.TF32Vec(i).exception
        BException := io.inB.TF32Vec(i).exception
        if (i < ReduceWidth/4/MinGroupSize) {
            FP4AScaleExceptionVec.zip(FP4BScaleExceptionVec).foreach {
                case (aScaleExceptionVec, bScaleExceptionVec) =>
                    AException := Mux(isFp4WithScaleOp, aScaleExceptionVec(i), AException)
                    BException := Mux(isFp4WithScaleOp, bScaleExceptionVec(i), BException)
            }
        }

        io.out.ExceptionVec(i).is_nan := AException.is_nan || BException.is_nan ||
            AException.is_inf && BException.is_zero ||
            BException.is_inf && AException.is_zero

        io.out.ExceptionVec(i).is_zero := AException.is_zero && !BException.is_inf ||
            BException.is_zero && !AException.is_inf

        io.out.ExceptionVec(i).is_inf := !AException.is_zero && BException.is_inf ||
            !BException.is_zero && AException.is_inf

        io.out.SignVec(i) := io.inA.TF32Vec(i).sign ^ io.inB.TF32Vec(i).sign
    }

    io.out.SignVec(ReduceWidth/8) := CDecode.sign
    MulExpVecSigned(ReduceWidth/8) := CDecode.exp.pad(9) //for debug

    io.out.ExceptionVec(ReduceWidth/8) := CDecode.exception
    val useDecodedCMantissa = Seq(
        Option.when(cuteFpeConfig.enableFp16Fp32)(io.opcode === FReduceComputeType.F16F16F32),
        Some(io.opcode === FReduceComputeType.BF16BF16F32),
        Option.when(cuteFpeConfig.enableTf32Fp32)(io.opcode === FReduceComputeType.TF32TF32F32),
        Option.when(cuteFpeConfig.enableFp32Fp32)(io.opcode === FReduceComputeType.Fp32F32),
        Option.when(cuteFpeConfig.enableMxfp8Fp32)(io.opcode === FReduceComputeType.Mxfp8e4m3F32),
        Option.when(cuteFpeConfig.enableMxfp8Fp32)(io.opcode === FReduceComputeType.Mxfp8e5m2F32),
        Option.when(cuteFpeConfig.enableNvfp4Fp32)(io.opcode === FReduceComputeType.Nvfp4F32),
        Option.when(cuteFpeConfig.enableMxfp4Fp32)(io.opcode === FReduceComputeType.Mxfp4F32),
        Some(io.opcode === FReduceComputeType.Fp8e4m3F32),
        Some(io.opcode === FReduceComputeType.Fp8e5m2F32)
    ).flatten.reduce(_ || _)

    io.out.CMantissa := Mux(useDecodedCMantissa, CDecode.signed_sig.pad(32), io.inC.asSInt)
    
    if (DEBUG_FP8) {
        printf("c mantissa: %x\n", io.out.CMantissa)
    }
}

// Pipe1的功能是将Pipe0的乘积向量尾数右移，得到尾数向量用于归约计算
class FReduceMACPipe1(implicit p: Parameters) extends CuteFpeModule {
    val io = IO(new Bundle{
        val in = Input(new FPipe0Result)
        val out = Output(new FPipe1Result)
    })

    // Main path
    // 1. Continue to find the maximum exponent
    val cmptreefp8p1 = Module(new CmpTreeP1(fp8cmptreelayers, ReduceWidth/8 + 1, 10))
    cmptreefp8p1.io.in := io.in.CmpTreefp8P0Result

    if (DEBUG_FP8)
    {printf("cmptreefp8p1.io.out: %x\n", cmptreefp8p1.io.out)}

    val productMaxExp = Wire(UInt(10.W))
    productMaxExp := cmptreefp8p1.io.out

    for(i <- 0 until ReduceWidth/8){
        io.out.RightShiftVec(i) := productMaxExp - io.in.MulExpVec(i)
        if (DEBUG_FP8)
        {printf("RightShiftVec[%d]: %x\n", i.U, io.out.RightShiftVec(i) )}
    }
    io.out.RightShiftVec(ReduceWidth/8) := productMaxExp - io.in.MulExpVec(ReduceWidth/8)

    // 2. Process exceptions
    // the next stage can determine the result based on HasNaN, HasPInf, HasNInf, OnlyNZero
    val NaNVec = Wire(UInt((ReduceWidth/8 + 1).W))
    val PInfVec = Wire(UInt((ReduceWidth/8 + 1).W))
    val NInfVec = Wire(UInt((ReduceWidth/8 + 1).W))
    val NZeroVec = Wire(UInt((ReduceWidth/8 + 1).W))

    NaNVec := Cat((0 until (ReduceWidth/8 + 1)).map(i => io.in.ExceptionVec(i).is_nan))
    PInfVec := Cat((0 until (ReduceWidth/8 + 1)).map(i => io.in.ExceptionVec(i).is_inf && !io.in.SignVec(i)))
    NInfVec := Cat((0 until (ReduceWidth/8 + 1)).map(i => io.in.ExceptionVec(i).is_inf && io.in.SignVec(i)))
    NZeroVec := Cat((0 until (ReduceWidth/8 + 1)).map(i => io.in.ExceptionVec(i).is_zero && io.in.SignVec(i)))

    val HasNaN = NaNVec.orR
    val HasPInf = PInfVec.orR
    val HasNInf = NInfVec.orR
    val OnlyNZero = NZeroVec.andR

    io.out.SumException := Cat(HasNaN, HasPInf, HasNInf, OnlyNZero)

    // FP4 Path
    if (cuteFpeConfig.enableFp4withsf) {
        val cmptreefp4p0 = Module(new CmpTreeP0(8, ReduceWidth/4/MinGroupSize + 1, 9))
        val ExpVec = Wire(Vec(ReduceWidth/4/MinGroupSize + 1, SInt(9.W)))
        for (i <- 0 until ReduceWidth/4/MinGroupSize + 1){
            ExpVec(i) := io.in.scaleFP.get(i).exp
            if (DEBUG_FP4 && cuteFpeConfig.enableFp4withsf) {
                printf("scaleFP[%d].exp: %x\n", i.U, io.in.scaleFP.get(i).exp)
            }
        }
        cmptreefp4p0.io.in := ExpVec.map(x => x.asUInt)
        io.out.CmpTreefp4P0Result.foreach(_ := cmptreefp4p0.io.out)

        // FP4Reduce sum
        for (i <- 0 until ReduceWidth/4/16){
            io.out.FP4ReduceRes.get(i) := io.in.FP4ReduceRes.get.slice(i * FP4P1AddNum, (i + 1) * FP4P1AddNum).reduce(_ + _)
            if (DEBUG_FP4) {
                printf("FP4ReduceRes[%d]: %x\n", i.U, io.out.FP4ReduceRes.get(i))
            }
        }

        io.out.scaleFP.zip(io.in.scaleFP).foreach{ case (out, in) => out := in }
    }

    io.out.Product0 := io.in.Product0
    io.out.Product1 := io.in.Product1
    // FP32: sum the four partial products into the 48-bit significand product
    // and keep bits 47..23, the same grid as a TF32 product shifted left by 3.
    io.out.fp32ProductOnTf32Grid.foreach { productOnTf32Grid =>
        for (element <- 0 until ReduceWidth/32) {
            val highTimesHigh = io.in.Product0(2 * element).asUInt(23, 0)
            val highTimesLow = io.in.Product0(2 * element + 1).asUInt(23, 0)
            val lowTimesHigh = io.in.Product1(2 * element).asUInt(23, 0)
            val lowTimesLow = io.in.Product1(2 * element + 1).asUInt(23, 0)
            val significandProduct = (highTimesHigh << 24) +& ((highTimesLow +& lowTimesHigh) << 12) +& lowTimesLow
            productOnTf32Grid(element) := significandProduct(47, 23)
        }
    }
    io.out.CMantissa := io.in.CMantissa
    io.out.MaxExp := productMaxExp
    io.out.opcode := io.in.opcode
    io.out.SignVec := io.in.SignVec
}

class FReduceMACPipe2(implicit p: Parameters) extends CuteFpeModule {
    val io = IO(new Bundle{
        val in = Input(new FPipe1Result)
        val out = Output(new FPipe2Result)
    })

    // Main path
    val Product1 = Wire(Vec(ReduceWidth/16, SInt(26.W)))
    val Product0 = Wire(Vec(ReduceWidth/16 + 1, SInt(26.W))) //每个向量的尾数右移结果

    val ProductRShift = Wire(Vec(ReduceWidth/8 + 1, SInt(26.W)))

    ProductRShift(ReduceWidth/8) := io.in.CMantissa(25, 0).asSInt >> io.in.RightShiftVec(ReduceWidth/8)

    //这里的问题是，如果c是负数，c为补码表示，右移完之后会变成全F，取负再相加后会变成1.与右移完之后变成0的情况不符
    if (DEBUG_FP8) {
        printf("ProductRShift_C: %x\n", ProductRShift(ReduceWidth/16))
    }
    
    val isFp32Op = cuteFpeConfig.enableFp32Fp32.B && io.in.opcode === FReduceComputeType.Fp32F32
    for(i <- 0 until ReduceWidth/16){
        val tempProduct0 = Wire(SInt(26.W))
        tempProduct0 := Cat(io.in.Product0(i)(22, 0), 0.U(3.W)).asSInt
        ProductRShift(i) := tempProduct0 >> io.in.RightShiftVec(i)
        // FP32 elements occupy the TF32 lanes; the other Product0 slots hold
        // their partial products and contribute nothing on their own.
        io.in.fp32ProductOnTf32Grid.foreach { productOnTf32Grid =>
            when(isFp32Op) {
                if (i < ReduceWidth/32) {
                    ProductRShift(i) := Cat(0.U(1.W), productOnTf32Grid(i)).asSInt >> io.in.RightShiftVec(i)
                } else {
                    ProductRShift(i) := 0.S
                }
            }
        }
        if (DEBUG_FP8) {
            printf("ProductRShift[%d]: %x\n", i.U,ProductRShift(i))
        }
    }

    for(i <- 0 until ReduceWidth/16){
        val tempProduct1 = Wire(SInt(26.W))
        tempProduct1 := Cat(io.in.Product1(i)(22, 0), 0.U(3.W)).asSInt
        ProductRShift(ReduceWidth/16 + i) := tempProduct1 >> io.in.RightShiftVec(ReduceWidth/16 + i)
        if (DEBUG_FP8) {
            printf("ProductRShift[%d]: %x\n", (i + ReduceWidth/16).U,ProductRShift(ReduceWidth/16 + i))
        }
    }

    for(i <- 0 until ReduceWidth/16 + 1){
        if(i != ReduceWidth/16){
            Product0(i) := Mux(io.in.opcode === FReduceComputeType.I8I8I32 || io.in.opcode === FReduceComputeType.I8U8I32 || io.in.opcode === FReduceComputeType.U8I8I32 || io.in.opcode === FReduceComputeType.U8U8I32,
                io.in.Product0(i).pad(26), ProductRShift(i))
        } else {
            Product0(i) := Mux(io.in.opcode === FReduceComputeType.I8I8I32 || io.in.opcode === FReduceComputeType.I8U8I32 || io.in.opcode === FReduceComputeType.U8I8I32 || io.in.opcode === FReduceComputeType.U8U8I32,
                0.S(26.W), ProductRShift(ReduceWidth/8))
        }
    }

    for (i <- 0 until ReduceWidth/16){
        Product1(i) := Mux(io.in.opcode === FReduceComputeType.Mxfp8e4m3F32 || io.in.opcode === FReduceComputeType.Mxfp8e5m2F32 || io.in.opcode === FReduceComputeType.Fp8e4m3F32 || io.in.opcode === FReduceComputeType.Fp8e5m2F32, ProductRShift(ReduceWidth/16 + i), io.in.Product1(i))
    }

    val ProductRShiftF = Wire(Vec(ReduceWidth/16, SInt((26 + log2Ceil(P2AddNum)).W)))
    val Int8Product = Wire(Vec(ReduceWidth/16, SInt((26 + log2Ceil(P2AddNum)).W)))

    for(i <- 0 until ReduceWidth/16){
        ProductRShiftF(i) := Mux(io.in.SignVec(i), -Product0(i), Product0(i)).pad(26 + log2Ceil(P2AddNum))
    }

    for(i <- 0 until ReduceWidth/16){
        Int8Product(i) := Mux(io.in.SignVec(ReduceWidth/16 + i), -Product1(i), Product1(i)).pad(26 + log2Ceil(P2AddNum))
    }

    val SumResult0 = Wire(Vec(P3AddNum, SInt((26 + log2Ceil(P2AddNum)).W)))
    val SumResult1 = Wire(Vec(P3AddNum, SInt((26 + log2Ceil(P2AddNum)).W)))
    for(i <- 0 until P3AddNum) {
        SumResult0(i) := ProductRShiftF.slice(i * P2AddNum, (i + 1) * P2AddNum).reduce(_ + _)
        SumResult1(i) := Int8Product.slice(i * P2AddNum, (i + 1) * P2AddNum).reduce(_ + _)
    }

    // FP4 path
    val FP4MaxExp = Option.when(cuteFpeConfig.enableFp4withsf)(Wire(UInt(9.W)))
    if (cuteFpeConfig.enableFp4withsf) {
        // 1. Get max exp for FP4 path
        val cmptreefp4p1 = Module(new CmpTreeP1(8, ReduceWidth/4/MinGroupSize + 1, 9))
        cmptreefp4p1.io.in := io.in.CmpTreefp4P0Result.get
        FP4MaxExp.get := cmptreefp4p1.io.out


        // 2. Shift for FP4
        val FP4Shift = Wire(Vec(ReduceWidth/4/MinGroupSize + 1, UInt(9.W)))
        for (i <- 0 until ReduceWidth/4/MinGroupSize + 1){
            FP4Shift(i) := FP4MaxExp.get.asUInt - io.in.scaleFP.get(i).exp.asUInt
        }

        // 3. Block dot multiply scale mantissa, and left shift
        for (i <- 0 until ReduceWidth/4/MinGroupSize){
            // 32 bits in tempFP4
            // reserve 3b on the left to reduce the impact of -1 accumulation caused by right shift of negative numbers
            val tempFP4 = Wire(SInt((32 + 3).W))
            tempFP4 := (io.in.FP4ReduceRes.get(i) * io.in.scaleFP.get(i).signed_sig).pad(32 + 3) <<
                Mux(
                    io.in.opcode === FReduceComputeType.Mxfp4F32, (15 + 3).U, (6 + 3).U
                )
            io.out.FP4ABShift.get(i) := tempFP4 >> FP4Shift(i)
            if (DEBUG_FP4 && cuteFpeConfig.enableFp4withsf) {
                printf("FP4Shift[%d]: %x\n", i.U, FP4Shift(i))
                printf("tempFP4[%d]: %x\n", i.U, tempFP4)
                printf("io.out.FP4ABShift[%d]: %x\n", i.U, io.out.FP4ABShift.get(i))
                printf("without pad io.out.FP4ABShift[%d]: %x\n", i.U, io.out.FP4ABShift.get(i)(34, 3))
            }
        }
        if (DEBUG_FP4 && cuteFpeConfig.enableFp4withsf) {
            printf("FP4Shift[8]: %x\n", FP4Shift(ReduceWidth/4/MinGroupSize))
            printf("io.out.FP4ABShift[8]: %x\n", io.out.FP4ABShift.get(ReduceWidth/4/MinGroupSize))
        }

        // 4. Process CMantissa for FP4 path
        val FP4signedCMantissa = Wire(SInt((32 + 3).W))
        FP4signedCMantissa := Cat(io.in.CMantissa, 0.U(3.W)).asSInt >> FP4Shift(ReduceWidth/4/MinGroupSize)
        io.out.FP4ABShift.get(ReduceWidth/4/MinGroupSize) := Mux(
          io.in.SignVec(ReduceWidth/8), -FP4signedCMantissa, FP4signedCMantissa
        )

        if (DEBUG_FP4) {
            printf("FP4MaxExp: %x\n", FP4MaxExp.get)
        }
    }

    io.out.ReduceRes0 := SumResult0
    io.out.ReduceRes1 := SumResult1
    io.out.CMantissa := Mux(io.in.opcode === FReduceComputeType.I8I8I32 || io.in.opcode === FReduceComputeType.I8U8I32 || io.in.opcode === FReduceComputeType.U8I8I32 || io.in.opcode === FReduceComputeType.U8U8I32,
        io.in.CMantissa, Mux(io.in.SignVec(ReduceWidth/8), -Product0(ReduceWidth/16), Product0(ReduceWidth/16)).pad(32))
    // Select MaxExp
    val isFp4WithScaleOp = io.in.opcode === FReduceComputeType.Nvfp4F32 || io.in.opcode === FReduceComputeType.Mxfp4F32
    io.out.MaxExp := io.in.MaxExp
    FP4MaxExp.foreach { fp4MaxExp =>
        io.out.MaxExp := Mux(isFp4WithScaleOp, fp4MaxExp.asUInt.pad(10) + 256.U, io.in.MaxExp)
    }
    if (DEBUG_FP4) {
        printf("io.out.MaxExp: %x\n", io.out.MaxExp)
    }
    io.out.opcode := io.in.opcode
    io.out.SumException := io.in.SumException
}

// Stage 3: Final reduction
class FReduceMACPipe3(implicit p: Parameters) extends CuteFpeModule {
    val io = IO(new Bundle{
        val in = Input(new FPipe2Result)
        val out = Output(new FPipe3Result)
    })

    val SumResult0 = Wire(Vec(P3AddNum, SInt((32 + 3).W)))
    val SumResult1 = Wire(Vec(P3AddNum, SInt((32).W)))
    val isFp4WithScaleOp = io.in.opcode === FReduceComputeType.Nvfp4F32 || io.in.opcode === FReduceComputeType.Mxfp4F32
    for(i <- 0 until P3AddNum) {
        val normalReduce = Cat(io.in.ReduceRes0(i).pad(32), 0.U(3.W)).asSInt
        SumResult0(i) := normalReduce
        io.in.FP4ABShift.foreach { fp4ABShift =>
            SumResult0(i) := Mux(isFp4WithScaleOp, fp4ABShift(i), normalReduce)
        }
        SumResult1(i) := io.in.ReduceRes1(i).pad(32)
    }

    val normalCAddend = Cat(io.in.CMantissa, 0.U(3.W)).asSInt
    val cAddend = Wire(SInt((32 + 3).W))
    cAddend := normalCAddend
    io.in.FP4ABShift.foreach { fp4ABShift =>
        cAddend := Mux(isFp4WithScaleOp, fp4ABShift(ReduceWidth / 4 / MinGroupSize), normalCAddend)
    }
    io.out.ReduceRes0 := (SumResult0.reduce(_ + _) + cAddend)(34, 3).asSInt
    
    io.out.ReduceRes1 := SumResult1.reduce(_ + _)

    if (DEBUG_FP8) {
        printf("CMantissa: %x\n", io.in.CMantissa)
    }

    io.out.CMantissa := io.in.CMantissa
    io.out.MaxExp := io.in.MaxExp
    io.out.opcode := io.in.opcode
    io.out.SumException := io.in.SumException
}

// Stage 4: Result selection, exception handling, normalization
class FReduceMACPipe4(implicit p: Parameters) extends CuteFpeModule {
    val io = IO(new Bundle{
        val in = Input(new FPipe3Result)
        val out = UInt(ResultWidth.W)
    })

    val ReduceResF = Wire(SInt(32.W))
    val ReduceResI = Wire(SInt(32.W))
    val ReduceRes  = Wire(SInt(32.W))
    val SumResult0 = Wire(SInt((32).W))
    val SumResult1 = Wire(SInt((32).W))
    
    SumResult0 := io.in.ReduceRes0.pad(32) //.pad(26 + log2Ceil(ReduceWidth/16))
    SumResult1 := io.in.ReduceRes1.pad(32)


    ReduceRes := Mux(
        (io.in.opcode === FReduceComputeType.I8I8I32 || io.in.opcode === FReduceComputeType.I8U8I32 ||
         io.in.opcode === FReduceComputeType.U8I8I32 || io.in.opcode === FReduceComputeType.U8U8I32 ||
         io.in.opcode === FReduceComputeType.Mxfp8e4m3F32 || io.in.opcode === FReduceComputeType.Mxfp8e5m2F32 ||
         io.in.opcode === FReduceComputeType.Fp8e4m3F32 || io.in.opcode === FReduceComputeType.Fp8e5m2F32),
        SumResult0 + SumResult1.pad(32), SumResult0
    )

    if (DEBUG_FP8 || DEBUG_FP4)
    {
        printf("SumResult0:%x\n", SumResult0)
        printf("SumResult1:%x\n", SumResult1)
        printf("ReduceRes:%x\n", ReduceRes)
    }

    ReduceResF := ReduceRes
    ReduceResI := ReduceRes

    val ExceptionBits = Wire(UInt(32.W))
    val IsException = Wire(Bool())
    IsException := io.in.SumException.orR || (ReduceRes === 0.S)

    ExceptionBits := Mux(io.in.SumException(3) || io.in.SumException(2) && io.in.SumException(1), 0x7FC00000.U,
        Mux(io.in.SumException(2), 0x7F800000.U, 
            Mux(io.in.SumException(1), BigInt("FF800000", 16).U, 0.U)
        )
    )

    //尾数符号位
    val ResultSign = Wire(Bool())
    ResultSign := ReduceResF(31)

    //去掉符号位，正直接转无符号数，负先有符号数取负再转无符号数
    val UnsignedSig = Wire(UInt(32.W))
    UnsignedSig := Mux(ResultSign, 
        (-ReduceResF.asSInt).asUInt, 
        ReduceResF.asUInt
    )
    if (DEBUG_FP8)
    {printf("UnsignedSig:%x\n", UnsignedSig)}

    val clz = Module(new CLZ(32))
    clz.io.in := UnsignedSig
    val LeadingZeros = Wire(UInt(5.W))
    LeadingZeros := clz.io.out
    if (DEBUG_FP8)
    {printf("LZ:%d\n", LeadingZeros)}
    //左移相同，该怎么左移就怎么左移,注意要把最高位的1省略掉作为规格化结果

    //修改一下这里规格化的逻辑
    //真正问题在乘法，乘法尾数是0，但阶码正常计算了
    //最终导致阶码比大小的时候，阶码较大的数尾数是0，不参与加法，而阶码第二大的数，尾数又要右移，导致丢精度，例如fp16_xtest.txt中的数据
    val ShiftMantissa = Wire(SInt(32.W))
    val ShiftExp = Wire(SInt(16.W))

    ShiftExp := (io.in.MaxExp - 511.U(10.W)).asSInt.pad(16) - LeadingZeros.pad(16).asSInt + 8.S(16.W)
    ShiftMantissa := Mux(LeadingZeros > 8.U, UnsignedSig.asSInt << (LeadingZeros - 8.U),
        UnsignedSig.asSInt >> (8.U - LeadingZeros) //这里右移导致移掉了低位的非零位，丢失信息，需要改
    )

    if (DEBUG_FP4) {
        printf("ShiftExp:%x\n", ShiftExp)
    }

    if (DEBUG_FP8 || DEBUG_FP4) {
        printf("ShiftMantissa:%x\n", ShiftMantissa)
    }

    val ResultExp = Wire(UInt(8.W))
    val ResultSig = Wire(UInt(24.W))
    ResultExp := Mux(ShiftExp < -126.S(16.W), 0.U, 
        Mux(ShiftExp > 127.S(16.W), 255.U, (ShiftExp + 127.S(16.W)).asUInt)
    )
    ResultSig := Mux(ShiftExp < -149.S(16.W), 0.U, 
        Mux(ShiftExp < -126.S(16.W), ShiftMantissa(23, 0).asUInt >> (-126.S(16.W) - ShiftExp).asUInt,
            Mux(ShiftExp > 127.S(16.W), 0.U, ShiftMantissa(23, 0).asUInt)
        )
    )

    val Result = Wire(UInt(ResultWidth.W))
    if (DEBUG_FP8 || DEBUG_FP4) {
        printf("Result:%x\n", Result)
    }
    Result := Cat(ResultSign, ResultExp, ResultSig(22, 0))
    io.out := Mux(
        (io.in.opcode === FReduceComputeType.I8I8I32 || io.in.opcode === FReduceComputeType.I8U8I32 ||
         io.in.opcode === FReduceComputeType.U8I8I32 || io.in.opcode === FReduceComputeType.U8U8I32),
        ReduceResI.asUInt, Mux(IsException, ExceptionBits, Result)
    )
}
