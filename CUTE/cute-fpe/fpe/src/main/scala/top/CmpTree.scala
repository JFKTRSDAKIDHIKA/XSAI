package cute

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config._

class CmpTreeP0Res(layers:Int, elemNum:Int, expWidth:Int)(implicit p: Parameters) extends CuteFpeBundle{
    val part1   = UInt((elemNum).W)   //第6层mask（也就是mask(5)）,用于算后续的mask
    val part2   = Vec(expWidth - layers, UInt((elemNum).W)) //6层以后的数据传到下一层
    val part3   = UInt(layers.W) //前6层的结果
}

class CmpTreeP0(layers:Int, elemNum:Int, expWidth:Int)(implicit p: Parameters) extends CuteFpeModule {
    val io = IO(new Bundle {
        val in    = Input(Vec(elemNum, UInt(expWidth.W)))
        val out   = Output(new CmpTreeP0Res(layers, elemNum, expWidth))
    })

  // 生成 in_vec_0: 9组，每组ReduceWidth/16位
    val inVec0 = Wire(Vec(expWidth, UInt((elemNum).W)))
    for (j <- 0 until expWidth) {
        inVec0(j) := Cat((0 until elemNum).reverse.map(i => io.in(i)(expWidth - 1 - j)))
    }

  // 生成 mask: 9组，每组ReduceWidth/16位
    val mask = Wire(Vec(layers, UInt((elemNum).W)))
    mask(0) := Mux(inVec0(0) === 0.U, ~inVec0(0), inVec0(0))
    for (i <- 1 until layers) {
        val prev = Wire(UInt((elemNum).W))
        val curr = Wire(UInt((elemNum).W))
        val prevandcurr = Wire(UInt((elemNum).W))
        prev := mask(i - 1)
        curr := inVec0(i)
        prevandcurr := (prev & curr)
        mask(i) := Mux(prevandcurr === 0.U, prev, prevandcurr)
    }

  // 生成 out: 9位
    io.out.part1 := mask(layers - 1)
    for(i <- 0 until (expWidth - layers)){
        io.out.part2(i) := inVec0(i + layers)
    }

    val res = Wire(Vec(layers, UInt(1.W)))
    for(i <- 0 until layers){
        val bit = Wire(UInt((elemNum).W))
        bit := mask(layers - 1) & inVec0(i)
        res(i) := Mux(bit === 0.U, 0.U, 1.U)
    }

    io.out.part3 := Cat((0 until layers).map(k => res(k)))
}

class CmpTreeP1(layers : Int, elemNum:Int, expWidth:Int)(implicit p: Parameters) extends CuteFpeModule {
    val io = IO(new Bundle {
        val in    = Input(new CmpTreeP0Res(layers, elemNum, expWidth))
        val out   = Output(UInt(expWidth.W))
    })

    val mask = Wire(Vec(expWidth - layers, UInt((elemNum).W)))
    for (i <- 0 until (expWidth - layers)) {
        val prev = Wire(UInt((elemNum).W))
        val curr = Wire(UInt((elemNum).W))
        val prevandcurr = Wire(UInt((elemNum).W))
        if(i != 0){
            prev := mask(i - 1)
        }else{
            prev := io.in.part1
        }
        curr := io.in.part2(i)
        prevandcurr := (prev & curr)
        mask(i) := Mux(prevandcurr === 0.U, prev, prevandcurr)
    }

    val res = Wire(Vec(expWidth - layers, UInt(1.W)))
    for(i <- 0 until (expWidth - layers)){
        val bit = Wire(UInt(elemNum.W))
        bit := mask(expWidth - 1 - layers) & io.in.part2(i)
        res(i) := Mux(bit === 0.U, 0.U, 1.U)
    }

    val partres = Wire(UInt((expWidth - layers).W))
    partres := Cat((0 until (expWidth - layers)).map(k => res(k)))

  // 生成 out: 9位
    io.out := Cat(io.in.part3, partres)
}
