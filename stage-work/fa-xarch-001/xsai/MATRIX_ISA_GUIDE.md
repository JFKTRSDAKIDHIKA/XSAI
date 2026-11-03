# XSAI matrix (CUTE) + RVV programmer's guide for a bare-metal FP32 FlashAttention kernel

Target: `DefaultMatrixFp32Config` (= `DefaultMatrixConfig` + `enableFp32Fp32`) in
`/das/ts01/home/jiashuao/AMENPU/XSAI` at HEAD `c1c8c6510`, plus the uncommitted working-tree
changes (`src/main/scala/top/Configs.scala`, `CUTE/src/main/scala/{TaskController,CUTEParameters}.scala`,
`CUTE/cute-fpe`, `difftest/.../mma_backend_cute_model.h`) that add the FP32 path. Researched on 2026-10-20 by reading the source only. No RTL was built or run.

Tag convention: `[V file:line]` = read in the source at that line. `[I]` = inferred, not proven by one line of code.
Path abbreviations:

| tag | path (relative to `XSAI/`) |
|---|---|
| `RC` | `rocket-chip/src/main/scala/rocket/Instructions.scala` |
| `MD` | `src/main/scala/xiangshan/backend/decode/MatrixDecoder.scala` |
| `DU` | `src/main/scala/xiangshan/backend/decode/DecodeUnit.scala` |
| `MC` | `src/main/scala/xiangshan/backend/decode/MCfgGen.scala` |
| `PK` | `src/main/scala/xiangshan/package.scala` |
| `MMA` | `src/main/scala/xiangshan/backend/fu/wrapper/Mma.scala` |
| `MSET` | `src/main/scala/xiangshan/backend/fu/wrapper/MSet.scala` |
| `MLS` | `src/main/scala/xiangshan/mem/pipeline/MlsUnit.scala` |
| `FEN` | `src/main/scala/xiangshan/backend/fu/Fence.scala` |
| `ACB` | `src/main/scala/xiangshan/backend/rob/AMUCtrlBuffer.scala` |
| `TC` | `CUTE/src/main/scala/TaskController.scala` |
| `TLE` | `CUTE/src/main/scala/TransposeLoadEngine.scala` |
| `CFG` | `src/main/scala/top/Configs.scala` |
| `PAR` | `src/main/scala/xiangshan/Parameters.scala` |

All encodings in section 1 were checked by machine. I compiled every macro in section 1.3 with
`/das/ts01/home/jiashuao/e2e/tools/riscv/bin/riscv64-unknown-elf-gcc` 15.1.0 / binutils 2.44 and
disassembled the result. A script then matched every 32-bit word against every `BitPat` in `RC`,
`src/main/scala/xiangshan/backend/decode/Instructions.scala` and `DU`. Each word matched exactly one
pattern, and that pattern was the intended instruction. [V, compile + match run]

---

## 0. Summary of the facts that matter for the kernel

* Registers: 4 tile registers `m0..m3`, which hold A or B. 4 accumulators, addressed as `m4..m7`. `mmacc` requires `md ∈ {4..7}` and `ms1, ms2 ∈ {0..3}`. [V MMA:49]
* FP32 limits for one `mmacc`: `mtilem ≤ 128`, `mtilen ≤ 128`, `mtilek ≤ 16` elements. Each tile row is 64 B. [V MMA:50-52, PAR:65-67, CFG:538-540]
* `mmacc md, ms1, ms2` computes `C[m][n] += Σ_k A[m][k] · B[n][k]`. B is held as **N rows × K**, the same layout as A. Therefore `S = Q·Kᵀ` loads K (row-major `[key][d]`) with plain `mlb`, and needs no transpose. For `O += P·V`, B must be Vᵀ. Load it with `mlbt` from row-major V (`mtilek ≤ 16` memory rows), or pre-transpose V and use `mlb`. [V TC:852-859, TLE:120-125, difftest `mma_backend_cpu.cpp:91-94`]
* Each tile register and each accumulator takes its element type from **mcfg entry = its own register number**. Set entries 0..7 to `12` (fp32) once with `msetcfg`. The reset value 0 means int4 and makes the MMA illegal. [V DU:1264, DU:1299, MC:53, MC:236]
* Strides are in **bytes**. The base and the stride must be **64-byte aligned**. Stores always write whole 64-byte lines, so a stored row should be a multiple of 64 B. [V TLE:124-125, CUTE2YGJK.scala:116-126, MultiChannelsCML.scala:69,724; I for the normal loaders]
* All matrix ops run in CUTE **after ROB commit**. Neither the core nor CUTE tracks memory dependences between matrix ops and core loads/stores. [V ACB:96-99, TC:17-24]
  * matrix store → core load: use `msc …; mrelease i; macquire i, n` before the RVV load.
  * core store → matrix load: put `mfence` (or `fence`) between the RVV store and the `mla`/`mlb`.
* `mrelease` waits only for **older matrix stores**. It does not wait for older matrix loads or `mmacc`. [V TC:1427-1430, difftest `checkers/matrix.cpp:167-174`]
* Bare metal: the emu runs the default flash code at `0x10000000`, which jumps to `0x80000000`, where `-i image.bin` is loaded. The program ends with `.insn i 0x6b,0,x0,a0,0` (`0x0005006b`); a0==0 means GOOD TRAP. UART TX is at `0x40600004`. FS and VS reset to Off, so `mstatus.FS` and `mstatus.VS` must be set before any FP or RVV instruction. The mstatus.MS check is commented out. `mtvec` resets to 0.
* RVV: VLEN = 128. `vfexp2.v` is `0x4e009057 | vs2<<20 | vd<<7`. It requires SEW 16 or 32, so set vtype inside the asm. Its FP32 error bound in the test is ≤ 2^13 ulp (relative error ≈ 1e-3).
* Build flags: `-march=rv64gcv_zfh_zvfh_zba_zbb_zbs_zicond -mabi=lp64d -mcmodel=medany`.

---

## 1. Instruction encodings

All matrix instructions use major opcode `0101011` = `0x2b` (custom-1) [V RC:312-356]. Field positions follow R-type:
`[31:25]=funct7  [24:20]=rs2  [19:15]=rs1  [14:12]=funct3  [11:7]=rd  [6:0]=0x2b`.

How matrix register numbers are encoded [V DU:768-773 `Imm_MATRIXREG = Cat(instr(22,20), instr(17,15), instr(9,7))`]:

* `md`/`ms` (destination or source of a load/store; destination of mmacc/mzero) = `instr[9:7]`. `instr[11:10]` must be 00.
* `ms1` (A of mmacc) = `instr[17:15]`. `instr[19:18]` must be 00.
* `ms2` (B of mmacc) = `instr[22:20]`. `instr[24:23]` must be 00.

So in `.insn r` form you write the matrix register number as `x<N>` in the rd/rs1/rs2 slot.

### 1.1 Encoding table (from the `BitPat`s in RC; bits that are not listed are fixed by the pattern)

| mnemonic | funct7 [31:25] | [24:20] | [19:15] | f3 | [11:7] | operands / meaning | source |
|---|---|---|---|---|---|---|---|
| `msetcfg` | `0100001` (0x21) | `00000` | rs1 = GPR value | 000 | `00`+entry[2:0] | mcfg[entry] := x[rs1] | RC:343 |
| `mgetcfg` | `0101000` (0x28) | `00000` | entry in rs1[2:0] | 000 | rd = GPR | x[rd] := mcfg[rs1[2:0]] | RC:317, DU:1259 |
| `msettilem` | `0010001` (0x11) | `00000` | rs1 = GPR | 000 | `00000` | mtilem := x[rs1] | RC:344 |
| `msettilen` | `0011001` (0x19) | `00000` | rs1 = GPR | 000 | `00000` | mtilen := x[rs1] | RC:346 |
| `msettilek` | `0001001` (0x09) | `00000` | rs1 = GPR | 000 | `00000` | mtilek := x[rs1] | RC:348 |
| `msettilemi` | `0010000` (0x10) | imm[9:5] | imm[4:0] | 000 | `00000` | mtilem := zext(imm10) | RC:345, DU:736-747 |
| `msettileni` | `0011000` (0x18) | imm[9:5] | imm[4:0] | 000 | `00000` | mtilen := zext(imm10) | RC:347 |
| `msettileki` | `0001000` (0x08) | imm[9:5] | imm[4:0] | 000 | `00000` | mtilek := zext(imm10) | RC:349 |
| `mla` | `0000000` (0x00) | rs2 = stride GPR | rs1 = base GPR | 001 | `00`+ms | tile ms ← A, mtilem rows × mtilek elems | RC:319 |
| `mlat` | `0100000` (0x20) | stride | base | 001 | `00`+ms | A from transposed memory (mtilek rows × mtilem elems) | RC:320 |
| `mlb` | `0001000` (0x08) | stride | base | 001 | `00`+ms | tile ms ← B, **mtilen rows × mtilek elems** | RC:322 |
| `mlbt` | `0101000` (0x28) | stride | base | 001 | `00`+ms | B from **mtilek rows × mtilen elems** (Kᵀ layout) | RC:323 |
| `mlc` | `0010000` (0x10) | stride | base | 001 | `00`+md | acc md ← mtilem rows × mtilen elems | RC:325 |
| `mlct` | `0110000` (0x30) | stride | base | 001 | `00`+md | transposed C load | RC:326 |
| `msa` | `0000001` (0x01) | stride | base | 001 | `00`+ms | **do not use**, see 1.2 | RC:335 |
| `msat` | `0100001` (0x21) | stride | base | 001 | `00`+ms | do not use | RC:336 |
| `msb` | `0001001` (0x09) | stride | base | 001 | `00`+ms | do not use | RC:338 |
| `msbt` | `0101001` (0x29) | stride | base | 001 | `00`+ms | do not use | RC:339 |
| `msc` | `0010001` (0x11) | stride | base | 001 | `00`+md | mem ← acc md, mtilem rows × mtilen elems | RC:341 |
| `msct` | `0110001` (0x31) | stride | base | 001 | `00`+md | transposed C store | RC:342 |
| `mla.whole` | `1000000` (0x40) | `00000` | base | 001 | `00`+ms | 128 rows × 64 B, stride fixed 64 B | RC:318 |
| `mlb.whole` | `1001000` (0x48) | `00000` | base | 001 | `00`+ms | 128 rows × 64 B, contiguous | RC:321 |
| `mlc.whole` | `1010000` (0x50) | `00000` | base | 001 | `00`+md | 128 rows × 128 e32, stride 512 B | RC:324 |
| `msc.whole` | `1010001` (0x51) | `00000` | base | 001 | `00`+md | inverse of mlc.whole | RC:340 |
| `msa.whole`, `msb.whole` | 0x41, 0x49 | | | 001 | | the pattern exists in RC but is **not in the decode table**, so the instruction is illegal | RC:334,337 vs MD:96-103 |
| `mmacc` | `0000000` | `00`+ms2 | `00`+ms1 | 010 | `00`+md | acc md += tile ms1 · tile ms2ᵀ | RC:327 |
| `mzero` | `0000000` | `00000` | `00000` | 011 | `00`+md | zero the whole acc (md 4..7) or tile (md 0..3) | RC:356 |
| `minit` | `0000000` | `00000` | `00000` | 000 | `00000` | mstatus.MS := Initial (01) | RC:329, `fu/wrapper/CSR.scala:55,86` |
| `msyncreset` | `1000000` (0x40) | sync idx (5 b) | `00000` | 100 | `00000` | msync[idx] := 0 | RC:350 |
| `mrelease` | `1001000` (0x48) | sync idx | `00000` | 100 | `00000` | msync[idx] += 1 when older matrix stores finish | RC:332 |
| `macquire` | `1010000` (0x50) | sync idx | rs1 = GPR threshold | 100 | `00000` | stall until msync[idx] ≥ x[rs1] | RC:312 |
| `mfence` | `1011000` (0x58) | `00000` | `00000` | 100 | `00000` | drain the core sbuffer | RC:316 |
| `mfence.pa`, `minval.pa` | 0x5f / 0x3f, opcode 0x73 | | | | | listed in RC (`HasMptCheck`) but not decoded by `MatrixDecoder` | RC:315 |

Ready-made words (fields set to 0 can be ORed in): msettilemi `0x2000002b | imm<<15`,
msettileni `0x3000002b | imm<<15`, msettileki `0x1000002b | imm<<15`, mmacc `0x0000202b | ms2<<20 | ms1<<15 | md<<7`,
mzero `0x0000302b | md<<7`, mfence `0xb000402b`, minit `0x0000002b`.

Words emitted by the compiled macros (disassembly of `/tmp/claude-1027/-das-ts01-home-jiashuao-AMENPU/d48364a8-456c-4c5f-9a34-b34060132d2b/scratchpad/enc/t.o`):
`msetcfg e0,a5 = 0x4207802b`, `msettilemi 128 = 0x2040002b`, `mla m0,(a0),a1 = 0x00b5102b`,
`mlb m2 = 0x10b5112b`, `mlbt m3 = 0x50b511ab`, `mlc m4 = 0x20b5122b`, `msc m6 = 0x22b5132b`,
`mmacc m4,m0,m1 = 0x0010222b`, `mzero m5 = 0x000032ab`, `mrelease 3 = 0x9030402b`,
`macquire 3,a2 = 0xa036402b`, `mfence = 0xb000402b`, `vfexp2.v v8,v9 = 0x4e909457`.

### 1.2 Operands and side effects of each instruction

**msetcfg** [V MD:59-61, DU:1256-1262, MC:91-105, MC:200-206, MSET:35-47]
* It writes mcfg entry `rd[2:0]` (8 entries [V MC:14]) with the full 64-bit value of x[rs1]. rd[4:3] must be 00 and rs2 must be 00000 (pattern).
* Raw value: `typeCode = raw[3:0]`, `tableSel = raw[7:4]` (must be 0), `raw[63:8]` is payload and ignored [V MC:204-206].
* Type codes [V MC:29-53]: 0 int4, 1 uint4, 2 int8, 3 uint8, 4 int32, 5 nvfp4, 6 mxfp4, 7 fp8e5m2, 8 fp8e4m3, 9 fp16, 10 bf16, 11 tf32, **12 fp32**.
* A code that is not supported or `tableSel ≠ 0` raises illegal-instruction on msetcfg itself [V MSET:42-46].
* msetcfg is decoded `blockBack = T, flushPipe = T`, so it serialises the pipeline and should run once at start-up [V MD:59-61].
* Entry ↔ register mapping:
  * MMA reads entries `rs1[2:0]` (A), `rs2[2:0]` (B), `rd[2:0]` (C), i.e. ms1, ms2 and md [V DU:1264].
  * A load/store reads entry `rd[2:0]`, i.e. the md/ms field [V DU:1299].
  * **So "entry e" is "the type of matrix register e".** For an FP32 kernel: `for e in 0..7: msetcfg e, 12`.
* MMA type legality for this config: A = fp32, B = fp32, C = fp32 is legal only when `enableFp32Fp32` is set [V MC:143,186, CFG:582-588]. TF32 (11/11/12) stays available as well.
* The reset value is 0 (int4) for all entries [V MC:236]. MMA with int4/int4/int4 is illegal (int4 needs C = int32 and `enableInt4Int32`). A matrix load with an unconfigured entry silently uses e4 width [V MLS:232-246].

**msettilem/n/k and the immediate forms** [V MD:66-72, MSET:13-25, DU:736-747, PK:461-497]
* `mtilex := x[rs1]` (register form) or `zext(instr[24:15])` (immediate form, 0..1023). No clamping happens here.
* Limits are checked by the instruction that **uses** the value. An out-of-range value raises illegal-instruction there [V MMA:50-52, MLS:226-271].
* These instructions are not serialising (`flushPipe = F, blockBack = F`) [V MD:67-72].
* mtilem/n/k are renamed registers. Each matrix instruction captures the values at rename, so they can be changed freely between instructions [V DU:1275-1277, DU:1290-1297].
* The values are readable with `csrr` at `mtilem = 0xcc4`, `mtilen = 0xcc5`, `mtilek = 0xcc6` [V RC:1390-1392, DU:1177-1179].

**Matrix loads and stores (mla/mlb/mlc/msc and variants)** [V MD:23-32, DU:1279-1302, MLS:194-271, MLS:553-578]
* rs1 is the base address (virtual; identical to physical in M-mode bare). rs2 is the **stride in bytes** between consecutive memory rows. md/ms is in `instr[9:7]`.
* Which mtile registers each instruction uses, with the memory shape for the non-transposed form:

| instr | `mtile0` (AmuLsuIO.row) | `mtile1` (AmuLsuIO.column) | memory rows × elements per row |
|---|---|---|---|
| mla | mtilem | mtilek | mtilem × mtilek |
| mlb | mtilek | mtilen | **mtilen × mtilek** (outer loop = column = N) |
| mlc / msc | mtilem | mtilen | mtilem × mtilen |

  Source: [V DU:1290-1297 (Mux1H on fuOpType(6,3): A→(m,k), B→(k,n), C→(m,n)); TC:729-738 (outer count of LoadB = column when not transposed, = row when transposed); TC:852-859 (B: N = column, K = row, `Stride_N = stride`)].

* Element width comes from mcfg[md/ms]. Whole-register forms force e8 for A/B and e32 for C, and ignore the mtile values [V MLS:203-204, MLS:562-575].
* Register index check:
  * A/B loads and stores need `ms ∈ 0..3`; C needs `md ∈ 4..7`. Otherwise illegal-instruction [V MLS:222-225].
  * Only `ms[1:0]` reaches CUTE [V TC:799, TC:851, TC:927].
* Size checks (illegal-instruction):
  * A: `mtilem ≤ 128`, `mtilek ≤ TRLEN/ew`.
  * B: `mtilen ≤ 128`, `mtilek ≤ TRLEN/ew`.
  * C: `mtilem ≤ 128`, `mtilen ≤ ARLEN/ew`.
  * For fp32: `mtilek ≤ 16`, `mtilen(C) ≤ 128` [V MLS:226-271].
* `msa/msat/msb/msbt`: they decode and pass the core checks. But the CUTE Store path always drives the **C** store engine (`CML_MicroTask_Config…StoreMicroTaskValid`) with `MatrixRegId = ms[1:0]`, whether or not `isacc` is set [V TC:1125-1165]. So `msa m1` would store **accumulator 5**, not tile 1 [I]. Do not use them.
* The CUTE scheduler does **not** order matrix memory ops by address. It orders them only by register dependences (RAW/WAW/WAR on tile and accumulator registers) [V TC:1415-1442]. Example: `msc X` followed by `mla X` without an intervening `mrelease`/`macquire` is a race [I].

**mmacc md, ms1, ms2** [V MD:107, MMA:19-75, DU:1263-1277]
* It reads mtilem, mtilen and mtilek, plus mcfg[ms1], mcfg[ms2] and mcfg[md].
* Illegal-instruction is raised when any of these holds: `md ∉ 4..7`, `ms1 ∉ 0..3`, `ms2 ∉ 0..3`, an mcfg triple that is not supported, `mtilem > 128`, `mtilen > 128`, `mtilen > ARLEN/ew(C)`, or `mtilek > TRLEN/ew(A)` [V MMA:45-52].
* It always accumulates. There is no form that overwrites C, so `mzero` (or `mlc`) the accumulator first.

**mzero md** [V MD:111, `fu/wrapper/Marith.scala`, TC:526-529, TC:982-995]
* md ≥ 4 zeroes the whole 128×128 e32 accumulator `md[1:0]`, independent of mtile and mcfg (`MatrixRegTensor_M = N = Tensor_MN`).
* md ≤ 3 zeroes tile `md` (ZeroTr).
* The scheduler orders it like a write to that register.

**msyncreset idx / mrelease idx / macquire idx, rs1** [V MD:114-131, FEN:66-80, FEN:102-149, `fu/wrapper/Mrelease.scala`, TC:578-599, TC:1427-1440]
* There are 32 counters `msyncRegs[0..31]`, 64 bits each, inside the core Fence unit [V PAR:172, FEN:68]. `csrr 0xcc7` (`msync`) returns 32, the number of counters [V RC:1393, `NewCSR/Unprivileged.scala:37-38,243-246`].
* The idx is `instr[24:20]`, the rs2 field used as a 5-bit immediate. For msyncreset and macquire it travels through rename as `imm = Cat(lsrc1, lsrc0)` and is read back as `imm(10,6)` [V `rename/Rename.scala:496-498`, FEN:79, FEN:127]. For mrelease it is `Imm_MSETVAL = instr[24:20]` [V DU:750-757].
* `msyncreset idx` (noSpec): `msync[idx] := 0`. If a release increments the same counter in the same cycle, the increment is dropped [V FEN:131-149].
* `mrelease idx` goes to CUTE after commit. CUTE issues it once **all older matrix *stores* in its window have completed**. "Completed" means every write response has come back from L2 [V TC:1427-1430, MultiChannelsCML.scala:812-819]. CUTE then signals the core, and the core does `msync[idx] += 1` [V FEN:136-149]. It does **not** wait for older mla/mlb/mlc/mmacc that have no store depending on them [V TC:1427-1430; also stated in `difftest/src/test/csrc/difftest/checkers/matrix.cpp:167-174`].
* `macquire idx, rs1`:
  * Decoded `noSpec = T, blockBack = T` [V MD:122-125]. It executes at the ROB head and spins until `msync[idx] ≥ x[rs1]` (unsigned compare) [V FEN:73, FEN:111].
  * Younger instructions are not dispatched until it completes [I from blockBack].
  * Usage pattern without resets: keep a software count `n`; after each `mrelease i`, do `n++; macquire i, n`.

**mfence** [V MD:126-130, FEN:103-110]
* noSpec + blockBack. It waits until the sbuffer is empty, i.e. all committed core stores are in L1D.
* It does **not** wait for CUTE. A plain RISC-V `fence` takes the same Fence-unit path (`s_wait` until `sbEmpty`) [V FEN:102,108].

**minit** [V MD:65-66, `fu/wrapper/CSR.scala:55,86`]
* A CSR op on mstatus (0x300) that writes `mstatus[26:25] (MS) := 01`. Optional, because the MS-off check is disabled (see 5.4).

### 1.3 Ready-to-use C header (compiled and verified)

```c
/* xsai_matrix.h -- XSAI CUTE matrix ISA, DefaultMatrixFp32Config.
 * Matrix register numbers (md/ms/ms1/ms2) and sync indices must be literal
 * integers (they are pasted into the .insn text).                          */
#pragma once
#include <stdint.h>
#define XM_STR_(x) #x
#define XM_STR(x) XM_STR_(x)

/* mcfg: entry e (0..7) = type of matrix register e; FP32 = 12 */
#define MCFG_FP32 12
#define MSETCFG(e, val)  __asm__ volatile(".insn r 0x2b, 0, 0x21, x" XM_STR(e) ", %0, x0" :: "r"((uint64_t)(val)) : "memory")
#define MGETCFG(e) ({ uint64_t _v; __asm__ volatile(".insn r 0x2b, 0, 0x28, %0, x" XM_STR(e) ", x0" : "=r"(_v)); _v; })

/* tile sizes in ELEMENTS */
#define MSETTILEM(v) __asm__ volatile(".insn r 0x2b, 0, 0x11, x0, %0, x0" :: "r"((uint64_t)(v)))
#define MSETTILEN(v) __asm__ volatile(".insn r 0x2b, 0, 0x19, x0, %0, x0" :: "r"((uint64_t)(v)))
#define MSETTILEK(v) __asm__ volatile(".insn r 0x2b, 0, 0x09, x0, %0, x0" :: "r"((uint64_t)(v)))
#define XM_TILEI(f7, imm) __asm__ volatile(".insn 4, %0" :: "i"((((f7) << 25) | (((imm) & 0x3ff) << 15) | 0x2b)))
#define MSETTILEMI(imm) XM_TILEI(0x10, imm)   /* imm: compile-time constant 0..1023 */
#define MSETTILENI(imm) XM_TILEI(0x18, imm)
#define MSETTILEKI(imm) XM_TILEI(0x08, imm)

/* loads/stores: base = byte address (64 B aligned), stride = bytes between rows (multiple of 64) */
#define XM_LS(f7, r, base, stride) __asm__ volatile(".insn r 0x2b, 1, " XM_STR(f7) ", x" XM_STR(r) ", %0, %1" \
        :: "r"(base), "r"((uint64_t)(stride)) : "memory")
#define XM_LSW(f7, r, base) __asm__ volatile(".insn r 0x2b, 1, " XM_STR(f7) ", x" XM_STR(r) ", %0, x0" :: "r"(base) : "memory")
#define MLA(ms, b, s)  XM_LS(0x00, ms, b, s)   /* ms 0..3: mtilem rows x mtilek elems           */
#define MLAT(ms, b, s) XM_LS(0x20, ms, b, s)   /* ms 0..3: memory = mtilek rows x mtilem elems */
#define MLB(ms, b, s)  XM_LS(0x08, ms, b, s)   /* ms 0..3: memory = mtilen rows x mtilek elems */
#define MLBT(ms, b, s) XM_LS(0x28, ms, b, s)   /* ms 0..3: memory = mtilek rows x mtilen elems */
#define MLC(md, b, s)  XM_LS(0x10, md, b, s)   /* md 4..7: mtilem rows x mtilen elems          */
#define MLCT(md, b, s) XM_LS(0x30, md, b, s)
#define MSC(md, b, s)  XM_LS(0x11, md, b, s)   /* md 4..7 */
#define MSCT(md, b, s) XM_LS(0x31, md, b, s)
/* MSA/MSAT/MSB/MSBT (f7 0x01/0x21/0x09/0x29) decode but store the ACCUMULATOR; not provided. */
#define MLA_WHOLE(ms, b) XM_LSW(0x40, ms, b)
#define MLB_WHOLE(ms, b) XM_LSW(0x48, ms, b)
#define MLC_WHOLE(md, b) XM_LSW(0x50, md, b)
#define MSC_WHOLE(md, b) XM_LSW(0x51, md, b)

/* compute */
#define MMACC(md, ms1, ms2) __asm__ volatile(".insn r 0x2b, 2, 0x00, x" XM_STR(md) ", x" XM_STR(ms1) ", x" XM_STR(ms2) ::: "memory")
#define MZERO(md) __asm__ volatile(".insn r 0x2b, 3, 0x00, x" XM_STR(md) ", x0, x0" ::: "memory")

/* sync */
#define MSYNCRESET(i) __asm__ volatile(".insn r 0x2b, 4, 0x40, x0, x0, x" XM_STR(i) ::: "memory")
#define MRELEASE(i)   __asm__ volatile(".insn r 0x2b, 4, 0x48, x0, x0, x" XM_STR(i) ::: "memory")
#define MACQUIRE(i, n) __asm__ volatile(".insn r 0x2b, 4, 0x50, x0, %0, x" XM_STR(i) :: "r"((uint64_t)(n)) : "memory")
#define MFENCE() __asm__ volatile(".insn r 0x2b, 4, 0x58, x0, x0, x0" ::: "memory")
#define MINIT()  __asm__ volatile(".insn r 0x2b, 0, 0x00, x0, x0, x0" ::: "memory")

/* vfexp2.v vd, vs2[, vm]: GAS macro that turns "vN" into N (GCC prints vector operands as vN) */
__asm__(
 ".macro vfexp2.v vd, vs2, vm=1\n"
 ".set .Lxm_vd, 32\n.set .Lxm_vs2, 32\n"
 ".irp n,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31\n"
 ".ifc \\vd,v\\n\n.set .Lxm_vd,\\n\n.endif\n"
 ".ifc \\vs2,v\\n\n.set .Lxm_vs2,\\n\n.endif\n"
 ".endr\n"
 ".if (.Lxm_vd > 31) || (.Lxm_vs2 > 31)\n.error \"vfexp2.v: operands must be v0..v31\"\n.endif\n"
 ".insn 4, 0x4c009057 | ((\\vm) << 25) | (.Lxm_vs2 << 20) | (.Lxm_vd << 7)\n"
 ".endm\n");
```

Notes on the header:
* `.insn r` cannot take vector register names. binutils 2.44 rejects `v8` in `.insn r`, which I verified. That is why `vfexp2.v` uses a GAS macro that emits `.insn 4, <word>`.
* The GPR operands (`%0`, `%1`) can be any register the compiler picks.
* The `"memory"` clobbers keep the compiler from moving C loads and stores across matrix ops. They provide **no** hardware ordering; section 3 covers that.

---

## 2. Architectural state and tile geometry (DefaultMatrixFp32Config)

| quantity | value | source |
|---|---|---|
| TLEN (one tile register) | 128·64·8 bit = **8 KiB** | [V CFG:538, PAR:65] |
| TRLEN (one tile row) | 64·8 bit = **64 B** | [V CFG:539] |
| MELEN | 32 | [V CFG:540] |
| ROWNUM = TLEN/TRLEN | **128** rows | [V PAR:674] |
| ARLEN (one accumulator row) = ROWNUM·MELEN | 4096 bit = 512 B = 128 × fp32 | [V PAR:675] |
| ALEN (one accumulator) | 128 × 128 × 32 bit = **64 KiB** | [V PAR:676] |
| CSRs `tlenb`/`trlenb`/`alenb` (0xcc1/2/3) | 8192 / 64 / 65536 | [V `NewCSR/Unprivileged.scala:22-34`, RC:1387-1389] |
| tile registers (A/B pool) | 4: `m0..m3` (any of them can hold A or B) | [V `CUTE/.../CUTEParameters.scala:795`, MMA:49] |
| accumulators | 4: `m4..m7` (CUTE C regs 0..3 = md[1:0]) | [V `CUTEParameters.scala:796`, TC:1072] |
| CUTE `Tensor_MN`/`Tensor_K` | 128 / 64 B | [V CFG:246-262, CFG:563-564] |
| PE array `Matrix_MN`, `ReduceWidthByte` | 8×8, 32 B (8 fp32 per reduce step) | [V `CUTEParameters.scala:294-301` (CUTE_8Tops, used through CUTE_128SCP), CFG:557] |
| CUTE issue window | 8 entries | [V `CUTEParameters.scala:621`] |
| core AMU control buffer | same size as the ROB; it back-pressures ROB enqueue when full | [V ACB:93-97] |
| msync counters | 32 | [V PAR:172] |

How mtilem/n/k are interpreted: they count **elements** of the operand type, not bytes.
* CUTE converts K to bytes with `computeKFromMsew` (`k << 2` for e32) [V TC:633-640].
* It then iterates `K_bytes / ReduceWidthByte` reduce steps [V TC:1076, 1085].
* Loaders use `bytes = dim << log2(ew)` and beats of 64 B, with a byte mask on the tail beat [V TC:653-676].

FP32 limits and granularity for one instruction:

| dim | legal maximum (checked in the core, otherwise illegal-instruction) | what CUTE actually computes (recommended granularity) |
|---|---|---|
| mtilem | 128 [V MMA:50, MLS:227] | rounded **up** to a multiple of 8 in the C path [V CDataController.scala:123]. Rows between mtilem and the round-up receive garbage partial sums [I]. Use a multiple of 8. |
| mtilen | 128 (and ≤ 128 = ARLEN/32) [V MMA:51, MLS:250-251] | rounded **down** to a multiple of 8 (`N/Matrix_MN`) [V ADataController.scala:113, CDataController.scala:173]. **Must be a multiple of 8.** For msc it should be a multiple of 16 (64 B rows, section 3.1). |
| mtilek | 16 (= 64 B / 4) [V MMA:52, MLS:255-256] | `(mtilek·4)/32` reduce steps, truncated [V TC:1076,1085]. **Must be 8 or 16.** Any other value silently drops the tail [I]. The difftest model would also flag the mismatch, because it models the tail lanes as valid [V `mma_backend_cpu.cpp:97-104`]. |

Partial tiles:
* A load with fewer rows writes only those rows. Bytes beyond `ew·cols` in the last 64 B beat are masked out of the register write [V TC:670-676, MultiChannelsCML.scala:458-471].
* Register contents outside the loaded region keep stale data [I]. That is harmless as long as the mtile values given to mmacc and msc cover only the loaded region.
* Every `mmacc` with fp32 is at most 128 × 128 × 16 = 262,144 MACs. With 8×8 PEs × 8 fp32 lanes per cycle, that is ≥ 512 CUTE cycles at full rate [I, peak only].

Tile register usage for D = 64 (FP32): a 128-row block of Q (or P) is **4 k-slices of 16 columns**, but only 4 tile registers exist. Plan one `mla`+`mlb` pair per `mmacc` and ping-pong `m0/m1` with `m2/m3`.

---

## 3. Memory semantics

### 3.1 Layout, stride, alignment

* Row-major. Memory row `r` of an operand starts at `base + r·stride`, and its elements are contiguous at `ew` bytes each.
  * A loader: `base + m·Stride_M + beat·64` [V MultiChannelsAML.scala:293 + TC:806].
  * B loader: `Tensor_Block_BaseAddr + n·Stride_N + kbeat·64` [V BMemoryLoader.scala:245].
  * C loader: `base + m·Stride_M + nbeat·64` [V CMemoryLoader.scala:376].
  * C store: `base + m·Stride_M + …` [V MultiChannelsCML.scala:716].
  * The stride is a byte count, taken from x[rs2] [V MLS:203-204, MLS:564].
* Every memory request is a 64 B TileLink `Get`/`Put` at `RequestAddr` [V CUTE2YGJK.scala:116-126]. TileLink requires the address to be aligned to the access size [I, TL rule], so **base and stride must be multiples of 64 B**. The transpose engine asserts exactly this: "transpose has the ordinary-load alignment contract" [V TLE:124-125].
* Stores: the C store engine always sends a full mask [V MultiChannelsCML.scala:69,724]. L2 accepts only full-line, full-mask `PutFullData` from the matrix unit [V `XSAICache/.../coupledL2/SinkA.scala:49-56`] [V CUTE2YGJK.scala:110-113].
  * A stored row of `mtilen·4` bytes that is not a multiple of 64 overwrites the rest of its last 64 B line with register contents [I].
  * Keep `mtilen` a multiple of 16 and the stride a multiple of 64 when storing.
* Address translation: only the base goes through the DTLB/PMP in the core pipeline [V MLS:145-148, MLS:563]. CUTE then walks rows by itself with physical addresses [I]. In M-mode bare, virtual = physical.
* `mlbt` (transpose engine) [V TC:854-899, TLE:120-125, TLE:278-290]:
  * The memory region is `mtilek` rows × `mtilen` elements.
  * Asserted: `mtilek·4 ≤ 64` (≤ 16 rows for fp32), `mtilen ≤ 128`, e4 not allowed.
  * It runs on the **A-loader hardware**, so `mla` and `mlbt` serialise against each other [V TC:557-559, TC:1460-1463]. A plain `mlb` of a pre-transposed Vᵀ runs on the separate B loader and can overlap with `mla`.

### 3.2 Coherence path

* CUTE's TileLink node is connected into the core's L2 inner bus (`l2top.inner.misc_l2_pmu := cute_logger := node`) [V `xiangshan/XSTile.scala:97-111`]. CUTE therefore goes through the private L2 (XSAICache CoupledL2), not around it.
* L2 is inclusive with a client directory:
  * A matrix `Get` probes L1D when the line is held in TRUNK (dirty-capable) state.
  * A matrix `PutFullData` always probes or invalidates L1D holders and leaves `clients = 0` [V `XSAICache/.../coupledL2/MainPipe.scala:256-260, 574-576`].
  * So L1D never returns stale data after a completed matrix store, and a matrix load sees data that has already reached L1D [I from those lines].
* What the hardware does **not** do:
  * Matrix ops reach CUTE only after ROB commit, in program order among themselves [V ACB:96-99, ACB:245-251].
  * Nothing in the core waits for the store buffer (sbuffer) before sending them. `AMUCtrlBuffer` has no sbuffer interlock, and `MlsUnit` has no store-forwarding or sbuffer check [V ACB whole file; MLS:91-648 has none].
  * Younger core loads can execute before an older matrix store has written memory.

### 3.3 Required sync sequences

(a) **mmacc → msc → RVV load of the stored data**
```c
MMACC(4, 0, 1);            // ... last k-slice
MSC(4, S_buf, ldS_bytes);  // goes to CUTE after commit
MRELEASE(0);               // completes after the msc's last write ack (TC:1427-1430, CML:812-819)
MACQUIRE(0, ++sync0);      // spins at ROB head until msync[0] >= sync0; younger loads wait (blockBack)
/* now RVV vle32 of S_buf is safe */
```
* `mfence` alone is **not** enough: it only drains the core sbuffer [V FEN:109].
* The msc does not need a separate wait for the mmacc. CUTE makes the store wait for the producing compute (RAW on the accumulator) [V TC:1432].

(b) **RVV store of P → mla of P**
```c
__riscv_vse32_v_f32m4(P_buf + ..., p, vl);   // ... all rows of P
MFENCE();                  // or "fence rw,rw": waits until the sbuffer is empty (FEN:103,109)
MLA(2, P_buf, ldP_bytes);  // CUTE's Get now finds the data in L1D (probe) or L2
```

(c) **WAR on memory buffers.** This is the hazard that is easy to miss. CUTE reads `P_buf`, `K` and `V` asynchronously, after commit. A later RVV store into `P_buf` (the next block's P) can land before CUTE has finished reading the old P. `mrelease` does not cover loads [V TC:1427-1430]. Two ways to make it safe:
1. **Double-buffer P (and S)** in memory, and make sure a store-bearing release happens between producer reuses.
2. Create a register dependence: load the next block's `K` into the **same tile register** that held the old `P`.
   * CUTE then makes that `mlb` wait for the old `mla P` to *complete* (WAW on the tile) [V TC:1433].
   * The next `mmacc S`, `msc S` and `mrelease` chain behind it [V TC:1431-1433].
   * So `macquire` on the next S release also proves that the old P has been read [I, follows from the dependence rules].

(d) **msc then mlc/mla of the same buffer** (e.g. rescaling O through memory) needs `mrelease`/`macquire` in between. Add `mfence` too if the core modifies the data in between.

Note: no matrix instruction scales accumulator rows. Rescaling O for an online softmax (`O *= exp(m_old − m_new)` per row) needs `msc` O → RVV → `mlc` O, i.e. 2 × 32 KiB per 128×64 block. The alternative is an algorithm that fixes the row maximum first (two-pass / max-first), so that O accumulates in an accumulator untouched [I, design implication].

---

## 4. MMA semantics

* `C[m][n] += Σ_{k<mtilek} A[m][k] · B[n][k]` for `m < mtilem`, `n < mtilen`. A and B are both stored in tile registers as *rows × K*. This is reproduced by the difftest CPU reference: `src1_row = src1 + i·row_bytes`, `src2_row = src2 + j·row_bytes`, dot product over k, accumulated into `src3[i][j]` [V `difftest/src/test/csrc/difftest/mma/backend/mma_backend_cpu.cpp:87-109`].
* Memory → register mapping:
  * `mlb` reads memory row `n` (`mtilen` of them) with `mtilek` elements each, i.e. B in memory is N × K row-major [V TC:852-859, TC:729-738].
  * `mlbt` reads memory row `k` with `mtilen` elements each, i.e. Bᵀ (K × N) row-major [V TLE:8, TLE:283-290, TC:854-856].
* For attention:
  * `S = Q · Kᵀ`: A = Q[q0:q0+M, 16c:16c+16] via `mla` (base = `Q + q0·256 + 64c`, stride = 256 B). B = K[j0:j0+N, 16c:16c+16] via `mlb` (base = `K + j0·256 + 64c`, stride 256 B). **No transpose needed.**
  * `O += P · V`: A = P[:, 16c:16c+16] via `mla`. B[n=d][k=j] = V[j][d] via `mlbt` from V rows j0+16c … +15 (base = `V + (j0+16c)·256`, stride 256 B, `mtilek = 16`, `mtilen = 64`). The alternative is `mlb` of a pre-transposed `Vt[d][j]`.
* C must be initialised, because there is no overwrite form. Use `MZERO(md)` (whole accumulator, cheap and independent of mtile) or `MLC`.
* Rounding mode bits are passed in the command (`rm = mfrm` for FP) [V MMA:63]. FP32 numerics, from the bit-exact difftest model, which DUT results are `memcmp`-checked against [V `mma_backend_cute_model.h:203-327`, `mma_backend_cpu.cpp:111`]:
  * Each 8-element K chunk (32 B) is reduced separately.
  * The 24×24-bit significand product is computed exactly, then **truncated** to 23 fraction bits.
  * All 8 products and C are aligned to the chunk's max exponent, with bits shifted out dropped.
  * They are summed in 28/32-bit two's-complement fixed point, and normalised to fp32 by truncation (no round-to-nearest).
  * NaN/Inf follow IEEE classes. Denormal results are flushed or truncated as in `normalize_fp32`.
  * Expect errors of a few ulp of the largest term per chunk, not an IEEE-exact sum.
  * Config comment: "CUTE runs it on the TF32 lanes at the TF32 rate, with full 24-bit significand products" [V CFG:579-581].

---

## 5. Bare-metal start-up on the XSAI emu

### 5.1 Reset and image loading

* Reset PC is `0x10000000` (flash; difftest `FIRST_INST_ADDRESS` for KMHV2) [V `difftest/config/config.h:54-58`, `src/test/scala/top/SimMMIO.scala:37,49`].
* With no `-F/--flash` argument, the emu fills the flash with [V `difftest/src/test/csrc/common/flash.cpp:63-75`]:
  `csrrsi x0, mnstatus, 8; addiw t0,zero,1; slli t1,t0,42; csrrc x0,mstatus,t1 (clear MDT); slli t0,t0,31; jr t0` → jumps to **0x80000000** in M-mode.
* `-i image` is loaded at `PMEM_BASE = 0x80000000` [V `difftest/config/config.h:48`, `common/ram.cpp:333-343`]. Raw `.bin`, gz, zstd or ELF are accepted.
* Run command, using the env from `stage-work/fa-xarch-001/xsai/env.sh`:
  ```
  $NOOP_HOME/build/emu -i fa.bin --no-diff -C 200000000 2>emu.err
  ```
  * `--no-diff`, `-C/--max-cycles` and `-i` are standard emu options [V `common/args.cpp:168-214,377`].
  * The local Ramulator2 build exits unless `CORALBENCH_RAMULATOR_LIB` and `CORALBENCH_RAMULATOR_CONFIG` are set [V difftest working-tree diff of `common/ram.cpp`].
  * `CORALBENCH_DUMP=<offset>:<bytes>:<path>` dumps *backing DRAM* at exit [V same diff]. Lines still dirty in the caches are not in that dump [I], so print results over the UART or check them in-program.
  * The bundled NEMU `.so` contains matrix mnemonics, but its FP32-MMA support was not checked. Use `--no-diff` [I].
* Ending the program: the XiangShan trap instruction `TRAP = 000000000000 rs1 000 00000 1101011`, code = x[rs1] [V DU:574-577]. Use `.insn i 0x6b, 0, x0, a0, 0` (= `0x0005006b`). a0 = 0 prints "HIT GOOD TRAP" [V `emu/emu.cpp:697-698`], anything else is a bad trap.
* UART: UARTLite at `0x40600000`, TX register at offset 4. Write the character with `sw`/`sb` to `0x40600004` [V `system/SoC.scala:140`, `device/AXI4UART.scala:40-49`, `SimMMIO.scala:50`]. A UART16550 also exists at `0x310b0000` [V SoC.scala:141].

### 5.2 CSR state that must be set

* `mstatus.FS` (bits 14:13) and `mstatus.VS` (bits 10:9) reset to **Off** [V `NewCSR/MachineLevel.scala:504-506`]. Decode raises illegal-instruction for FP and vector ops while they are Off [V DU:970-976].
  * Set both before any compiler-generated code runs. GCC 15 with `v` in `-march` may emit vector code for struct copies and memset.
* `mstatus.MS` (bits 26:25) resets Off [V MachineLevel.scala:515], but the `msIsOff` illegal check is **commented out** in decode [V DU:977]. Setting it (or `MINIT`) is harmless and future-proof.
* `mtvec` resets to 0 [V `NewCSR/CSRBundles.scala:18-19`]. Without a handler, any illegal-instruction (e.g. a bad mtile) jumps to address 0. Install a handler that bad-traps with `mcause`.
* Cycle counter: `rdcycle` (`csrr cycle`, 0xc00) or `csrr mcycle` (0xb00). `mcountinhibit.CY` resets 0, so the counter runs [V MachineLevel.scala:333-338, 701-705].

### 5.3 Start-up file and linker script (compiled; entry verified at 0x80000000)

```asm
/* crt.S */
  .section .text.init, "ax"
  .globl _start
_start:
  la    t0, trap_entry
  csrw  mtvec, t0
  li    t0, (1 << 13) | (1 << 9) | (1 << 25)   /* FS=Initial, VS=Initial, MS=Initial */
  csrs  mstatus, t0
  .option push
  .option norelax
  la    gp, __global_pointer$
  .option pop
  la    sp, _stack_top
  la    t0, _bss_start
  la    t1, _bss_end
1: bgeu t0, t1, 2f
  sd    zero, 0(t0)
  addi  t0, t0, 8
  j     1b
2: call  main
  .insn i 0x6b, 0, x0, a0, 0          /* XiangShan trap: a0 == 0 -> GOOD TRAP */
3: j 3b
  .align 2
trap_entry:
  csrr  a0, mcause
  ori   a0, a0, 0x100                 /* nonzero -> BAD TRAP, code shows mcause */
  .insn i 0x6b, 0, x0, a0, 0
4: j 4b
```
```ld
/* link.ld */
OUTPUT_ARCH(riscv)
ENTRY(_start)
SECTIONS
{
  . = 0x80000000;
  .text : { *(.text.init) *(.text .text.*) }
  .rodata : ALIGN(64) { *(.rodata .rodata.* .srodata .srodata.*) }
  .data : ALIGN(64) { *(.data .data.*) }
  .sdata : { __global_pointer$ = . + 0x800; *(.sdata .sdata.*) }
  .bss (NOLOAD) : ALIGN(64) { _bss_start = .; *(.sbss .sbss.* .bss .bss.* COMMON) . = ALIGN(8); _bss_end = .; }
  . = ALIGN(4096);
  . += 0x10000;
  _stack_top = .;
  _end = .;
}
```
Build (verified to link):
```
T=/das/ts01/home/jiashuao/e2e/tools/riscv/bin
$T/riscv64-unknown-elf-gcc -O2 -march=rv64gcv_zfh_zvfh_zba_zbb_zbs_zicond -mabi=lp64d -mcmodel=medany \
   -ffreestanding -fno-builtin -nostdlib -nostartfiles -T link.ld crt.S fa.c -lgcc -o fa.elf
$T/riscv64-unknown-elf-objcopy -O binary fa.elf fa.bin
```
* `-mcmodel=medany` is required, because code at 0x80000000 is outside the ±2 GiB window around 0 that `medlow` can address [I, standard RISC-V code-model rule].
* The toolchain has a single multilib (default `rv64imafdc…/lp64d`) [V `-print-multi-lib`]. Linking its `libgcc` with RV64GCV objects works.
* Give 64-byte alignment (`__attribute__((aligned(64)))`) to every matrix buffer.

### 5.4 Old test `/das/ts01/home/jiashuao/e2e/xsai-matrix-test/matrix_test.c`: defects found

1. `MSETCFG` builds `(0x421U<<20)|…`, which sets rs2 = 1. `MSETCFG` requires rs2 = 00000 [V RC:343]. The word `0x4212802b` matches **no** pattern in RC or DU (checked), so it raises illegal-instruction, and with mtvec = 0 it jumps to address 0. Correct value: `0x420U<<20`, or use the macro in 1.3.
2. Its reference computes `Σ_k A[i·K+k]·B[k·N+j]`, i.e. B as K × N, but loads B with `mlb`. `mlb` treats memory as N × K. The reference must use `B[j·K+k]`, or the test must use `mlbt`.
3. FS and VS are never enabled, there is no `mtvec` handler, and BSS is not cleared.
4. Everything else matches the decoder: MZERO, MMACC (ms2 in [22:20], ms1 in [17:15], md in [9:7]), MSETTILE{M,K,N}I, MSYNCRESET, MRELEASE, MACQUIRE, MINIT, MLA/MLB/MSC, and entries 0/1/4 holding the types of m0/m1/m4.

---

## 6. RVV

* VLEN = 128, ELEN = 64 [V PAR:60-61]. The ISA string includes `v, zfh, zvfh, zvfhmin, zba, zbb, zbs, zicond, zfa, zvbb, …` [V PAR:381-394]. For e32: m1 = 4 elements, m8 = 32 elements.
* `vfexp2.v vd, vs2, vm` (custom, Zvfexp) [V `backend/decode/Instructions.scala:33-35`, `VecDecoder.scala:620`]:
  * Fields: `funct6 = 010011`, `vm = instr[25]` (1 = unmasked), `vs2 = [24:20]`, `[19:15] = 00001` (sub-opcode in the VFUNARY1 slot), `funct3 = 001` (OPFVV), `vd = [11:7]`, opcode `1010111`.
  * Word = `0x4c009057 | vm<<25 | vs2<<20 | vd<<7`; unmasked `0x4e009057 | vs2<<20 | vd<<7`.
  * Enabled because `HasVfexp2 = true` by default [V PAR:85, VecDecoder.scala:835-839].
  * Semantics: elementwise `vd[i] = 2^vs2[i]` in the current SEW. SEW must be 16 (fp16) or 32 (fp32), otherwise illegal-instruction [V `VecExceptionGen.scala:120-122, 205`]. Rounding follows `frm` [V `fu/wrapper/VFExp2.scala` (`mod.io.rm := rm`)].
  * Special cases, per the reference in the unit test: NaN → canonical NaN, +inf → +inf, −inf → +0, **subnormal input → 1.0** [V `yunsuan/src/test/scala/vector/vexp2/VfExp2UlpSpec.scala:388-405`].
  * Accuracy: the FP32 test asserts `bitLength(ulp_error) ≤ 13`, i.e. error < 2^13 ulp (relative error ≲ 2^-10 ≈ 1e-3) against `StrictMath.pow(2,x)`, for all rounding modes. FP16 and BF16 are ≤ 1 ulp [V VfExp2UlpSpec.scala:766-777, 964-1030].
  * For softmax: `exp(x − m) = vfexp2(vfmul(x − m, log2e))`.
* **GCC pitfall, verified by compiling.** GCC's VSETVL pass may leave a different SEW/LMUL active before an inline asm. Example: `vsetivli e8,mf4` was chosen ahead of a `vle32`. A bare `vfexp2.v` would then run with SEW = 8 and trap. Always set vtype inside the asm and declare the clobbers; GCC then re-inserts its own `vsetvli` afterwards (verified):
  ```c
  static inline vfloat32m4_t vfexp2_f32m4(vfloat32m4_t x, size_t vl) {
    vfloat32m4_t y;
    __asm__ ("vsetvli zero, %2, e32, m4, ta, ma\n\tvfexp2.v %0, %1"
             : "=&vr"(y) : "vr"(x), "r"(vl) : "vl", "vtype");
    return y;
  }
  ```
* march for `/das/ts01/home/jiashuao/e2e/tools/riscv/bin/riscv64-unknown-elf-gcc` (GCC 15.1.0): `-march=rv64gcv_zfh_zvfh_zba_zbb_zbs_zicond -mabi=lp64d -mcmodel=medany` (verified to compile and link). `zfa`, `zbc` and `zvbb` are also in the core's ISA string if wanted [V PAR:390-394]. The core has VLEN = 128, matching GCC's `zvl128b` default for `v`.

---

## 7. Suggested FP32 FA block schedule (Mq ≤ 128 query rows, Nk = 128 keys per block, D = 64)

Set-up, once:
`MSETCFG(0..7, 12)`, `MSYNCRESET(0)` (or keep a running count), and enable FS/VS in crt.

Per key block j:

1. **S block**:
   * Set `MSETTILEM(Mq); MSETTILENI(128); MSETTILEKI(16); MZERO(4)`.
   * For c in 0..3, ping-ponging tiles {0,1} / {2,3}:
     `MLA(a, Q + q0·256 + 64c, 256); MLB(b, K + j0·256 + 64c, 256); MMACC(4, a, b)`.
   * Then `MSC(4, S_buf[j&1], 512); MRELEASE(0); MACQUIRE(0, ++n)`.
2. **Softmax on RVV**, reading `S_buf` rows (stride 512 B):
   * Compute the row max and `p = vfexp2((s − m)·log2e)` (fp32), plus the row sums.
   * Write `P_buf[j&1]` with row stride 512 B, 64 B aligned.
   * Then `MFENCE()`.
3. **PV**:
   * Set `MSETTILENI(64); MSETTILEKI(16)`.
   * For c in 0..7: `MLA(p, P_buf + 64c, 512); MLBT(v, V + (j0 + 16c)·256, 256); MMACC(5, p, v)`.
   * This accumulates into the O accumulator (m5).
   * Online-softmax rescaling of O needs an `msc`/`mlc` round trip (section 3.3, note at the end). Alternatively use a max-first pass, so that m5 accumulates directly. Divide by the row sum at the end.
4. Store O with `MSC(5, O + q0·256, 256)`, then `MRELEASE`/`MACQUIRE` before the scalar check or print.

Hazards to respect:
* `mlbt` and `mla` share the A loader (3.1).
* Matrix ops have no address ordering among themselves (1.2).
* WAR on `P_buf`/`S_buf` (3.3 c).
* `mtilek` must be 8 or 16, and `mtilen` a multiple of 8, or 16 when stored (2).
