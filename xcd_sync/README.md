# xcd id + cross-workgroup atomic sync cost

Unit tests for two things, mainly aimed at multi-XCD CDNA parts (gfx942 /
gfx950) but arch-portable down to gfx90a and up to gfx12:

* reading the XCD (XCC) id from inside a kernel, and checking the block -> XCD
  mapping is what we think it is
* how long one grid-wide sync built out of a single `atomicAdd` actually takes

Every test launches **256 blocks x 1 thread**. With one block per CU all blocks
are resident at once, which is what makes it legal for a block to spin-wait on
the others.

```
sh rebuild.sh                  # defaults to gfx950 -> build/gfx950/test.exe
ARCH=gfx1250 sh rebuild.sh     #              gfx1250 -> build/gfx1250/test.exe

./build/gfx950/test.exe                    # default 64 trace iters, 10000 loop iters
./build/gfx950/test.exe 64 10000           # [trace_iters] [loop_iters]
```

The build dir is per-arch so the two can coexist. `-save-temps` leaves the ISA
next to the binary as `build/<arch>/main.hip-hip-amdgcn-amd-amdhsa-<arch>.s`.

Exit code is non-zero if any check fails. Builds clean for gfx90a / gfx942 /
gfx950 / gfx1201 / gfx1250; only gfx950 has been run (see the gfx1250 section
at the bottom).

## the tests

**test 1 - xcd id.** `s_getreg(HW_REG_XCC_ID)` per block, then four checks:
every id is `< hipDeviceAttributeNumberOfXccs`, the 256 blocks spread evenly
over the XCDs, the mapping is the round-robin `blockIdx.x % num_xcc`, and the
physical `(xcd, se, sa, cu)` location is unique (i.e. no CU got two blocks).
On an arch with no XCC_ID the first three are reported as SKIP; the last one
still runs and doubles as a check that the hwreg field layout is right, since
wrong bit offsets make the ids collide.

**test 2 - one sync, traced.** Each block stamps `wall_clock64()` before the
`atomicAdd` and after the spin, for every iteration, and the host reconstructs
the timeline. `wall_clock64()` is a constant device-wide 100 MHz clock, so
timestamps taken on different XCDs are directly comparable — `clock64()` is
not, don't use it here. The test also verifies no block was released before the
last block arrived.

**test 3 - N syncs in one launch.** The same barrier `loop_iters` times inside
one kernel, timing only the total, so per-iteration stores don't perturb the
measurement and the launch overhead is divided away. Also runs a 1-iteration
kernel to measure the launch overhead separately and subtract it.

**test 4 - reference.** `N` contended `atomicAdd` on the same address with
nobody waiting, plus the cost of `wall_clock64()` itself. These are the floor
the barrier can't go below.

The barrier counter is never reset: iteration `i` waits for it to reach
`(i+1) * gridDim.x`. Monotonic, so no second buffer and no reset race. The spin
is bounded by a ~2s deadline (clock checked once per 1024 spins, so it doesn't
show in the numbers) — a barrier that isn't actually device coherent then fails
the test instead of hanging the GPU.

## results, MI350X (gfx950, 256 CU, 8 XCD), ROCm 7.2

```
test 1  all four checks pass, blocks land round-robin: 0 1 2 3 4 5 6 7 0 1 ...

test 2  last arrive -> last release      3.67 us   (the actual sync cost)
        arrival skew                     0.22 us
        release skew                     0.22 us
        first arrive -> last release     3.88 us

test 3  10000 syncs, 3.651 us / sync     (launch overhead 7.8 us, amortized out)

test 4  256 contended atomicAdd, no wait 2.90 us / round
        wall_clock64()                   0.02 us / read
```

Two things worth noting:

* **the barrier is atomic-contention bound, not spin bound.** 256 `atomicAdd`
  to one address cost 2.90 us on their own (~11 ns each, fully serialized at
  the coherence point); the full barrier is 3.65 us. So ~80% of a grid-wide
  sync is just serializing the arrivals, and only ~0.75 us is
  notice-and-release. Spreading the counter over several addresses (tree /
  per-XCD sub-counters) is the lever, not a smarter spin.
* **release skew is small (0.22 us) and roughly XCD independent.** The
  per-XCD release delay is 0.07-0.15 us, i.e. no XCD is systematically last
  out; the far-L2 read latency dominates and it's about the same for everyone.

## gfx1250

Builds clean, **but has not been run** - there is no gfx1250 in the machine
this was written on, so everything below is from reading the generated ISA
only. Two things change vs gfx9.

*The atomics are fine.* `__HIP_MEMORY_SCOPE_AGENT` lowers to an explicit
device scope on both the arrive and the spin, which is exactly what the
barrier needs:

```
global_atomic_add_u32 v0, v1, s[4:5] scope:SCOPE_DEV
global_load_b32       v1, v0, s[4:5] scope:SCOPE_DEV
```

`wall_clock64()` also exists, lowered to `s_sendmsg_rtn_b64
sendmsg(MSG_RTN_GET_REALTIME)` rather than `s_memrealtime`. That is a message
round trip, so it is likely more expensive than the 20 ns/read measured on
gfx950 - test 4 measures it, check that number before trusting test 2's skew
columns.

*The hw registers are completely different, and get it wrong silently.* There
is no `HW_REG_XCC_ID` on gfx1250, and `HW_REG_HW_ID` does not exist either -
it was split into `HW_ID1`(23) / `HW_ID2`(24) back in gfx10. A stale register
number still assembles, it just decodes to an unrelated register: on gfx1250
reg 4 is `HW_REG_WAVE_STATE_PRIV` and reg 20 is `HW_REG_IB_STS2`, so the
first version of this test was cheerfully reading garbage. Confirmed with

```
llvm-mc -arch=amdgcn -mcpu=gfx1250   # HW_REG_HW_ID / HW_REG_XCC_ID: rejected
```

So on gfx10+ the code reads `HW_ID1` and reports `(se, sa, wgp, simd)`, and
`get_xcc_id()` returns 0 with test 1's XCD checks marked SKIP.

Note HIP's own `__smid()` is wrong here too: `amd_device_functions.h` only
special-cases `__GFX10__`/`__GFX11__` and falls back to `HW_ID 4` otherwise,
so it reads `WAVE_STATE_PRIV` on `__GFX12__`. Don't use it on gfx1250.

Two things to check first on real hardware:

* the `HW_ID1` field offsets used for gfx12 are inherited from gfx10/gfx11 and
  are **unverified** on gfx1250. The "physical location unique" check is the
  canary - if it fails, the layout moved. The raw `HW_ID1` of block 0 is
  printed so it can be eyeballed against the ISA doc.
* gfx1250 may have fewer than 256 CUs. The barrier tests need all 256 blocks
  resident at once; if they aren't, the spin can't complete. The program warns
  when `GRID_SIZE > multiProcessorCount`, and the spin deadline turns it into
  a clean failure rather than a hang, but the timing numbers are then
  meaningless - lower `GRID_SIZE` to the CU count.

## note on `sc1`

On gfx942/gfx950 each XCD has its own L2, so a device-wide sync needs accesses that reach the
common coherence point. `__hip_atomic_*` at `__HIP_MEMORY_SCOPE_AGENT` is what
does that. Checking the generated ISA (`-save-temps` leaves the `.s` in
`build/`):

```
global_atomic_add v0, v1, s[4:5]            # no sc1
global_load_dword v1, v0, s[4:5] sc1        # sc1 -> past the local L2
```

The spin load gets `sc1` as expected. The non-returning atomic does *not*, yet
the barrier is verifiably correct (test 2 checks nobody leaves early, and the
final counter matches) — on gfx950 the atomic goes to the coherence point
anyway. Worth re-checking on other targets before relying on it - gfx1250 does
mark it explicitly (`scope:SCOPE_DEV`, see above). If it were XCD-local, each
XCD would count in its own L2 copy and the counter would never reach 256.
