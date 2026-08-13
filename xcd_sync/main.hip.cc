// unit tests around cross-workgroup sync on multi-XCD CDNA parts.
//
// all tests launch GRID_SIZE(=256) blocks x 1 thread, so every block is
// resident at the same time and it is safe for a block to spin-wait on the
// others.
//
//   test 1 : read the XCD(XCC) id from inside the kernel, and check the
//            block -> XCD mapping is the expected round-robin one
//   test 2 : cost of one grid-wide sync built out of a single atomicAdd
//            (arrive, then spin until the counter reaches GRID_SIZE)
//   test 3 : same sync repeated N times inside one kernel, so the kernel
//            launch overhead is amortized away
//   test 4 : reference number - N contended atomicAdd with no waiting
//
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <algorithm>
#include <vector>

#define HIP_CALL(call)                                                         \
    do {                                                                       \
        hipError_t err = call;                                                 \
        if(err != hipSuccess) {                                                \
            printf("[hiperror](%d) %s, fail to call %s at %s:%d\n", (int)err,  \
                   hipGetErrorString(err), #call, __FILE__, __LINE__);         \
            exit(1);                                                           \
        }                                                                      \
    } while(0)

#define GRID_SIZE  256
#define BLOCK_SIZE 1

// ---------------------------------------------------------------------------
// hw register layout. this differs a lot between families, and getting it
// wrong is silent - a bogus register number still assembles, it just decodes
// to a completely unrelated register:
//
//   gfx9  : HW_REG_HW_ID(4) holds CU/SE, and gfx942/gfx950 add their own
//           HW_REG_XCC_ID(20)
//   gfx10+: HW_REG_HW_ID is gone, split into HW_ID1(23) / HW_ID2(24), and
//           there is no XCC_ID at all. on gfx1250 register 4 decodes as
//           HW_REG_WAVE_STATE_PRIV and 20 as HW_REG_IB_STS2.
//
// don't use HIP's HW_ID / __smid() here: its header only special cases
// __GFX10__/__GFX11__ and falls back to HW_ID=4 for everything else, so it is
// itself wrong on __GFX12__.
// ---------------------------------------------------------------------------
#if defined(__GFX10__) || defined(__GFX11__) || defined(__GFX12__)

#define HAS_XCC_ID 0
#define C_HW_ID 23 // HW_ID1
// field layout below is the gfx10/gfx11 HW_ID1 one. it is NOT verified on
// gfx1250 - the "unique location per block" check in test 1 is what catches it
// if the layout moved, and the raw HW_ID1 is dumped so it can be eyeballed.
#define C_HW_ID_SIMD_ID_SIZE   2
#define C_HW_ID_SIMD_ID_OFFSET 8
#define C_HW_ID_WGP_ID_SIZE    4
#define C_HW_ID_WGP_ID_OFFSET  10
#define C_HW_ID_SA_ID_SIZE     1
#define C_HW_ID_SA_ID_OFFSET   16
#define C_HW_ID_SE_ID_SIZE     3
#define C_HW_ID_SE_ID_OFFSET   18

#else

#define C_HW_ID              4
#define C_HW_ID_CU_ID_SIZE   4
#define C_HW_ID_CU_ID_OFFSET 8
#if defined(__gfx908__) || defined(__gfx90a__)
#define C_HW_ID_SE_ID_SIZE 3
#else
#define C_HW_ID_SE_ID_SIZE 2 // 4 SE per XCC on gfx942/gfx950
#endif
#define C_HW_ID_SE_ID_OFFSET 13

#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__)
#define HAS_XCC_ID             1
#define C_XCC_ID               20
#define C_XCC_ID_XCC_ID_SIZE   4
#define C_XCC_ID_XCC_ID_OFFSET 0
#else
#define HAS_XCC_ID 0
#endif

#endif

#define GETREG(FIELD) __builtin_amdgcn_s_getreg(GETREG_IMMED(FIELD##_SIZE - 1, FIELD##_OFFSET, C_HW_ID))

__device__ static inline uint32_t get_xcc_id()
{
#if HAS_XCC_ID
    return __builtin_amdgcn_s_getreg(
        GETREG_IMMED(C_XCC_ID_XCC_ID_SIZE - 1, C_XCC_ID_XCC_ID_OFFSET, C_XCC_ID));
#else
    return 0; // no such register on this arch
#endif
}

__device__ static inline uint32_t get_se_id() { return GETREG(C_HW_ID_SE_ID); }

// shader array within the SE. only gfx10+ splits an SE that way.
__device__ static inline uint32_t get_sa_id()
{
#if defined(__GFX10__) || defined(__GFX11__) || defined(__GFX12__)
    return GETREG(C_HW_ID_SA_ID);
#else
    return 0;
#endif
}

// smallest addressable compute unit the wave sits on. gfx9 reports a CU
// directly; gfx10+ reports a WGP plus the SIMD inside it.
__device__ static inline uint32_t get_cu_id()
{
#if defined(__GFX10__) || defined(__GFX11__) || defined(__GFX12__)
    return (GETREG(C_HW_ID_WGP_ID) << C_HW_ID_SIMD_ID_SIZE) | GETREG(C_HW_ID_SIMD_ID);
#else
    return GETREG(C_HW_ID_CU_ID);
#endif
}

__device__ static inline uint32_t get_hw_id_raw()
{
    return __builtin_amdgcn_s_getreg(GETREG_IMMED(32 - 1, 0, C_HW_ID));
}

// ---------------------------------------------------------------------------
// device scope atomics. on MI300/MI350 each XCD has its own L2, so a plain
// (workgroup/wavefront scope) access is *not* enough - agent scope is what
// makes the compiler emit the sc1 bit that takes the access past the local L2
// to the coherence point shared by all XCDs.
// ---------------------------------------------------------------------------
__device__ static inline uint32_t atomic_add_dev(uint32_t* p, uint32_t v)
{
    return __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ static inline uint32_t atomic_load_dev(const uint32_t* p)
{
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// counting barrier over all blocks of the grid. the counter is never reset,
// iteration i simply waits for it to reach (i + 1) * gridDim.x, which keeps
// the whole thing race free without a second buffer.
//
// the spin is bounded by `deadline` (a wall_clock64() stamp) so a broken
// barrier reports a failure instead of hanging the GPU. the clock is only
// read once every 1024 spins, so it does not show up in the measurement.
__device__ static inline bool grid_barrier(uint32_t* cnt, uint32_t target, uint64_t deadline)
{
    atomic_add_dev(cnt, 1);
    uint32_t spin = 0;
    while(atomic_load_dev(cnt) < target) {
        if((++spin & 0x3ff) == 0 && (uint64_t)wall_clock64() > deadline) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// test 1
// ---------------------------------------------------------------------------
__global__ void xcd_id_kernel(uint32_t* p_xcc,
                              uint32_t* p_se,
                              uint32_t* p_cu,
                              uint32_t* p_key,
                              uint32_t* p_raw,
                              uint32_t* p_caps)
{
    uint32_t bid = blockIdx.x;
    uint32_t xcc = get_xcc_id();
    uint32_t se  = get_se_id();
    uint32_t sa  = get_sa_id();
    uint32_t cu  = get_cu_id();

    p_xcc[bid] = xcc;
    p_se[bid]  = se;
    p_cu[bid]  = cu;
    p_raw[bid] = get_hw_id_raw();
    // packed physical location, unique per CU/SIMD on every arch
    p_key[bid] = (xcc << 24) | (se << 16) | (sa << 8) | cu;

    if(bid == 0) p_caps[0] = HAS_XCC_ID;
}

// ---------------------------------------------------------------------------
// test 2/3 : timestamp every arrive/release so the host can look at the skew.
// wall_clock64() is a constant, device wide clock, so timestamps taken on
// different XCDs are directly comparable (clock64() is not).
// ---------------------------------------------------------------------------
__global__ void barrier_trace_kernel(uint32_t* cnt,
                                     uint64_t* p_enter,
                                     uint64_t* p_exit,
                                     uint32_t* p_xcc,
                                     uint32_t* p_timeout,
                                     int iters,
                                     uint64_t timeout_ticks)
{
    uint32_t bid      = blockIdx.x;
    uint32_t timeout  = 0;
    p_xcc[bid]        = get_xcc_id();
    uint64_t deadline = wall_clock64() + timeout_ticks;

    for(int i = 0; i < iters; i++) {
        uint64_t t0 = wall_clock64();
        if(!grid_barrier(cnt, (uint32_t)(i + 1) * gridDim.x, deadline)) timeout++;
        uint64_t t1 = wall_clock64();

        p_enter[(size_t)i * gridDim.x + bid] = t0;
        p_exit[(size_t)i * gridDim.x + bid]  = t1;
    }
    p_timeout[bid] = timeout;
}

// same barrier, but only the total is timed - no per iteration stores to
// perturb the measurement, and the launch overhead is spread over `iters`.
__global__ void barrier_loop_kernel(uint32_t* cnt,
                                    uint64_t* p_begin,
                                    uint64_t* p_end,
                                    uint32_t* p_xcc,
                                    uint32_t* p_timeout,
                                    int iters,
                                    uint64_t timeout_ticks)
{
    uint32_t bid     = blockIdx.x;
    uint32_t timeout = 0;
    p_xcc[bid]       = get_xcc_id();

    uint64_t t0       = wall_clock64();
    uint64_t deadline = t0 + timeout_ticks;
    for(int i = 0; i < iters; i++) {
        if(!grid_barrier(cnt, (uint32_t)(i + 1) * gridDim.x, deadline)) timeout++;
    }
    uint64_t t1 = wall_clock64();

    p_begin[bid]   = t0;
    p_end[bid]     = t1;
    p_timeout[bid] = timeout;
}

// ---------------------------------------------------------------------------
// test 4 : `iters` contended atomicAdd on one address, nobody waits. this is
// the floor the barrier above can never go below.
// ---------------------------------------------------------------------------
__global__ void atomic_only_kernel(uint32_t* cnt,
                                   uint64_t* p_begin,
                                   uint64_t* p_end,
                                   uint32_t* p_sink,
                                   int iters)
{
    uint32_t bid = blockIdx.x;
    uint32_t acc = 0;

    uint64_t t0 = wall_clock64();
    for(int i = 0; i < iters; i++) {
        acc += atomic_add_dev(cnt, 1);
    }
    uint64_t t1 = wall_clock64();

    p_begin[bid] = t0;
    p_end[bid]   = t1;
    p_sink[bid]  = acc; // keep the loop alive
}

// cost of the two wall_clock64() reads themselves
__global__ void clock_overhead_kernel(uint64_t* p_begin, uint64_t* p_end, int iters)
{
    uint64_t acc = 0;
    uint64_t t0  = wall_clock64();
    for(int i = 0; i < iters; i++) {
        acc += wall_clock64();
    }
    uint64_t t1 = wall_clock64();

    p_begin[blockIdx.x] = t0;
    p_end[blockIdx.x]   = t1 + (acc & 0); // keep the loop alive
}

// ---------------------------------------------------------------------------
// host helpers
// ---------------------------------------------------------------------------
static double g_tick_ns = 10.0; // wall clock period, filled in from the device
static int    g_num_xcc = 1;
static int    g_num_cu  = 0;
static int    g_fail    = 0;

static void report(const char* name, bool ok, const char* detail = nullptr)
{
    printf("[%s] %s%s%s\n", ok ? " OK " : "FAIL", name, detail ? " : " : "",
           detail ? detail : "");
    if(!ok) g_fail++;
}

static void skip(const char* name, const char* why)
{
    printf("[SKIP] %s : %s\n", name, why);
}

static double to_us(double ticks) { return ticks * g_tick_ns / 1000.0; }

// the spinning barrier gives up after this many wall clock ticks (~2s), so a
// barrier that is not actually device coherent fails the test instead of
// hanging the GPU until the watchdog fires
static uint64_t timeout_ticks() { return (uint64_t)(2.0e9 / g_tick_ns); }

static bool check_no_timeout(const std::vector<uint32_t>& timeout)
{
    uint32_t total = 0;
    for(uint32_t t : timeout) total += t;
    char buf[192];
    snprintf(buf, sizeof(buf),
             total ? "%u timed out - the atomic is probably not device coherent across XCD"
                   : "%u spin timeout",
             total);
    report("no block timed out in the spin", total == 0, buf);
    return total == 0;
}

struct stat_t {
    double mean, med, min, max;
};

static stat_t summarize(std::vector<double> v)
{
    stat_t s{0, 0, 0, 0};
    if(v.empty()) return s;
    std::sort(v.begin(), v.end());
    double sum = 0;
    for(double x : v) sum += x;
    s.mean = sum / v.size();
    s.med  = v[v.size() / 2];
    s.min  = v.front();
    s.max  = v.back();
    return s;
}

// ---------------------------------------------------------------------------
// test 1 : xcd id
// ---------------------------------------------------------------------------
static void test_xcd_id()
{
    printf("\n--- test 1 : read XCD id from kernel (%d blocks x %d thread) ---\n",
           GRID_SIZE, BLOCK_SIZE);

    uint32_t *d_xcc, *d_se, *d_cu, *d_key, *d_raw, *d_caps;
    HIP_CALL(hipMalloc(&d_xcc, GRID_SIZE * sizeof(uint32_t)));
    HIP_CALL(hipMalloc(&d_se, GRID_SIZE * sizeof(uint32_t)));
    HIP_CALL(hipMalloc(&d_cu, GRID_SIZE * sizeof(uint32_t)));
    HIP_CALL(hipMalloc(&d_key, GRID_SIZE * sizeof(uint32_t)));
    HIP_CALL(hipMalloc(&d_raw, GRID_SIZE * sizeof(uint32_t)));
    HIP_CALL(hipMalloc(&d_caps, sizeof(uint32_t)));

    xcd_id_kernel<<<dim3(GRID_SIZE), dim3(BLOCK_SIZE)>>>(d_xcc, d_se, d_cu, d_key, d_raw, d_caps);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());

    std::vector<uint32_t> xcc(GRID_SIZE), se(GRID_SIZE), cu(GRID_SIZE), key(GRID_SIZE),
        raw(GRID_SIZE);
    uint32_t caps = 0;
    HIP_CALL(hipMemcpy(xcc.data(), d_xcc, GRID_SIZE * sizeof(uint32_t), hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(se.data(), d_se, GRID_SIZE * sizeof(uint32_t), hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(cu.data(), d_cu, GRID_SIZE * sizeof(uint32_t), hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(key.data(), d_key, GRID_SIZE * sizeof(uint32_t), hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(raw.data(), d_raw, GRID_SIZE * sizeof(uint32_t), hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(&caps, d_caps, sizeof(uint32_t), hipMemcpyDeviceToHost));

    if(!caps) {
        // gfx10/11/12 have no HW_REG_XCC_ID at all, so there is nothing to
        // read and the round-robin question does not apply
        skip("xcd id", "this arch has no HW_REG_XCC_ID, get_xcc_id() returns 0");
        if(g_num_xcc != 1)
            report("num_xcc consistent with arch", false,
                   "HIP reports >1 XCC but the arch exposes no XCC_ID");
    } else {
        // 1.a every id must be inside [0, num_xcc)
        bool in_range = true;
        for(int i = 0; i < GRID_SIZE; i++)
            if((int)xcc[i] >= g_num_xcc) in_range = false;
        {
            char buf[128];
            snprintf(buf, sizeof(buf), "all xcd id < hipDeviceAttributeNumberOfXccs(%d)",
                     g_num_xcc);
            report("xcd id in range", in_range, buf);
        }

        // 1.b GRID_SIZE blocks should spread evenly over the XCDs
        std::vector<int> hist(g_num_xcc, 0);
        for(int i = 0; i < GRID_SIZE; i++)
            if((int)xcc[i] < g_num_xcc) hist[xcc[i]]++;

        int  expect_per_xcc = GRID_SIZE / g_num_xcc;
        bool even           = true;
        for(int x = 0; x < g_num_xcc; x++)
            if(hist[x] != expect_per_xcc) even = false;
        {
            char buf[256];
            int  n = snprintf(buf, sizeof(buf), "expect %d blocks/xcd, got", expect_per_xcc);
            for(int x = 0; x < g_num_xcc && n < (int)sizeof(buf); x++)
                n += snprintf(buf + n, sizeof(buf) - n, " %d", hist[x]);
            report("blocks evenly spread over xcd", even, buf);
        }

        // 1.c the dispatcher hands blocks to the XCDs round-robin, so on a
        //     full GRID_SIZE launch block b is expected to land on
        //     xcd (b % num_xcc)
        int mismatch = 0;
        for(int i = 0; i < GRID_SIZE; i++)
            if((int)xcc[i] != i % g_num_xcc) mismatch++;
        {
            char buf[128];
            snprintf(buf, sizeof(buf), "blockIdx.x %% %d, %d/%d mismatch", g_num_xcc, mismatch,
                     GRID_SIZE);
            report("block -> xcd mapping is round-robin", mismatch == 0, buf);
        }

        printf("  xcd of first 16 blocks: ");
        for(int i = 0; i < 16; i++) printf("%u ", xcc[i]);
        printf("...\n");
    }

    // with 1 block per CU the physical location has to be unique. this also
    // doubles as a check that the hwreg field layout is right for the arch:
    // if the offsets are wrong the ids collide.
    if(GRID_SIZE <= g_num_cu) {
        std::vector<uint32_t> sorted = key;
        std::sort(sorted.begin(), sorted.end());
        bool uniq = std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end();
        report("physical location unique across blocks", uniq,
               "1 block per CU, no CU reused (also validates the hwreg layout)");
    } else {
        skip("physical location unique across blocks", "grid is larger than the CU count");
    }

    printf("  block0 : xcd %u se %u cu %u, raw HW_ID%s 0x%08x\n", xcc[0], se[0], cu[0],
           caps ? "" : "1", raw[0]);

    HIP_CALL(hipFree(d_xcc));
    HIP_CALL(hipFree(d_se));
    HIP_CALL(hipFree(d_cu));
    HIP_CALL(hipFree(d_key));
    HIP_CALL(hipFree(d_raw));
    HIP_CALL(hipFree(d_caps));
}

// ---------------------------------------------------------------------------
// test 2 : one sync, traced
// ---------------------------------------------------------------------------
static void test_barrier_trace(int iters)
{
    printf("\n--- test 2 : cost of one atomicAdd grid sync (%d blocks, %d iters traced) ---\n",
           GRID_SIZE, iters);

    uint32_t* d_cnt;
    uint32_t* d_xcc;
    uint32_t* d_timeout;
    uint64_t *d_enter, *d_exit;
    size_t    n = (size_t)iters * GRID_SIZE;

    HIP_CALL(hipMalloc(&d_cnt, 32 * sizeof(uint32_t))); // own cacheline
    HIP_CALL(hipMalloc(&d_xcc, GRID_SIZE * sizeof(uint32_t)));
    HIP_CALL(hipMalloc(&d_timeout, GRID_SIZE * sizeof(uint32_t)));
    HIP_CALL(hipMalloc(&d_enter, n * sizeof(uint64_t)));
    HIP_CALL(hipMalloc(&d_exit, n * sizeof(uint64_t)));
    HIP_CALL(hipMemset(d_cnt, 0, 32 * sizeof(uint32_t)));

    barrier_trace_kernel<<<dim3(GRID_SIZE), dim3(BLOCK_SIZE)>>>(
        d_cnt, d_enter, d_exit, d_xcc, d_timeout, iters, timeout_ticks());
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());

    std::vector<uint64_t> enter(n), leave(n);
    std::vector<uint32_t> xcc(GRID_SIZE), timeout(GRID_SIZE);
    uint32_t              cnt = 0;
    HIP_CALL(hipMemcpy(enter.data(), d_enter, n * sizeof(uint64_t), hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(leave.data(), d_exit, n * sizeof(uint64_t), hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(xcc.data(), d_xcc, GRID_SIZE * sizeof(uint32_t), hipMemcpyDeviceToHost));
    HIP_CALL(
        hipMemcpy(timeout.data(), d_timeout, GRID_SIZE * sizeof(uint32_t), hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(&cnt, d_cnt, sizeof(uint32_t), hipMemcpyDeviceToHost));

    report("counter reached iters * grid", cnt == (uint32_t)iters * GRID_SIZE, nullptr);
    if(!check_no_timeout(timeout)) return;

    // every block must leave iteration i after the last block entered it,
    // otherwise the barrier let somebody through early
    int    broken = 0;
    std::vector<double> lat, arrive_skew, release_skew, span;
    for(int i = 0; i < iters; i++) {
        uint64_t e_min = ~0ull, e_max = 0, x_min = ~0ull, x_max = 0;
        for(int b = 0; b < GRID_SIZE; b++) {
            uint64_t e = enter[(size_t)i * GRID_SIZE + b];
            uint64_t x = leave[(size_t)i * GRID_SIZE + b];
            e_min      = std::min(e_min, e);
            e_max      = std::max(e_max, e);
            x_min      = std::min(x_min, x);
            x_max      = std::max(x_max, x);
        }
        if(x_min < e_max) broken++; // somebody exited before the last arrival
        if(i == 0) continue;        // iteration 0 pays the cold cache
        lat.push_back((double)(x_max - e_max));
        arrive_skew.push_back((double)(e_max - e_min));
        release_skew.push_back((double)(x_max - x_min));
        span.push_back((double)(x_max - e_min));
    }
    {
        char buf[128];
        snprintf(buf, sizeof(buf), "%d/%d iterations released a block early", broken, iters);
        report("barrier actually synchronizes", broken == 0, buf);
    }

    stat_t s_lat = summarize(lat), s_arr = summarize(arrive_skew);
    stat_t s_rel = summarize(release_skew), s_span = summarize(span);
    printf("  %-28s %8s %8s %8s %8s\n", "(us)", "mean", "median", "min", "max");
    printf("  %-28s %8.3f %8.3f %8.3f %8.3f\n", "last arrive -> last release",
           to_us(s_lat.mean), to_us(s_lat.med), to_us(s_lat.min), to_us(s_lat.max));
    printf("  %-28s %8.3f %8.3f %8.3f %8.3f\n", "arrival skew", to_us(s_arr.mean),
           to_us(s_arr.med), to_us(s_arr.min), to_us(s_arr.max));
    printf("  %-28s %8.3f %8.3f %8.3f %8.3f\n", "release skew", to_us(s_rel.mean),
           to_us(s_rel.med), to_us(s_rel.min), to_us(s_rel.max));
    printf("  %-28s %8.3f %8.3f %8.3f %8.3f\n", "first arrive -> last release",
           to_us(s_span.mean), to_us(s_span.med), to_us(s_span.min), to_us(s_span.max));

    // where does the release land per XCD? (relative to the earliest exit of
    // that iteration, averaged over the traced iterations)
    if(iters > 1 && g_num_xcc > 1) {
        std::vector<double> per_xcc(g_num_xcc, 0.0);
        std::vector<int>    per_cnt(g_num_xcc, 0);
        for(int i = 1; i < iters; i++) {
            uint64_t x_min = ~0ull;
            for(int b = 0; b < GRID_SIZE; b++)
                x_min = std::min(x_min, leave[(size_t)i * GRID_SIZE + b]);
            for(int b = 0; b < GRID_SIZE; b++) {
                if((int)xcc[b] >= g_num_xcc) continue;
                per_xcc[xcc[b]] += (double)(leave[(size_t)i * GRID_SIZE + b] - x_min);
                per_cnt[xcc[b]]++;
            }
        }
        printf("  release delay per xcd (us):");
        for(int x = 0; x < g_num_xcc; x++)
            printf(" %d:%.3f", x, per_cnt[x] ? to_us(per_xcc[x] / per_cnt[x]) : 0.0);
        printf("\n");
    }

    HIP_CALL(hipFree(d_cnt));
    HIP_CALL(hipFree(d_xcc));
    HIP_CALL(hipFree(d_timeout));
    HIP_CALL(hipFree(d_enter));
    HIP_CALL(hipFree(d_exit));
}

// ---------------------------------------------------------------------------
// test 3 : many syncs in one launch, launch overhead amortized
// ---------------------------------------------------------------------------
static void test_barrier_loop(int iters)
{
    printf("\n--- test 3 : %d back-to-back syncs in one launch ---\n", iters);

    uint32_t* d_cnt;
    uint32_t* d_xcc;
    uint32_t* d_timeout;
    uint64_t *d_begin, *d_end;
    HIP_CALL(hipMalloc(&d_cnt, 32 * sizeof(uint32_t)));
    HIP_CALL(hipMalloc(&d_xcc, GRID_SIZE * sizeof(uint32_t)));
    HIP_CALL(hipMalloc(&d_timeout, GRID_SIZE * sizeof(uint32_t)));
    HIP_CALL(hipMalloc(&d_begin, GRID_SIZE * sizeof(uint64_t)));
    HIP_CALL(hipMalloc(&d_end, GRID_SIZE * sizeof(uint64_t)));

    hipEvent_t ev_start, ev_stop;
    HIP_CALL(hipEventCreate(&ev_start));
    HIP_CALL(hipEventCreate(&ev_stop));

    double kernel_ms_1 = 0.0, kernel_ms_n = 0.0;

    // 1 iteration and N iterations, so the launch overhead can be subtracted
    for(int pass = 0; pass < 2; pass++) {
        int it = (pass == 0) ? 1 : iters;
        for(int rep = 0; rep < 3; rep++) { // warm up + timed
            HIP_CALL(hipMemset(d_cnt, 0, 32 * sizeof(uint32_t)));
            HIP_CALL(hipDeviceSynchronize());
            HIP_CALL(hipEventRecord(ev_start, 0));
            barrier_loop_kernel<<<dim3(GRID_SIZE), dim3(BLOCK_SIZE)>>>(
                d_cnt, d_begin, d_end, d_xcc, d_timeout, it, timeout_ticks());
            HIP_CALL(hipEventRecord(ev_stop, 0));
            HIP_CALL(hipGetLastError());
            HIP_CALL(hipDeviceSynchronize());
            float ms = 0;
            HIP_CALL(hipEventElapsedTime(&ms, ev_start, ev_stop));
            if(rep == 2) (pass == 0 ? kernel_ms_1 : kernel_ms_n) = ms;
        }
    }

    std::vector<uint64_t> begin(GRID_SIZE), end(GRID_SIZE);
    std::vector<uint32_t> timeout(GRID_SIZE);
    uint32_t              cnt = 0;
    HIP_CALL(hipMemcpy(begin.data(), d_begin, GRID_SIZE * sizeof(uint64_t), hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(end.data(), d_end, GRID_SIZE * sizeof(uint64_t), hipMemcpyDeviceToHost));
    HIP_CALL(
        hipMemcpy(timeout.data(), d_timeout, GRID_SIZE * sizeof(uint32_t), hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(&cnt, d_cnt, sizeof(uint32_t), hipMemcpyDeviceToHost));

    report("counter reached iters * grid", cnt == (uint32_t)iters * GRID_SIZE, nullptr);
    if(!check_no_timeout(timeout)) return;

    uint64_t b_min = ~0ull, e_max = 0;
    for(int b = 0; b < GRID_SIZE; b++) {
        b_min = std::min(b_min, begin[b]);
        e_max = std::max(e_max, end[b]);
    }
    double in_kernel_us = to_us((double)(e_max - b_min));
    printf("  in-kernel  : %.3f us total, %.4f us / sync\n", in_kernel_us, in_kernel_us / iters);
    printf("  wall(event): %.3f us total, %.4f us / sync\n", kernel_ms_n * 1000.0,
           kernel_ms_n * 1000.0 / iters);
    printf("  launch ovh : %.3f us (1-iter kernel), amortized cost/sync = %.4f us\n",
           kernel_ms_1 * 1000.0, (kernel_ms_n - kernel_ms_1) * 1000.0 / (iters - 1));

    HIP_CALL(hipEventDestroy(ev_start));
    HIP_CALL(hipEventDestroy(ev_stop));
    HIP_CALL(hipFree(d_cnt));
    HIP_CALL(hipFree(d_xcc));
    HIP_CALL(hipFree(d_timeout));
    HIP_CALL(hipFree(d_begin));
    HIP_CALL(hipFree(d_end));
}

// ---------------------------------------------------------------------------
// test 4 : reference numbers
// ---------------------------------------------------------------------------
static void test_reference(int iters)
{
    printf("\n--- test 4 : reference (no waiting) ---\n");

    uint32_t *d_cnt, *d_sink;
    uint64_t *d_begin, *d_end;
    HIP_CALL(hipMalloc(&d_cnt, 32 * sizeof(uint32_t)));
    HIP_CALL(hipMalloc(&d_sink, GRID_SIZE * sizeof(uint32_t)));
    HIP_CALL(hipMalloc(&d_begin, GRID_SIZE * sizeof(uint64_t)));
    HIP_CALL(hipMalloc(&d_end, GRID_SIZE * sizeof(uint64_t)));
    HIP_CALL(hipMemset(d_cnt, 0, 32 * sizeof(uint32_t)));

    auto span_us = [&]() {
        std::vector<uint64_t> begin(GRID_SIZE), end(GRID_SIZE);
        HIP_CALL(hipMemcpy(begin.data(), d_begin, GRID_SIZE * sizeof(uint64_t),
                           hipMemcpyDeviceToHost));
        HIP_CALL(hipMemcpy(end.data(), d_end, GRID_SIZE * sizeof(uint64_t), hipMemcpyDeviceToHost));
        uint64_t b_min = ~0ull, e_max = 0;
        for(int b = 0; b < GRID_SIZE; b++) {
            b_min = std::min(b_min, begin[b]);
            e_max = std::max(e_max, end[b]);
        }
        return to_us((double)(e_max - b_min));
    };

    atomic_only_kernel<<<dim3(GRID_SIZE), dim3(BLOCK_SIZE)>>>(d_cnt, d_begin, d_end, d_sink, iters);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
    uint32_t cnt = 0;
    HIP_CALL(hipMemcpy(&cnt, d_cnt, sizeof(uint32_t), hipMemcpyDeviceToHost));
    double atomic_us = span_us();
    report("no atomicAdd lost", cnt == (uint32_t)iters * GRID_SIZE, nullptr);
    printf("  %d x %d contended atomicAdd, no wait : %.3f us, %.4f us / round\n", iters, GRID_SIZE,
           atomic_us, atomic_us / iters);

    clock_overhead_kernel<<<dim3(GRID_SIZE), dim3(BLOCK_SIZE)>>>(d_begin, d_end, iters);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
    double clk_us = span_us();
    printf("  %d x wall_clock64()                  : %.3f us, %.4f us / read\n", iters, clk_us,
           clk_us / iters);

    HIP_CALL(hipFree(d_cnt));
    HIP_CALL(hipFree(d_sink));
    HIP_CALL(hipFree(d_begin));
    HIP_CALL(hipFree(d_end));
}

int main(int argc, char** argv)
{
    int iters_trace = 64;
    int iters_loop  = 10000;
    if(argc >= 2) iters_trace = atoi(argv[1]);
    if(argc >= 3) iters_loop = atoi(argv[2]);

    int dev = 0;
    HIP_CALL(hipSetDevice(dev));
    hipDeviceProp_t prop;
    HIP_CALL(hipGetDeviceProperties(&prop, dev));

    int wall_khz = 0;
    HIP_CALL(hipDeviceGetAttribute(&g_num_xcc, hipDeviceAttributeNumberOfXccs, dev));
    HIP_CALL(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, dev));
    if(wall_khz <= 0) wall_khz = 100000; // 100MHz fallback
    g_tick_ns = 1.0e6 / (double)wall_khz;
    g_num_cu  = prop.multiProcessorCount;

    printf("device : %s (%s), %d CU, %d XCC\n", prop.name, prop.gcnArchName,
           prop.multiProcessorCount, g_num_xcc);
    printf("wall clock : %d kHz (%.2f ns / tick)\n", wall_khz, g_tick_ns);
    printf("launch : %d blocks x %d thread\n", GRID_SIZE, BLOCK_SIZE);

    // the barrier tests spin, so every block has to be resident at once
    if(GRID_SIZE > prop.multiProcessorCount) {
        printf("WARNING: %d blocks > %d CU, the spinning barrier may deadlock\n", GRID_SIZE,
               prop.multiProcessorCount);
    }

    test_xcd_id();
    test_barrier_trace(iters_trace);
    test_barrier_loop(iters_loop);
    test_reference(iters_loop);

    printf("\n%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
           g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
