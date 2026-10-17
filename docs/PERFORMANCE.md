# PERFORMANCE.md: single-core CPU saturation during serve decode

Status: root cause confirmed. The 100 percent core is a ROCm runtime bug, not
tessera code. The app-side O(N^2) streaming cost is fixed and built. The
runtime fix (a ROCm upgrade) is pending and needs the owner.

## Problem

`tessera-cli serve` pegs one CPU core during decode. The decode rate also
falls after about 20000 generated tokens. The target run uses the Qwen 3.8
27B GGUF on GPU 0 with thinking on and no completion cap. Generation reaches
more than 26000 tokens.

## Reference command

    ./cmake-build-rocm/tessera-cli serve \
      --model ~/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf \
      --gpus 0 --draft-block 2 --speculate --context 262144 --kv-q8 \
      --host 0.0.0.0 --port 8080 --enable-thinking

Prompt: "Build a detailed voxel-style Japanese garden in Three.js, with a
pagoda, tiny villagers, a flying dragon and interactive details."

## Confirmed root cause (the 100 percent core)

The hot thread is the ROCm runtime's `rocr::core::Runtime::AsyncEventsLoop`,
not a tessera thread.

Evidence:

- The server runs in the `tessera:rocm` container. Inside it PID 1 is the
  server with six threads.
- Per-thread CPU over a two second window: tid 8 at about 95 percent, with
  about 60 percent system time; tid 21 at about 12 percent user time; all
  others idle. GPU 0 is about 100 percent busy, GPU 1 idle.
- Thread start times (ticks since boot, HZ 100) place tid 8 and tid 9 about
  0.5 second after the main thread, during HIP and HSA init. The app's own
  threads appear minutes later (first request). So tid 8 is a runtime thread,
  not the scheduler or an HTTP worker.
- `wchan` for the hot thread toggles between running and `kfd_wait_on_events`.
  That is the `AMDKFD_IOC_WAIT_EVENTS` ioctl the async-events loop uses.
- `src/backends/rocm/rocm_backend.cpp:301-304` already notes this residual:
  "HSA AsyncEventsLoop polls KFD WAIT_EVENTS (~660k/s, one core) during any
  activity (legacy-rocm-build#6522); HSA_TOOLS_DISABLE_REGISTER=1 has no
  effect." `rg` confirms tessera registers no HSA async handler itself.
- The spun thread was captured with a backtrace in the upstream report
  (ROCm/legacy-rocm-build#6406): `rocr::core::Runtime::AsyncEventsLoop`. The
  loop re-waits on a KFD event that is already signaled, so the wait returns
  in about 12 microseconds and the loop never sleeps.
- Related upstream issue: ROCm/rocm-systems#7860. A ROCm contributor states
  the fix is a backoff in that polling fallback, ROCm/rocm-systems#7898. It
  landed in the nightly builds from 2026-08-22 (ROCm 10.1.0a, TheRock).

Installed runtime: ROCm 7.2.0, `libhsa-runtime64.so.1.18.70200`,
build id `1.18.0-rocm-rel-7.2-43-fc0010cf6a`. This build predates the fix.

## Fix for the 100 percent core

Upgrade the ROCm runtime to a build that includes ROCm/rocm-systems#7898 (a
backoff in the AsyncEventsLoop polling fallback). This is an environment
change, not a tessera code change. Applying the backoff patch to
`libhsa-runtime64` in place is the alternative. No environment variable
disables the spin: `HSA_ENABLE_INTERRUPT`, `HSA_ENABLE_WAIT_ANY` and
`HSA_TOOLS_DISABLE_REGISTER` have no effect. `HSA_ENABLE_MWAITX=1` only lowers
power, it does not stop the loop.

## App-side host work to reduce (does not fix the runtime spin)

These reduce the other host cost and the tok/s falloff. They are still worth
doing and are independent of the runtime fix.

1. Host work that grows with output length. `ThinkStreamer::Push` in
   `src/serve/tools/tool_call.cpp` re-allocates and re-scans the whole
   accumulated text every token. It is O(N) per token and O(N^2) total, and it
   crosses the glibc 128 KiB mmap threshold near 20000 to 30000 output tokens.
   This is the app-side cost that lines up with the observed tok/s falloff
   past 20k.
2. KV cache growth copies the whole cache at each power-of-two row boundary
   (`src/core/decode_internal.hpp:525-559`). The 16384 to 32768 step is a
   large device copy in one token.
3. Per-token HIP submissions. The decode issues on the order of 1000 kernel
   launches per token from one thread (`src/backends/rocm/rocm_backend.cpp:688`
   says about 500). Reducing these lowers GPU-side gaps and event churn. It
   does not stop the runtime spin.

## Plan (todo)

- [x] Confirm the hot thread and function with read-only inspection. Done.
      Result: the ROCm runtime `AsyncEventsLoop`, a runtime bug. Confirmed
      again on the rebuilt binary: the hot thread starts about 0.07 second
      after the process, runs about 94 percent CPU (mostly system time) and
      waits in `kfd_wait_on_events`.
- [ ] Upgrade or patch the ROCm runtime to include ROCm/rocm-systems#7898.
      Environment change. Needs the owner. This is the only fix for the
      100 percent core; no tessera code change can stop a runtime spin.
- [x] Make `ThinkStreamer` incremental. Done. `Push` now folds only the new
      bytes (`Fold`), tracks the last unmatched opener and a resumable closer
      scan, and never copies or scans the whole accumulated text again. Files:
      `src/serve/tools/tool_call.hpp` and `src/serve/tools/tool_call.cpp`.
      Validated with a 200000-case randomized equivalence check against the
      original algorithm (reasoning and content concatenations equal after
      `Finish`), then built.
- [x] Smooth KV cache growth. Evaluated, no change. The growth copy is
      geometric, so it is amortized O(1) per token and bursts only at the
      power-of-two rows. At a q8 KV that burst is a few hundred megabytes of
      device copy across the full-attention layers, a few milliseconds, once
      per doubling. It is not the observed sustained falloff, and preallocating
      the whole context would raise memory for every short run. Left as is.
- [x] Reduce per-token HIP submissions. Evaluated, no change. The app thread
      idles in a blocking stream wait while the GPU runs; the dispatch is not
      CPU-bound. Reducing launches would not stop the runtime spin. A fusion
      or graph refactor is high risk and does not address the root cause, so it
      is not done here.
- [x] Follow the hard rules. No new tests. Root-cause fixes only. Comments at
      most 4 lines.

## Verification

- Confirm the runtime fix: after the ROCm upgrade or patch, run the reference
  command and watch a long generation. One core must stay idle. `top -H` on
  the server must show no thread near 100 percent during decode.
- App-side: reconfigure and build ROCm. There is no test suite in the tree
  (removed in commit d5d053d), so `ctest` has no cases to run; the streamer
  change was checked with the randomized equivalence harness. Run the
  reference command and confirm the streaming thread's CPU does not grow with
  output length and that tok/s does not collapse after about 20000 tokens.

Command to rebuild and test:

    cmake -B cmake-build-rocm -DTESSERA_BACKEND=rocm && cmake --build cmake-build-rocm -j
    ctest --test-dir cmake-build-rocm --output-on-failure   # no cases in tree

Container and process inspection commands used:

    docker exec <container> ps -eLf
    docker exec <container> sh -c 'for t in /proc/1/task/*; do echo $t $(cat $t/wchan); done'
    ps -eLo pid,tid,pcpu,stat,comm -p <pid>
    # start times: field 22 of /proc/<pid>/task/<tid>/stat
