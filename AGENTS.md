# AGENTS.md — tessera working guidelines

This file is the working guide for this repository: how to approach bugs,
where fixes must live, how to verify them. Read it before starting an
investigation and keep it current after meaningful changes.

Tessera is a C++20 LLM inference engine with two interchangeable compute
backends — Vulkan (cross-vendor GPU) and ROCm (AMD) — that runs open-weight
models. First-class targets: Qwen 3.8 27B in GGUF, and Qwen 3.8 in MXFP4
with and without DFlash2 speculative decoding.

## Engineering Principles

Write for a capable developer who never saw this code: they must be able to
read, fix, and extend it without asking.

- Prefer clarity over cleverness. Explicit beats implicit; minimize magic. If a
  trick needs explaining, write the plain version.
- Complexity budget: small functions, cohesive modules. No hidden globals —
  state flows through explicitly constructed and owned objects (Engine, Model,
  Backend); there is no process-wide singleton, keep it that way.
- Concurrency: write every component with concurrent use in mind. Prefer
  immutable inputs and explicit outputs over shared mutable state. Document
  the thread-safety contract on shared interfaces: safe for concurrent use,
  externally synchronized, or confined to a single thread.
- The pillars this project is judged on:
  1. Extensibility: new LLM architectures arrive as data, not as core code
     changes.
  2. Portability: the core stays independent of both Vulkan and ROCm so the
     backends can be refactored, swapped, or extended without touching it.
  3. Performance: every line of code serves inference speed. Choose the
     faster implementation even when it takes more code. Hot paths avoid
     needless copies, allocations, and abstraction overhead.
- Backend parity: Vulkan and ROCm implement the same kernels. Every kernel
  exists on both backends with the same contract, the same element order,
  and the same numerics. A kernel that lands on one backend lands on the
  other in the same change. Only the numerical tolerance may differ per
  backend.

## Architecture Rules

### Backend abstraction
- The core defines a single abstract device/backend interface; all device work
  (buffer allocation, kernel launch, synchronization, host<->device copies)
  goes through it.
- `src/backends/vulkan/` and `src/backends/rocm/` are the only places allowed
  to include vendor headers (vulkan.hpp, hip/hiprtc, rocblas, ...). No vendor
  type may cross the backend boundary.
- Core code must compile with neither SDK present. If a core file needs a
  vendor header, the abstraction is wrong: extend the interface instead.
- One active backend per binary, selected at configure time
  (`TESSERA_BACKEND=vulkan|rocm`).
- Verify behavior in the backend the change targets. A fix proven on Vulkan may
  not exercise the same code path as ROCm, and vice versa — state which
  backend was used.
- Numerical tolerance differs slightly between backends: assert per-backend
  tolerance, never a single hard-coded epsilon.

### Architectures are pluggable modules
- Architecture specific behavior lives in one polymorphic module per
  architecture. The module owns the config parse, the weight name map, the
  layer assembly (which generic kernels run, in what order, on which shapes)
  and any genuinely architecture specific kernel.
- Layout: `src/models/<arch>/` (for example
  `src/models/qwen3_5/{config.cpp,weights.cpp,layers.cpp}`) behind one
  `Architecture` interface in `include/tessera/architecture.hpp`, selected at
  model load time and constructed by a factory in
  `src/models/registry.cpp`. `src/core/` keeps the backend agnostic engine,
  scheduler and generic kernels, and never names a model.
- Generic code calls the interface. It has no `if (model == "qwen3")` branches
  in kernels, attention or memory management.
- A new architecture is a new `src/models/<arch>/` module plus one
  registration line in `src/models/registry.cpp`, with no edits under
  `src/core/`. A fix that belongs to one architecture lives in that module; a
  fix that belongs to a shared kernel lives in `src/core/numerics/` or
  `src/backends/*/kernels/`.
- A capability shared by several architectures is a generic kernel in
  `src/core/numerics/` and `src/backends/*/kernels/`, added with tests. A
  capability that only one architecture needs is a kernel in
  `src/models/<arch>/kernels/`, behind the same backend interface.
- The extensibility test: a new architecture is a module under
  `src/models/<arch>/` (config, weights, assembly) plus a test that runs it on
  the configured backend, with zero edits under `src/core/`. If the module
  needs a new generic kernel, that kernel lands in `src/core/numerics/` with
  tests, not inlined in the module.

### Speculative decoding
- Draft/verification algorithms (DFlash2 today; DFlash, DSpark, others later)
  plug in behind a speculative-decoding strategy interface, selected at model
  load time. Off by default.
- DFlash2 (reference): extends DFlash with local dynamic convolution (grouped
  causal convolutions wrapping attention and the MLP inside the parallel draft
  block, never crossing an anchor-block boundary) and a candidate selector
  (low-rank, predecessor-conditioned transition scores that re-rank the unary
  top-K). Requires the full verifier vocabulary.
- The non-speculative path is the reference baseline: it must remain stable
  (within per-backend tolerance) when the strategy is off, and disabling the
  strategy must be a clean code path, not a stub.

## Hard Rules

These are hard rules, set by the project owner:

1. **Naming (Google style)**

   - Types (class/struct/enum/alias/concept): PascalCase
   - Functions and methods: camelCase
   - Variables, parameters, members: snake_case; private members with trailing
     underscore (`buffer_`)
   - Namespaces: snake_case
   - Files: `snake_case.hpp` / `snake_case.cpp`; test files `snake_test.cpp`
   - Macros: SCREAMING_SNAKE_CASE — avoid macros altogether where possible

2. **No duplicate functions** — every piece of logic has exactly one canonical
   implementation. Shared logic is extracted into a single free function owned
   by the appropriate module; never copy-paste-adapt.

3. **No trivial pass-through methods** — do not write a method whose body only
   calls one other method and returns its result; call the target directly
   instead. Only allowed when the wrapper adds a contract: synchronization,
   caching, a stable public API over changing internals, or virtual dispatch.

4. **One responsibility per file, under 600 lines** — one primary
   class/component per header/source pair; every file stays under 600 lines
   (hard limit). When a file approaches it, split responsibilities — do not
   append. Keep only a few files per folder (roughly five or fewer); when a
   module outgrows that, split it into subfolders by concern.

5. **Modern C++20** — RAII; const-correctness; `[[nodiscard]]` on anything
   returning an error/status or a new object; `std::span` instead of raw
   pointer+length pairs; include-what-you-use; forward declarations in headers;
   `#pragma once`.

6. **Error handling** — exceptions only for programmer errors and fatal startup
   failures. Runtime I/O, weight loading/parsing, and device errors use
   result-style returns (`std::expected`) or `std::error_code`, following the
   style of the module being touched. Exceptions never cross thread
   boundaries.

7. **Formatting** — the committed `.clang-format` (LLVM base) is the source of
   truth; run clang-format before every commit. 4-space indent, 100-column
   limit.

8. **No band-aid fixes** — root-cause fixes only. Never ship shims,
   workarounds, or half-assed patches that paper over a symptom. If the
   correct fix turns out larger than expected, do it properly (splitting files
   as rule 4 requires); if unsure of the right approach, ask the project owner
   instead of committing a hack.

9. **Tests for every function added** — when a function is added or changed,
   add or update unit tests covering it in the same change, so no regression
   slips through. Tests double as documentation: a reader can learn the API
   call patterns from them, so cover usage of the public surface. Cover edge
   cases too — empty/zero/largest inputs, malformed weight headers, error paths
   — not just the happy path. A task is not done until the new tests are in
   place and the full suite passes.

10. **No unsolicited docs files** — do not create new files under `docs/` (or
    new `.md` files anywhere) unless the user explicitly asks or an approved
    plan calls for them.

11. **Comments are max 4 lines** — a comment (single `//` line or a `/* */`
    block) may span at most 4 consecutive lines. Prefer fewer short lines; say
    what is non-obvious, never restate the code. Exception: public-API doc
    comments in headers (rule 12) are exempt from the 4-line cap; inline
    comments inside function bodies are not.

12. **Doc comments on every public API** — every public type and function in
    `include/tessera/` carries a brief `//` block comment directly above it:
    what it does, its preconditions and contracts, and a short usage example
    for non-obvious APIs. Do not introduce doxygen-style markup. Tricky
    private logic gets an inline comment at the definition, within the rule 11
    cap.

13. **No magic numbers** — every literal falls in exactly one bucket:

    - Format constants (GGUF magic/version, MXFP4 block layout, DFlash2
      checkpoint-contract parameters): named constants, never configurable —
      never move these to external parameters.
    - Tunables (timeouts, buffer sizes, batch caps): named file-local
      constants; a value that must vary per deployment comes through the
      engine options — never an inline literal.
    - Style: file-local `constexpr` with kCamelCase names
      (`kDefaultBlockTokens`).

14. **Logs are actionable** — every diagnostic line carries identifying context
    in a message prefix (`engine 1: ...`, `kernel attention: ...`) and states
    what happened and what it means, so the line alone suggests the next
    action. Handle every error — never swallow silently; log each one once, at
    the right level, where it is handled. Use the existing diagnostic channels;
    do not invent a new logging framework.

15. **Security** — validate all external input at the boundary: GGUF header
    magic and sizes, weight tensor metadata, model config fields; parsers
    reject malformed input rather than pushing the check onto callers. Secrets
    and credentials come only from the environment — never in code, tests, or
    the repo.

16. **Deterministic where feasible** — tests use fixed RNG seeds and never
    depend on wall-clock time (inject or parameterize time instead). Never
    iterate a `std::unordered_map` and depend on its order.

17. **Credit external code** — record borrowed code or ideas in `CREDITS.md`
    (project, version, license, where used) in the same change.

18. **Documentation style** — write markdown in plain standard English in
    the style of ASD-STE100 Simplified Technical English: short
    sentences, one idea per sentence, no idioms, no contractions. Do not
    use em dashes; use a period, a comma, or a new sentence instead.

19. **Repeated garbage is a bug, not noise** — when a streamed prompt produces
    visibly wrong output, for example a loop, a tail that never ends, repeated
    special tokens, or a turn that should have closed, treat it as a defect.
    Stop the run and root-cause it in the same session. Do not dismiss it as a
    bad sample and do not generate past it. Research how the reference runtimes
    handle the case (https://codeberg.org/ggz14/radiance-vllm-mxfp4 first, then
    hipfire, then vLLM, llama.cpp, the model card and the HuggingFace docs),
    implement the
    mechanism that prevents it, and add a regression test. This
    project finds most of its bugs by streaming prompts, so the fix is part of
    the task, not an extra.

20. **Protect the machine before every command** — estimate the device and
    host memory a command will use before you run it. The 27B models need tens
    of gigabytes, so overlapping heavy steps can exhaust memory and make the
    workstation unusable.

    - Run one heavy command at a time: one model load, one GPU test or one
      build. Wait for it to finish and free its memory before the next starts.
    - Only one GPU must be free. Leave the other device alone if another owner
      already holds it (for example a reference vLLM container). Do not stop a
      busy device's server to free both; run the command on the free GPU.
    - Never start two model loads together. Never put a heavy run in a shell
      loop, a background job, or one command with a build or a full-checkpoint
      Python step. These have crashed the workstation:
      `for w in 5 8 16 64; do TESSERA_TEST_MODEL=... ./tessera-tests ...; done`
      and `cmake --build ... -j && ./tessera-tests ... && python ...`.
    - A sweep runs in one process, not one process per value. Do not load the
      27B target and the draft for every value of a sweep.
    - When you only need a small value, do not load a full model to get it. Use
      `TESSERA_TEST_*` paths that point at the model you need.
    - Cap every command's timeout at 5 minutes. A test, build or run that needs
      longer is a signal something is wrong (too heavy, a slow kernel, a
      runaway); stop and fix it instead of waiting. A GPU test that used to
      finish in under a minute but now takes five means the change made it
      slow, not that the wait is expected.
    - Run a model server in the foreground. A full serve (hipfire, vLLM, or the
      radiance container that compiles gfx1201 kernels and loads the 27B target
      plus the draft) holds tens of GB of device and host memory for the whole
      Backgrounding it and polling it from other commands has crashed the
      workstation. Start it, then wait; do not run anything else until it exits
      and its memory is freed.
    - Keep reference and diagnostic Python scripts memory-light. Load one
      tensor at a time, keep float32 (never float64), and never `.float()` or
      copy a whole multi-GB checkpoint at once. A script that materializes the
      full checkpoint in host memory next to a running server has crashed the
      workstation. A 2 GB FP8 checkpoint becomes many gigabytes once `.float()`
      copies a [5120, 25600] tensor; free each tensor before loading the next.
    - If a command could exceed the machine's memory, split it or ask the
      project owner first.

21. **Match the reference runtime** - the behavior contract is the served
    reference runtime, `radiance-vllm-mxfp4` first, then hipfire, then vLLM.
    When a
    tessera result differs, read the reference source, find the exact
    difference, and change tessera to match. Do not ask the project owner
    which behavior is correct and do not guess. The served target container
    (`r9700-qwen3.8-mxfp4-g0a`, launcher
    `Quackster/vllm-gfx1201-launchers`) and its captured outputs are the
    reference. A numerical or acceptance difference from the reference is a
    defect to root-cause, not a property of the model.

## Working Principles

Repository-specific constraints on top of the hard rules:

- Keep `docs/PROGRESS.md` current as development progresses: after each
  finished milestone, record what was completed and refresh the "Next"
  list so the file always reflects reality.
- Keep the `README.md` features table current as development progresses:
  after each finished milestone, update the Area, State, and Detail rows
  so the table matches `docs/PROGRESS.md`.
- Hook up the CLI whenever a new feature lands: if an engine capability is
  added (a kernel, a decode path, a loader), expose it through `tools/cli`
  in the same change so the feature is usable from the command line.
- Search with `rg` before assuming a behavior is missing.
- Research model architecture online: use papers, model cards, and
  public specs for algorithm details. Prefer the reference at
  https://codeberg.org/ggz14/radiance-vllm-mxfp4 first, then hipfire, then
  vLLM, for algorithm behavior. Do not read the local llama.cpp checkout when
  designing or porting model logic.
- Do not encode model names into tests: a "qwen3" test is a test of the
  generic attention path fed with qwen3-shaped parameters.
- When a numerical regression appears, check whether the reference baseline
  (non-speculative path) changed first. A drifting baseline usually means a
  kernel or scheduler change, not a speculative-strategy bug.
- If a diagnostic hardcoded path is used briefly to prove a cause, remove it
  before finalizing. The permanent fix must obey the architecture rules above.
- Test prompts by streaming the output and watch the whole stream. Stop at the
  first sign of garbage (a loop, a repeated special token, a tail that never
  ends) and fix the root cause before continuing. A model must stop when it
  emits its declared end-of-turn or end-of-text token; a run that streams past
  that token is a defect in the engine, not a property of the model.

## Git

- Commit messages: subject + body as the work warrants.
- No `Co-Authored-By:` or any machine-attribution trailers.
- Commit every milestone once finished: when a feature (or other
  self-contained piece of work) is complete and the full test suite passes,
  make a git commit for it right away — do not let finished milestones pile up
  uncommitted.
- A milestone commit must include source code changes. Do not make a commit
  of only markdown (`.md`) files, unless told to commit them explicitly.

## Building and Testing

    cmake -B cmake-build-vulkan -DTESSERA_BACKEND=vulkan
    cmake -B cmake-build-rocm    -DTESSERA_BACKEND=rocm
    cmake --build cmake-build-vulkan -j
    ctest --test-dir cmake-build-vulkan --output-on-failure

Do not create new CMake test executables or new top-level test files. All
runtime coverage goes into the single test target; add test functions to the
`tests/*.cpp` sources compiled into it. New suites only fragment ctest output
and bloat configure time.

## Where Things Live

Provisional layout, to be updated as the tree stabilizes:

- `include/tessera/` — public C++ API (Engine, Model, backend interface,
  speculative-decoding strategy interface).
- `src/core/` — backend-agnostic engine, model definitions, scheduler, generic
  kernel launch planning, weight loaders (GGUF, MXFP4).
- `src/backends/vulkan/` — Vulkan implementation of the backend interface; the
  only place that includes Vulkan headers.
- `src/backends/rocm/` — ROCm/HIP implementation of the backend interface; the
  only place that includes hip/roc headers.
- `src/spec/` — speculative-decoding strategies (dflash2; future algorithms).
- `tools/cli/` — thin CLI front-end; all logic stays in the library.
- `tests/` — GoogleTest suite, single test binary.
- `docs/PROGRESS.md` — progress tracker; updated as development progresses.

## Model Data

Model weights live in `~/models/`, one flat directory per model variant — not
in the HuggingFace cache. When a model is fetched (e.g. `huggingface-cli`),
flatten it into `~/models/` under the name below and delete the cache entry
so the cache never grows a second copy.

Naming: `<family>-<size>-<format>[-<variant>]`, matching the on-disk layout.
For GGUF, the quantization lives in the file name, not the folder name.

Current inventory (the first-class targets):

- `~/models/Qwen3.8-27B-GGUF/` — `Qwen3.8-27B-UD-Q4_K_M.gguf`,
  `mmproj-BF16.gguf`, `MTP/mtp-Qwen3.8-27B-Q4_0.gguf`, `config.json`,
  `README.md`.
- `~/models/Qwen3.8-27B-MXFP4-MTPFP8/` — MXFP4 model with fp8 MTP
  (`model.safetensors`, tokenizer/processor configs, `config.json`).
- `~/models/Qwen3.8-27B-DFlash2-FP8/` — DFlash2 draft checkpoint (FP8
  `model.safetensors` + `config.json`). The Qwen 3.8 DFlash2 drafter is
  tested against the Qwen 3.8 27B MXFP4 target, not the GGUF. The aux
  target hidden states the drafter consumes come from the MXFP4 target.

Model paths are runtime configuration (CLI flag / engine option). Never
hard-code `~/models/...` paths in code or tests; tests take model paths as
parameters.
