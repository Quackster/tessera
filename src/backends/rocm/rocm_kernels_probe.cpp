#include "backends/rocm/rocm_kernels.hpp"

namespace tessera::backends::rocm {

// Built-in "device_probe": exercises the shared device primitives in
// rocm_device.hpp so a host test can check them against scalar
// references. One wave of 32 lanes. Buffers: 0 in (f32), 1 swap (f32),
// 2 bf16 round trip (f32), 3 byte gather (u32). Grid 1, block 32.
__global__ void DeviceProbeKernel(const float* in, float* swap, float* bf16,
                                  unsigned int* gather) {
  const unsigned int i = threadIdx.x;
  const float value = in[i];
  swap[i] = Swap16(value);
  bf16[i] = Bf16ToF32(F32ToBf16(value));
  // The upper 16 bits carry markers the gather must drop.
  const unsigned int w0 = (i * 7u + 3u) | 0xA5A50000u;
  const unsigned int w1 = (i * 11u + 5u) | 0x5A5A0000u;
  gather[i] = ByteGather(w0, w1, static_cast<int>(i & 1u));
}

}  // namespace tessera::backends::rocm
