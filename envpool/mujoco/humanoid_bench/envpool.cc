// Copyright (c) 2026 ApexRL contributors.
// Licensed under the Apache License, Version 2.0.

#include "envpool/core/py_envpool.h"
#include "envpool/mujoco/humanoid_bench/humanoid_bench.h"

using HumanoidBenchEnvSpec = PyEnvSpec<humanoid_bench::HumanoidBenchEnvSpec>;
using HumanoidBenchEnvPool = PyEnvPool<humanoid_bench::HumanoidBenchEnvPool>;
using HumanoidBenchPixelEnvSpec = PyEnvSpec<humanoid_bench::HumanoidBenchPixelEnvSpec>;
using HumanoidBenchPixelEnvPool = PyEnvPool<humanoid_bench::HumanoidBenchPixelEnvPool>;

PYBIND11_MODULE(humanoid_bench_envpool, m) {
  REGISTER(m, HumanoidBenchEnvSpec, HumanoidBenchEnvPool)
  REGISTER(m, HumanoidBenchPixelEnvSpec, HumanoidBenchPixelEnvPool)
}
