// Copyright 2021 Garena Online Private Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <array>
#include <utility>

#include "envpool/atari/atari_env.h"
#include "envpool/core/py_envpool.h"

using AtariEnvSpec = PyEnvSpec<atari::AtariEnvSpec>;
using AtariEnvPool = PyEnvPool<atari::AtariEnvPool>;

using AtariContinuousEnvSpec = PyEnvSpec<atari::AtariContinuousEnvSpec>;
using AtariContinuousEnvPool = PyEnvPool<atari::AtariContinuousEnvPool>;

PYBIND11_MODULE(atari_envpool, m) {
  REGISTER(m, AtariEnvSpec, AtariEnvPool)
  REGISTER(m, AtariContinuousEnvSpec, AtariContinuousEnvPool)
  m.def("_continuous_action", [](const std::array<float, 3>& action,
                                 float threshold) {
    atari::ValidateContinuousAction(action.data());
    if (!std::isfinite(threshold)) {
      throw std::invalid_argument("continuous_action_threshold must be finite");
    }
    auto converted = atari::ConvertContinuousAction(action.data(), threshold);
    return std::make_pair(converted.action, converted.paddle_strength);
  });
}
