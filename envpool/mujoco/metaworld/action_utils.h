// Copyright 2026 Garena Online Private Limited
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

#ifndef ENVPOOL_MUJOCO_METAWORLD_ACTION_UTILS_H_
#define ENVPOOL_MUJOCO_METAWORLD_ACTION_UTILS_H_

#include <algorithm>

namespace metaworld {

inline float MocapActionDelta(float action) {
  // MetaWorld v3 multiplies the clipped float32 action by action_scale before
  // adding it to MuJoCo's float64 mocap position. Preserve that rounding.
  return std::clamp(action, -1.0F, 1.0F) * 0.01F;
}

}  // namespace metaworld

#endif  // ENVPOOL_MUJOCO_METAWORLD_ACTION_UTILS_H_
