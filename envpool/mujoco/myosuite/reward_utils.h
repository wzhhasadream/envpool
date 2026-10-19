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

#ifndef ENVPOOL_MUJOCO_MYOSUITE_REWARD_UTILS_H_
#define ENVPOOL_MUJOCO_MYOSUITE_REWARD_UTILS_H_

#include <cmath>

namespace myosuite {

// MyoSuite 2.11.6 WalkEnvV0._get_ref_rotation_rew compares quaternion
// components directly. Preserve its sign sensitivity and do not normalize or
// replace this with a quaternion angular distance.
inline double ReferenceRotationReward(const double* current,
                                      const double* target) {
  double squared_norm = 0.0;
  for (int axis = 0; axis < 4; ++axis) {
    const double delta = 5.0 * (current[axis] - target[axis]);
    squared_norm += delta * delta;
  }
  return std::exp(-std::sqrt(squared_norm));
}

}  // namespace myosuite

#endif  // ENVPOOL_MUJOCO_MYOSUITE_REWARD_UTILS_H_
