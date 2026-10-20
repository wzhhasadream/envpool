/*
 * Copyright 2026 Garena Online Private Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ENVPOOL_ATARI_CONTINUOUS_ACTION_H_
#define ENVPOOL_ATARI_CONTINUOUS_ACTION_H_

#include <array>
#include <cmath>
#include <stdexcept>

namespace atari {

inline constexpr float kContinuousPi = 3.14159265358979323846f;

struct ContinuousAction {
  int action;
  float paddle_strength;
};

inline void ValidateContinuousAction(const float* action) {
  for (int i = 0; i < 3; ++i) {
    if (!std::isfinite(action[i])) {
      throw std::invalid_argument("Atari continuous actions must be finite");
    }
  }
  if (action[0] < 0.0f || action[0] > 1.0f || action[1] < -kContinuousPi ||
      action[1] > kContinuousPi || action[2] < 0.0f || action[2] > 1.0f) {
    throw std::invalid_argument(
        "Atari continuous actions must be inside [0, 1] x [-pi, pi] x [0, 1]");
  }
}

// ALE 0.11.2 Python AtariEnv.step/map_action_idx, at the pinned commit
// 72a87826916aa0b696096c991faa982992ff80ec (env.py lines 279-300, 397-453).
// Both comparisons use +threshold, intentionally matching that version rather
// than introducing a symmetric dead zone. The radius also controls paddles.
// NumPy and libm can round float32 trig differently at a threshold boundary;
// see the documented precision caveat and the oracle boundary report.
inline ContinuousAction ConvertContinuousAction(const float* action,
                                                float threshold) {
  const float x = action[0] * std::cos(action[1]);
  const float y = action[0] * std::sin(action[1]);
  const int horizontal =
      static_cast<int>(x > threshold) - static_cast<int>(x < threshold);
  const int vertical =
      static_cast<int>(y > threshold) - static_cast<int>(y < threshold);
  // Rows: left, center, right. Columns: down, center, up.
  constexpr std::array<std::array<int, 3>, 3> directions = {
      {{9, 4, 7}, {5, 0, 2}, {8, 3, 6}}};
  int index = directions[horizontal + 1][vertical + 1];
  if (action[2] > threshold) {
    index = index == 0 ? 1 : index + 8;
  }
  return {index, action[0]};
}

}  // namespace atari

#endif  // ENVPOOL_ATARI_CONTINUOUS_ACTION_H_
