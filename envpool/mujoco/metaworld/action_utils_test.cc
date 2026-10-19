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

#include "envpool/mujoco/metaworld/action_utils.h"

#include <gtest/gtest.h>

#include <array>

TEST(MetaWorldActionUtilsTest, MatchesPinnedFloat32ActionScaling) {
  // Evaluated with the original v3.0.0 SawyerXYZEnv.set_xyz_action and float32
  // actions, starting from a zero float64 mocap position with unclipped bounds.
  const std::array<float, 8> actions{0.0F,   1.0F,       -1.0F, 0.7F,
                                     -0.99F, 0.1234567F, 2.0F,  -2.0F};
  const std::array<double, 8> expected{0.0,
                                       0.0099999997764825821,
                                       -0.0099999997764825821,
                                       0.00699999975040555,
                                       -0.0098999999463558197,
                                       0.0012345670256763697,
                                       0.0099999997764825821,
                                       -0.0099999997764825821};
  for (std::size_t i = 0; i < actions.size(); ++i) {
    EXPECT_EQ(static_cast<double>(metaworld::MocapActionDelta(actions[i])),
              expected[i]);
  }
}
