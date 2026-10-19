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

#include "envpool/mujoco/myosuite/reward_utils.h"

#include <gtest/gtest.h>

#include <array>

TEST(MyoSuiteRewardUtilsTest, ReferenceRotationMatchesPinnedOracle) {
  // Values evaluated by MyoSuite 2.11.6 WalkEnvV0._get_ref_rotation_rew
  // (commit 05cb84678373f91271004f99602ebbf01e57d1a1). In particular, q and -q
  // have different rewards even though they represent the same orientation.
  const std::array<double, 4> target{1.0, 0.0, 0.0, 0.0};
  const std::array<std::array<double, 4>, 4> current{{
      {1.0, 0.0, 0.0, 0.0},
      {0.0, 1.0, 0.0, 0.0},
      {-1.0, 0.0, 0.0, 0.0},
      {0.5, 0.5, 0.5, 0.5},
  }};
  const std::array<double, 4> expected{1.0, 0.00084932570471916947,
                                       4.5399929762484854e-05,
                                       0.006737946999085467};
  for (std::size_t i = 0; i < current.size(); ++i) {
    EXPECT_DOUBLE_EQ(
        myosuite::ReferenceRotationReward(current[i].data(), target.data()),
        expected[i]);
  }
}

TEST(MyoSuiteRewardUtilsTest, ReferenceRotationDoesNotNormalizeInputs) {
  const std::array<double, 4> current{0.25, -0.5, 0.75, 1.0};
  const std::array<double, 4> target{-0.5, 0.25, 1.0, 0.75};
  EXPECT_DOUBLE_EQ(
      myosuite::ReferenceRotationReward(current.data(), target.data()),
      0.0037343931739807368);
}
