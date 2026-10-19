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

#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
extern "C" int __stdcall MjStlDecoderDllMain(void* hinst, DWORD reason,
                                             void* reserved);
#endif

#include "envpool/mujoco/dmc/dog.h"

namespace mujoco_dmc {
namespace {

void InitializeMeshDecoder() {
#ifdef _WIN32
  // This executable has no Python extension DllMain to register the decoder.
  MjStlDecoderDllMain(nullptr, DLL_PROCESS_ATTACH, nullptr);
#endif
}

TArray<int> EnvIds(const std::vector<int>& ids) {
  TArray<int> array(Spec<int>({static_cast<int>(ids.size())}));
  for (std::size_t i = 0; i < ids.size(); ++i) {
    array[i] = ids[i];
  }
  return array;
}

DogEnv::Action Action(const std::vector<int>& ids, int step,
                      int original_slot = -1) {
  int count = static_cast<int>(ids.size());
  std::vector<Array> values{EnvIds(ids), EnvIds(ids),
                            Array(Spec<mjtNum>({count, 38}))};
  DogEnv::Action action(values);
  for (int row = 0; row < count; ++row) {
    int slot = original_slot < 0 ? ids[row] : original_slot;
    for (int joint = 0; joint < 38; ++joint) {
      action["action"_][row][joint] = ((step + slot + joint) % 7 - 3) * 0.1;
    }
  }
  return action;
}

void ExpectRowEqual(const std::vector<Array>& actual, int row,
                    const std::vector<Array>& expected) {
  const auto keys = DogEnvSpec::StateSpec::AllKeys();
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (keys[i] == "info:env_id" || keys[i] == "info:players.env_id") {
      continue;
    }
    SCOPED_TRACE(keys[i]);
    Array left = actual[i][row];
    Array right = expected[i][0];
    ASSERT_EQ(left.Shape(), right.Shape());
    ASSERT_EQ(left.element_size, right.element_size);
    EXPECT_EQ(
        std::memcmp(left.Data(), right.Data(), left.size * left.element_size),
        0);
  }
}

template <typename SpecType>
void CheckInvalidAssignments() {
  auto config = SpecType::kDefaultConfig;
  config["num_envs"_] = 3;
  config["task_name"_] = std::string("run");
  config["env_task_names"_] = std::vector<std::string>{"run", "stand"};
  EXPECT_THROW(SpecType spec(config), std::invalid_argument);
  config["env_task_names"_] = std::vector<std::string>{"run", "stand", "jump"};
  EXPECT_THROW(SpecType spec(config), std::invalid_argument);
  config["env_task_names"_] = std::vector<std::string>{"run", "stand", "fetch"};
  EXPECT_THROW(SpecType spec(config), std::invalid_argument);
  config["env_task_names"_] = std::vector<std::string>{"run", "stand", "trot"};
  config["env_max_episode_steps"_] = std::vector<int>{7, 11};
  EXPECT_THROW(SpecType spec(config), std::invalid_argument);
  for (int invalid : {0, -1}) {
    config["env_max_episode_steps"_] = std::vector<int>{7, invalid, 5};
    EXPECT_THROW(SpecType spec(config), std::invalid_argument);
  }
  config["env_max_episode_steps"_] = std::vector<int>{7, 11, 5};
  EXPECT_NO_THROW(SpecType spec(config));
}

TEST(DmcDogMultitaskTest, ValidatesStateAndPixelAssignments) {
  CheckInvalidAssignments<DogEnvSpec>();
  CheckInvalidAssignments<DogPixelEnvSpec>();
  EXPECT_EQ(DogTaskIndex("fetch"), 0);
  EXPECT_EQ(DogTaskIndex("run"), 1);
  EXPECT_EQ(DogTaskIndex("stand"), 2);
  EXPECT_EQ(DogTaskIndex("trot"), 3);
  EXPECT_EQ(DogTaskIndex("walk"), 4);
}

void CheckMixedPool(bool asynchronous) {
  InitializeMeshDecoder();
  const std::vector<std::string> tasks{"run", "stand", "trot"};
  const std::vector<int> seeds{17, 41, 109};
  const std::vector<int> horizons{7, 11, 5};
  auto config = DogEnvSpec::kDefaultConfig;
  config["num_envs"_] = 3;
  config["num_threads"_] = 2;
  config["batch_size"_] = asynchronous ? 1 : 3;
  config["env_seed"_] = seeds;
  config["task_name"_] = std::string("run");
  config["env_task_names"_] = tasks;
  config["env_max_episode_steps"_] = horizons;
  config["action_repeat"_] = 3;
  config["frame_stack"_] = 2;
  config["friction_multiplier"_] = 0.35;
  DogEnvSpec mixed_spec(config);
  DogEnvPool mixed(mixed_spec);
  std::array<std::unique_ptr<DogEnvPool>, 3> references;
  std::array<std::vector<Array>, 3> expected;
  for (int slot = 0; slot < 3; ++slot) {
    auto standalone = config;
    standalone["num_envs"_] = 1;
    standalone["num_threads"_] = 1;
    standalone["batch_size"_] = 1;
    standalone["env_seed"_] = std::vector<int>{seeds[slot]};
    standalone["task_name"_] = tasks[slot];
    standalone["max_episode_steps"_] = horizons[slot];
    standalone["env_task_names"_] = std::vector<std::string>{};
    standalone["env_max_episode_steps"_] = std::vector<int>{};
    references[slot] = std::make_unique<DogEnvPool>(DogEnvSpec(standalone));
    references[slot]->Reset(EnvIds({0}));
    expected[slot] = references[slot]->Recv();
  }

  auto check_received = [&]() {
    std::array<bool, 3> seen{};
    for (int received = 0; received < (asynchronous ? 3 : 1); ++received) {
      auto state = mixed.Recv();
      const auto* ids = static_cast<int*>(state[0].Data());
      int rows = static_cast<int>(state[0].Shape(0));
      for (int row = 0; row < rows; ++row) {
        ASSERT_GE(ids[row], 0);
        ASSERT_LT(ids[row], 3);
        EXPECT_FALSE(seen[ids[row]]);
        seen[ids[row]] = true;
        ExpectRowEqual(state, row, expected[ids[row]]);
      }
    }
    EXPECT_EQ(seen, (std::array<bool, 3>{true, true, true}));
  };

  mixed.Reset(EnvIds({0, 1, 2}));
  check_received();
  // Different horizons, non-divisible repeats, and several automatic resets.
  for (int step = 0; step < 16; ++step) {
    for (int slot = 0; slot < 3; ++slot) {
      references[slot]->Send(Action({0}, step, slot));
      expected[slot] = references[slot]->Recv();
    }
    mixed.Send(Action({0, 1, 2}, step));
    check_received();
    if (step == 5 || step == 12) {
      mixed.Reset(EnvIds({2}));
      references[2]->Reset(EnvIds({0}));
      expected[2] = references[2]->Recv();
      auto reset = mixed.Recv();
      ASSERT_EQ(reset[0].Shape(0), 1);
      EXPECT_EQ(*static_cast<int*>(reset[0].Data()), 2);
      ExpectRowEqual(reset, 0, expected[2]);
    }
  }
}

TEST(DmcDogMultitaskTest, MixedPoolMatchesStandaloneWithSubsetResets) {
  CheckMixedPool(false);
}

TEST(DmcDogMultitaskTest, AsyncPoolPreservesIdentityAndStandaloneDynamics) {
  CheckMixedPool(true);
}

}  // namespace
}  // namespace mujoco_dmc
