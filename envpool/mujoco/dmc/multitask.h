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

#ifndef ENVPOOL_MUJOCO_DMC_MULTITASK_H_
#define ENVPOOL_MUJOCO_DMC_MULTITASK_H_

#include <array>
#include <stdexcept>
#include <string>
#include <string_view>

#include "envpool/core/dict.h"

namespace mujoco_dmc {

struct DmcTaskMetadata {
  std::string_view domain;
  std::string_view name;
  int index;
  int group;
};

// Task IDs are domain-local, in alphabetical task-name order. The registry
// coverage test keeps this native table in sync with all registered DMC tasks.
// Groups express articulated topology AND meaningful observation semantics,
// not XML byte equality. Manipulator scenes differ but use the same arm,
// object pose/velocity representation, and action contract. Quadruped walk/run
// differ only in floor extent. Each slot still loads its own original model.
inline const DmcTaskMetadata& DmcTask(const std::string& domain,
                                      const std::string& name) {
  static constexpr std::array<DmcTaskMetadata, 51> tasks = {{
      {"acrobot", "swingup", 0, 0},
      {"acrobot", "swingup_sparse", 1, 0},
      {"ball_in_cup", "catch", 0, 0},
      {"cartpole", "balance", 0, 0},
      {"cartpole", "balance_sparse", 1, 0},
      {"cartpole", "swingup", 2, 0},
      {"cartpole", "swingup_sparse", 3, 0},
      {"cartpole", "three_poles", 4, 1},
      {"cartpole", "two_poles", 5, 2},
      {"cheetah", "run", 0, 0},
      {"dog", "fetch", 0, 1},
      {"dog", "run", 1, 0},
      {"dog", "stand", 2, 0},
      {"dog", "trot", 3, 0},
      {"dog", "walk", 4, 0},
      {"finger", "spin", 0, 1},
      {"finger", "turn_easy", 1, 0},
      {"finger", "turn_hard", 2, 0},
      {"fish", "swim", 0, 0},
      {"fish", "upright", 1, 1},
      {"hopper", "hop", 0, 0},
      {"hopper", "stand", 1, 0},
      {"humanoid", "run", 0, 0},
      {"humanoid", "run_pure_state", 1, 1},
      {"humanoid", "stand", 2, 0},
      {"humanoid", "walk", 3, 0},
      {"humanoid_CMU", "run", 0, 0},
      {"humanoid_CMU", "stand", 1, 0},
      {"humanoid_CMU", "walk", 2, 0},
      {"lqr", "lqr_2_1", 0, 0},
      {"lqr", "lqr_6_2", 1, 1},
      {"manipulator", "bring_ball", 0, 0},
      {"manipulator", "bring_peg", 1, 0},
      {"manipulator", "insert_ball", 2, 0},
      {"manipulator", "insert_peg", 3, 0},
      {"pendulum", "swingup", 0, 0},
      {"point_mass", "easy", 0, 0},
      {"point_mass", "hard", 1, 0},
      {"quadruped", "escape", 0, 1},
      {"quadruped", "fetch", 1, 2},
      {"quadruped", "run", 2, 0},
      {"quadruped", "walk", 3, 0},
      {"reacher", "easy", 0, 0},
      {"reacher", "hard", 1, 0},
      {"stacker", "stack_2", 0, 0},
      {"stacker", "stack_4", 1, 1},
      {"swimmer", "swimmer15", 0, 0},
      {"swimmer", "swimmer6", 1, 1},
      {"walker", "run", 0, 0},
      {"walker", "stand", 1, 0},
      {"walker", "walk", 2, 0},
  }};
  for (const auto& task : tasks) {
    if (task.domain == domain && task.name == name) {
      return task;
    }
  }
  throw std::invalid_argument("Unknown task_name " + name + " for dmc " +
                              domain + ".");
}

inline int DmcTaskIndex(const std::string& domain, const std::string& name) {
  return DmcTask(domain, name).index;
}

template <typename Config>
void ValidateDmcTasks(const Config& config, const std::string& domain) {
  const auto& reference = DmcTask(domain, config["task_name"_]);
  const auto& tasks = config["env_task_names"_];
  const auto& horizons = config["env_max_episode_steps"_];
  const auto count = static_cast<std::size_t>(config["num_envs"_]);
  if (!tasks.empty() && tasks.size() != count) {
    throw std::invalid_argument(
        "env_task_names must contain one task for each env");
  }
  if (!horizons.empty() && horizons.size() != count) {
    throw std::invalid_argument(
        "env_max_episode_steps must contain one horizon for each env");
  }
  // These domains share articulated topology and action bounds, but state
  // observations select different meaningful keys. Only their pixel specs
  // remove that semantic difference; legacy extra state keys are not padding
  // permission. Topology-varying domains remain separated even with pixels.
  constexpr bool from_pixels =
      any_match<decltype("render_width"_), typename Config::Keys>::value;
  const bool pixel_semantics =
      from_pixels &&
      (domain == "finger" || domain == "fish" || domain == "humanoid");
  for (const auto& name : tasks) {
    const auto& task = DmcTask(domain, name);
    if (task.group != reference.group && !pixel_semantics) {
      throw std::invalid_argument(
          "Incompatible DMC " + domain +
          " tasks: " + std::string(reference.name) + " and " + name +
          " have different model topology or observation semantics");
    }
  }
  for (int horizon : horizons) {
    if (horizon <= 0) {
      throw std::invalid_argument("env_max_episode_steps must be positive");
    }
  }
}

template <typename Config>
const std::string& DmcTaskName(const Config& config, int env_id) {
  const auto& tasks = config["env_task_names"_];
  return tasks.empty() ? config["task_name"_] : tasks.at(env_id);
}

template <typename Config>
int DmcMaxEpisodeSteps(const Config& config, int env_id) {
  const auto& horizons = config["env_max_episode_steps"_];
  return horizons.empty() ? config["max_episode_steps"_] : horizons.at(env_id);
}

}  // namespace mujoco_dmc

#endif  // ENVPOOL_MUJOCO_DMC_MULTITASK_H_
