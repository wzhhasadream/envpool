/*
 * Copyright 2021 Garena Online Private Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ENVPOOL_ATARI_ATARI_ENV_H_
#define ENVPOOL_ATARI_ATARI_ENV_H_

#include <algorithm>
#include <deque>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ale_interface.hpp"
#include "envpool/atari/continuous_action.h"
#include "envpool/atari/task_catalog.h"
#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/utils/image_process.h"

namespace atari {

auto TurnOffVerbosity() {
  ale::Logger::setMode(ale::Logger::Error);
  return true;
}

static bool verbosity_off = TurnOffVerbosity();

auto GetRomPath(const std::string& base_path, const std::string& task) {
  std::stringstream ss;
  // hardcode path here :(
  ss << base_path << "/atari/roms/" << task << ".bin";
  return ss.str();
}

// Canonical indices refer to the pinned ALE catalog, never to slot order.
int AtariTaskIndex(const std::string& task) {
  auto found = std::find(kAtariTaskNames.begin(), kAtariTaskNames.end(), task);
  if (found == kAtariTaskNames.end()) {
    throw std::invalid_argument("Unknown Atari task " + task);
  }
  return static_cast<int>(found - kAtariTaskNames.begin());
}

template <typename Config>
void ValidateAtariTasks(const Config& config) {
  AtariTaskIndex(config["task"_]);
  const auto& tasks = config["env_task_names"_];
  const auto& horizons = config["env_max_episode_steps"_];
  const auto count = static_cast<std::size_t>(config["num_envs"_]);
  if (!tasks.empty() && tasks.size() != count) {
    throw std::invalid_argument(
        "env_task_names must contain one task for each env");
  }
  if (!tasks.empty() && !config["continuous"_] &&
      !config["full_action_space"_]) {
    throw std::invalid_argument(
        "Atari multitask pools require full_action_space=True (18 actions)");
  }
  if (!horizons.empty() && horizons.size() != count) {
    throw std::invalid_argument(
        "env_max_episode_steps must contain one horizon for each env");
  }
  for (const auto& task : tasks) {
    AtariTaskIndex(task);
  }
  for (int horizon : horizons) {
    if (horizon <= 0) {
      throw std::invalid_argument("env_max_episode_steps must be positive");
    }
  }
}

template <typename Config>
const std::string& AtariTaskName(const Config& config, int env_id) {
  const auto& tasks = config["env_task_names"_];
  return tasks.empty() ? config["task"_] : tasks.at(env_id);
}

template <typename Config>
int AtariMaxEpisodeSteps(const Config& config, int env_id) {
  const auto& horizons = config["env_max_episode_steps"_];
  return horizons.empty() ? config["max_episode_steps"_] : horizons.at(env_id);
}

template <bool Continuous>
class AtariEnvFnsImpl {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "stack_num"_.Bind(4), "frame_skip"_.Bind(4), "noop_max"_.Bind(30),
        "zero_discount_on_life_loss"_.Bind(false), "episodic_life"_.Bind(false),
        "reward_clip"_.Bind(false), "use_fire_reset"_.Bind(true),
        "img_height"_.Bind(84), "img_width"_.Bind(84),
        "task"_.Bind(std::string("pong")),
        "env_task_names"_.Bind(std::vector<std::string>{}),
        "env_max_episode_steps"_.Bind(std::vector<int>{}), "mode"_.Bind(-1),
        "difficulty"_.Bind(-1), "full_action_space"_.Bind(false),
        "continuous"_.Bind(Continuous),
        "continuous_action_threshold"_.Bind(0.5f),
        "repeat_action_probability"_.Bind(0.0f),
        "use_inter_area_resize"_.Bind(true), "gray_scale"_.Bind(true));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    ValidateAtariTasks(conf);
    if (conf["continuous"_] != Continuous) {
      throw std::invalid_argument(
          "continuous must match the selected Atari action-space binding");
    }
    if (!std::isfinite(conf["continuous_action_threshold"_])) {
      throw std::invalid_argument("continuous_action_threshold must be finite");
    }
    return MakeDict(
        "obs"_.Bind(
            Spec<uint8_t>({conf["stack_num"_] * (conf["gray_scale"_] ? 1 : 3),
                           conf["img_height"_], conf["img_width"_]},
                          {0, 255})),
        "info:task_id"_.Bind(
            Spec<int>({-1}, {0, static_cast<int>(kAtariTaskNames.size()) - 1})),
        "info:lives"_.Bind(Spec<int>({-1})),
        "info:reward"_.Bind(Spec<float>({-1})),
        "info:terminated"_.Bind(Spec<int>({-1}, {0, 1})),
        "info:ram"_.Bind(Spec<uint8_t>({128}, {0, 255})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    std::vector<std::string> tasks = conf["env_task_names"_];
    if (tasks.empty()) {
      tasks.push_back(conf["task"_]);
    }
    std::sort(tasks.begin(), tasks.end());
    tasks.erase(std::unique(tasks.begin(), tasks.end()), tasks.end());
    int action_size = 0;
    // Validate each selected ROM before any asynchronous workers start. In a
    // mixed pool, equal action counts alone do not imply equal action meaning.
    for (const auto& task : tasks) {
      ale::ALEInterface env;
      env.loadROM(GetRomPath(conf["base_path"_], task));
      if (conf["mode"_] >= 0) {
        env.setMode(conf["mode"_]);
      }
      if (conf["difficulty"_] >= 0) {
        env.setDifficulty(conf["difficulty"_]);
      }
      auto actions = conf["full_action_space"_] || Continuous
                         ? env.getLegalActionSet()
                         : env.getMinimalActionSet();
      action_size = static_cast<int>(actions.size());
      if (!Continuous && !conf["env_task_names"_].empty()) {
        if (action_size != 18) {
          throw std::invalid_argument(
              "Atari multitask pools require the canonical 18 ALE actions; " +
              task + " has " + std::to_string(action_size));
        }
        for (int action = 0; action < action_size; ++action) {
          if (static_cast<int>(actions[action]) != action) {
            throw std::invalid_argument(
                "Atari multitask pools require canonical ALE action ordering");
          }
        }
      }
    }
    if constexpr (Continuous) {
      return MakeDict(
          "action"_.Bind(Spec<float>({-1, 3}, {{0.0f, -kContinuousPi, 0.0f},
                                               {1.0f, kContinuousPi, 1.0f}})));
    } else {
      return MakeDict("action"_.Bind(Spec<int>({-1}, {0, action_size - 1})));
    }
  }
};

using AtariEnvFns = AtariEnvFnsImpl<false>;
using AtariContinuousEnvFns = AtariEnvFnsImpl<true>;
using AtariEnvSpec = EnvSpec<AtariEnvFns>;
using AtariContinuousEnvSpec = EnvSpec<AtariContinuousEnvFns>;
using FrameSpec = Spec<uint8_t>;

template <bool Continuous>
class AtariEnvImpl : public Env<EnvSpec<AtariEnvFnsImpl<Continuous>>>,
                     public RenderableEnv {
  using Base = Env<EnvSpec<AtariEnvFnsImpl<Continuous>>>;

 public:
  using Spec = typename Base::Spec;
  using Action = typename Base::Action;

 protected:
  using Base::Allocate;
  using Base::force_reset_;
  using Base::gen_;
  using Base::seed_;
  using Base::spec_;

  const int kRawHeight = 210;
  const int kRawWidth = 160;
  const int kRawSize = kRawWidth * kRawHeight;
  std::unique_ptr<ale::ALEInterface> env_;
  ale::ActionVect action_set_;
  int max_episode_steps_, elapsed_step_, stack_num_, frame_skip_, task_index_;
  bool fire_reset_{false}, reward_clip_, zero_discount_on_life_loss_;
  bool gray_scale_, episodic_life_, use_inter_area_resize_;
  bool done_{true};
  int lives_;
  FrameSpec raw_spec_, resize_spec_, transpose_spec_;
  std::deque<Array> stack_buf_;
  std::vector<Array> maxpool_buf_;
  Array resize_img_;
  Array render_img_;
  std::uniform_int_distribution<> dist_noop_;
  std::string rom_path_;

 public:
  AtariEnvImpl(const Spec& spec, int env_id)
      : Base(spec, env_id),
        env_(new ale::ALEInterface()),
        max_episode_steps_(AtariMaxEpisodeSteps(spec.config, env_id)),
        elapsed_step_(max_episode_steps_),
        stack_num_(spec.config["stack_num"_]),
        frame_skip_(spec.config["frame_skip"_]),
        task_index_(AtariTaskIndex(AtariTaskName(spec.config, env_id))),
        reward_clip_(spec.config["reward_clip"_]),
        zero_discount_on_life_loss_(spec.config["zero_discount_on_life_loss"_]),
        gray_scale_(spec.config["gray_scale"_]),
        episodic_life_(spec.config["episodic_life"_]),
        use_inter_area_resize_(spec.config["use_inter_area_resize"_]),
        raw_spec_({kRawHeight, kRawWidth, gray_scale_ ? 1 : 3}),
        resize_spec_({spec.config["img_height"_], spec.config["img_width"_],
                      gray_scale_ ? 1 : 3}),
        transpose_spec_({gray_scale_ ? 1 : 3, spec.config["img_height"_],
                         spec.config["img_width"_]}),
        resize_img_(resize_spec_),
        render_img_(FrameSpec({kRawHeight, kRawWidth, 3})),
        dist_noop_(0, spec.config["noop_max"_] - 1),
        rom_path_(GetRomPath(spec.config["base_path"_],
                             AtariTaskName(spec.config, env_id))) {
    env_->setFloat("repeat_action_probability",
                   spec.config["repeat_action_probability"_]);
    env_->setInt("random_seed", seed_);
    env_->loadROM(rom_path_);
    if (spec.config["mode"_] >= 0) {
      env_->setMode(spec.config["mode"_]);
    }
    if (spec.config["difficulty"_] >= 0) {
      env_->setDifficulty(spec.config["difficulty"_]);
    }
    if (spec.config["full_action_space"_] || Continuous) {
      action_set_ = env_->getLegalActionSet();
    } else {
      action_set_ = env_->getMinimalActionSet();
    }
    if (spec.config["use_fire_reset"_]) {
      // https://github.com/sail-sg/envpool/issues/221
      for (auto a : action_set_) {
        if (a == 1) {
          fire_reset_ = true;
        }
      }
    }
    // init buf
    for (int i = 0; i < 2; ++i) {
      maxpool_buf_.emplace_back(raw_spec_);
    }
    for (int i = 0; i < stack_num_; ++i) {
      stack_buf_.emplace_back(transpose_spec_);
    }
  }

  void Reset() override {
    int noop = dist_noop_(gen_) + 1 - static_cast<int>(fire_reset_);
    bool push_all = false;
    if (force_reset_ || !episodic_life_ || env_->game_over() ||
        elapsed_step_ >= max_episode_steps_) {
      env_->reset_game();
      elapsed_step_ = 0;
      push_all = true;
    }
    while ((noop--) != 0) {
      env_->act(static_cast<ale::Action>(0));
      if (env_->game_over()) {
        env_->reset_game();
        push_all = true;
      }
    }
    if (fire_reset_) {
      env_->act(static_cast<ale::Action>(1));
    }
    uint8_t* ale_screen_data = env_->getScreen().getArray();
    auto* ptr = static_cast<uint8_t*>(maxpool_buf_[0].Data());
    if (gray_scale_) {
      env_->theOSystem->colourPalette().applyPaletteGrayscale(
          ptr, ale_screen_data, kRawSize);
    } else {
      env_->theOSystem->colourPalette().applyPaletteRGB(ptr, ale_screen_data,
                                                        kRawSize);
    }
    PushStack(push_all, false);
    done_ = false;
    lives_ = env_->lives();
    WriteState(0.0, 1.0, 0.0);
  }

  void Step(const Action& action) override {
    float reward = 0.0;
    done_ = false;
    ale::Action ale_action;
    float paddle_strength = 1.0f;
    if constexpr (Continuous) {
      const auto* values = static_cast<const float*>(action["action"_].Data());
      auto decoded = ConvertContinuousAction(
          values, spec_.config["continuous_action_threshold"_]);
      ale_action = static_cast<ale::Action>(decoded.action);
      paddle_strength = decoded.paddle_strength;
    } else {
      int index = action["action"_];
      ale_action = action_set_[index];
    }
    int skip_id = frame_skip_;
    int pooled_frame_count = std::min(frame_skip_, 2);
    for (; skip_id > 0 && !done_; --skip_id) {
      reward += env_->act(ale_action, paddle_strength);
      done_ = env_->game_over();
      if (skip_id <= pooled_frame_count) {
        uint8_t* ale_screen_data = env_->getScreen().getArray();
        auto* ptr = static_cast<uint8_t*>(
            maxpool_buf_[pooled_frame_count - skip_id].Data());
        if (gray_scale_) {
          env_->theOSystem->colourPalette().applyPaletteGrayscale(
              ptr, ale_screen_data, kRawSize);
        } else {
          env_->theOSystem->colourPalette().applyPaletteRGB(
              ptr, ale_screen_data, kRawSize);
        }
      }
    }
    // push the maxpool outcome to the stack_buf
    PushStack(false, frame_skip_ > 1 && skip_id == 0);
    ++elapsed_step_;
    done_ |= (elapsed_step_ >= max_episode_steps_);
    if (episodic_life_ && 0 < env_->lives() && env_->lives() < lives_) {
      done_ = true;
    }
    float discount;
    if (zero_discount_on_life_loss_) {
      discount = static_cast<float>(lives_ == env_->lives() && !done_);
    } else {
      discount = 1.0f - static_cast<float>(done_);
    }
    float info_reward = reward;
    if (reward_clip_) {
      if (reward > 0) {
        reward = 1;
      } else if (reward < 0) {
        reward = -1;
      }
    }
    lives_ = env_->lives();
    WriteState(reward, discount, info_reward);
  }

  bool IsDone() override { return done_; }

  std::pair<int, int> RenderSize(int width, int height) const override {
    return {width > 0 ? width : kRawWidth, height > 0 ? height : kRawHeight};
  }

  void Render(int width, int height, int /*camera_id*/,
              unsigned char* rgb) override {
    uint8_t* ale_screen_data = env_->getScreen().getArray();
    auto* ptr = static_cast<uint8_t*>(render_img_.Data());
    env_->theOSystem->colourPalette().applyPaletteRGB(ptr, ale_screen_data,
                                                      kRawSize);
    Array output(FrameSpec({height, width, 3}), reinterpret_cast<char*>(rgb));
    if (width == kRawWidth && height == kRawHeight) {
      output.Assign(render_img_);
    } else {
      Resize(render_img_, &output, use_inter_area_resize_);
    }
  }

 protected:
  int CurrentMaxEpisodeSteps() const override { return max_episode_steps_; }

 private:
  void WriteState(float reward, float discount, float info_reward) {
    auto state = Allocate();
    state["discount"_] = discount;
    state["trunc"_] = done_ && (elapsed_step_ >= max_episode_steps_);
    state["reward"_] = reward;
    state["info:task_id"_] = task_index_;
    state["info:lives"_] = lives_;
    state["info:reward"_] = info_reward;
    state["info:terminated"_] = env_->game_over();
    const auto& ram = env_->getRAM();
    state["info:ram"_].Assign(ram.array(), ram.size());
    // overwrite current_step to make sure
    // episodic_life == True behaves correctly
    // see Issue #179
    state["elapsed_step"_] = elapsed_step_;
    for (int i = 0; i < stack_num_; ++i) {
      state["obs"_]
          .Slice(gray_scale_ ? i : i * 3, gray_scale_ ? i + 1 : (i + 1) * 3)
          .Assign(stack_buf_[i]);
    }
  }

  /**
   * FrameStack env wrapper implementation.
   *
   * The original gray scale image are saved inside maxpool_buf_.
   * The stacked result is in stack_buf_ where len(stack_buf_) == stack_num_.
   *
   * At reset time, we need to clear all data in stack_buf_ with push_all =
   * true and maxpool = false (there is only one observation); at step time,
   * we push max(maxpool_buf_[0], maxpool_buf_[1]) at the end of
   * stack_buf_, and pop the first item in stack_buf_, with push_all = false
   * and maxpool = true.
   *
   * @param push_all whether to use the most recent observation to write all
   *   of the data in stack_buf_.
   * @param maxpool whether to perform maxpool operation on the last two
   *   observation. Maybe there is only one?
   */
  void PushStack(bool push_all, bool maxpool) {
    auto* ptr = static_cast<uint8_t*>(maxpool_buf_[0].Data());
    if (maxpool) {
      auto* ptr1 = static_cast<uint8_t*>(maxpool_buf_[1].Data());
      for (std::size_t i = 0; i < maxpool_buf_[0].size; ++i) {
        ptr[i] = std::max(ptr[i], ptr1[i]);
      }
    }
    Resize(maxpool_buf_[0], &resize_img_, use_inter_area_resize_);
    Array tgt = std::move(*stack_buf_.begin());
    ptr = static_cast<uint8_t*>(tgt.Data());
    stack_buf_.pop_front();
    if (gray_scale_) {
      tgt.Assign(resize_img_);
    } else {
      auto* ptr1 = static_cast<uint8_t*>(resize_img_.Data());
      // tgt = resize_img_.transpose(1, 2, 0)
      // tgt[i, j, k] = resize_img_[j, k, i]
      std::size_t h = resize_img_.Shape(0);
      std::size_t w = resize_img_.Shape(1);
      for (std::size_t j = 0; j < h; ++j) {
        for (std::size_t k = 0; k < w; ++k) {
          for (std::size_t i = 0; i < 3; ++i) {
            ptr[i * h * w + j * w + k] = ptr1[j * w * 3 + k * 3 + i];
          }
        }
      }
    }
    std::size_t size = tgt.size;
    stack_buf_.push_back(std::move(tgt));
    if (push_all) {
      for (auto& s : stack_buf_) {
        auto* ptr_s = static_cast<uint8_t*>(s.Data());
        if (ptr != ptr_s) {
          std::memcpy(ptr_s, ptr, size);
        }
      }
    }
  }
};

using AtariEnv = AtariEnvImpl<false>;
using AtariContinuousEnv = AtariEnvImpl<true>;
using AtariEnvPool = AsyncEnvPool<AtariEnv>;

// Check the entire batch on the caller thread. Throwing from an asynchronous
// environment worker would terminate the process and could partly step a batch.
class AtariContinuousEnvPool : public AsyncEnvPool<AtariContinuousEnv> {
  using Base = AsyncEnvPool<AtariContinuousEnv>;

 public:
  explicit AtariContinuousEnvPool(const Spec& spec) : Base(spec) {}

  void Send(const Action& action) { Send(action.template AllValues<Array>()); }

  void Send(const std::vector<Array>& action) override {
    ValidateActions(action);
    Base::Send(action);
  }

  void Send(std::vector<Array>&& action) override {
    ValidateActions(action);
    Base::Send(std::move(action));
  }

 private:
  static void ValidateActions(const std::vector<Array>& action) {
    if (action.size() != 3 || action[2].element_size != sizeof(float) ||
        action[2].ndim != 2 || action[2].Shape(1) != 3 || action[0].ndim != 1 ||
        action[2].Shape(0) != action[0].Shape(0)) {
      throw std::invalid_argument(
          "Atari continuous action batch must have shape (num_envs, 3)");
    }
    const auto* values = static_cast<const float*>(action[2].Data());
    for (std::size_t i = 0; i < action[2].Shape(0); ++i) {
      ValidateContinuousAction(values + i * 3);
    }
  }
};

}  // namespace atari

#endif  // ENVPOOL_ATARI_ATARI_ENV_H_
