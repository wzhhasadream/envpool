// Copyright (c) 2026 ApexRL contributors.
// Licensed under the Apache License, Version 2.0.

#ifndef ENVPOOL_MUJOCO_HUMANOID_BENCH_HUMANOID_BENCH_H_
#define ENVPOOL_MUJOCO_HUMANOID_BENCH_HUMANOID_BENCH_H_

#include "envpool/core/async_envpool.h"
#include "envpool/mujoco/humanoid_bench/task.h"

namespace humanoid_bench {

class HumanoidBenchEnvFns {
 public:
  static auto DefaultConfig() {
    return MakeDict(
        "task"_.Bind(std::string("walk")), "xml_file"_.Bind(std::string()),
        "camera_name"_.Bind(std::string("cam_default")),
        "robot_dof"_.Bind(26), "obs_dim"_.Bind(51), "action_dim"_.Bind(19),
        "nq"_.Bind(26), "nv"_.Bind(25), "nbody"_.Bind(0), "ngeom"_.Bind(0),
        "frame_skip"_.Bind(10), "frame_stack"_.Bind(1),
        "action_repeat"_.Bind(1),
        "stand_height"_.Bind(1.65), "reset_noise_scale"_.Bind(0.01),
        "reward_threshold"_.Bind(700.0), "debug_state"_.Bind(false),
        "metric_names"_.Bind(std::vector<std::string>{}),
        "friction_multiplier"_.Bind(1.0));
  }
  template <typename Config>
  static auto StateSpec(const Config& conf) {
    const double inf = std::numeric_limits<double>::infinity();
    const bool debug = conf["debug_state"_];
    return MakeDict(
        "obs"_.Bind(envpool::mujoco::StackSpec(Spec<double>({conf["obs_dim"_]}, {-inf, inf}), conf["frame_stack"_])),
        "info:metrics"_.Bind(Spec<double>({kMetricCount}, {-inf, inf})),
        "info:per_timestep_reward"_.Bind(Spec<double>({-1})),
        "info:terminated"_.Bind(Spec<bool>({-1})),
        "info:qpos"_.Bind(Spec<double>({debug ? conf["nq"_] : 0}, {-inf, inf})),
        "info:qvel"_.Bind(Spec<double>({debug ? conf["nv"_] : 0}, {-inf, inf})),
        "info:qacc_warmstart"_.Bind(Spec<double>({debug ? conf["nv"_] : 0}, {-inf, inf})),
        "info:body_pos"_.Bind(Spec<double>({debug ? conf["nbody"_] * 3 : 0}, {-inf, inf})),
        "info:body_quat"_.Bind(Spec<double>({debug ? conf["nbody"_] * 4 : 0}, {-inf, inf})),
        "info:geom_rgba"_.Bind(Spec<float>({debug ? conf["ngeom"_] * 4 : 0}, {-inf, inf})),
        "info:task_state"_.Bind(Spec<double>({debug ? kTaskStateSize : 0}, {-inf, inf})));
  }
  template <typename Config>
  static auto ActionSpec(const Config& conf) {
    return MakeDict("action"_.Bind(Spec<float>({-1, conf["action_dim"_]}, {-1.0f, 1.0f})));
  }
};

using HumanoidBenchEnvSpec = EnvSpec<HumanoidBenchEnvFns>;
using HumanoidBenchPixelEnvSpec = EnvSpec<envpool::mujoco::PixelObservationEnvFns<HumanoidBenchEnvFns>>;

template <typename SpecT, bool kPixels>
class HumanoidBenchEnv : public Env<SpecT>, public Task {
  using Base = Env<SpecT>;
  using Base::spec_;
  bool debug_;
  double last_reward_{0};

 public:
  using Spec = SpecT;
  using Action = typename Base::Action;
  HumanoidBenchEnv(const Spec& spec, int id)
      : Base(spec, id), Task(
            std::string(spec.config["base_path"_]) + "/assets/" + std::string(spec.config["xml_file"_]),
            spec.config["task"_], spec.config["robot_dof"_], spec.config["frame_skip"_],
            spec.config["max_episode_steps"_], spec.config["frame_stack"_],
            envpool::mujoco::RenderWidthOrDefault<kPixels>(spec.config),
            envpool::mujoco::RenderHeightOrDefault<kPixels>(spec.config),
            spec.config["camera_name"_], spec.config["stand_height"_],
            spec.config["reset_noise_scale"_], spec.config["metric_names"_]),
        debug_(spec.config["debug_state"_]) {
    const double friction = spec.config["friction_multiplier"_];
    if (spec.config["action_repeat"_] < 1) throw std::invalid_argument("action_repeat must be positive");
    if (!std::isfinite(friction) || friction < 0) throw std::invalid_argument("friction_multiplier must be finite and nonnegative");
    for (int i = 0; i < model_->ngeom * 3; ++i) model_->geom_friction[i] *= friction;
    if (model_->nq != spec.config["nq"_] || model_->nv != spec.config["nv"_] || model_->nu != spec.config["action_dim"_])
      throw std::invalid_argument("HumanoidBench asset dimensions differ from the pinned registry");
  }
  bool IsDone() override { return done_; }
  void Reset() override {
    elapsed_step_ = 0;
    done_ = false;
    last_reward_ = 0;
    ResetTask(&this->gen_);
    Observe();
    WriteState(0.0, true);
  }
  void Step(const Action& action) override {
    double reward = 0;
    bool terminated = false;
    for (int i = 0; i < spec_.config["action_repeat"_]; ++i) {
      Simulate(static_cast<float*>(action["action"_].Data()));
      Observe();
      metrics_.fill(std::numeric_limits<double>::quiet_NaN());
      last_reward_ = Reward();
      reward += last_reward_;
      terminated = Terminated();
      ++elapsed_step_;
      done_ = terminated || elapsed_step_ >= max_episode_steps_;
      if (done_) break;
    }
    WriteState(reward, false, terminated);
  }

 private:
  void WriteState(double reward, bool reset, bool terminated = false) {
    auto state = Base::Allocate();
    state["reward"_] = static_cast<float>(reward);
    state["info:per_timestep_reward"_] = last_reward_;
    state["info:terminated"_] = terminated;
    state["discount"_] = terminated ? 0.0f : 1.0f;
    state["trunc"_] = !reset && elapsed_step_ >= max_episode_steps_;
    state["elapsed_step"_] = elapsed_step_;
    if constexpr (kPixels) {
      auto pixels = state["obs:pixels"_];
      RenderPixels(&pixels, reset);
    } else {
      auto observation = state["obs"_];
      frame_stack_buffer_.Assign("obs", &observation, obs_.data(), obs_.size(), reset);
    }
    state["info:metrics"_].Assign(metrics_.data(), kMetricCount);
    if (debug_) {
      state["info:qpos"_].Assign(data_->qpos, model_->nq);
      state["info:qvel"_].Assign(data_->qvel, model_->nv);
      state["info:qacc_warmstart"_].Assign(data_->qacc_warmstart, model_->nv);
      state["info:body_pos"_].Assign(model_->body_pos, model_->nbody * 3);
      state["info:body_quat"_].Assign(model_->body_quat, model_->nbody * 4);
      state["info:geom_rgba"_].Assign(model_->geom_rgba, model_->ngeom * 4);
      ExportTaskState();
      state["info:task_state"_].Assign(task_state_.data(), kTaskStateSize);
    }
  }
};

using HumanoidBenchEnvPool = AsyncEnvPool<HumanoidBenchEnv<HumanoidBenchEnvSpec, false>>;
using HumanoidBenchPixelEnvPool = AsyncEnvPool<HumanoidBenchEnv<HumanoidBenchPixelEnvSpec, true>>;

}  // namespace humanoid_bench
#endif
