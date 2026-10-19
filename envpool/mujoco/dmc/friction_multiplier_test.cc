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

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "envpool/mujoco/dmc/acrobot.h"
#include "envpool/mujoco/dmc/ball_in_cup.h"
#include "envpool/mujoco/dmc/cartpole.h"
#include "envpool/mujoco/dmc/cheetah.h"
#include "envpool/mujoco/dmc/dog.h"
#include "envpool/mujoco/dmc/finger.h"
#include "envpool/mujoco/dmc/fish.h"
#include "envpool/mujoco/dmc/hopper.h"
#include "envpool/mujoco/dmc/humanoid.h"
#include "envpool/mujoco/dmc/humanoid_CMU.h"
#include "envpool/mujoco/dmc/lqr.h"
#include "envpool/mujoco/dmc/manipulator.h"
#include "envpool/mujoco/dmc/mujoco_env.h"
#include "envpool/mujoco/dmc/pendulum.h"
#include "envpool/mujoco/dmc/point_mass.h"
#include "envpool/mujoco/dmc/quadruped.h"
#include "envpool/mujoco/dmc/reacher.h"
#include "envpool/mujoco/dmc/stacker.h"
#include "envpool/mujoco/dmc/swimmer.h"
#include "envpool/mujoco/dmc/walker.h"

namespace mujoco_dmc {
namespace {

std::string SlidingBoxXml(bool explicit_pair) {
  return std::string(R"(
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 .1" condim="6"
          friction=".8 .04 .006"/>
    <body pos="0 0 .049">
      <freejoint/>
      <geom name="box" type="box" size=".05 .05 .05" mass="1"
            condim="6" friction=".2 .01 .002"/>
    </body>
  </worldbody>
)") +
         (explicit_pair ? R"(
  <contact>
    <pair geom1="floor" geom2="box" condim="6"
          friction=".3 .7 .02 .01 .04"/>
  </contact>
)"
                        : "") +
         "</mujoco>";
}

class FrictionTestEnv : public MujocoEnv {
 public:
  explicit FrictionTestEnv(double friction_multiplier = 1.0,
                           bool explicit_pair = false, int action_repeat = 1,
                           int horizon = 1000, int n_sub_steps = 1)
      : MujocoEnv("envpool", SlidingBoxXml(explicit_pair), n_sub_steps, horizon,
                  1, 84, 84, -1, action_repeat, friction_multiplier) {}

  mjModel* Model() { return model_; }
  mjData* Data() { return data_; }

  void TaskInitializeEpisode() override { data_->qvel[0] = 1.0; }
  float TaskGetReward() override { return static_cast<float>(data_->qpos[0]); }
};

class ControlStepTestEnv : public FrictionTestEnv {
 public:
  int before_steps{0}, after_steps{0}, before_substeps{0}, after_substeps{0};
  int terminal_at{0};

  ControlStepTestEnv(int action_repeat, int horizon)
      : FrictionTestEnv(1.0, false, action_repeat, horizon, 2) {}

  int ElapsedStep() const { return elapsed_step_; }
  float Reward() const { return reward_; }
  float LastReward() const { return last_reward_; }
  float Discount() const { return discount_; }
  bool Done() const { return done_; }

  void TaskInitializeEpisode() override {
    FrictionTestEnv::TaskInitializeEpisode();
    before_steps = after_steps = before_substeps = after_substeps = 0;
  }
  void TaskBeforeStep(const mjtNum* action) override {
    ++before_steps;
    MujocoEnv::TaskBeforeStep(action);
  }
  void TaskAfterStep() override { ++after_steps; }
  void TaskBeforeSubStep(const mjtNum* action) override { ++before_substeps; }
  void TaskAftersSubStep() override { ++after_substeps; }
  bool TaskShouldTerminateEpisode() override {
    return terminal_at > 0 && elapsed_step_ >= terminal_at;
  }
};

TEST(DmcActionRepeatTest, MatchesProductionControlStepsAndPartialHorizon) {
  for (int repeat : {1, 3, 9}) {
    for (int horizon : {5, 17}) {
      SCOPED_TRACE(::testing::Message() << repeat << ":" << horizon);
      ControlStepTestEnv reference(1, horizon);
      ControlStepTestEnv repeated(repeat, horizon);
      for (int episode = 0; episode < 3; ++episode) {
        reference.ControlReset();
        repeated.ControlReset();
        EXPECT_EQ(repeated.Reward(), 0.0f);
        EXPECT_EQ(repeated.LastReward(), 0.0f);
        EXPECT_EQ(repeated.ElapsedStep(), 0);
        EXPECT_FALSE(repeated.Done());
        const mjtNum action = 0;
        while (!repeated.Done()) {
          double sum = 0;
          for (int i = 0; i < repeat && !reference.Done(); ++i) {
            reference.ControlStep(&action);
            sum += reference.Reward();
          }
          repeated.ControlStep(&action);
          EXPECT_EQ(repeated.Reward(), static_cast<float>(sum));
          EXPECT_EQ(repeated.LastReward(), reference.Reward());
          EXPECT_EQ(repeated.ElapsedStep(), reference.ElapsedStep());
          EXPECT_EQ(repeated.Done(), reference.Done());
          EXPECT_EQ(repeated.Discount(), reference.Discount());
          EXPECT_EQ(repeated.before_steps, reference.before_steps);
          EXPECT_EQ(repeated.after_steps, reference.after_steps);
          EXPECT_EQ(repeated.before_substeps, reference.before_substeps);
          EXPECT_EQ(repeated.after_substeps, reference.after_substeps);
          for (int i = 0; i < reference.Model()->nq; ++i) {
            EXPECT_EQ(repeated.Data()->qpos[i], reference.Data()->qpos[i]);
          }
          for (int i = 0; i < reference.Model()->nv; ++i) {
            EXPECT_EQ(repeated.Data()->qvel[i], reference.Data()->qvel[i]);
          }
        }
        EXPECT_EQ(repeated.ElapsedStep(), horizon);
        EXPECT_EQ(repeated.before_steps, horizon);
        EXPECT_EQ(repeated.before_substeps, 2 * horizon);
        EXPECT_EQ(repeated.Discount(), 1.0f);
      }
    }
  }
}

TEST(DmcActionRepeatTest, StopsAtTaskTerminationAndPreservesHorizonPrecedence) {
  for (int horizon : {2, 10}) {
    ControlStepTestEnv env(9, horizon);
    env.terminal_at = 2;
    env.ControlReset();
    const mjtNum action = 0;
    env.ControlStep(&action);
    EXPECT_TRUE(env.Done());
    EXPECT_EQ(env.ElapsedStep(), 2);
    EXPECT_EQ(env.before_steps, 2);
    EXPECT_EQ(env.before_substeps, 4);
    EXPECT_EQ(env.Discount(), horizon == 2 ? 1.0f : 0.0f);
  }
}

std::array<double, 4> InvalidMultipliers() {
  return {-1.0, std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::infinity(),
          -std::numeric_limits<double>::infinity()};
}

template <typename TaskFns, typename PixelFns = PixelObservationEnvFns<TaskFns>>
void CheckSpecFrictionConfig() {
  auto config = EnvSpec<TaskFns>::kDefaultConfig;
  EXPECT_EQ(config["friction_multiplier"_], 1.0);
  for (double multiplier : {0.0, 0.25, 1.0, 3.0}) {
    config["friction_multiplier"_] = multiplier;
    EXPECT_NO_THROW(TaskFns::StateSpec(config));
  }
  for (double multiplier : InvalidMultipliers()) {
    config["friction_multiplier"_] = multiplier;
    EXPECT_THROW(TaskFns::StateSpec(config), std::invalid_argument);
  }
  auto pixel_config = EnvSpec<PixelFns>::kDefaultConfig;
  EXPECT_EQ(pixel_config["friction_multiplier"_], 1.0);
  for (double multiplier : {0.0, 0.25, 1.0, 3.0}) {
    pixel_config["friction_multiplier"_] = multiplier;
    EXPECT_NO_THROW(PixelFns::StateSpec(pixel_config));
  }
  for (double multiplier : InvalidMultipliers()) {
    pixel_config["friction_multiplier"_] = multiplier;
    EXPECT_THROW(PixelFns::StateSpec(pixel_config), std::invalid_argument);
  }
}

TEST(DmcFrictionMultiplierTest, EveryStateAndPixelDomainValidatesConfig) {
  CheckSpecFrictionConfig<AcrobotEnvFns>();
  CheckSpecFrictionConfig<BallInCupEnvFns>();
  CheckSpecFrictionConfig<CartpoleEnvFns>();
  CheckSpecFrictionConfig<CheetahEnvFns>();
  CheckSpecFrictionConfig<DogEnvFns>();
  CheckSpecFrictionConfig<FingerEnvFns>();
  CheckSpecFrictionConfig<FishEnvFns>();
  CheckSpecFrictionConfig<HopperEnvFns>();
  CheckSpecFrictionConfig<HumanoidEnvFns>();
  CheckSpecFrictionConfig<HumanoidCMUEnvFns>();
  CheckSpecFrictionConfig<LqrEnvFns>();
  CheckSpecFrictionConfig<ManipulatorEnvFns>();
  CheckSpecFrictionConfig<PendulumEnvFns>();
  CheckSpecFrictionConfig<PointMassEnvFns>();
  CheckSpecFrictionConfig<QuadrupedEnvFns, QuadrupedPixelEnvFns>();
  CheckSpecFrictionConfig<ReacherEnvFns>();
  CheckSpecFrictionConfig<StackerEnvFns>();
  CheckSpecFrictionConfig<SwimmerEnvFns>();
  CheckSpecFrictionConfig<WalkerEnvFns>();
}

template <typename PixelFns>
void CheckPixelCamera(int expected_camera) {
  auto config = EnvSpec<PixelFns>::kDefaultConfig;
  EXPECT_EQ(config["render_camera_id"_], expected_camera);
  EXPECT_EQ(config["render_width"_], 84);
  EXPECT_EQ(config["render_height"_], 84);
  config["render_width"_] = 96;
  config["render_height"_] = 64;
  auto state_spec = PixelFns::StateSpec(config);
  EXPECT_EQ(state_spec["obs:pixels"_].shape, std::vector<int>({3, 64, 96}));
  for (int camera : {-1, 0, 2}) {
    config["render_camera_id"_] = camera;
    EXPECT_EQ(RenderCameraIdOrDefault<true>(config), camera);
    EXPECT_NO_THROW(PixelFns::StateSpec(config));
  }
}

TEST(DmcPixelCameraTest, DomainDefaultsAndExplicitOverrides) {
  CheckPixelCamera<AcrobotPixelEnvFns>(0);
  CheckPixelCamera<BallInCupPixelEnvFns>(0);
  CheckPixelCamera<CartpolePixelEnvFns>(0);
  CheckPixelCamera<CheetahPixelEnvFns>(0);
  CheckPixelCamera<DogPixelEnvFns>(0);
  CheckPixelCamera<FingerPixelEnvFns>(0);
  CheckPixelCamera<FishPixelEnvFns>(0);
  CheckPixelCamera<HopperPixelEnvFns>(0);
  CheckPixelCamera<HumanoidPixelEnvFns>(0);
  CheckPixelCamera<HumanoidCMUPixelEnvFns>(0);
  CheckPixelCamera<LqrPixelEnvFns>(0);
  CheckPixelCamera<ManipulatorPixelEnvFns>(0);
  CheckPixelCamera<PendulumPixelEnvFns>(0);
  CheckPixelCamera<PointMassPixelEnvFns>(0);
  CheckPixelCamera<QuadrupedPixelEnvFns>(2);
  CheckPixelCamera<ReacherPixelEnvFns>(0);
  CheckPixelCamera<StackerPixelEnvFns>(0);
  CheckPixelCamera<SwimmerPixelEnvFns>(0);
  CheckPixelCamera<WalkerPixelEnvFns>(0);
}

class BallInCupInspectionEnv : public BallInCupEnv {
 public:
  using BallInCupEnv::BallInCupEnv;
  mjModel* Model() { return model_; }
  mjData* Data() { return data_; }
};

TEST(DmcBallInCupTest, CatchRewardUsesBallGeomRadiusAtTargetBoundary) {
  BallInCupEnvSpec spec;
  BallInCupInspectionEnv env(spec, 0);
  env.PhysicsReset();
  mjModel* model = env.Model();
  mjData* data = env.Data();
  const int ball_body = mj_name2id(model, mjOBJ_XBODY, "ball");
  const int ball_geom = mj_name2id(model, mjOBJ_GEOM, "ball");
  const int target = mj_name2id(model, mjOBJ_SITE, "target");
  ASSERT_NE(ball_body, ball_geom);
  const mjtNum radius = model->geom_size[3 * ball_geom];
  ASSERT_NE(radius, model->geom_size[3 * ball_body]);
  // Test the task reward directly at both sides of its strict x/z boundary.
  // The body and geom intentionally have different indices in the real model.
  for (int axis : {0, 2}) {
    mjtNum limit = model->site_size[3 * target + axis] - radius;
    for (mjtNum sign : {-1.0, 1.0}) {
      for (mjtNum offset : {-1.0e-6, 1.0e-6}) {
        for (int coordinate : {0, 2}) {
          data->xpos[3 * ball_body + coordinate] =
              data->site_xpos[3 * target + coordinate];
        }
        data->xpos[3 * ball_body + axis] += sign * (limit + offset);
        EXPECT_EQ(env.TaskGetReward(), offset < 0 ? 1.0f : 0.0f);
      }
    }
  }
}

class FingerInspectionEnv : public FingerEnv {
 public:
  using FingerEnv::FingerEnv;
  mjModel* Model() { return model_; }
};

TEST(DmcFingerTest, SpinHidesOnlyTargetAndTipAlpha) {
  auto config = FingerEnvSpec::kDefaultConfig;
  config["task_name"_] = std::string("spin");
  FingerEnvSpec spec(config);
  FingerInspectionEnv env(spec, 0);
  mjModel* model = env.Model();
  const int target = mj_name2id(model, mjOBJ_SITE, "target");
  const int tip = mj_name2id(model, mjOBJ_SITE, "tip");
  ASSERT_GE(target, 0);
  ASSERT_GE(tip, 0);
  std::vector<float> original(model->site_rgba,
                              model->site_rgba + 4 * model->nsite);
  for (int episode = 0; episode < 3; ++episode) {
    env.ControlReset();
    for (int site = 0; site < model->nsite; ++site) {
      for (int channel = 0; channel < 4; ++channel) {
        const bool hidden_alpha =
            channel == 3 && (site == target || site == tip);
        EXPECT_EQ(model->site_rgba[4 * site + channel],
                  hidden_alpha ? 0.0f : original[4 * site + channel]);
      }
    }
  }
}

TEST(DmcFrictionMultiplierTest, ConstructorRejectsInvalidMultiplier) {
  for (double multiplier : InvalidMultipliers()) {
    EXPECT_THROW(FrictionTestEnv env(multiplier), std::invalid_argument);
  }
}

TEST(DmcFrictionMultiplierTest, ScalesEveryCoefficientWithoutResetCompounding) {
  for (bool explicit_pair : {false, true}) {
    FrictionTestEnv baseline(1.0, explicit_pair);
    for (double multiplier : {0.0, 0.25, 1.0, 3.0}) {
      SCOPED_TRACE(::testing::Message() << explicit_pair << ":" << multiplier);
      FrictionTestEnv env(multiplier, explicit_pair);
      for (int episode = 0; episode < 12; ++episode) {
        for (int i = 0; i < 3 * baseline.Model()->ngeom; ++i) {
          EXPECT_EQ(env.Model()->geom_friction[i],
                    baseline.Model()->geom_friction[i] * multiplier);
        }
        for (int i = 0; i < 5 * baseline.Model()->npair; ++i) {
          EXPECT_EQ(env.Model()->pair_friction[i],
                    baseline.Model()->pair_friction[i] * multiplier);
        }
        env.ControlReset();
        const mjtNum action = 0;
        for (int step = 0; step < 5; ++step) {
          env.ControlStep(&action);
        }
      }
    }
  }
}

TEST(DmcFrictionMultiplierTest, DefaultMatchesUnmodifiedCompiledModel) {
  FrictionTestEnv env;
  std::string xml = SlidingBoxXml(false);
  mjVFS vfs;
  mj_defaultVFS(&vfs);
  ASSERT_EQ(mj_addBufferVFS(&vfs, "baseline.xml", xml.data(), xml.size()), 0);
  std::array<char, 1000> error{};
  std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model(
      mj_loadXML("baseline.xml", &vfs, error.data(), error.size()),
      &mj_deleteModel);
  mj_deleteVFS(&vfs);
  ASSERT_NE(model, nullptr) << error.data();
  ASSERT_EQ(env.Model()->ngeom, model->ngeom);
  for (int i = 0; i < 3 * model->ngeom; ++i) {
    EXPECT_EQ(env.Model()->geom_friction[i], model->geom_friction[i]);
  }
  FrictionTestEnv explicit_one(1.0);
  for (int episode = 0; episode < 3; ++episode) {
    env.ControlReset();
    explicit_one.ControlReset();
    const mjtNum action = 0;
    for (int step = 0; step < 200; ++step) {
      env.ControlStep(&action);
      explicit_one.ControlStep(&action);
      for (int i = 0; i < model->nq; ++i) {
        EXPECT_EQ(env.Data()->qpos[i], explicit_one.Data()->qpos[i]);
      }
      for (int i = 0; i < model->nv; ++i) {
        EXPECT_EQ(env.Data()->qvel[i], explicit_one.Data()->qvel[i]);
      }
    }
  }
}

TEST(DmcFrictionMultiplierTest, ExplicitPairOverridesGeomContactFriction) {
  for (double multiplier : {0.25, 1.0, 3.0}) {
    FrictionTestEnv env(multiplier, true);
    env.ControlReset();
    ASSERT_EQ(env.Model()->npair, 1);
    ASSERT_GT(env.Data()->ncon, 0);
    for (int i = 0; i < env.Data()->ncon; ++i) {
      for (int j = 0; j < 5; ++j) {
        EXPECT_EQ(env.Data()->contact[i].friction[j],
                  env.Model()->pair_friction[j]);
      }
    }
    EXPECT_NE(env.Model()->pair_friction[0], env.Model()->geom_friction[0]);
    EXPECT_NE(env.Model()->pair_friction[0], env.Model()->pair_friction[1]);
  }
}

TEST(DmcFrictionMultiplierTest, ChangesContactDynamics) {
  for (bool explicit_pair : {false, true}) {
    FrictionTestEnv frictionless(0.0, explicit_pair);
    FrictionTestEnv baseline(1.0, explicit_pair);
    frictionless.ControlReset();
    baseline.ControlReset();
    const mjtNum action = 0;
    for (int step = 0; step < 200; ++step) {
      frictionless.ControlStep(&action);
      baseline.ControlStep(&action);
    }
    EXPECT_GT(frictionless.Data()->qpos[0], baseline.Data()->qpos[0] + 0.05);
    EXPECT_TRUE(std::isfinite(frictionless.Data()->qpos[0]));
    EXPECT_TRUE(std::isfinite(baseline.Data()->qpos[0]));
  }
}

}  // namespace
}  // namespace mujoco_dmc
