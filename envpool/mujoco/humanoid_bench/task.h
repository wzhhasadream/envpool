// Task logic translated from HumanoidBench 0.2.1 (MIT).
// Copyright (c) 2024 Robot Learning Lab (RLL) lab @ UC Berkeley.
// See third_party/humanoid_bench/LICENSE for the original license.

#ifndef ENVPOOL_MUJOCO_HUMANOID_BENCH_TASK_H_
#define ENVPOOL_MUJOCO_HUMANOID_BENCH_TASK_H_

#include <array>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "envpool/mujoco/gym/mujoco_env.h"

namespace humanoid_bench {

constexpr int kMetricCount = 128;
constexpr int kTaskStateSize = 64;
using Vec3 = std::array<double, 3>;

class Task : public mujoco_gym::MujocoEnv {
 protected:
  std::string task_;
  std::string render_key_;
  struct RenderContext;
  std::shared_ptr<RenderContext> render_context_;
  int robot_dof_, camera_id_, stage_{0}, spoon_steps_{0};
  double stand_height_, randomness_;
  bool truck_initialized_{false};
  Vec3 goal_{}, head_pos0_{};
  std::array<Vec3, 6> placement_goals_{};
  std::array<int, 6> bookshelf_objects_{{-19, -22, -15, -20, -23, -23}};
  std::array<double, 5> initial_zs_{};
  std::vector<int> truck_, picked_, table_;
  std::unordered_map<std::string, int> bodies_, sites_, geoms_, sensors_;
  std::unordered_map<std::string, int> metric_ids_;
  std::array<double, kMetricCount> metrics_{};
  std::array<double, kTaskStateSize> task_state_{};
  std::vector<double> obs_, control_;

 public:
  Task(const std::string& xml, const std::string& task, int robot_dof,
       int frame_skip, int max_steps, int frame_stack, int width, int height,
       const std::string& camera_name, double stand_height, double randomness,
       const std::vector<std::string>& metric_names);
  void ResetTask(std::mt19937* gen);
  void Observe();
  double Reward();
  bool Terminated();
  void Simulate(const float* action);
  void ExportTaskState();
  void Render(int width, int height, int camera_id, unsigned char* rgb) override;
  void RenderPixels(Array* target, bool reset);

 private:
  int Id(mjtObj type, const char* name) const;
  const double* Body(const char* name) const;
  const double* Site(const char* name) const;
  const double* Geom(const char* name) const;
  const double* Sensor(const char* name) const;
  double Upright() const;
  double SmallControl() const;
  double Stand(double upright_bound = 0.9, double upright_margin = 1.9,
               double height_margin_fraction = 0.25);
  double Info(const char* name, double value);
  bool Contact(int geom) const;
  bool UponTable(int package) const;
  Vec3 SpoonGoal() const;
  double MazeReward(double stable);
  double CabinetReward(double stable);
  double TruckReward();
  double ManipulationReward(double stable);
  void RenderFrame(int width, int height, int camera_id, unsigned char* rgb);
};

}  // namespace humanoid_bench
#endif
