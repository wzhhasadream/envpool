// Task logic translated from HumanoidBench 0.2.1 (MIT).
// Copyright (c) 2024 Robot Learning Lab (RLL) lab @ UC Berkeley.
// See third_party/humanoid_bench/LICENSE for the original license.

#include "envpool/mujoco/humanoid_bench/task.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <mutex>
#include <stdexcept>

#include "envpool/mujoco/dmc/utils.h"

namespace humanoid_bench {
struct Task::RenderContext {
  std::mutex mutex;
  envpool::mujoco::OffscreenRenderer renderer;
};

namespace {
constexpr double inf = std::numeric_limits<double>::infinity();
double T(double x, double low, double high, double margin) {
  return mujoco_dmc::RewardTolerance(x, low, high, margin);
}
double L(double x, double low, double high, double margin, double value = 0.1) {
  return mujoco_dmc::RewardTolerance(x, low, high, margin, value, mujoco_dmc::SigmoidType::kLinear);
}
double Dist(const double* a, const double* b, int n = 3) {
  double sum = 0;
  for (int i = 0; i < n; ++i) sum += (a[i] - b[i]) * (a[i] - b[i]);
  return std::sqrt(sum);
}
void EulerQuat(double* quat, double roll, double pitch, double yaw) {
  const double cr = std::cos(roll / 2), cp = std::cos(pitch / 2), cy = std::cos(yaw / 2);
  const double sr = std::sin(roll / 2), sp = std::sin(pitch / 2), sy = std::sin(yaw / 2);
  quat[0] = cr * cp * cy + sr * sp * sy;
  quat[1] = sr * cp * cy - cr * sp * sy;
  quat[2] = cr * sp * cy + sr * cp * sy;
  quat[3] = cr * cp * sy - sr * sp * cy;
}
}  // namespace

Task::Task(const std::string& xml, const std::string& task, int robot_dof,
           int frame_skip, int max_steps, int frame_stack, int width, int height,
           const std::string& camera_name, double stand_height, double randomness,
           const std::vector<std::string>& metric_names)
    : MujocoEnv(xml, frame_skip, true, max_steps, frame_stack, width, height),
      task_(task), render_key_(xml), robot_dof_(robot_dof), camera_id_(mj_name2id(model_, mjOBJ_CAMERA, camera_name.c_str())),
      stand_height_(stand_height), randomness_(randomness), control_(model_->nu) {
  if (metric_names.size() > kMetricCount) throw std::invalid_argument("Too many HumanoidBench metrics");
  for (std::size_t i = 0; i < metric_names.size(); ++i) metric_ids_[metric_names[i]] = i;
  for (int i = 0; i < model_->nbody; ++i) if (const char* name = mj_id2name(model_, mjOBJ_BODY, i)) bodies_[name] = i;
  for (int i = 0; i < model_->nsite; ++i) if (const char* name = mj_id2name(model_, mjOBJ_SITE, i)) sites_[name] = i;
  for (int i = 0; i < model_->ngeom; ++i) if (const char* name = mj_id2name(model_, mjOBJ_GEOM, i)) geoms_[name] = i;
  for (int i = 0; i < model_->nsensor; ++i) if (const char* name = mj_id2name(model_, mjOBJ_SENSOR, i)) sensors_[name] = model_->sensor_adr[i];
}

int Task::Id(mjtObj type, const char* name) const {
  const auto& names = type == mjOBJ_BODY ? bodies_ : type == mjOBJ_SITE ? sites_ : geoms_;
  return names.at(name);
}
const double* Task::Body(const char* name) const { return data_->xpos + 3 * bodies_.at(name); }
const double* Task::Site(const char* name) const { return data_->site_xpos + 3 * sites_.at(name); }
const double* Task::Geom(const char* name) const { return data_->geom_xpos + 3 * geoms_.at(name); }
const double* Task::Sensor(const char* name) const { return data_->sensordata + sensors_.at(name); }
double Task::Upright() const { return data_->xmat[9 * bodies_.at("torso_link") + 8]; }
double Task::Info(const char* name, double value) { metrics_[metric_ids_.at(name)] = value; return value; }
double Task::SmallControl() const {
  double control = 0;
  for (int i = 0; i < model_->nu; ++i)
    control += mujoco_dmc::RewardTolerance(data_->actuator_force[i], 0, 0, 10, 0, mujoco_dmc::SigmoidType::kQuadratic);
  return (4 + control / model_->nu) / 5;
}
double Task::Stand(double upright_bound, double upright_margin, double height_margin_fraction) {
  const double standing = Info("standing", T(Site("head")[2], stand_height_, inf, stand_height_ * height_margin_fraction));
  const double upright = Info("upright", L(Upright(), upright_bound, inf, upright_margin, 0));
  return Info("stand_reward", standing * upright);
}
bool Task::Contact(int geom) const {
  for (int i = 0; i < data_->ncon; ++i) if (data_->contact[i].geom1 == geom || data_->contact[i].geom2 == geom) return true;
  return false;
}

void Task::ResetTask(std::mt19937* gen) {
  auto uniform = [&](double low, double high) { return std::uniform_real_distribution<double>(low, high)(*gen); };
  has_cached_render_ = false;
  metrics_.fill(std::numeric_limits<double>::quiet_NaN());
  mj_resetDataKeyframe(model_, data_, 0);
  mj_forward(model_, data_);
  for (int i = 0; i < model_->nq; ++i) data_->qpos[i] += uniform(-randomness_, randomness_);
  mj_forward(model_, data_);
  if (task_ == "reach") goal_ = {uniform(-2, 2), uniform(-2, 2), uniform(0.2, 2)};
  if (task_ == "push") goal_ = {uniform(0.7, 1.0), uniform(-0.5, 0.5), 1.0};
  if (task_ == "sit_hard") {
    data_->qpos[0] = uniform(0.2, 0.4);
    data_->qpos[1] = uniform(-0.15, 0.15);
    EulerQuat(data_->qpos + 3, 0, 0, uniform(-1.8, 1.8));
    mj_forward(model_, data_);
  }
  if (task_ == "maze" || task_ == "kitchen" || task_ == "basketball") stage_ = 0;
  if (task_ == "cabinet") stage_ = 1;
  if (task_ == "window") std::copy_n(Site("head"), 3, head_pos0_.begin());
  // Upstream sets highbar reset noise to zero after the first reset.
  if (task_.find("highbar_") == 0) randomness_ = 0;
  if (task_ == "cube") {
    for (double* quat : {data_->qpos + model_->nq - 11, data_->qpos + model_->nq - 4, model_->body_quat + (model_->nbody - 1) * 4}) {
      const double r = uniform(-3.14, 3.14), p = uniform(-3.14, 3.14), y = uniform(-3.14, 3.14);
      EulerQuat(quat, r, p, y);
    }
    mj_forward(model_, data_);
  }
  if (task_ == "basketball") {
    const double angle = uniform(0, 1.45) * (uniform(0, 1) < 0.5 ? 1 : -1);
    data_->qvel[model_->nv - 6] = -7.5 * std::cos(angle);
    data_->qvel[model_->nv - 5] = -7.5 * std::sin(angle);
    data_->qpos[model_->nq - 7] = 1.5 * std::cos(angle);
    data_->qpos[model_->nq - 6] = 1.5 * std::sin(angle);
    data_->qpos[model_->nq - 5] = 0.3924 + Site("left_hand")[2];
    mj_forward(model_, data_);
  }
  if (task_ == "package") {
    data_->qpos[model_->nq - 7] = uniform(-2, 2);
    data_->qpos[model_->nq - 6] = uniform(-2, 2);
    data_->qpos[model_->nq - 5] = 0.35;
    mj_forward(model_, data_);
    double* destination = model_->body_pos + (model_->nbody - 1) * 3;
    destination[0] = uniform(-2, 2); destination[1] = uniform(-2, 2); destination[2] = 0;
    // Upstream changes body_pos after forward; preserve its reset observation.
  }
  if (task_ == "room") {
    for (int i = -7; i < 0; ++i) {
      data_->qpos[model_->nq + i * 7] = uniform(-3.5 + i + 7, -3.5 + i + 8);
      data_->qpos[model_->nq + i * 7 + 1] = uniform(1.2, 3.5) * (uniform(0, 1) < 0.5 ? 1 : -1);
      if (i == -4) {
        const double quat[] = {0.0733422, 0.0519076, -0.240058, -0.966591};
        std::copy_n(quat, 4, data_->qpos + model_->nq + i * 7 + 3);
      }
    }
    mj_forward(model_, data_);
  }
  if (task_.find("bookshelf_") == 0) {
    stage_ = 0;
    placement_goals_ = {{{0.75, -0.25, 1.55}, {0.8, 0.05, 0.95}, {0.8, -0.25, 0.95}, {0.85, 0.05, 0.35}, {0.85, -0.25, 0.35}, {0.85, -0.25, 0.35}}};
    if (task_ == "bookshelf_hard") {
      std::shuffle(placement_goals_.begin(), placement_goals_.begin() + 5, *gen);
      placement_goals_[5] = placement_goals_[4];
      std::array<int, 12> objects;
      std::iota(objects.begin(), objects.end(), -24);
      std::shuffle(objects.begin(), objects.end(), *gen);
      std::copy_n(objects.begin(), 5, bookshelf_objects_.begin());
      bookshelf_objects_[5] = bookshelf_objects_[4];
    }
    const std::vector<std::string> names = {"book_a", "book_b", "book_c", "book_d", "book_e", "trophy", "book_flat", "soda_can", "book_thick", "chalk", "snow_globe", "headphone"};
    const int counts[] = {1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 2, 5};
    const double colors[5][3] = {{1, .8, .8}, {1, .5, .5}, {.9, .2, .2}, {.5, 0, 0}, {.2, 0, 0}};
    for (int object = 0; object < 12; ++object) for (int part = 0; part < counts[object]; ++part) {
      const std::string suffix = std::string("_") + static_cast<char>('a' + part);
      const int vision = geoms_.at(names[object] + "_vision" + suffix), actual = geoms_.at(names[object] + suffix);
      for (int axis = 0; axis < 3; ++axis) model_->geom_rgba[vision * 4 + axis] = model_->geom_rgba[actual * 4 + axis] = .8f;
      model_->geom_rgba[vision * 4 + 3] = 0; model_->geom_rgba[actual * 4 + 3] = 1;
    }
    for (int i = 0; i < 5; ++i) {
      const int object = bookshelf_objects_[i] + 24;
      std::copy_n(placement_goals_[i].data(), 3, model_->body_pos + (model_->nbody + bookshelf_objects_[i] + 12) * 3);
      for (int part = 0; part < counts[object]; ++part) {
        const std::string suffix = std::string("_") + static_cast<char>('a' + part);
        const int vision = geoms_.at(names[object] + "_vision" + suffix), actual = geoms_.at(names[object] + suffix);
        for (int axis = 0; axis < 3; ++axis) model_->geom_rgba[vision * 4 + axis] = model_->geom_rgba[actual * 4 + axis] = colors[i][axis];
        model_->geom_rgba[vision * 4 + 3] = .2f; model_->geom_rgba[actual * 4 + 3] = 1;
      }
    }
  }
}

Vec3 Task::SpoonGoal() const {
  const double angle = spoon_steps_ * (2 * M_PI / 40);
  return {0.75 + std::cos(angle) * .06, -.1 + std::sin(angle) * .06, .95};
}
void Task::Observe() {
  obs_.clear();
  auto append = [&](const double* x, int n) { obs_.insert(obs_.end(), x, x + n); };
  if (task_ == "reach" || task_ == "push" || task_ == "package") {
    append(data_->qpos, robot_dof_); append(data_->qvel, robot_dof_ - 1);
    if (task_ == "package") {
      append(Site("destination_loc"), 3); append(data_->qpos + model_->nq - 7, 7); append(data_->qvel + model_->nv - 6, 6);
      append(Site("left_hand"), 3); append(Site("right_hand"), 3);
    } else {
      append(Site("left_hand"), 3); append(goal_.data(), 3);
      if (task_ == "push") { append(data_->qpos + model_->nq - 7, 3); append(data_->qvel + model_->body_dofadr[bodies_.at("object")], 3); }
    }
  } else if (task_ == "kitchen") {
    append(data_->qpos, robot_dof_); append(data_->qpos + model_->nq - 21, 21);
  } else {
    append(data_->qpos, model_->nq); append(data_->qvel, model_->nv);
    if (task_ == "cube") append(data_->xquat + bodies_.at("target_cube") * 4, 4);
    if (task_.find("bookshelf_") == 0) obs_.push_back(bookshelf_objects_[stage_]);
    if (task_ == "spoon") { const Vec3 goal = SpoonGoal(); append(goal.data(), 3); }
  }
}
void Task::Simulate(const float* action) {
  for (int i = 0; i < model_->nu; ++i) {
    // Gymnasium stores the original actuator ranges as float32.
    const float low = static_cast<float>(model_->actuator_ctrlrange[i * 2]);
    const float high = static_cast<float>(model_->actuator_ctrlrange[i * 2 + 1]);
    control_[i] = (action[i] + 1.0f) / 2.0f * (high - low) + low;
  }
  MujocoStep(control_.data());
}

double Task::Reward() {
  if (task_ == "reach") {
    const double distance = Info("hand_dist", Dist(Site("left_hand"), goal_.data()));
    const double healthy = Info("healthy_reward", data_->xmat[17] * 5);
    double motion = 0;
    for (int i = 0; i < robot_dof_ - 1; ++i) motion += data_->qvel[i] * data_->qvel[i];
    Info("motion_penalty", motion);
    return healthy - .0001 * motion + Info("reward_close", distance < 1 ? 5 : 0) + Info("reward_success", distance < .05 ? 10 : 0);
  }
  if (task_ == "push") {
    const double distance = Info("target_dist", Dist(data_->qpos + model_->nq - 7, goal_.data()));
    const double hand = Info("hand_dist", Dist(Site("left_hand"), data_->qpos + model_->nq - 7));
    Info("success", distance < .05);
    return -.1 * hand - distance + Info("reward_success", distance < .05 ? 1000 : 0);
  }
  if (task_ == "kitchen") {
    const int indices[4][7] = {{13}, {14, 15, 16, 17, 18, 19, 20}, {2, 3}, {8, 9}};
    const double goals[4][7] = {{-.75}, {-.23, .75, 1, .99, 0, 0, -.06}, {-.88, -.01}, {-.69, -.05}};
    const int sizes[] = {1, 7, 2, 2};
    const int previous = stage_;
    while (stage_ < 4) {
      double distance = 0;
      for (int i = 0; i < sizes[stage_]; ++i) {
        const double delta = data_->qpos[model_->nq - 21 + indices[stage_][i]] - goals[stage_][i];
        distance += delta * delta;
      }
      if (std::sqrt(distance) >= .3) break;
      ++stage_;
    }
    Info("success_subtasks", stage_); Info("success", stage_ == 4);
    return stage_ - previous;
  }
  if (task_ == "truck") return TruckReward();
  const double small = Info("small_control", SmallControl());
  if (task_.find("highbar_") == 0) {
    const double upright = Info("upright_reward", L(-Upright(), .9, inf, 1.9, 0));
    const double feet = Info("feet_reward", (1 + L((Site("left_foot")[2] + Site("right_foot")[2]) / 2, 4.8, inf, 2, 0)) / 2);
    return upright * feet * small;
  }
  if (task_ == "crawl") {
    const double height = stand_height_ < 1.5 ? .6 : .8;
    const double move = Info("move", (5 * L(Sensor("pelvis_subtreelinvel")[0], 1, inf, 1, 0) + 1) / 6);
    const double head = Info("crawling_head", T(Site("head")[2], height - .2, height + .2, 1));
    const double crawling = Info("crawling", T(Site("imu")[2], height - .2, height + .2, 1));
    const double target[] = {.75, 0, .65, 0};
    const double rotation = T(Dist(data_->xquat + bodies_.at("pelvis") * 4, target, 4), 0, 0, 1);
    const double tunnel = Info("in_tunnel", T(Site("imu")[1], -1, 1, 0));
    return (.1 * small + .25 * std::min(crawling, head) + .4 * move + .25 * rotation) * tunnel;
  }
  if (task_.find("sit_") == 0) {
    const double sitting = Info("sitting", T(data_->qpos[2], .68, .72, .2));
    const double on_chair = T(data_->qpos[0] - Body("chair")[0], -.19, .19, .2) * T(data_->qpos[1] - Body("chair")[1], 0, 0, .1);
    const double posture = Info("sitting_posture", T(Site("head")[2] - Site("imu")[2], .35, .45, .3));
    const double upright = Info("upright", L(Upright(), .95, inf, .9, 0));
    const double sit = Info("sit_reward", (.5 * sitting + .5 * on_chair) * upright * posture);
    const double* velocity = Sensor("pelvis_subtreelinvel");
    return small * sit * Info("dont_move", (T(velocity[0], 0, 0, 2) + T(velocity[1], 0, 0, 2)) / 2);
  }
  double stand;
  if (task_.find("balance_") == 0) {
    const double standing = Info("standing", T(Site("head")[2], stand_height_ + .37, inf, stand_height_ / 4));
    const double upright = Info("upright", L(Upright(), .9, inf, 1.9, 0));
    stand = Info("stand_reward", standing * upright);
  } else if (task_ == "stair" || task_ == "slide") {
    const double standing = Info("standing", T(Site("head")[2] - Site("left_foot")[2], 1.2, inf, .45) * T(Site("head")[2] - Site("right_foot")[2], 1.2, inf, .45));
    stand = Info("stand_reward", standing * Info("upright", L(Upright(), .5, inf, 1.9, 0)));
  } else {
    stand = Stand((task_ == "hurdle" || task_ == "package" || task_ == "powerlift") ? .8 : .9,
                  task_ == "door" ? .9 : 1.9, task_ == "powerlift" ? 1.0 / 3 : .25);
  }
  const double stable = stand * small;
  if (task_ == "stand" || task_.find("balance_") == 0 || task_ == "cube") {
    const double* velocity = Sensor("pelvis_subtreelinvel");
    const double still = Info("dont_move", (T(velocity[0], 0, 0, 2) + T(velocity[1], 0, 0, 2)) / 2);
    if (task_ != "cube") return stable * still;
    const double* target = data_->xquat + bodies_.at("target_cube") * 4;
    const double alignment = Info("orientation_alignment_reward", (T(Dist(data_->xquat + bodies_.at("left_cube_to_rotate") * 4, target, 4), 0, 0, .3) + T(Dist(data_->xquat + bodies_.at("right_cube_to_rotate") * 4, target, 4), 0, 0, .3)) / 2);
    const double close = Info("cube_closeness_reward", (T(Dist(Site("left_hand"), Body("left_cube_to_rotate")), 0, .1, .5) + T(Dist(Site("right_hand"), Body("right_cube_to_rotate")), 0, .1, .5)) / 2);
    return .2 * stable * still + .5 * alignment + .3 * close;
  }
  if (task_ == "walk" || task_ == "run" || task_ == "hurdle" || task_ == "pole" || task_ == "stair" || task_ == "slide") {
    const double speed = task_ == "run" || task_ == "hurdle" ? 5 : task_ == "pole" ? .5 : 1;
    const double move = Info("move", (5 * L(Sensor("pelvis_subtreelinvel")[0], speed, inf, speed, 0) + 1) / 6);
    double discount = 1;
    if (task_ == "hurdle") {
      if (Contact(geoms_.at("left_barrier_collision")) || Contact(geoms_.at("right_barrier_collision")) || Contact(geoms_.at("behind_barrier_collision"))) discount = .1;
      Info("wall_collision_discount", discount);
    }
    if (task_ == "pole") {
      for (int i = 0; i < data_->ncon; ++i) {
        const int a = data_->contact[i].geom1, b = data_->contact[i].geom2;
        const char *an = mj_id2name(model_, mjOBJ_GEOM, a), *bn = mj_id2name(model_, mjOBJ_GEOM, b);
        if (a != 0 && b != 0 && ((an && std::strstr(an, "pole_r")) || (bn && std::strstr(bn, "pole_r")))) discount = .1;
      }
      Info("collision_discount", discount);
      return (.5 * stable + .5 * move) * discount;
    }
    return stable * move * discount;
  }
  if (task_ == "maze") return MazeReward(stable);
  if (task_ == "cabinet") return CabinetReward(stable);
  return ManipulationReward(stable);
}

double Task::ManipulationReward(double stable) {
  if (task_ == "door") {
    const double angle = data_->qpos[model_->nq - 2];
    const double door = Info("door_openness_reward", std::min(1.0, angle * std::abs(angle)));
    const double hatch = Info("door_hatch_openness_reward", L(data_->qpos[model_->nq - 1], .75, 2, .75));
    const double hand = Info("hand_hatch_proximity_reward", L(std::min(Dist(Body("door_hatch"), Site("left_hand")), Dist(Body("door_hatch"), Site("right_hand"))), 0, .25, 1));
    const double passage = Info("passage_reward", L(Site("imu")[0], 1.2, inf, 1, 0));
    return .1 * stable + .45 * door + .05 * hatch + .05 * hand + .35 * passage;
  }
  if (task_ == "powerlift") return .2 * stable + .8 * Info("reward_dumbbell_lifted", T(Body("dumbbell")[2], 1.9, 2.1, 2));
  if (task_ == "package") {
    const double* box = data_->qpos + model_->nq - 7;
    const double distance = Info("dist_package_destination", Dist(box, Site("destination_loc")));
    const double left = Info("dist_hand_package_left", Dist(Site("left_hand"), box));
    const double right = Info("dist_hand_package_right", Dist(Site("right_hand"), box));
    const double height = Info("package_height", std::min(box[2], 1.0));
    Info("success", distance < .1);
    return stable - 3 * distance - (left + right) * .1 + height + (distance < .1 ? 1000 : 0);
  }
  if (task_ == "room") {
    const char* names[] = {"chair", "trophy", "headphone", "package_a", "package_b", "snow_globe"};
    double variance[2] = {};
    for (int axis = 0; axis < 2; ++axis) {
      double mean = 0;
      for (const char* name : names) mean += Body(name)[axis] / 6;
      for (const char* name : names) variance[axis] += std::pow(Body(name)[axis] - mean, 2) / 6;
    }
    return .2 * stable + .8 * Info("room_object_organized", T(std::max(variance[0], variance[1]), 0, 0, 3));
  }
  if (task_ == "insert_small" || task_ == "insert_normal") {
    const double target = Info("cube_target_reward", (L(Dist(Site("block_peg_a"), Site("peg_a")), 0, 0, .5) + L(Dist(Site("block_peg_b"), Site("peg_b")), 0, 0, .5)) / 2);
    const double height = Info("peg_height_reward", (L(Site("peg_a")[2] - 1.1, 0, 0, .15) + L(Site("peg_b")[2] - 1.1, 0, 0, .15)) / 2);
    const double hand = Info("hand_tool_proximity_reward", T(std::min(Dist(Site("left_hand"), Site("peg_a")), Dist(Site("right_hand"), Site("peg_b"))), 0, .2, .5));
    return (.5 * stable + .5 * target) * (.5 * height + .5 * hand);
  }
  if (task_.find("bookshelf_") == 0) {
    const double* object = data_->xpos + (model_->nbody + bookshelf_objects_[stage_]) * 3;
    const double distance = Info("obj_goal_dist", Dist(object, placement_goals_[stage_].data()));
    const double proximity = Info("reward_proximity", L(distance, 0, .15, 1));
    const double hand = Info("reward_hand_proximity", std::exp(-std::min(Dist(Site("left_hand"), object), Dist(Site("right_hand"), object))));
    double reward = .2 * stable + .4 * proximity + .4 * hand;
    if (distance < .15) reward += 100 * ++stage_;
    Info("success_subtasks", stage_); Info("success", stage_ == 5);
    return reward;
  }
  if (task_ == "basketball") {
    const double* ball = Body("basketball");
    const double hand = Info("reward_hand_proximity", T(std::max(Dist(Site("left_hand"), ball), Dist(Site("right_hand"), ball)), 0, .2, 1));
    const double distance = Dist(ball, Site("hoop_center"));
    const double hoop = Info("reward_ball_success", L(distance, 0, 0, 7));
    if (stage_ == 0 && Contact(geoms_.at("basketball_collision"))) stage_ = 1;
    Info("success_subtasks", stage_); Info("success", distance < .05);
    return (stage_ ? .15 * stable + .05 * hand + .8 * hoop : .5 * stable + .5 * hand) + (distance < .05 ? 1000 : 0);
  }
  if (task_ == "window") {
    double contact = 1;
    for (const char* name : {"wipe_contact_site_a", "wipe_contact_site_b", "wipe_contact_site_c", "wipe_contact_site_d", "wipe_contact_site_e"}) contact = std::min(contact, L(Site(name)[0], .92, .92, .4));
    Info("window_contact_reward", contact);
    double filter = 0;
    const int pane = geoms_.at("window_pane_collision"), wipe = geoms_.at("window_wipe_collision");
    for (int i = 0; i < data_->ncon; ++i) {
      const auto& c = data_->contact[i];
      if ((c.geom1 == pane && c.geom2 == wipe) || (c.geom2 == pane && c.geom1 == wipe)) filter = 1;
    }
    Info("window_contact_filter", filter);
    const double hand = Info("hand_tool_proximity_reward", std::min(T(Dist(Site("left_hand"), Body("window_wiping_tool")), 0, .2, .5), T(Dist(Site("right_hand"), Body("window_wiping_tool")), 0, .2, .5)));
    const double moving = Info("moving_wipe_reward", T(std::abs(Sensor("window_wiping_tool_subtreelinvel")[2]), .5, .5, .5));
    const double head = T(Dist(Site("head"), head_pos0_.data()), .4, .4, .1);
    const double manipulation = .2 * stable * head + .4 * moving + .4 * hand;
    return .5 * manipulation + .5 * Info("window_contact_total_reward", filter * contact);
  }
  if (task_ == "spoon") {
    const double hand = Info("hand_tool_proximity_reward", T(std::min(Dist(Site("left_hand"), Geom("spoon_handle")), Dist(Site("right_hand"), Geom("spoon_handle"))), 0, .2, .5));
    const Vec3 target = SpoonGoal();
    std::copy_n(target.data(), 3, data_->site_xpos + sites_.at("goal") * 3);
    const double spin = Info("spoon_spinning_reward", T(Dist(Geom("spoon_plate"), target.data()), 0, 0, .15));
    const double *plate = Geom("spoon_plate"), *cup = Body("cup");
    const double in_cup = Info("reward_spoon_in_cup", std::abs(plate[0] - cup[0]) < .1 && std::abs(plate[1] - cup[1]) < .1 && std::abs(plate[2] - cup[2] - .1) < .1);
    ++spoon_steps_;
    return .15 * stable + .25 * hand + .25 * in_cup + .35 * spin;
  }
  throw std::logic_error("Unimplemented HumanoidBench task: " + task_);
}

double Task::MazeReward(double stable) {
  const double checkpoints[5][3] = {{0, 0, 1}, {3, 0, 1}, {3, 6, 1}, {6, 6, 1}, {6, 6, 1}};
  const double directions[5][2] = {{1, 0}, {1, 0}, {0, 1}, {1, 0}, {0, 1}};
  const char* intersections[] = {"intersection_a", "intersection_b", "intersection_c", "intersection_d"};
  auto color = [&](int index, bool green) {
    float* rgba = model_->geom_rgba + geoms_.at(intersections[index]) * 4;
    rgba[0] = green ? 0 : 1; rgba[1] = green ? 1 : 0; rgba[2] = 0; rgba[3] = 1;
  };
  if (stage_ == 0) color(0, true);
  const int previous = stage_;
  if (Dist(Body("pelvis"), checkpoints[stage_], 2) < .4) {
    if (stage_ < 4) color(stage_, true);
    if (stage_ < 3) color(stage_ + 1, false);
    stage_ = std::min(stage_ + 1, 4);
  }
  double discount = 1;
  const int wall = geoms_.at("block_collision_00");
  for (int i = 0; i < data_->ncon; ++i) if (data_->contact[i].geom1 >= wall || data_->contact[i].geom2 >= wall) discount = .1;
  Info("wall_collision_discount", discount);
  const double* velocity = Sensor("pelvis_subtreelinvel");
  const double move = Info("move", (5 * (stage_ == 4 ? 1 : L(velocity[0] - 2 * directions[stage_][0], 0, 0, 1, 0) * L(velocity[1] - 2 * directions[stage_][1], 0, 0, 1, 0)) + 1) / 6);
  const double proximity = Info("checkpoint_proximity_reward", T(Dist(checkpoints[stage_], Site("imu"), 2), 0, 0, 1));
  const double bonus = Info("stage_convert_reward", previous < stage_ ? stage_ * 100 : 0);
  Info("success_subtasks", stage_);
  return (.2 * stable + .4 * move + .4 * proximity) * discount + bonus;
}

double Task::CabinetReward(double stable) {
  double reward = 1000;
  bool complete = false;
  if (stage_ == 1 || stage_ == 2) {
    reward = Info("door_openness_reward", std::abs(data_->qpos[model_->nq - (stage_ == 1 ? 30 : 33)] / (stage_ == 1 ? .4 : .45)));
    complete = reward > .95;
  } else if (stage_ == 3 || stage_ == 4) {
    const double* cube = Body(stage_ == 3 ? "drawer_cube" : "lateral_cabinet_cube");
    const double doors = std::max(std::min(1.0, std::abs(data_->qpos[model_->nq - 32])), std::min(1.0, std::abs(data_->qpos[model_->nq - 31])));
    const double door = Info("door_openness_reward", stage_ == 3 ? doors : std::min(1.0, std::abs(data_->qpos[model_->nq - 29])));
    if (stage_ == 4) Info("secondary_door_openness_reward", doors);
    const double height = stage_ == 3 ? .94 : 1.54;
    const double horizontal = (L(cube[0] - .9, -.3, .3, .3) + L(cube[1], -.6, .6, .3)) / 2;
    const double vertical = L(cube[2] - height, -.15, .15, .3);
    const double proximity = Info("drawer_cube_proximity_reward", .3 * horizontal + .7 * vertical);
    reward = .5 * proximity + .5 * door;
    complete = cube[0] >= .9 - .3 && cube[0] <= .9 + .3 && cube[1] >= -.6 && cube[1] <= .6 && cube[2] >= height - .15 && cube[2] <= height + .15;
  }
  if (stage_ < 5) { Info("subtask_complete", complete); reward = .2 * stable + .8 * reward; }
  if (complete) reward += 100 * stage_++;
  Info("success_subtasks", stage_ - 1); Info("success", stage_ == 5);
  return reward;
}

bool Task::UponTable(int package) const {
  const std::string name = std::string("package_") + static_cast<char>('a' + package);
  const double* position = Body(name.c_str());
  return position[0] < 2.35 && position[1] < -1.35 && position[0] > 1.65 && position[1] > -2.05 && position[2] > .5;
}
double Task::TruckReward() {
  auto package = [&](int i) { const std::string name = std::string("package_") + static_cast<char>('a' + i); return Body(name.c_str()); };
  if (!truck_initialized_) {
    truck_initialized_ = true;
    truck_ = {0, 1, 2, 3, 4};
    for (int i = 0; i < 5; ++i) initial_zs_[i] = package(i)[2];
  }
  double reward = 0;
  // Match upstream's list mutation semantics: an erased item skips its successor.
  for (std::size_t i = 0; i < truck_.size(); ++i) if (package(truck_[i])[2] > initial_zs_[truck_[i]] + .1) {
    picked_.push_back(truck_[i]); truck_.erase(truck_.begin() + i); reward += 100;
  }
  for (std::size_t i = 0; i < picked_.size(); ++i) if (UponTable(picked_[i])) {
    table_.push_back(picked_[i]); picked_.erase(picked_.begin() + i); reward += 100;
  }
  for (std::size_t i = 0; i < table_.size(); ++i) if (!UponTable(table_[i])) {
    picked_.push_back(table_[i]); table_.erase(table_.begin() + i); reward -= 100;
  }
  const double upright = Info("upright", L(Upright(), .9, inf, 1.9, 0));
  double truck_distance = inf, picked_distance = inf, table_distance = inf;
  for (int i : truck_) truck_distance = std::min(truck_distance, Dist(package(i), data_->qpos));
  for (int i : picked_) { picked_distance = std::min(picked_distance, Dist(package(i), data_->qpos)); table_distance = std::min(table_distance, Dist(package(i), Body("table"))); }
  const double rt = Info("reward_robot_package_truck", truck_.empty() ? 0 : L(truck_distance, 0, .2, 4, 0));
  const double rp = Info("reward_robot_package_picked_up", picked_.empty() ? 0 : L(picked_distance, 0, .2, 4, 0));
  const double pt = Info("reward_package_table", picked_.empty() ? 0 : L(table_distance, 0, .2, 4, 0));
  Info("packages_on_truck", truck_.size()); Info("packages_picked_up", picked_.size()); Info("packages_on_table", table_.size());
  Info("success_subtasks", table_.size()); Info("success", table_.size() == 5);
  return reward + upright * (1 + rt + rp + pt) + (table_.size() == 5 ? 1000 : 0);
}

bool Task::Terminated() {
  const double height = data_->qpos[2];
  if (task_ == "reach" || task_ == "crawl") return false;
  if (task_ == "push") return Dist(data_->qpos + model_->nq - 7, goal_.data()) < .05;
  if (task_ == "package") return Dist(data_->qpos + model_->nq - 7, Site("destination_loc")) < .1;
  if (task_ == "cabinet") return stage_ == 5;
  if (task_ == "kitchen") return stage_ == 4;
  if (task_ == "truck") return table_.size() == 5;
  if (task_.find("highbar_") == 0) return Site("head")[2] < 2;
  if (task_ == "stair" || task_ == "slide") return Upright() < .1;
  if (task_ == "pole" || task_.find("sit_") == 0) return height < .5;
  if (task_ == "room") return height < .3;
  if (task_ == "door" || task_ == "spoon") return height < .58;
  if (task_ == "window") return height < .58 || Body("window_wiping_tool")[2] < .58;
  if (task_ == "basketball") return data_->qpos[model_->nq - 5] < .5 || height < .5 || Dist(Body("basketball"), Site("hoop_center")) < .05;
  if (task_ == "cube" || task_.find("bookshelf_") == 0 || task_.find("insert_") == 0) {
    const bool insert = task_.find("insert_") == 0;
    if (height < (insert ? .5 : .58)) { Info("terminated_reason", 0); return true; }
    if (task_ == "cube" && (Body("left_cube_to_rotate")[2] < .6 || Body("right_cube_to_rotate")[2] < .6)) { Info("terminated_reason", 1); return true; }
    if (insert && (Site("block_peg_a")[2] < .5 || Site("block_peg_b")[2] < .5 || Site("peg_a")[2] < .5 || Site("peg_b")[2] < .5)) { Info("terminated_reason", 1); return true; }
    if (task_.find("bookshelf_") == 0) {
      if (stage_ == 5) { Info("terminated_reason", 1); return true; }
      if (data_->xpos[(model_->nbody + bookshelf_objects_[stage_]) * 3 + 2] < .5) { Info("terminated_reason", 2); return true; }
    }
    return false;
  }
  if (task_.find("balance_") == 0) {
    if (height < .8) return true;
    const int sphere = geoms_.at("pivot_sphere_collision"), board = geoms_.at("stand_board_collision");
    for (int i = 0; i < data_->ncon; ++i) {
      const int a = data_->contact[i].geom1, b = data_->contact[i].geom2;
      if ((a == sphere || b == sphere) && a != 0 && b != 0 && a != board && b != board) return true;
      if ((a == 0 || b == 0) && a != sphere && b != sphere) return true;
    }
    return false;
  }
  return height < .2;
}

void Task::ExportTaskState() {
  task_state_.fill(0);
  std::copy_n(goal_.data(), 3, task_state_.begin());
  std::copy_n(head_pos0_.data(), 3, task_state_.begin() + 3);
  task_state_[6] = stage_; task_state_[7] = spoon_steps_;
  for (int i = 0; i < 6; ++i) {
    task_state_[8 + i] = bookshelf_objects_[i];
    std::copy_n(placement_goals_[i].data(), 3, task_state_.begin() + 14 + i * 3);
  }
}

void Task::RenderFrame(int width, int height, int camera_id, unsigned char* rgb) {
  if (!render_context_) {
    static std::mutex cache_mutex;
    static std::unordered_map<std::string, std::weak_ptr<RenderContext>> cache;
    std::lock_guard<std::mutex> lock(cache_mutex);
    render_context_ = cache[render_key_].lock();
    if (!render_context_) {
      render_context_ = std::make_shared<RenderContext>();
      cache[render_key_] = render_context_;
    }
  }
  // Share immutable mesh/texture uploads and the upstream 16384px shadow map.
  // Physics remains parallel; GL calls use one context at a time per model.
  std::lock_guard<std::mutex> lock(render_context_->mutex);
  mjvGeom marker{};
  const bool target = task_ == "reach" || task_ == "push";
  if (target) {
    const double size[] = {.05, .05, .05}, mat[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const float rgba[] = {.8f, .28f, .28f, 1};
    mjv_initGeom(&marker, mjGEOM_SPHERE, size, goal_.data(), mat, rgba);
  }
  render_context_->renderer.Render(model_, data_, width, height, camera_id < 0 ? camera_id_ : camera_id, rgb,
                    nullptr, nullptr, target ? &marker : nullptr, target ? 1 : 0);
}
void Task::Render(int width, int height, int camera_id, unsigned char* rgb) {
  if (!CopyCachedRender(width, height, camera_id, rgb)) RenderFrame(width, height, camera_id, rgb);
}
void Task::RenderPixels(Array* target, bool reset) {
  unsigned char* rgb = pixel_frame_stack_buffer_.Prepare("obs:pixels", render_width_, render_height_);
  RenderFrame(render_width_, render_height_, render_camera_id_, rgb);
  UpdateCachedRender(render_width_, render_height_, render_camera_id_, rgb);
  pixel_frame_stack_buffer_.Commit("obs:pixels", target, render_width_, render_height_, reset);
}

}  // namespace humanoid_bench
