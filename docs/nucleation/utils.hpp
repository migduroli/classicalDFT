#pragma once

#include "dft/algorithms/dynamics.hpp"
#include "dft/algorithms/minimization.hpp"
#include "dft/algorithms/saddle_point.hpp"
#include "dft/config/parser.hpp"
#include "dft/fields.hpp"
#include "dft/functionals/bulk/coexistence.hpp"
#include "dft/functionals/functional.hpp"
#include "dft/functionals/functionals.hpp"
#include "dft/grid.hpp"
#include "dft/init.hpp"
#include "dft/math/spline.hpp"
#include "dft/physics/potentials.hpp"
#include "dft/physics/walls.hpp"
#include "dft/types.hpp"

#include <algorithm>
#include <armadillo>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <format>
#include <functional>
#include <iostream>
#include <memory>
#include <numbers>
#include <optional>
#include <print>
#include <stdexcept>
#include <string>
#include <vector>

namespace nucleation {

  // Text utilities for TOML token parsing.

  [[nodiscard]] inline auto normalize_token(std::string_view token) -> std::string {
    std::string out;
    out.reserve(token.size());
    for (unsigned char ch : token) {
      if (std::isalnum(ch)) {
        out.push_back(static_cast<char>(std::toupper(ch)));
      }
    }
    return out;
  }

  [[nodiscard]] inline auto slugify_token(std::string_view token) -> std::string {
    std::string out;
    out.reserve(token.size() + 8);

    auto append_separator = [&]() {
      if (!out.empty() && out.back() != '_')
        out.push_back('_');
    };

    for (std::size_t i = 0; i < token.size(); ++i) {
      unsigned char ch = static_cast<unsigned char>(token[i]);
      if (!std::isalnum(ch)) {
        append_separator();
        continue;
      }

      bool upper = std::isupper(ch);
      if (upper && i > 0) {
        unsigned char prev = static_cast<unsigned char>(token[i - 1]);
        bool prev_lower_or_digit = std::islower(prev) || std::isdigit(prev);
        if (prev_lower_or_digit)
          append_separator();
      }

      out.push_back(static_cast<char>(std::tolower(ch)));
    }

    while (!out.empty() && out.back() == '_') {
      out.pop_back();
    }

    return out;
  }

  // Abbreviate well-known model names to short uppercase tags.

  [[nodiscard]] inline auto abbreviate(std::string_view name) -> std::string {
    static const std::array<std::pair<std::string_view, std::string_view>, 8> table{{
        {"LENNARDJONES", "LJ"},
        {"TENWOLDEFRENKEL", "TWF"},
        {"WANGRAMIREZDOBNIKARFRENKEL", "WRDF"},
        {"WEEKSCHANDLERANDERSEN", "WCA"},
        {"BARKERHENDERSON", "BH"},
        {"WHITEBEAR", "WB"},
        {"WHITEBEARII", "WBII"},
        {"LJ93", "LJ93"},
    }};
    auto key = normalize_token(name);
    for (const auto& [pattern, abbr] : table) {
      if (key == pattern)
        return std::string{abbr};
    }
    return key;
  }

  [[nodiscard]] inline auto wall_axis(const std::string& name) -> int {
    auto token = normalize_token(name);
    if (token == "X" || token == "0" || token == "100")
      return 0;
    if (token == "Y" || token == "1" || token == "010")
      return 1;
    if (token == "Z" || token == "2" || token == "001")
      return 2;
    throw std::runtime_error("Unknown wall normal: " + name);
  }

  // Configuration structs mirroring the TOML layout.

  struct GridConfig {
    double dx;
    std::array<double, 3> box_size;
    double predictor_dx{0.0}; // Coarser grid for the periodic predictor (0 = same as dx).
  };

  struct PotentialConfig {
    std::string name;
    double sigma;
    double epsilon;
    double cutoff;
    double alpha{50.0};

    [[nodiscard]] auto build() const -> dft::physics::potentials::Potential {
      if (name == "LennardJones")
        return dft::physics::potentials::make_lennard_jones(sigma, epsilon, cutoff);
      if (name == "TenWoldeFrenkel")
        return dft::physics::potentials::make_ten_wolde_frenkel(sigma, epsilon, cutoff, alpha);
      if (name == "WangRamirezDobnikarFrenkel")
        return dft::physics::potentials::make_wang_ramirez_dobnikar_frenkel(sigma, epsilon, cutoff);
      throw std::runtime_error("Unknown potential: " + name);
    }
  };

  struct SplitConfig {
    std::string interaction{"BarkerHenderson"};
    std::string hard_sphere_diameter; // defaults to interaction if absent
  };

  struct FunctionalConfig {
    std::string name;
    double temperature;
  };

  struct ModelConfig {
    GridConfig grid;
    PotentialConfig potential;
    SplitConfig split;
    FunctionalConfig functional;
  };

  struct DropletConfig {
    double radius;
    double supersaturation{1.1};
  };

  struct SeedConfig {
    std::string kind{"Droplet"};
    std::string structure{"FCC"};
    std::string orientation{"001"};
    std::array<long, 3> unit_cells{8, 8, 8};
    double gaussian_width{0.25};
    double modulation_amplitude{0.35};
    std::array<double, 3> envelope_radii{0.0, 0.0, 0.0};
    std::array<double, 3> offset{0.35, -0.20, 0.15};
    double interface_width{0.25};

    [[nodiscard]] auto uses_crystal() const -> bool {
      auto k = normalize_token(kind);
      return k == "CRYSTAL" || k == "CRYSTALSEED" || k == "CRYSTALPERTURBEDDROPLET";
    }

    [[nodiscard]] auto uses_smooth() const -> bool {
      auto k = normalize_token(kind);
      return k == "SMOOTH" || k == "CAP" || k == "WETTING" || k == "ELLIPSOID";
    }

    [[nodiscard]] auto resolved_radii(double radius) const -> std::array<double, 3> {
      std::array<double, 3> radii = envelope_radii;
      if (radii[0] <= 0.0)
        radii[0] = 1.10 * radius;
      if (radii[1] <= 0.0)
        radii[1] = 0.92 * radius;
      if (radii[2] <= 0.0)
        radii[2] = 0.78 * radius;
      return radii;
    }
  };

  struct WallConfig {
    std::string kind{"None"};
    std::string normal{"z"};
    std::string side{"lower"};
    double sigma{1.0};
    double epsilon{0.0};
    double density{1.0};
    double cutoff{0.0};

    [[nodiscard]] auto build() const -> dft::physics::walls::WallPotential {
      if (!is_active())
        return {};
      return {
          dft::physics::walls::LJ93{.sigma = sigma, .epsilon = epsilon, .density = density, .cutoff = cutoff},
          wall_axis(normal),
          is_lower()
      };
    }

    [[nodiscard]] auto is_active() const -> bool {
      auto k = normalize_token(kind);
      return k == "LJ93" && epsilon > 0.0;
    }

    [[nodiscard]] auto is_lower() const -> bool {
      auto token = normalize_token(side);
      if (token == "LOWER" || token == "MIN" || token == "LEFT")
        return true;
      if (token == "UPPER" || token == "MAX" || token == "RIGHT")
        return false;
      throw std::runtime_error("Unknown wall side: " + side);
    }
  };

  struct GravityConfig {
    double strength{0.0};

    [[nodiscard]] auto build(const dft::physics::walls::WallPotential& wall) const -> dft::physics::Gravity {
      return dft::physics::Gravity{.strength = strength};
    }

    [[nodiscard]] auto is_active() const -> bool { return std::abs(strength) > 1e-30; }
  };

  struct OutputConfig {
    std::string directory;
  };

  struct FireConfig {
    double dt;
    double dt_max;
    double alpha_start;
    double alpha_fac;
    double force_tolerance;
    int max_steps;
    int log_interval;

    [[nodiscard]] auto build_minimizer(bool homogeneous_boundary) const -> dft::algorithms::minimization::Minimizer {
      return {
          .fire =
              {.dt = dt,
               .dt_max = dt_max,
               .alpha_start = alpha_start,
               .f_alpha = alpha_fac,
               .force_tolerance = force_tolerance,
               .max_steps = max_steps},
          .param = dft::algorithms::minimization::Unbounded{.rho_min = 1e-99},
          .use_homogeneous_boundary = homogeneous_boundary,
          .log_interval = log_interval,
      };
    }
  };

  struct EigenConfig {
    double tolerance;
    int max_iterations;
    double hessian_eps;
    int log_interval;
  };

  struct DdftConfig {
    double dt;
    double dt_max;
    double fp_tolerance;
    int fp_max_iterations;
    double diffusion_coefficient;
    int n_steps;
    int snapshot_interval;
    int log_interval;
    double perturb_scale;
  };

  struct NucleationConfig {
    ModelConfig model;
    DropletConfig droplet;
    SeedConfig seed;
    WallConfig wall;
    GravityConfig gravity;
    OutputConfig output;
    FireConfig fire;
    EigenConfig eigen;
    DdftConfig ddft;

    [[nodiscard]] auto profile_axis() const -> int { return wall.is_active() ? wall_axis(wall.normal) : 0; }

    [[nodiscard]] auto cross_section_axes() const -> std::array<int, 2> {
      int axis = profile_axis();
      return {0, axis == 0 ? 1 : axis};
    }

    [[nodiscard]] auto export_directory() const -> std::string {
      if (!output.directory.empty())
        return output.directory;

      // {Potential}_{Split}_{Scenario}_T{temperature}_S{supersaturation}
      auto pot = abbreviate(model.potential.name);
      auto split = abbreviate(model.split.interaction);
      auto scenario = wall.is_active() ? std::string{"Wall"}
          : seed.uses_crystal()        ? std::string{"Crystal"}
                                       : std::string{"Droplet"};
      return std::format("{}_{}_{}_{}", pot, split, scenario, fmt_params());
    }

    [[nodiscard]] auto fmt_params() const -> std::string {
      auto strip = [](std::string s) {
        // Remove trailing zeros: 0.50 -> 0.5, 1.00 -> 1
        if (s.find('.') != std::string::npos) {
          s.erase(s.find_last_not_of('0') + 1, std::string::npos);
          if (s.back() == '.')
            s.pop_back();
        }
        return s;
      };
      return std::format(
          "T{}_S{}",
          strip(std::format("{:.2f}", model.functional.temperature)),
          strip(std::format("{:.2f}", droplet.supersaturation))
      );
    }
  };

  [[nodiscard]] inline auto read_config(const std::string& path) -> NucleationConfig {
    auto cfg = dft::config::parse_config(path);
    using dft::config::get;

    auto get_or = [&]<typename T>(const std::string& key, T fallback) -> T {
      try {
        return get<T>(cfg, key);
      } catch (...) {
        return fallback;
      }
    };

    auto get_or_vec3 = [&](const std::string& key, std::array<double, 3> fallback) -> std::array<double, 3> {
      try {
        return get<std::array<double, 3>>(cfg, key);
      } catch (...) {
        return fallback;
      }
    };

    auto get_or_shape3 = [&](const std::string& key, std::array<long, 3> fallback) -> std::array<long, 3> {
      try {
        return get<std::array<long, 3>>(cfg, key);
      } catch (...) {
        return fallback;
      }
    };

    auto split_interaction = get_or("model.split.interaction", std::string{"BarkerHenderson"});

    double box_length = get_or("model.grid.box_length", 0.0);
    std::array<double, 3> default_box = (box_length > 0.0) ? std::array<double, 3>{box_length, box_length, box_length}
                                                           : std::array<double, 3>{0.0, 0.0, 0.0};
    auto box_size = get_or_vec3("model.grid.box_size", default_box);
    if (std::ranges::any_of(box_size, [](double length) { return length <= 0.0; })) {
      throw std::runtime_error("model.grid requires either positive box_length or box_size = [Lx, Ly, Lz]");
    }

    return {
        .model =
            {.grid =
                 {.dx = get<double>(cfg, "model.grid.dx"),
                  .box_size = box_size,
                  .predictor_dx = get_or("model.grid.predictor_dx", 0.0)},
             .potential =
                 {.name = get_or("model.potential.name", std::string{"LennardJones"}),
                  .sigma = get<double>(cfg, "model.potential.sigma"),
                  .epsilon = get<double>(cfg, "model.potential.epsilon"),
                  .cutoff = get<double>(cfg, "model.potential.cutoff"),
                  .alpha = get_or("model.potential.alpha", 50.0)},
             .split =
                 {.interaction = split_interaction,
                  .hard_sphere_diameter = get_or("model.split.hard_sphere_diameter", split_interaction)},
             .functional =
                 {.name = get<std::string>(cfg, "model.functional.name"),
                  .temperature = get<double>(cfg, "model.functional.temperature")}},
        .droplet =
            {.radius = get<double>(cfg, "droplet.radius"), .supersaturation = get_or("droplet.supersaturation", 1.1)},
        .seed =
            {.kind = get_or("seed.kind", std::string{"Droplet"}),
             .structure = get_or("seed.structure", std::string{"FCC"}),
             .orientation = get_or("seed.orientation", std::string{"001"}),
             .unit_cells = get_or_shape3("seed.unit_cells", {8, 8, 8}),
             .gaussian_width = get_or("seed.gaussian_width", 0.25),
             .modulation_amplitude = get_or("seed.modulation_amplitude", 0.35),
             .envelope_radii = get_or_vec3("seed.envelope_radii", {0.0, 0.0, 0.0}),
             .offset = get_or_vec3("seed.offset", {0.35, -0.20, 0.15}),
             .interface_width = get_or("seed.interface_width", 0.25)},
        .wall =
            {.kind = get_or("wall.kind", std::string{"None"}),
             .normal = get_or("wall.normal", std::string{"z"}),
             .side = get_or("wall.side", std::string{"lower"}),
             .sigma = get_or("wall.sigma", 1.0),
             .epsilon = get_or("wall.epsilon", 0.0),
             .density = get_or("wall.density", 1.0),
             .cutoff = get_or("wall.cutoff", 0.0)},
        .gravity = {.strength = get_or("gravity.strength", 0.0)},
        .output = {.directory = get_or("output.directory", std::string{})},
        .fire =
            {.dt = get<double>(cfg, "fire.dt"),
             .dt_max = get<double>(cfg, "fire.dt_max"),
             .alpha_start = get<double>(cfg, "fire.alpha_start"),
             .alpha_fac = get<double>(cfg, "fire.alpha_fac"),
             .force_tolerance = get<double>(cfg, "fire.force_tolerance"),
             .max_steps = get<int>(cfg, "fire.max_steps"),
             .log_interval = get<int>(cfg, "fire.log_interval")},
        .eigen =
            {.tolerance = get<double>(cfg, "eigen.tolerance"),
             .max_iterations = get<int>(cfg, "eigen.max_iterations"),
             .hessian_eps = get<double>(cfg, "eigen.hessian_eps"),
             .log_interval = get<int>(cfg, "eigen.log_interval")},
        .ddft =
            {.dt = get<double>(cfg, "ddft.dt"),
             .dt_max = get<double>(cfg, "ddft.dt_max"),
             .fp_tolerance = get<double>(cfg, "ddft.fp_tolerance"),
             .fp_max_iterations = get<int>(cfg, "ddft.fp_max_iterations"),
             .diffusion_coefficient = get<double>(cfg, "ddft.diffusion_coefficient"),
             .n_steps = get<int>(cfg, "ddft.n_steps"),
             .snapshot_interval = get<int>(cfg, "ddft.snapshot_interval"),
             .log_interval = get<int>(cfg, "ddft.log_interval"),
             .perturb_scale = get<double>(cfg, "ddft.perturb_scale")},
    };
  }

  [[nodiscard]] inline auto split_scheme(const std::string& name) -> dft::physics::potentials::SplitScheme {
    return dft::physics::potentials::parse_split_scheme(name);
  }

  struct RadialSnapshot {
    double time{0.0};
    std::vector<double> r;
    std::vector<double> rho;
  };

  using SliceSnapshot = dft::Slice1D;

  using PathwayPoint = dft::algorithms::dynamics::PathwayPoint;

  using DynamicsResult = dft::algorithms::dynamics::DynamicsAnalysis;

  using ClusterInfo = dft::algorithms::saddle_point::ConstrainedResult;

  [[nodiscard]] inline auto
  shifted_coordinate(const dft::Grid& grid, int axis, long index, const std::optional<WallConfig>& wall = std::nullopt)
      -> double {
    if (wall && wall->is_active() && axis == wall_axis(wall->normal)) {
      return wall->build().distance(grid, index);
    }
    return index * grid.dx - grid.box_size[axis] / 2.0;
  }

  [[nodiscard]] inline auto seed_axis_displacement(
      const dft::Grid& grid,
      int axis,
      long index,
      double center,
      const std::optional<WallConfig>& wall = std::nullopt
  ) -> double {
    double delta = index * grid.dx - center;
    if (wall && wall->is_active() && axis == wall_axis(wall->normal))
      return delta;
    return dft::Grid::minimum_image(delta, grid.box_size[axis]);
  }

  // Step function: rho_in inside radius R, rho_out outside.

  [[nodiscard]] inline auto seed_center(const dft::Grid& grid, const NucleationConfig& cfg) -> std::array<double, 3> {
    std::array<double, 3> center = {grid.box_size[0] / 2.0, grid.box_size[1] / 2.0, grid.box_size[2] / 2.0};
    if (!cfg.wall.is_active())
      return center;

    int axis = wall_axis(cfg.wall.normal);
    double extent = (cfg.seed.uses_crystal() || cfg.seed.uses_smooth())
        ? cfg.seed.resolved_radii(cfg.droplet.radius)[axis]
        : cfg.droplet.radius;
    double contact = cfg.wall.build().attachment_distance(grid);
    double min_center = extent + contact;
    double max_center = grid.box_size[axis] - extent - contact;
    if (min_center > max_center) {
      center[axis] = 0.5 * grid.box_size[axis];
      return center;
    }
    center[axis] = cfg.wall.is_lower() ? min_center : max_center;
    return center;
  }

  [[nodiscard]] inline auto ellipsoidal_envelope(
      const dft::Grid& grid,
      const std::array<double, 3>& radii,
      double interface_width,
      const std::array<double, 3>& center,
      const std::optional<WallConfig>& wall = std::nullopt
  ) -> arma::vec {
    auto n = static_cast<arma::uword>(grid.total_points());
    arma::vec envelope(n, arma::fill::zeros);

    double r_min = std::min({radii[0], radii[1], radii[2]});
    double width = std::max(interface_width, 0.5 * grid.dx);

    for (long ix = 0; ix < grid.shape[0]; ++ix) {
      double dx = seed_axis_displacement(grid, 0, ix, center[0], wall);
      for (long iy = 0; iy < grid.shape[1]; ++iy) {
        double dy = seed_axis_displacement(grid, 1, iy, center[1], wall);
        for (long iz = 0; iz < grid.shape[2]; ++iz) {
          double dz = seed_axis_displacement(grid, 2, iz, center[2], wall);
          double ell = std::sqrt(
              (dx * dx) / (radii[0] * radii[0]) + (dy * dy) / (radii[1] * radii[1]) + (dz * dz) / (radii[2] * radii[2])
          );
          double signed_distance = (ell - 1.0) * r_min;
          envelope(static_cast<arma::uword>(grid.flat_index(ix, iy, iz))) = 0.5
              * (1.0 - std::tanh(signed_distance / width));
        }
      }
    }

    return envelope;
  }

  [[nodiscard]] inline auto
  lattice_field(const dft::Grid& grid, const SeedConfig& seed, const std::optional<WallConfig>& wall = std::nullopt)
      -> arma::vec {
    auto lattice = dft::build_lattice(
        dft::parse_structure(seed.structure),
        dft::parse_orientation(seed.orientation),
        {
            seed.unit_cells[0],
            seed.unit_cells[1],
            seed.unit_cells[2],
        }
    );

    arma::rowvec3 box = {grid.box_size[0], grid.box_size[1], grid.box_size[2]};
    arma::mat positions = lattice.scaled_positions(box);
    positions.col(0).transform([&](double x) { return dft::Grid::wrap_periodic(x + seed.offset[0], grid.box_size[0]); }
    );
    positions.col(1).transform([&](double y) { return dft::Grid::wrap_periodic(y + seed.offset[1], grid.box_size[1]); }
    );
    positions.col(2).transform([&](double z) {
      double shifted = z + seed.offset[2];
      if (wall && wall->is_active() && wall_axis(wall->normal) == 2)
        return std::clamp(shifted, 0.0, grid.box_size[2] - 0.5 * grid.dx);
      return dft::Grid::wrap_periodic(shifted, grid.box_size[2]);
    });

    arma::vec field(static_cast<arma::uword>(grid.total_points()), arma::fill::zeros);

    double sigma = std::max(seed.gaussian_width, 0.5 * grid.dx);
    double inv_two_sigma2 = 0.5 / (sigma * sigma);
    double cutoff = 3.0 * sigma;
    long span = std::max(1L, static_cast<long>(std::ceil(cutoff / grid.dx)));

    for (arma::uword atom = 0; atom < positions.n_rows; ++atom) {
      double x0 = positions(atom, 0);
      double y0 = positions(atom, 1);
      double z0 = positions(atom, 2);

      long ix0 = static_cast<long>(std::llround(x0 / grid.dx));
      long iy0 = static_cast<long>(std::llround(y0 / grid.dx));
      long iz0 = static_cast<long>(std::llround(z0 / grid.dx));

      for (long dix = -span; dix <= span; ++dix) {
        long ix = (ix0 + dix) % grid.shape[0];
        if (ix < 0)
          ix += grid.shape[0];
        double dx = seed_axis_displacement(grid, 0, ix, x0, wall);
        if (std::abs(dx) > cutoff)
          continue;

        for (long diy = -span; diy <= span; ++diy) {
          long iy = (iy0 + diy) % grid.shape[1];
          if (iy < 0)
            iy += grid.shape[1];
          double dy = seed_axis_displacement(grid, 1, iy, y0, wall);
          if (std::abs(dy) > cutoff)
            continue;

          for (long diz = -span; diz <= span; ++diz) {
            long iz = iz0 + diz;
            if (wall && wall->is_active() && wall_axis(wall->normal) == 2) {
              if (iz < 0 || iz >= grid.shape[2])
                continue;
            } else {
              iz %= grid.shape[2];
              if (iz < 0)
                iz += grid.shape[2];
            }
            double dz = seed_axis_displacement(grid, 2, iz, z0, wall);
            if (std::abs(dz) > cutoff)
              continue;

            double r2 = dx * dx + dy * dy + dz * dz;
            field(static_cast<arma::uword>(grid.flat_index(ix, iy, iz))) += std::exp(-r2 * inv_two_sigma2);
          }
        }
      }
    }

    return field;
  }

  [[nodiscard]] inline auto normalized_lattice_field(
      const dft::Grid& grid,
      const SeedConfig& seed,
      const arma::vec& envelope,
      const std::optional<WallConfig>& wall = std::nullopt
  ) -> arma::vec {
    arma::vec field = lattice_field(grid, seed, wall);
    arma::uvec active = arma::find(envelope > 0.05);
    if (active.is_empty())
      return field;

    double max_val = field.elem(active).max();
    if (max_val > 1e-30)
      field /= max_val;
    return field;
  }

  // Wall and gravity field construction now delegates to library types.

  [[nodiscard]] inline auto crystalline_seed(
      const dft::Grid& grid,
      const SeedConfig& seed,
      double radius,
      double rho_in,
      double rho_out,
      const std::array<double, 3>& center,
      const std::optional<WallConfig>& wall = std::nullopt
  ) -> arma::vec {
    auto radii = seed.resolved_radii(radius);
    arma::vec envelope = ellipsoidal_envelope(grid, radii, seed.interface_width, center, wall);

    if (wall && wall->is_active()) {
      arma::vec wf = wall->build().field(grid);
      arma::uvec repulsive = arma::find(wf > 0.0);
      envelope.elem(repulsive).zeros();
    }

    arma::vec base = rho_out + (rho_in - rho_out) * envelope;
    arma::vec mod = normalized_lattice_field(grid, seed, envelope, wall);

    arma::uvec active = arma::find(envelope > 0.05);
    if (!active.is_empty()) {
      double mean = arma::mean(mod.elem(active));
      mod.elem(active) -= mean;
      double max_abs = arma::abs(mod.elem(active)).max();
      if (max_abs > 1e-30)
        mod /= max_abs;
    }

    arma::vec rho = base + seed.modulation_amplitude * (envelope % mod);
    return arma::clamp(rho, 1e-18, arma::datum::inf);
  }

  [[nodiscard]] inline auto smooth_seed(
      const dft::Grid& grid,
      const SeedConfig& seed,
      double radius,
      double rho_in,
      double rho_out,
      const std::array<double, 3>& center,
      const std::optional<WallConfig>& wall = std::nullopt
  ) -> arma::vec {
    auto radii = seed.resolved_radii(radius);
    arma::vec envelope = ellipsoidal_envelope(grid, radii, seed.interface_width, center, wall);
    arma::vec rho = rho_out + (rho_in - rho_out) * envelope;
    return arma::clamp(rho, 1e-18, arma::datum::inf);
  }

  [[nodiscard]] inline auto smooth_seed_on_background(
      const dft::Grid& grid,
      const SeedConfig& seed,
      double radius,
      double rho_in,
      const arma::vec& background,
      const std::array<double, 3>& center,
      const std::optional<WallConfig>& wall = std::nullopt
  ) -> arma::vec {
    auto radii = seed.resolved_radii(radius);
    arma::vec envelope = ellipsoidal_envelope(grid, radii, seed.interface_width, center, wall);
    arma::vec target = background;
    target.transform([rho_in](double rho_bg) { return std::max(rho_bg, rho_in); });
    arma::vec rho = background + envelope % (target - background);
    return arma::clamp(rho, 1e-18, arma::datum::inf);
  }

  [[nodiscard]] inline auto crystalline_seed_on_background(
      const dft::Grid& grid,
      const SeedConfig& seed,
      double radius,
      double rho_in,
      const arma::vec& background,
      const std::array<double, 3>& center,
      const std::optional<WallConfig>& wall = std::nullopt
  ) -> arma::vec {
    auto radii = seed.resolved_radii(radius);
    arma::vec envelope = ellipsoidal_envelope(grid, radii, seed.interface_width, center, wall);

    if (wall && wall->is_active()) {
      arma::vec wf = wall->build().field(grid);
      arma::uvec repulsive = arma::find(wf > 0.0);
      envelope.elem(repulsive).zeros();
    }

    arma::vec target = background;
    target.transform([rho_in](double rho_bg) { return std::max(rho_bg, rho_in); });
    arma::vec base = background + envelope % (target - background);

    arma::vec mod = normalized_lattice_field(grid, seed, envelope, wall);
    arma::uvec active = arma::find(envelope > 0.05);
    if (!active.is_empty()) {
      double mean = arma::mean(mod.elem(active));
      mod.elem(active) -= mean;
      double max_abs = arma::abs(mod.elem(active)).max();
      if (max_abs > 1e-30)
        mod /= max_abs;
    }

    arma::vec rho = base + seed.modulation_amplitude * (envelope % mod);
    return arma::clamp(rho, 1e-18, arma::datum::inf);
  }

  // Wall and gravity field construction now delegates to library types.

  [[nodiscard]] inline auto make_initial_density(
      const NucleationConfig& cfg,
      const dft::Grid& grid,
      const arma::vec& r,
      double rho_in,
      double rho_out
  ) -> arma::vec {
    auto center = seed_center(grid, cfg);
    if (cfg.seed.uses_crystal()) {
      return crystalline_seed(
          grid,
          cfg.seed,
          cfg.droplet.radius,
          rho_in,
          rho_out,
          center,
          cfg.wall.is_active() ? std::optional<WallConfig>{cfg.wall} : std::nullopt
      );
    }
    if (cfg.seed.uses_smooth() || cfg.wall.is_active()) {
      return smooth_seed(
          grid,
          cfg.seed,
          cfg.droplet.radius,
          rho_in,
          rho_out,
          center,
          cfg.wall.is_active() ? std::optional<WallConfig>{cfg.wall} : std::nullopt
      );
    }
    return dft::StepProfile{.radius = cfg.droplet.radius, .rho_in = rho_in, .rho_out = rho_out}.apply(r);
  }

  // Build the combined external field (wall + gravity).

  [[nodiscard]] inline auto build_external_field(
      const NucleationConfig& cfg,
      const dft::Grid& grid,
      const dft::physics::walls::WallPotential& wall_potential
  ) -> arma::vec {
    arma::vec field = wall_potential.field(grid);
    if (cfg.gravity.is_active()) {
      field += cfg.gravity.build(wall_potential).field(grid, wall_potential);
      std::println(std::cout, "Gravity: g*={:.6f}", cfg.gravity.strength);
    }
    return field;
  }

  // Seed a droplet onto a non-uniform background density.

  [[nodiscard]] inline auto seed_on_background(
      const dft::Grid& grid,
      const NucleationConfig& cfg,
      double rho_l,
      const arma::vec& background,
      const std::array<double, 3>& center,
      const dft::math::CubicSpline* predictor_spline = nullptr,
      double predictor_cutoff = 0.0
  ) -> arma::vec {
    if (cfg.seed.uses_crystal()) {
      return crystalline_seed_on_background(grid, cfg.seed, cfg.droplet.radius, rho_l, background, center, cfg.wall);
    }

    if (!predictor_spline) {
      return smooth_seed_on_background(grid, cfg.seed, cfg.droplet.radius, rho_l, background, center, cfg.wall);
    }

    arma::vec rho = background;
    arma::vec distance = grid.radial_distances(center);
    for (arma::uword i = 0; i < rho.n_elem; ++i) {
      double r_value = distance(i);
      if (r_value > predictor_cutoff)
        continue;
      rho(i) += std::max((*predictor_spline)(r_value), 0.0);
    }
    return arma::clamp(rho, 1e-18, arma::datum::inf);
  }

  [[nodiscard]] inline auto
  extract_profile_slice(const arma::vec& rho, const dft::Grid& grid, const NucleationConfig& cfg) -> SliceSnapshot {
    int axis = cfg.profile_axis();
    auto slice = grid.line_slice(rho, axis);

    if (cfg.wall.is_active()) {
      for (long i = 0; i < grid.shape[axis]; ++i) {
        slice.x[static_cast<std::size_t>(i)] = shifted_coordinate(grid, axis, i, cfg.wall);
      }
    }

    return slice;
  }

  using Slice2D = dft::Slice2D;

  // Extract a 2D plane slice through the centre of the box.

  [[nodiscard]] inline auto extract_plane_slice(
      const arma::vec& rho,
      const dft::Grid& grid,
      const std::array<int, 2>& axes,
      const std::optional<WallConfig>& wall = std::nullopt,
      std::optional<long> fixed_index = std::nullopt
  ) -> Slice2D {
    if (axes[0] == axes[1])
      throw std::runtime_error("Plane axes must be distinct");

    int fixed_axis = 3 - axes[0] - axes[1];
    std::array<long, 3> index = {grid.shape[0] / 2, grid.shape[1] / 2, grid.shape[2] / 2};
    long fixed = fixed_index.value_or(grid.shape[fixed_axis] / 2);
    fixed = std::clamp(fixed, 0L, grid.shape[fixed_axis] - 1);
    long nx = grid.shape[axes[0]];
    long ny = grid.shape[axes[1]];
    double cx = grid.box_size[axes[0]] / 2.0;
    double cy = grid.box_size[axes[1]] / 2.0;

    auto sny = static_cast<std::size_t>(ny);
    auto snx = static_cast<std::size_t>(nx);
    std::vector<std::vector<double>> xg(sny, std::vector<double>(snx));
    std::vector<std::vector<double>> yg(sny, std::vector<double>(snx));
    std::vector<std::vector<double>> zg(sny, std::vector<double>(snx));

    for (long iy = 0; iy < ny; ++iy) {
      index[axes[1]] = iy;
      for (long ix = 0; ix < nx; ++ix) {
        index[axes[0]] = ix;
        index[fixed_axis] = fixed;
        xg[static_cast<std::size_t>(iy)][static_cast<std::size_t>(ix)] = wall
            ? shifted_coordinate(grid, axes[0], ix, *wall)
            : ix * grid.dx - cx;
        yg[static_cast<std::size_t>(iy)][static_cast<std::size_t>(ix)] = wall
            ? shifted_coordinate(grid, axes[1], iy, *wall)
            : iy * grid.dx - cy;
        zg[static_cast<std::size_t>(iy)][static_cast<std::size_t>(ix)] =
            rho(static_cast<arma::uword>(grid.flat_index(index[0], index[1], index[2])));
      }
    }

    return {
        .time = 0.0,
        .nx = nx,
        .ny = ny,
        .x_label = dft::Grid::axis_name(axes[0]),
        .y_label = dft::Grid::axis_name(axes[1]),
        .x = std::move(xg),
        .y = std::move(yg),
        .z = std::move(zg),
    };
  }

  [[nodiscard]] inline auto extract_xy_slice(
      const arma::vec& rho,
      const dft::Grid& grid,
      std::optional<long> fixed_z = std::nullopt,
      const std::optional<WallConfig>& wall = std::nullopt
  ) -> Slice2D {
    return extract_plane_slice(rho, grid, {0, 1}, wall, fixed_z);
  }

  [[nodiscard]] inline auto extract_xz_slice(
      const arma::vec& rho,
      const dft::Grid& grid,
      std::optional<long> fixed_y = std::nullopt,
      const std::optional<WallConfig>& wall = std::nullopt
  ) -> Slice2D {
    return extract_plane_slice(rho, grid, {0, 2}, wall, fixed_y);
  }

  [[nodiscard]] inline auto extract_yz_slice(
      const arma::vec& rho,
      const dft::Grid& grid,
      std::optional<long> fixed_x = std::nullopt,
      const std::optional<WallConfig>& wall = std::nullopt
  ) -> Slice2D {
    return extract_plane_slice(rho, grid, {1, 2}, wall, fixed_x);
  }

  [[nodiscard]] inline auto
  extract_cross_section(const arma::vec& rho, const dft::Grid& grid, const NucleationConfig& cfg) -> Slice2D {
    return extract_plane_slice(
        rho,
        grid,
        cfg.cross_section_axes(),
        cfg.wall.is_active() ? std::optional<WallConfig>{cfg.wall} : std::nullopt
    );
  }

  // Wall-aware center density: at the seed center (not the box center).

  [[nodiscard]] inline auto center_density(const arma::vec& rho, const dft::Grid& grid, const NucleationConfig& cfg)
      -> double {
    if (!cfg.wall.is_active())
      return grid.center_value(rho);

    auto center = seed_center(grid, cfg);
    return grid.value_at(rho, center);
  }

  // Extract x-slice snapshots and (R_eff, Omega, rho_center) pathway from a DDFT simulation result.
  // Wall-aware variant that uses config-dependent profile axis and center density.

  [[nodiscard]] inline auto extract_dynamics(
      const dft::algorithms::dynamics::SimulationResult& sim,
      const dft::Grid& grid,
      const arma::vec& /*r*/,
      double rho_background,
      double delta_rho,
      const NucleationConfig& cfg
  ) -> DynamicsResult {
    double dv = grid.cell_volume();
    DynamicsResult result;
    for (const auto& snap : sim.snapshots) {
      auto prof = extract_profile_slice(snap.densities[0], grid, cfg);
      prof.time = snap.time;
      result.profiles.push_back(std::move(prof));
      double R = dft::effective_radius(snap.densities[0], rho_background, delta_rho, dv);
      result.pathway.push_back({
          .radius = R,
          .energy = snap.energy,
          .rho_center = center_density(snap.densities[0], grid, cfg),
      });
    }
    return result;
  }

  // Result of the critical-cluster search algorithm.

  using CriticalClusterResult = dft::algorithms::saddle_point::WallRampResult;

  // Find the critical cluster. Delegates entirely to the library's
  // WallRampSearch, providing nucleation-specific seed generation
  // callbacks.

  [[nodiscard]] inline auto find_critical_cluster(
      const dft::functionals::Functional& func,
      const NucleationConfig& cfg,
      const dft::physics::walls::WallPotential& wall_potential,
      const arma::vec& wall_field,
      const std::array<double, 3>& seed_origin,
      double rho_v,
      double rho_l,
      double rho_out,
      double mu_out
  ) -> CriticalClusterResult {
    bool has_wall = cfg.wall.is_active();
    auto minimizer = cfg.fire.build_minimizer(!has_wall);

    auto search = dft::algorithms::saddle_point::WallRampSearch{
        .minimizer = minimizer,
        .max_retries = has_wall ? 4 : 0,
        .droplet_radius = cfg.droplet.radius,
        .predictor_dx = cfg.model.grid.predictor_dx,
    };

    auto make_seed = [&](const dft::Grid& grid, const arma::vec& r, double rho_l_seed, double rho_out_seed) {
      return make_initial_density(cfg, grid, r, rho_l_seed, rho_out_seed);
    };

    auto make_bg_seed = [&](const dft::Grid& grid,
                            const arma::vec& background,
                            const std::array<double, 3>& center,
                            const dft::math::CubicSpline* spline,
                            double cutoff) {
      return seed_on_background(grid, cfg, rho_l, background, center, spline, cutoff);
    };

    return search
        .find(func, wall_potential, wall_field, make_seed, make_bg_seed, seed_origin, rho_v, rho_l, rho_out, mu_out);
  }

} // namespace nucleation
