#pragma once

#include "cdft/core/types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>

#ifdef DFT_HAS_MATPLOTLIB
#include "matplotlibcpp.h"
#endif

namespace cdft::geometry {

  // ── Grid point indexing ───────────────────────────────────────────────────

  enum class Direction : int { X = 0, Y = 1, Z = 2 };
  enum class Plane { XY, XZ, YZ };

  // ── UniformGrid: replaces the entire geometry hierarchy ───────────────────

  template <std::size_t Dim>
  struct UniformGrid {
    double spacing = 1.0;
    std::array<double, Dim> dimensions = {};
    std::array<double, Dim> origin = {};

    // ── Derived sizes ───────────────────────────────────────────────────

    [[nodiscard]] constexpr std::array<long, Dim> shape() const {
      std::array<long, Dim> s{};
      for (std::size_t d = 0; d < Dim; ++d) {
        s[d] = static_cast<long>(std::floor((dimensions[d] + 1e-8 * spacing) / spacing)) + 1;
      }
      return s;
    }

    [[nodiscard]] constexpr long total_vertices() const {
      auto s = shape();
      long n = 1;
      for (auto v : s) n *= v;
      return n;
    }

    [[nodiscard]] constexpr long total_elements() const {
      auto s = shape();
      long n = 1;
      for (auto v : s) n *= (v - 1);
      return n;
    }

    // ── Volumes ─────────────────────────────────────────────────────────

    [[nodiscard]] constexpr double cell_volume() const {
      double v = 1.0;
      for (std::size_t d = 0; d < Dim; ++d) v *= spacing;
      return v;
    }

    [[nodiscard]] constexpr double total_volume() const {
      double v = 1.0;
      for (auto d : dimensions) v *= d;
      return v;
    }

    // ── Coordinate access ───────────────────────────────────────────────

    [[nodiscard]] constexpr std::array<double, Dim> position(const std::array<long, Dim>& index) const {
      std::array<double, Dim> x{};
      for (std::size_t d = 0; d < Dim; ++d) {
        x[d] = origin[d] + static_cast<double>(index[d]) * spacing;
      }
      return x;
    }

    // ── Cartesian <-> flat index ────────────────────────────────────────

    [[nodiscard]] constexpr long flat_index(const std::array<long, Dim>& index) const {
      auto s = shape();
      long result = 0;
      long stride = 1;
      for (int d = static_cast<int>(Dim) - 1; d >= 0; --d) {
        long idx = index[static_cast<std::size_t>(d)];
        if (idx < 0) idx += s[static_cast<std::size_t>(d)];
        result += idx * stride;
        stride *= s[static_cast<std::size_t>(d)];
      }
      return result;
    }

    [[nodiscard]] constexpr std::array<long, Dim> cartesian_index(long global) const {
      auto s = shape();
      std::array<long, Dim> result{};
      for (int d = static_cast<int>(Dim) - 1; d >= 0; --d) {
        result[static_cast<std::size_t>(d)] = global % s[static_cast<std::size_t>(d)];
        global /= s[static_cast<std::size_t>(d)];
      }
      return result;
    }

    // ── Periodic wrapping ───────────────────────────────────────────────

    [[nodiscard]] constexpr std::array<double, Dim> wrap(const std::array<double, Dim>& x) const {
      std::array<double, Dim> result{};
      for (std::size_t d = 0; d < Dim; ++d) {
        result[d] = std::fmod(x[d], dimensions[d]);
        if (result[d] < 0.0) result[d] += dimensions[d];
      }
      return result;
    }

    // ── Plotting (2D and 3D) ────────────────────────────────────────────

#ifdef DFT_HAS_MATPLOTLIB
    void plot(const std::string& path = "", bool interactive = true) const {
      namespace plt = matplotlibcpp;

      auto s = shape();
      std::vector<double> xs, ys;

      if constexpr (Dim == 2) {
        for (long i = 0; i < s[0]; ++i) {
          for (long j = 0; j < s[1]; ++j) {
            xs.push_back(origin[0] + static_cast<double>(i) * spacing);
            ys.push_back(origin[1] + static_cast<double>(j) * spacing);
          }
        }
      } else if constexpr (Dim == 3) {
        long k = 0;  // project onto z=0 slice
        for (long i = 0; i < s[0]; ++i) {
          for (long j = 0; j < s[1]; ++j) {
            xs.push_back(origin[0] + static_cast<double>(i) * spacing);
            ys.push_back(origin[1] + static_cast<double>(j) * spacing);
          }
        }
        (void)k;
      }

      if (!xs.empty()) {
        double x_min = *std::min_element(xs.begin(), xs.end());
        double x_max = *std::max_element(xs.begin(), xs.end());
        double y_min = *std::min_element(ys.begin(), ys.end());
        double y_max = *std::max_element(ys.begin(), ys.end());
        double x_pad = 0.1 * (x_max - x_min);
        double y_pad = 0.1 * (y_max - y_min);

        plt::scatter(xs, ys, 10.0);
        plt::xlim(x_min - x_pad, x_max + x_pad);
        plt::ylim(y_min - y_pad, y_max + y_pad);

        if (!path.empty()) {
          plt::save(path);
        }
        if (interactive) {
          plt::show();
        }
      }
    }
#else
    void plot(const std::string& /*path*/ = "", bool /*interactive*/ = true) const {}
#endif
  };

  // ── Pythonic aliases ──────────────────────────────────────────────────────

  using Grid2D = UniformGrid<2>;
  using Grid3D = UniformGrid<3>;

}  // namespace cdft::geometry
