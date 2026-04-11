#ifndef DFT_PLOTTING_MATPLOTLIB_HPP
#define DFT_PLOTTING_MATPLOTLIB_HPP

#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#ifdef DFT_HAS_MATPLOTLIB
#include "matplotlibcpp.h"
#endif

namespace dft::plotting {

  struct ContourfData {
    std::vector<std::vector<double>> x;
    std::vector<std::vector<double>> y;
    std::vector<std::vector<double>> z;
  };

  struct ContourfOptions {
    int width{600};
    int height{600};
    std::variant<int, std::vector<double>> levels{64};
    std::string cmap{"viridis"};
    std::optional<double> vmin;
    std::optional<double> vmax;
    std::optional<double> alpha;
    std::string extend{"neither"};
    std::string xlabel;
    std::string ylabel;
    std::string title;
    bool square_axes{true};
    bool colorbar{true};
    double shrink{0.82};
    double pad{0.03};
  };

  struct ContourfLine {
    std::vector<double> x;
    std::vector<double> y;
    std::string color{"black"};
    std::string linestyle{"-"};
    double linewidth{1.0};
  };

  struct ContourfPanelsOptions {
    int width{800};
    int height{1100};
    int columns{1};
    bool shared_colorbar{true};
    std::string colorbar_orientation{"vertical"};
    double shrink{0.92};
    double pad{0.03};
    std::string suptitle;
    double suptitle_y{0.985};
    double left{0.10};
    double right{0.88};
    double bottom{0.06};
    double top{0.96};
    double wspace{0.18};
    double hspace{0.28};
  };

  using SubplotLayout = ContourfPanelsOptions;

  struct ContourfPanel {
    ContourfData field;
    ContourfOptions options;
    std::vector<ContourfLine> lines;

    void save(const std::string& filepath) const;
    void show() const;
  };

  // Multi-panel figure: owns panels + layout, renders via .save() / .show().

  struct Figure {
    std::vector<ContourfPanel> panels;
    ContourfPanelsOptions layout;

    void save(const std::string& filepath) const;
    void show() const;
  };

#ifdef DFT_HAS_MATPLOTLIB
  namespace detail {

    [[nodiscard]] inline auto to_pylist_1d(const std::vector<double>& values) -> PyObject* {
      PyObject* list = PyList_New(static_cast<Py_ssize_t>(values.size()));
      for (std::size_t i = 0; i < values.size(); ++i) {
        PyList_SetItem(list, static_cast<Py_ssize_t>(i), PyFloat_FromDouble(values[i]));
      }
      return list;
    }

    [[nodiscard]] inline auto to_pylist_2d(const std::vector<std::vector<double>>& values) -> PyObject* {
      PyObject* list = PyList_New(static_cast<Py_ssize_t>(values.size()));
      for (std::size_t i = 0; i < values.size(); ++i) {
        PyObject* row = PyList_New(static_cast<Py_ssize_t>(values[i].size()));
        for (std::size_t j = 0; j < values[i].size(); ++j) {
          PyList_SetItem(row, static_cast<Py_ssize_t>(j), PyFloat_FromDouble(values[i][j]));
        }
        PyList_SetItem(list, static_cast<Py_ssize_t>(i), row);
      }
      return list;
    }

    [[nodiscard]] inline auto import_module(const char* name) -> PyObject* {
      PyObject* module = PyImport_ImportModule(name);
      if (!module) {
        PyErr_Print();
        throw std::runtime_error(std::string("Failed to import Python module: ") + name);
      }
      return module;
    }

    [[nodiscard]] inline auto get_attr(PyObject* object, const char* name) -> PyObject* {
      PyObject* attr = PyObject_GetAttrString(object, name);
      if (!attr) {
        PyErr_Print();
        throw std::runtime_error(std::string("Failed to access Python attribute: ") + name);
      }
      return attr;
    }

    inline void set_dict_double(PyObject* dict, const char* key, double value) {
      PyObject* obj = PyFloat_FromDouble(value);
      PyDict_SetItemString(dict, key, obj);
      Py_DECREF(obj);
    }

    inline void set_dict_long(PyObject* dict, const char* key, long value) {
      PyObject* obj = PyLong_FromLong(value);
      PyDict_SetItemString(dict, key, obj);
      Py_DECREF(obj);
    }

    inline void set_dict_string(PyObject* dict, const char* key, const std::string& value) {
      PyObject* obj = PyUnicode_FromString(value.c_str());
      PyDict_SetItemString(dict, key, obj);
      Py_DECREF(obj);
    }

    inline void set_dict_levels(PyObject* dict, const std::variant<int, std::vector<double>>& levels) {
      if (std::holds_alternative<int>(levels)) {
        set_dict_long(dict, "levels", std::get<int>(levels));
        return;
      }

      const auto& values = std::get<std::vector<double>>(levels);
      PyObject* list = PyList_New(static_cast<Py_ssize_t>(values.size()));
      for (std::size_t i = 0; i < values.size(); ++i) {
        PyList_SetItem(list, static_cast<Py_ssize_t>(i), PyFloat_FromDouble(values[i]));
      }
      PyDict_SetItemString(dict, "levels", list);
      Py_DECREF(list);
    }

    inline void set_axes_equal(PyObject* ax) {
      PyObject* fn = get_attr(ax, "set_aspect");
      PyObject* args = PyTuple_New(1);
      PyTuple_SetItem(args, 0, PyUnicode_FromString("equal"));
      PyObject* kwargs = PyDict_New();
      set_dict_string(kwargs, "adjustable", "box");
      PyObject* res = PyObject_Call(fn, args, kwargs);
      if (!res)
        PyErr_Print();
      Py_XDECREF(res);
      Py_DECREF(kwargs);
      Py_DECREF(args);
      Py_DECREF(fn);
    }

    [[nodiscard]] inline auto current_axes(PyObject* pyplot) -> PyObject* {
      PyObject* gca_fn = get_attr(pyplot, "gca");
      PyObject* ax = PyObject_CallObject(gca_fn, nullptr);
      Py_DECREF(gca_fn);
      if (!ax) {
        PyErr_Print();
        throw std::runtime_error("matplotlib gca() call failed");
      }
      return ax;
    }

    [[nodiscard]] inline auto select_subplot(PyObject* pyplot, int rows, int columns, int index) -> PyObject* {
      PyObject* subplot_fn = get_attr(pyplot, "subplot");
      PyObject* args = PyTuple_New(3);
      PyTuple_SetItem(args, 0, PyLong_FromLong(rows));
      PyTuple_SetItem(args, 1, PyLong_FromLong(columns));
      PyTuple_SetItem(args, 2, PyLong_FromLong(index));
      PyObject* ax = PyObject_CallObject(subplot_fn, args);
      Py_DECREF(args);
      Py_DECREF(subplot_fn);
      if (!ax) {
        PyErr_Print();
        throw std::runtime_error("matplotlib subplot() call failed");
      }
      return ax;
    }

    inline void call_axes_string_method(PyObject* ax, const char* method, const std::string& value) {
      if (value.empty())
        return;

      PyObject* fn = get_attr(ax, method);
      PyObject* args = PyTuple_New(1);
      PyTuple_SetItem(args, 0, PyUnicode_FromString(value.c_str()));
      PyObject* res = PyObject_CallObject(fn, args);
      if (!res)
        PyErr_Print();
      Py_XDECREF(res);
      Py_DECREF(args);
      Py_DECREF(fn);
    }

    inline void set_axes_text(PyObject* ax, const ContourfOptions& options) {
      call_axes_string_method(ax, "set_xlabel", options.xlabel);
      call_axes_string_method(ax, "set_ylabel", options.ylabel);
      call_axes_string_method(ax, "set_title", options.title);
    }

    inline void plot_line(PyObject* ax, const ContourfLine& line) {
      PyObject* plot_fn = get_attr(ax, "plot");
      PyObject* args = PyTuple_New(2);
      PyTuple_SetItem(args, 0, to_pylist_1d(line.x));
      PyTuple_SetItem(args, 1, to_pylist_1d(line.y));
      PyObject* kwargs = PyDict_New();
      set_dict_string(kwargs, "color", line.color);
      set_dict_string(kwargs, "linestyle", line.linestyle);
      set_dict_double(kwargs, "linewidth", line.linewidth);
      PyObject* res = PyObject_Call(plot_fn, args, kwargs);
      if (!res)
        PyErr_Print();
      Py_XDECREF(res);
      Py_DECREF(kwargs);
      Py_DECREF(args);
      Py_DECREF(plot_fn);
    }

    [[nodiscard]] inline auto render_contourf(
        PyObject* pyplot,
        PyObject* cm,
        PyObject* ax,
        const ContourfData& field,
        const ContourfOptions& options
    ) -> PyObject* {
      PyObject* contourf_fn = get_attr(pyplot, "contourf");

      PyObject* args = PyTuple_New(3);
      PyTuple_SetItem(args, 0, to_pylist_2d(field.x));
      PyTuple_SetItem(args, 1, to_pylist_2d(field.y));
      PyTuple_SetItem(args, 2, to_pylist_2d(field.z));

      PyObject* kwargs = PyDict_New();
      PyObject* cmap = get_attr(cm, options.cmap.c_str());
      PyDict_SetItemString(kwargs, "cmap", cmap);
      Py_DECREF(cmap);
      set_dict_levels(kwargs, options.levels);
      if (options.vmin)
        set_dict_double(kwargs, "vmin", *options.vmin);
      if (options.vmax)
        set_dict_double(kwargs, "vmax", *options.vmax);
      if (options.alpha)
        set_dict_double(kwargs, "alpha", *options.alpha);
      set_dict_string(kwargs, "extend", options.extend);

      PyObject* res = PyObject_Call(contourf_fn, args, kwargs);
      Py_DECREF(kwargs);
      Py_DECREF(args);
      Py_DECREF(contourf_fn);
      if (!res) {
        PyErr_Print();
        throw std::runtime_error("matplotlib contourf call failed");
      }

      if (options.square_axes)
        set_axes_equal(ax);
      set_axes_text(ax, options);
      return res;
    }

    inline void add_colorbar(
        PyObject* pyplot,
        PyObject* mappable,
        PyObject* axes,
        double shrink,
        double pad,
        const std::string& orientation,
        PyObject* cax = nullptr
    ) {
      PyObject* colorbar_fn = get_attr(pyplot, "colorbar");
      PyObject* cb_args = PyTuple_New(1);
      Py_INCREF(mappable);
      PyTuple_SetItem(cb_args, 0, mappable);
      PyObject* cb_kwargs = PyDict_New();
      if (cax) {
        Py_INCREF(cax);
        PyDict_SetItemString(cb_kwargs, "cax", cax);
        Py_DECREF(cax);
      } else {
        Py_INCREF(axes);
        PyDict_SetItemString(cb_kwargs, "ax", axes);
        Py_DECREF(axes);
      }
      set_dict_double(cb_kwargs, "shrink", shrink);
      set_dict_double(cb_kwargs, "pad", pad);
      if (!orientation.empty())
        set_dict_string(cb_kwargs, "orientation", orientation);
      PyObject* cb = PyObject_Call(colorbar_fn, cb_args, cb_kwargs);
      if (!cb)
        PyErr_Print();
      Py_XDECREF(cb);
      Py_DECREF(cb_kwargs);
      Py_DECREF(cb_args);
      Py_DECREF(colorbar_fn);
    }

    inline void adjust_subplots(PyObject* pyplot, const ContourfPanelsOptions& options) {
      PyObject* adjust_fn = get_attr(pyplot, "subplots_adjust");
      PyObject* args = PyTuple_New(0);
      PyObject* kwargs = PyDict_New();
      set_dict_double(kwargs, "left", options.left);
      set_dict_double(kwargs, "right", options.right);
      set_dict_double(kwargs, "bottom", options.bottom);
      set_dict_double(kwargs, "top", options.top);
      set_dict_double(kwargs, "wspace", options.wspace);
      set_dict_double(kwargs, "hspace", options.hspace);
      PyObject* res = PyObject_Call(adjust_fn, args, kwargs);
      if (!res)
        PyErr_Print();
      Py_XDECREF(res);
      Py_DECREF(kwargs);
      Py_DECREF(args);
      Py_DECREF(adjust_fn);
    }

    [[nodiscard]] inline auto add_axes_rect(PyObject* pyplot, double left, double bottom, double width, double height)
        -> PyObject* {
      PyObject* gcf_fn = get_attr(pyplot, "gcf");
      PyObject* fig = PyObject_CallObject(gcf_fn, nullptr);
      Py_DECREF(gcf_fn);
      if (!fig) {
        PyErr_Print();
        throw std::runtime_error("matplotlib gcf() call failed");
      }

      PyObject* add_axes_fn = get_attr(fig, "add_axes");
      PyObject* args = PyTuple_New(1);
      PyObject* rect = PyList_New(4);
      PyList_SetItem(rect, 0, PyFloat_FromDouble(left));
      PyList_SetItem(rect, 1, PyFloat_FromDouble(bottom));
      PyList_SetItem(rect, 2, PyFloat_FromDouble(width));
      PyList_SetItem(rect, 3, PyFloat_FromDouble(height));
      PyTuple_SetItem(args, 0, rect);
      PyObject* ax = PyObject_CallObject(add_axes_fn, args);
      Py_DECREF(args);
      Py_DECREF(add_axes_fn);
      Py_DECREF(fig);
      if (!ax) {
        PyErr_Print();
        throw std::runtime_error("matplotlib add_axes() call failed");
      }
      return ax;
    }

    inline void set_suptitle(PyObject* pyplot, const std::string& title, double y) {
      if (title.empty())
        return;

      PyObject* suptitle_fn = get_attr(pyplot, "suptitle");
      PyObject* args = PyTuple_New(1);
      PyTuple_SetItem(args, 0, PyUnicode_FromString(title.c_str()));
      PyObject* kwargs = PyDict_New();
      set_dict_double(kwargs, "y", y);
      PyObject* res = PyObject_Call(suptitle_fn, args, kwargs);
      if (!res)
        PyErr_Print();
      Py_XDECREF(res);
      Py_DECREF(kwargs);
      Py_DECREF(args);
      Py_DECREF(suptitle_fn);
    }

  } // namespace detail

  inline void contourf_impl(
      const ContourfData& field,
      const std::optional<std::string>& filepath,
      const ContourfOptions& options = {}
  ) {
    namespace plt = matplotlibcpp;
    plt::figure_size(options.width, options.height);

    PyObject* pyplot = detail::import_module("matplotlib.pyplot");
    PyObject* cm = detail::import_module("matplotlib.cm");
    PyObject* ax = detail::current_axes(pyplot);
    PyObject* res = detail::render_contourf(pyplot, cm, ax, field, options);
    if (options.colorbar)
      detail::add_colorbar(pyplot, res, ax, options.shrink, options.pad, "vertical");

    Py_DECREF(ax);
    Py_DECREF(res);
    Py_DECREF(cm);
    Py_DECREF(pyplot);

    plt::tight_layout();
    if (filepath) {
      plt::save(*filepath);
      plt::clf();
      plt::close();
    }
  }

  inline void contourf(const ContourfData& field, const ContourfOptions& options = {}) {
    contourf_impl(field, std::nullopt, options);
  }

  inline void contourf(const ContourfData& field, const std::string& filepath, const ContourfOptions& options = {}) {
    contourf_impl(field, filepath, options);
  }

  inline void contourf_panels_impl(
      const std::vector<ContourfPanel>& panels,
      const std::optional<std::string>& filepath,
      const ContourfPanelsOptions& options = {}
  ) {
    if (panels.empty()) {
      throw std::invalid_argument("contourf_panels requires at least one panel");
    }

    namespace plt = matplotlibcpp;
    plt::figure_size(options.width, options.height);

    PyObject* pyplot = detail::import_module("matplotlib.pyplot");
    PyObject* cm = detail::import_module("matplotlib.cm");
    std::vector<PyObject*> axes;
    std::vector<PyObject*> mappables;
    axes.reserve(panels.size());
    mappables.reserve(panels.size());

    auto cleanup = [&]() {
      for (PyObject* mappable : mappables)
        Py_DECREF(mappable);
      for (PyObject* ax : axes)
        Py_DECREF(ax);
      Py_DECREF(cm);
      Py_DECREF(pyplot);
    };

    try {
      int columns = options.columns <= 0 ? 1 : options.columns;
      if (columns > static_cast<int>(panels.size()))
        columns = static_cast<int>(panels.size());
      int rows =
          static_cast<int>((panels.size() + static_cast<std::size_t>(columns) - 1) / static_cast<std::size_t>(columns));

      bool wants_colorbar = false;
      for (std::size_t i = 0; i < panels.size(); ++i) {
        PyObject* ax = detail::select_subplot(pyplot, rows, columns, static_cast<int>(i) + 1);
        PyObject* mappable = detail::render_contourf(pyplot, cm, ax, panels[i].field, panels[i].options);
        for (const auto& line : panels[i].lines) {
          detail::plot_line(ax, line);
        }

        wants_colorbar = wants_colorbar || panels[i].options.colorbar;
        if (!options.shared_colorbar && panels[i].options.colorbar) {
          detail::add_colorbar(
              pyplot,
              mappable,
              ax,
              panels[i].options.shrink,
              panels[i].options.pad,
              options.colorbar_orientation
          );
        }

        axes.push_back(ax);
        mappables.push_back(mappable);
      }

      detail::set_suptitle(pyplot, options.suptitle, options.suptitle_y);
      detail::adjust_subplots(pyplot, options);

      if (options.shared_colorbar && wants_colorbar) {
        if (options.colorbar_orientation == "horizontal") {
          double span = options.right - options.left;
          double colorbar_width = 0.76 * span;
          double colorbar_left = options.left + 0.12 * span;
          double colorbar_height = 0.026;
          double colorbar_bottom = std::max(0.04, 0.40 * options.bottom);
          PyObject* cax =
              detail::add_axes_rect(pyplot, colorbar_left, colorbar_bottom, colorbar_width, colorbar_height);
          detail::add_colorbar(
              pyplot,
              mappables.back(),
              axes.back(),
              options.shrink,
              options.pad,
              options.colorbar_orientation,
              cax
          );
          Py_DECREF(cax);
        } else {
          PyObject* ax_list = PyList_New(static_cast<Py_ssize_t>(axes.size()));
          for (std::size_t i = 0; i < axes.size(); ++i) {
            Py_INCREF(axes[i]);
            PyList_SetItem(ax_list, static_cast<Py_ssize_t>(i), axes[i]);
          }
          detail::add_colorbar(
              pyplot,
              mappables.back(),
              ax_list,
              options.shrink,
              options.pad,
              options.colorbar_orientation
          );
          Py_DECREF(ax_list);
        }
      }
    } catch (...) {
      cleanup();
      throw;
    }

    cleanup();
    if (filepath) {
      plt::save(*filepath);
      plt::clf();
      plt::close();
    }
  }

  inline void contourf_panels(const std::vector<ContourfPanel>& panels, const ContourfPanelsOptions& options = {}) {
    contourf_panels_impl(panels, std::nullopt, options);
  }

  inline void contourf_panels(
      const std::vector<ContourfPanel>& panels,
      const std::string& filepath,
      const ContourfPanelsOptions& options = {}
  ) {
    contourf_panels_impl(panels, filepath, options);
  }

  // Method implementations

  inline void ContourfPanel::save(const std::string& filepath) const {
    contourf_impl(field, filepath, options);
  }

  inline void ContourfPanel::show() const {
    contourf_impl(field, std::nullopt, options);
  }

  inline void Figure::save(const std::string& filepath) const {
    contourf_panels_impl(panels, filepath, layout);
  }

  inline void Figure::show() const {
    contourf_panels_impl(panels, std::nullopt, layout);
  }

#else

  inline void contourf(const ContourfData&, const ContourfOptions& = {}) {
    throw std::runtime_error("Matplotlib support is not enabled in this build");
  }

  inline void contourf(const ContourfData&, const std::string&, const ContourfOptions& = {}) {
    throw std::runtime_error("Matplotlib support is not enabled in this build");
  }

  inline void contourf_panels(const std::vector<ContourfPanel>&, const ContourfPanelsOptions& = {}) {
    throw std::runtime_error("Matplotlib support is not enabled in this build");
  }

  inline void
  contourf_panels(const std::vector<ContourfPanel>&, const std::string&, const ContourfPanelsOptions& = {}) {
    throw std::runtime_error("Matplotlib support is not enabled in this build");
  }

  inline void ContourfPanel::save(const std::string&) const {
    throw std::runtime_error("Matplotlib support is not enabled in this build");
  }

  inline void ContourfPanel::show() const {
    throw std::runtime_error("Matplotlib support is not enabled in this build");
  }

  inline void Figure::save(const std::string&) const {
    throw std::runtime_error("Matplotlib support is not enabled in this build");
  }

  inline void Figure::show() const {
    throw std::runtime_error("Matplotlib support is not enabled in this build");
  }

#endif

} // namespace dft::plotting

#endif // DFT_PLOTTING_MATPLOTLIB_HPP
