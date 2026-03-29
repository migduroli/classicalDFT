#pragma once

#include <string>
#include <vector>

namespace cdft::viz {

  enum class ExportFormat { JPG, PNG, EPS, PS, PDF };

  enum class Symbol {
    CIRCLE = 1,
    SQUARE,
    DIAMOND,
    TRIANGLE_UP,
    TRIANGLE_LEFT,
    TRIANGLE_DOWN,
    TRIANGLE_RIGHT,
    PLUS,
    CROSS,
    STAR,
  };

  enum class Color {
    WHITE = 0,
    BLACK,
    RED,
    GREEN,
    BLUE,
    YELLOW,
    BROWN,
    GREY,
    VIOLET,
    CYAN,
    MAGENTA,
    ORANGE,
    INDIGO,
    MAROON,
    TURQUOISE,
    DARKGREEN,
  };

  enum class LineStyle {
    NONE = 0,
    SOLID = 1,
    DOTTED,
    DASHED_SHORT,
    DASHED_LONG,
    DASH_DOT_SHORT,
    DASH_DOT_LONG,
    DOUBLE_DASH_DOT_SHORT,
    DOUBLE_DASH_DOT_LONG,
  };

  enum class Axis { X, Y };

}  // namespace cdft::viz
