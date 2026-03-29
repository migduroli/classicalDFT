#include "cdft/config.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace cdft::io {

  nlohmann::json ConfigParser::parse_ini(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open INI file: " + path);
    }

    nlohmann::json result;
    std::string current_section;
    std::string line;

    while (std::getline(file, line)) {
      auto start = line.find_first_not_of(" \t\r\n");
      if (start == std::string::npos) continue;
      line = line.substr(start);

      if (line[0] == ';' || line[0] == '#') continue;

      if (line[0] == '[') {
        auto end = line.find(']');
        if (end == std::string::npos) {
          throw std::runtime_error("Malformed section in INI file: " + line);
        }
        current_section = line.substr(1, end - 1);
        continue;
      }

      auto eq = line.find('=');
      if (eq == std::string::npos) continue;

      auto key = line.substr(0, eq);
      auto value = line.substr(eq + 1);

      auto trim = [](std::string& s) {
        auto a = s.find_first_not_of(" \t");
        auto b = s.find_last_not_of(" \t\r\n");
        s = (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
      };
      trim(key);
      trim(value);

      nlohmann::json json_value;
      try {
        json_value = std::stod(value);
      } catch (...) {
        json_value = value;
      }

      if (!current_section.empty()) {
        result[current_section][key] = json_value;
      } else {
        result[key] = json_value;
      }
    }
    return result;
  }

  ConfigParser::ConfigParser(std::string path, ConfigFormat format)
      : path_(std::move(path)), format_(format) {
    switch (format_) {
      case ConfigFormat::INI:
        data_ = parse_ini(path_);
        break;
      case ConfigFormat::JSON: {
        std::ifstream f(path_);
        if (!f.is_open()) {
          throw std::runtime_error("Cannot open JSON file: " + path_);
        }
        data_ = nlohmann::json::parse(f);
        break;
      }
    }
  }

}  // namespace cdft::io
