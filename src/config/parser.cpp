#include "dft/config/parser.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace dft::config {

  namespace {

    auto parse_ini(const std::string& path) -> nlohmann::json {
      std::ifstream file(path);
      if (!file.is_open()) {
        throw std::runtime_error("Cannot open INI file: " + path);
      }

      nlohmann::json result;
      std::string current_section;
      std::string line;

      while (std::getline(file, line)) {
        auto start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos)
          continue;
        line = line.substr(start);

        if (line[0] == ';' || line[0] == '#')
          continue;

        if (line[0] == '[') {
          auto end = line.find(']');
          if (end == std::string::npos) {
            throw std::runtime_error("Malformed section in INI file: " + line);
          }
          current_section = line.substr(1, end - 1);
          continue;
        }

        auto eq = line.find('=');
        if (eq == std::string::npos)
          continue;

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

  }  // namespace

  auto parse_config(const std::string& path, FileType type) -> nlohmann::json {
    switch (type) {
      case FileType::INI:
        return parse_ini(path);
      case FileType::JSON: {
        std::ifstream f(path);
        if (!f.is_open()) {
          throw std::runtime_error("Cannot open JSON file: " + path);
        }
        return nlohmann::json::parse(f);
      }
    }
    throw std::runtime_error("Unknown file type");
  }

}  // namespace dft::config
