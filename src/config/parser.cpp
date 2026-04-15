#include "dft/config/parser.hpp"

#include <format>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <toml++/toml.hpp>

namespace dft::config {

  namespace {

    auto parse_ini(const std::string& path) -> nlohmann::json {
      std::ifstream file(path);
      if (!file.is_open()) {
        throw std::runtime_error(std::format("Cannot open INI file: {}", path));
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
            throw std::runtime_error(std::format("Malformed section in INI file: {}", line));
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
          // Handle dotted sections: [a.b.c] → result["a"]["b"]["c"][key].
          nlohmann::json* node = &result;
          std::istringstream section_stream(current_section);
          std::string section_token;
          while (std::getline(section_stream, section_token, '.')) {
            node = &(*node)[section_token];
          }
          (*node)[key] = json_value;
        } else {
          result[key] = json_value;
        }
      }

      return result;
    }

    auto toml_to_json(const toml::node& node) -> nlohmann::json {
      if (auto* tbl = node.as_table()) {
        nlohmann::json obj = nlohmann::json::object();
        for (auto& [k, v] : *tbl) {
          obj[std::string(k)] = toml_to_json(v);
        }
        return obj;
      }
      if (auto* arr = node.as_array()) {
        nlohmann::json jarr = nlohmann::json::array();
        for (auto& v : *arr) {
          jarr.push_back(toml_to_json(v));
        }
        return jarr;
      }
      if (auto* v = node.as_string())
        return nlohmann::json(v->get());
      if (auto* v = node.as_integer())
        return nlohmann::json(v->get());
      if (auto* v = node.as_floating_point())
        return nlohmann::json(v->get());
      if (auto* v = node.as_boolean())
        return nlohmann::json(v->get());
      return nlohmann::json{};
    }

    auto parse_toml(const std::string& path) -> nlohmann::json {
      try {
        auto tbl = toml::parse_file(path);
        return toml_to_json(tbl);
      } catch (const toml::parse_error& err) {
        throw std::runtime_error(std::format("TOML parse error in {}: {}", path, std::string(err.description())));
      }
    }

  } // namespace

  auto parse_config(const std::string& path, FileType type) -> nlohmann::json {
    switch (type) {
      case FileType::INI:
        return parse_ini(path);
      case FileType::JSON: {
        std::ifstream f(path);
        if (!f.is_open()) {
          throw std::runtime_error(std::format("Cannot open JSON file: {}", path));
        }
        return nlohmann::json::parse(f);
      }
      case FileType::TOML:
        return parse_toml(path);
    }
    throw std::runtime_error("Unknown file type");
  }

  auto parse_config(const std::string& path) -> nlohmann::json {
    auto ext_pos = path.rfind('.');
    if (ext_pos != std::string::npos) {
      auto ext = path.substr(ext_pos);
      if (ext == ".toml")
        return parse_config(path, FileType::TOML);
      if (ext == ".json")
        return parse_config(path, FileType::JSON);
    }
    return parse_config(path, FileType::INI);
  }

} // namespace dft::config
