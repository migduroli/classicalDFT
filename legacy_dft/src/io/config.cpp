#include "classicaldft_bits/io/config.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace dft::io::config {

  nlohmann::json ConfigParser::parse_ini(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open INI file: " + path);
    }

    nlohmann::json result;
    std::string current_section;
    std::string line;

    while (std::getline(file, line)) {
      // Trim whitespace
      auto start = line.find_first_not_of(" \t\r\n");
      if (start == std::string::npos)
        continue;
      line = line.substr(start);

      // Skip comments
      if (line[0] == ';' || line[0] == '#')
        continue;

      // Section header
      if (line[0] == '[') {
        auto end = line.find(']');
        if (end == std::string::npos) {
          throw std::runtime_error("Malformed section in INI file: " + line);
        }
        current_section = line.substr(1, end - 1);
        continue;
      }

      // Key = value
      auto eq = line.find('=');
      if (eq == std::string::npos)
        continue;

      auto key = line.substr(0, eq);
      auto value = line.substr(eq + 1);

      // Trim key and value
      auto trim = [](std::string& s) {
        auto a = s.find_first_not_of(" \t");
        auto b = s.find_last_not_of(" \t\r\n");
        s = (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
      };
      trim(key);
      trim(value);

      // Try to parse as number, otherwise store as string
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

  ConfigParser::ConfigParser() {
    read_config_file(config_file_path_, config_file_type_);
  }

  ConfigParser::ConfigParser(std::string config_file, const FileType& file_type)
      : config_file_path_(std::move(config_file)), config_file_type_(file_type) {
    read_config_file(config_file_path_, config_file_type_);
  }

  void ConfigParser::set_config_file_path(const std::string& config_file) {
    config_file_path_ = config_file;
  }

  void ConfigParser::set_config_file_type(const FileType& file_type) {
    config_file_type_ = file_type;
  }

  void ConfigParser::read_config_file(const std::string& config_file, const FileType& file_type) {
    set_config_file_path(config_file);
    set_config_file_type(file_type);

    switch (file_type) {
      case FileType::INI:
        data_ = parse_ini(config_file);
        break;
      case FileType::JSON: {
        std::ifstream f(config_file);
        if (!f.is_open()) {
          throw std::runtime_error("Cannot open JSON file: " + config_file);
        }
        data_ = nlohmann::json::parse(f);
        break;
      }
    }
  }
}  // namespace dft::io::config