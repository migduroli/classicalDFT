#ifndef CLASSICALDFT_IO_CONFIG_H
#define CLASSICALDFT_IO_CONFIG_H

#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>

namespace dft::io::config {

  enum class FileType { INI = 0, JSON };

  const std::string DEFAULT_CONFIG_FILE_NAME = "config.ini";
  const FileType DEFAULT_CONFIG_FILE_TYPE = FileType::INI;

  class ConfigParser {
   private:
    std::string config_file_path_ = DEFAULT_CONFIG_FILE_NAME;
    FileType config_file_type_ = DEFAULT_CONFIG_FILE_TYPE;
    nlohmann::json data_{};

    static nlohmann::json parse_ini(const std::string& path);

   public:
    ConfigParser();
    explicit ConfigParser(std::string config_file, const FileType& file_type = DEFAULT_CONFIG_FILE_TYPE);

    void set_config_file_path(const std::string& config_file);
    void set_config_file_type(const FileType& file_type);
    void read_config_file(const std::string& config_file, const FileType& file_type);

    [[nodiscard]] const std::string& config_file_path() const { return config_file_path_; }
    [[nodiscard]] const FileType& config_file_type() const { return config_file_type_; }
    [[nodiscard]] const nlohmann::json& data() const { return data_; }

    template <typename T>
    [[nodiscard]] T get(const std::string& dotted_path) const {
      const nlohmann::json* node = &data_;
      std::string token;
      std::istringstream stream(dotted_path);
      while (std::getline(stream, token, '.')) {
        if (!node->contains(token)) {
          throw std::runtime_error("Key not found: " + dotted_path);
        }
        node = &(*node)[token];
      }
      return node->get<T>();
    }
  };
}  // namespace dft::io::config
#endif  // CLASSICALDFT_IO_CONFIG_H
