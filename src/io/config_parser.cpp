#include "dft_lib/io/config_parser.h"

#include "dft_lib/io/console.h"

#include <boost/property_tree/info_parser.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <utility>

namespace dft_core {
  namespace config_parser {
    ConfigParser::ConfigParser() {
      this->read_config_file(this->config_file_path(), this->config_file_type());
    }

    ConfigParser::ConfigParser(std::string config_file, const FileType& file_type)
        : config_file_path_(std::move(config_file)), config_file_type_(file_type) {
      this->read_config_file(this->config_file_path(), this->config_file_type());
    }

    void ConfigParser::set_config_file_path(const std::string& config_file) {
      config_file_path_ = config_file;
    }

    void ConfigParser::set_config_file_type(const FileType& file_type) {
      config_file_type_ = file_type;
    }

    void ConfigParser::read_config_file(const std::string& config_file, const FileType& file_type) {
      this->set_config_file_path(config_file);
      this->set_config_file_type(file_type);

      switch (file_type) {
        case FileType::INI:
          boost::property_tree::ini_parser::read_ini(this->config_file_path(), this->config_tree_);
          break;
        case FileType::JSON:
          boost::property_tree::json_parser::read_json(this->config_file_path(), this->config_tree_);
          break;
        case FileType::INFO:
          boost::property_tree::info_parser::read_info(this->config_file_path(), this->config_tree_);
          break;
        case FileType::XML:
          boost::property_tree::xml_parser::read_xml(this->config_file_path(), this->config_tree_);
          break;
      }
    }
  }  // namespace config_parser
}  // namespace dft_core