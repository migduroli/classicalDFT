#include "classicaldft_bits/io/config.h"

#include <gtest/gtest.h>

using namespace dft_core;

// region Cttors:

TEST(config_parser, default_cttor_works_ok) {
  std::string expected_file_path = "config.ini";
  auto expected_file_type = config_parser::FileType::INI;

  auto config = config_parser::ConfigParser();
  ASSERT_STREQ(config.config_file_path().c_str(), expected_file_path.c_str());
  ASSERT_EQ(config.config_file_type(), expected_file_type);
}

TEST(config_parser, specific_cttor_works_ok) {
  std::vector<config_parser::FileType> types{config_parser::FileType::INI, config_parser::FileType::JSON};
  std::vector<std::string> files{"config.ini", "config.json"};

  for (size_t i = 0; i < types.size(); ++i) {
    auto config = config_parser::ConfigParser(files[i], types[i]);
    ASSERT_STREQ(config.config_file_path().c_str(), files[i].c_str());
    ASSERT_EQ(config.config_file_type(), types[i]);
  }
}

TEST(config_parser, get_works_ok) {
  std::vector<config_parser::FileType> types{config_parser::FileType::INI, config_parser::FileType::JSON};
  std::vector<std::string> files{"config.ini", "config.json"};

  for (size_t i = 0; i < types.size(); ++i) {
    auto config = config_parser::ConfigParser(files[i], types[i]);

    double expected_double_value = 10;
    std::string expected_string_value = "a_text_string";

    auto actual_string_value = config.get<std::string>("default.StringValue");
    auto actual_double_value = config.get<double>("default.DoubleValue");

    ASSERT_DOUBLE_EQ(actual_double_value, expected_double_value);
    ASSERT_STREQ(actual_string_value.c_str(), expected_string_value.c_str());
  }
}

// endregion

// region Exceptions:

TEST(config_parser, cttor_throws_on_missing_ini) {
  EXPECT_THROW(config_parser::ConfigParser("config_not.ini", config_parser::FileType::INI), std::runtime_error);
}

TEST(config_parser, cttor_throws_on_missing_json) {
  EXPECT_THROW(config_parser::ConfigParser("config_not.json", config_parser::FileType::JSON), std::runtime_error);
}

TEST(config_parser, get_throws_on_missing_key) {
  auto config = config_parser::ConfigParser("config.ini");
  EXPECT_THROW(config.get<std::string>("nonexistent.key"), std::runtime_error);
}

// endregion

// endregion
