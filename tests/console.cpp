#include "dft/console.hpp"

#include <catch2/catch_test_macros.hpp>
#include <sstream>

using namespace dft::console;

TEST_CASE("bold wraps message with ANSI bold codes", "[console]") {
  auto result = format::bold("test");
  CHECK(result == "\x1b[1mtest\x1b[0m");
}

TEST_CASE("blink wraps message with ANSI blink codes", "[console]") {
  auto result = format::blink("test");
  CHECK(result == "\033[33;5;7mtest\033[0m");
}

TEST_CASE("now_str returns 19-character timestamp", "[console]") {
  auto ts = now_str();
  CHECK(ts.size() == 19);
  CHECK(ts[4] == '-');
  CHECK(ts[7] == '-');
  CHECK(ts[10] == ' ');
  CHECK(ts[13] == ':');
  CHECK(ts[16] == ':');
}

TEST_CASE("write outputs to stdout", "[console]") {
  std::ostringstream captured;
  auto* old_buf = std::cout.rdbuf(captured.rdbuf());
  write("hello");
  std::cout.rdbuf(old_buf);
  CHECK(captured.str() == "hello");
}

TEST_CASE("write_line outputs with newline", "[console]") {
  std::ostringstream captured;
  auto* old_buf = std::cout.rdbuf(captured.rdbuf());
  write_line("line");
  std::cout.rdbuf(old_buf);
  CHECK(captured.str() == "line\n");
}

TEST_CASE("info output contains Info marker", "[console]") {
  std::ostringstream captured;
  auto* old_buf = std::cout.rdbuf(captured.rdbuf());
  info("test message");
  std::cout.rdbuf(old_buf);
  CHECK(captured.str().find("[i] Info:") != std::string::npos);
  CHECK(captured.str().find("test message") != std::string::npos);
}

TEST_CASE("warning output contains Warning marker", "[console]") {
  std::ostringstream captured;
  auto* old_buf = std::cout.rdbuf(captured.rdbuf());
  warning("test warning");
  std::cout.rdbuf(old_buf);
  CHECK(captured.str().find("[?] Warning:") != std::string::npos);
}

TEST_CASE("error output contains Error marker", "[console]") {
  std::ostringstream captured;
  auto* old_buf = std::cout.rdbuf(captured.rdbuf());
  error("test error");
  std::cout.rdbuf(old_buf);
  CHECK(captured.str().find("[!] Error:") != std::string::npos);
}

TEST_CASE("debug output contains Debug marker", "[console]") {
  std::ostringstream captured;
  auto* old_buf = std::cout.rdbuf(captured.rdbuf());
  debug("test debug");
  std::cout.rdbuf(old_buf);
  CHECK(captured.str().find("[+] Debug:") != std::string::npos);
}

TEST_CASE("color constants are non-empty ANSI codes", "[console]") {
  CHECK_FALSE(color::RED.empty());
  CHECK_FALSE(color::GREEN.empty());
  CHECK_FALSE(color::YELLOW.empty());
  CHECK_FALSE(color::CYAN.empty());
  CHECK_FALSE(color::MAGENTA.empty());
  CHECK_FALSE(color::RESET.empty());
}
