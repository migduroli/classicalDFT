#ifndef DFT_CONSOLE_HPP
#define DFT_CONSOLE_HPP

#include <chrono>
#include <iomanip>
#include <iostream>
#include <print>
#include <string>

namespace dft::console {

  namespace color {

    inline const std::string RED{"\033[0;31m"};
    inline const std::string GREEN{"\033[1;32m"};
    inline const std::string YELLOW{"\033[1;33m"};
    inline const std::string CYAN{"\033[0;36m"};
    inline const std::string MAGENTA{"\033[0;35m"};
    inline const std::string RESET{"\033[0m"};

  }  // namespace color

  namespace format {

    [[nodiscard]] inline auto bold(const std::string& msg) -> std::string {
      return "\x1b[1m" + msg + "\x1b[0m";
    }

    [[nodiscard]] inline auto blink(const std::string& msg) -> std::string {
      return "\033[33;5;7m" + msg + "\033[0m";
    }

  }  // namespace format

  template <class T>
  void write(const T& msg) {
    std::print(std::cout, "{}", msg);
  }

  template <class T>
  void write_line(const T& msg) {
    std::println(std::cout, "{}", msg);
  }

  template <class T>
  void write_line(const std::initializer_list<T>& msg) {
    for (const auto& m : msg) {
      std::println(std::cout, "{}", m);
    }
  }

  inline void new_line() { std::println(std::cout, ""); }

  inline void pause() { std::cin.ignore(); }

  [[nodiscard]] inline auto read_line() -> std::string {
    std::string out;
    std::cin >> out;
    return out;
  }

  inline void wait() {
    write_line("Press enter to continue...");
    std::cin.ignore();
  }

  [[nodiscard]] inline auto now_str() -> std::string {
    std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::string s(19, '\0');
    std::strftime(&s[0], s.size(), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return s;
  }

  inline void info(const std::string& msg) {
    write_line(color::GREEN + now_str() + " | " + "[i] Info: " + msg + color::RESET);
  }

  inline void warning(const std::string& msg) {
    write_line(color::YELLOW + now_str() + " | " + "[?] Warning: " + msg + color::RESET);
  }

  inline void error(const std::string& msg) {
    write_line(color::RED + now_str() + " | " + "[!] Error: " + msg + color::RESET);
  }

  inline void debug(const std::string& msg) {
    write_line(color::CYAN + now_str() + " | " + "[+] Debug: " + msg + color::RESET);
  }

}  // namespace dft::console

#endif  // DFT_CONSOLE_HPP
