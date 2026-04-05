#include <dftlib>

using namespace dft;

int main() {
  console::info("Hello world");
  console::warning("This is a warning");
  console::error("This is an error!!");
  console::debug("This is a debugging message");
  console::write_line(console::format::bold("Bold text"));
  console::write_line(console::format::blink("Blinking"));
}
