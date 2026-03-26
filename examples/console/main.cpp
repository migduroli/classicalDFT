#include "dft_lib/dft_lib.h"

int main(int argc, char **argv)
{
  console::Info("Hello world");
  console::Warning("This is a warning");
  console::Error("This is an error!!");
  console::Debug("This is a debugging message");
  console::WriteLine(console::format::Bold("Bold text"));
  console::WriteLine(console::format::Blink("Blinking"));
  console::Wait();
}