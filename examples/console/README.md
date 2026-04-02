# Console: terminal formatting utilities

Demonstrates the ANSI terminal formatting utilities in the `console` namespace.

## What this example does

Prints coloured log messages at four severity levels (info, warning, error,
debug) and applies text formatting (bold, blink). All output uses ANSI escape
codes for terminal colouring.

## Key API functions used

| Function | Purpose |
|----------|---------|
| `console::info()` | green info message |
| `console::warning()` | yellow warning |
| `console::error()` | red error |
| `console::debug()` | cyan debug message |
| `console::format::bold()` | bold text |
| `console::format::blink()` | blinking text |

## Build and run

```bash
make run
```
