# Console: terminal formatting utilities

## Purpose

This example demonstrates the ANSI terminal formatting utilities provided by
the `console` namespace. These utilities are used throughout the library and
all docs to produce structured, colour-coded terminal output.

The `console` module wraps ANSI escape codes (SGR sequences) in a simple API
so that log messages are visually distinguishable by severity, and text can
be styled without manually constructing escape sequences.

## ANSI escape codes

ANSI SGR (Select Graphic Rendition) codes control text attributes in
terminals that support them. The general format is:

```
ESC[<code>m
```

where `ESC` is the byte `0x1B` (or `\033` in C). The codes used here are:

| Code | Effect |
|------|--------|
| `0` | Reset all attributes |
| `1` | Bold / increased intensity |
| `5` | Blink (slow) |
| `32` | Green foreground (info) |
| `33` | Yellow foreground (warning) |
| `31` | Red foreground (error) |
| `36` | Cyan foreground (debug) |

Each `console::info()`, `console::warning()`, `console::error()`, and
`console::debug()` call wraps the message in the corresponding colour code
and appends the reset code `ESC[0m` at the end, so it does not bleed into
subsequent output.

---

## Step-by-step code walkthrough

```cpp
console::info("Hello world");         // "[INFO] Hello world"  in green
console::warning("This is a warning"); // "[WARNING] ..."       in yellow
console::error("This is an error!!");  // "[ERROR] ..."         in red
console::debug("This is a debugging message"); // "[DEBUG] ..." in cyan

console::write_line(console::format::bold("Bold text"));
console::write_line(console::format::blink("Blinking"));
```

- `console::info(msg)` — prefixes with `[INFO]`, prints in green.
- `console::warning(msg)` — prefixes with `[WARNING]`, prints in yellow.
- `console::error(msg)` — prefixes with `[ERROR]`, prints in red.
- `console::debug(msg)` — prefixes with `[DEBUG]`, prints in cyan.
- `console::write_line(msg)` — prints a line without a severity prefix.
- `console::format::bold(msg)` — wraps the text in bold SGR codes.
- `console::format::blink(msg)` — wraps the text in blink SGR codes
  (terminal support varies).

## Build and run

```bash
make run-local
```
