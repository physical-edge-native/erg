# erg

Robot control split over the parts of a robot body, built on
[Dyne](https://github.com/physical-edge-native/dyne) and the
[Physical-Edge Native Architecture](https://physical-edge-native.github.io/).

**Status: just started. Nothing here runs yet.**

## What this repository is for

A legged robot is usually controlled by one policy that sees the whole body.
erg designs a four-legged robot whose legs each carry their own small controller.
Each controller sees only its own leg's sensors and the values exchanged with its
neighbours, in the same way that Dyne splits a physical field into regions that
exchange only the values on their borders.

The aim is to measure how this differs from a central policy on the same body:
walking performance, computation and communication per part, and behaviour under
message loss, delay, and parts that stop or come back while running. We make no
claim that the split controller is better; the comparison is the result.

Plan (subject to change):

- simulation in MuJoCo first, on CPUs; the robot description is kept loadable by
  Isaac Lab for later use;
- the robot is designed so that it can be built from hobby servos and ESP32-class
  microcontrollers;
- leg controllers are trained in Python and run as Dyne WebAssembly solvers, so the
  same bits run on a PC and on a microcontroller.

Work is tracked in GitHub Issues.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
