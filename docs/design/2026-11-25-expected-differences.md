# Expected differences between split and central control, and how each is tested

Date: 2026-11-25. Issue: #10. Decided by the maintainer; this record was written with an AI assistant.

erg does not claim that split control (one controller per leg) is better than one central policy overall.
It may claim a specific difference when that difference was written down here **before** the experiment, was
measured under the stated conditions, and is reported together with what split control costs or loses.
This record lists those differences, the advantages and the disadvantages alike.

## Forms compared

- **Split**: one microcontroller per leg, reading and commanding its own three servos on its own bus; legs exchange
  messages with their neighbours (the ring front-left, front-right, hind-right, hind-left), as in the robot design
  record ([2026-11-24](2026-11-24-robot-and-comparison.md)). In simulation, the other link settings (none, all) are
  also run.
- **Central**: one microcontroller with all 12 servos on one bus and no radio; one policy sees all sensors.

Both forms use the same body, the same sensors, the same training budget (environment steps) and at least 5
training seeds.

## When a difference counts as shown

- Simulation: the difference of the means over seeds is larger than twice its standard error, and has the expected
  sign. Otherwise the result is reported as "no difference shown", which is a result too.
- Hardware: at least 10 repeated trials per form; the same rule over trials. Quantities measured once by nature
  (cost, mass, wire length) are reported as they are, with how they were measured.

## Expected advantages of split control

| # | Expectation | What is measured | Refuted if | Reported alongside |
|---|---|---|---|---|
| 1 | Keeps standing and walking when one leg's controller or link stops | falls per minute and speed-tracking error after a controller stops (its leg holds its last position); the same when a link drops all messages | after one controller stops, split falls as often as central does when its one microcontroller stops | performance without faults; the central form's behaviour when its microcontroller stops (everything stops) |
| 2 | A cheaper microcontroller is enough | inference time per control step on each candidate board; the cheapest set of boards that meets 20 ms | the cheapest board that meets 20 ms for a leg's policy is the same class as for the central policy, so four of them cost more for nothing | total cost of four boards versus one |
| 3 | The control period can be shorter | time to read three servos versus twelve on one bus at 1 Mbps; the shortest period each form can keep on hardware | the shortest period kept by split is not shorter than central's | message delay between legs, which central does not have |
| 4 | A leg's controller can be removed and put back while running | time from power-up of the returning controller until it controls its leg again (Dyne's leave and rejoin rules); speed-tracking error during that time | the robot falls or must stop for the returning controller | how walking suffers while the leg is missing |
| 5 | A leg reacts faster to its own contact and load | time from a contact or load change (foot touch-down, a bump) to a change of that leg's joint commands | split's reaction time is not shorter than central's | — |
| 6 | A faulty sensor affects only its own leg | the deviation of each leg's commands, and falls, when one leg's sensors are frozen or given noise | the other legs' commands deviate as much under split as under central | the faulty leg's own deviation |
| 7 | Wiring and assembly are simpler | wire length and count, connectors, and assembly time of the built robot | not smaller for split | parts added by split (boards, radios, power to each board) |
| 8 | Computation and communication per controller do not grow with the number of legs | parameters, multiply-adds and bytes per control step per controller, for four and six legs (a six-legged model in simulation only) | they grow with the number of legs under split | the central policy's growth, and the total for the whole robot |

## Expected disadvantages, or quantities where either side may come out ahead

| Quantity | Expectation | What is measured |
|---|---|---|
| Power of the computers and radios | split draws more (four boards with radios on, against one without radio); expected to be a few percent of the servos' power | current of the computer supply and of the servo supply, measured separately, standing and walking |
| Cost of the on-board computers | split costs more (four boards and four IMUs) | actual parts cost |
| Fast whole-body motion | split is worse: legs see each other's state at least one control step late | sudden stop, sharp turn, recovery from a push; large jumps are likely beyond these servos and are not planned |
| Fine control | split may be better within a leg (foot position) and worse for the body as a whole (body posture) | foot position error and body posture error, separately |
| Robot mass | about the same (estimated 10–20 g more for split, about 1 %) | weighed parts of each form |

## Decisions recorded with this list (2026-11-25)

- **Where training happens.** Policies are trained on a PC, only in simulation. Data from the running robot is not
  sent to the PC; measurements to identify the servos are calibration of parts and are the one exception. Learning
  on the legs' own microcontrollers is future work.
- **Contract between neighbouring legs.** Whatever messages are lost or late, each controller's commands stay finite
  and inside the joint limits. This is checked in simulation with message loss from 0 to 100 %. How much walking
  suffers is measured, not promised.
- **A message that did not arrive.** The receiving leg uses the last value received and its age in control steps
  (capped), as inputs to its policy.
- **Use of Dyne from Python.** Dyne's virtual network (loss, delay, jitter) and message format are used through a
  thin erg crate that depends on the Dyne crates by path, built to WebAssembly and called with wasmtime, as Dyne's
  `mlp-actor` is.
- **Training compute.** Whether split control lowers the computation needed for training is studied in gauss, not
  in erg (see the [scope record](2026-11-24-scope.md)).

## Other options considered

- **Measure everything and decide afterwards what to claim.** Simpler, but a claim picked after seeing the results
  is not evidence.
- **Make no claims at all.** Safe, but the maintainer wants the advantages that do appear to be stated with their
  grounds.
