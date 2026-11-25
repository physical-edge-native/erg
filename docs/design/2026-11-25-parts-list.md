# Parts list for the first robot

Date: 2026-11-25. Issue: #23. Decided by the maintainer; this record was written with an AI assistant.

The list itself is [`hardware/bom.csv`](../../hardware/bom.csv), one row per part. It follows the robot design record
([2026-11-24-robot-and-comparison.md](2026-11-24-robot-and-comparison.md)). `tests/test_parts_list.py` checks that
its quantities match the MuJoCo model.

## Decided (2026-11-25)

| Point | Decision | Reason |
|---|---|---|
| Order of purchase | **Stage 1**: one leg plus spares (3 servos and 1 spare, 1 microcontroller, 1 IMU, 1 servo bus adapter, 1 foot switch and 1 spare). **Stage 2**: the other three legs | The first hardware measurements (time to read three servos on one bus; inference time of Dyne's learned solver on the microcontroller) need only one leg. If inference does not fit in the 20 ms control period, the choice of microcontroller may change before stage 2 |
| IMU board | Adafruit LSM6DSOX (product 4438) | Same chip (LSM6DSOX) as Dyne's IMU record of 2026-11-21, so its analysis applies; I2C with a plug-in cable (STEMMA QT / Qwiic) |
| Servo bus interface | Waveshare Bus Servo Adapter (A), one per leg | The STS3215 sends and receives on one wire; this board already has the circuit that switches between the two, and passes the servo power through |
| Frame | Printed in PLA on a 3D printer available to the maintainer (Bambu Lab X2D); changed on 2026-11-25 from an outside printing service | The printer became available; only PLA is at hand. The frame is designed in #25 ([2026-11-25-first-leg-frame.md](2026-11-25-first-leg-frame.md)) |
| Where the list lives | In this public repository, with list prices and their sources | Anyone can rebuild the robot from it. Prices actually paid are kept private |
| Availability | Prefer parts that are unlikely to go out of stock; give a second source where one exists | Stock of the servo at one Japanese shop was seen both in stock and sold out on the same day |
| Power for stage 1 | A bench power supply with an adjustable current limit, set to 7.4 V | The servo's stall current is not clear from the translated datasheet (both 650 mA and 2.5 A appear); the limit keeps the first runs safe, and the same supply measures the current needed to choose the battery |

## How to read `hardware/bom.csv`

- `stage`: 1 or 2, as above.
- `role`: what the part is for. The test counts `servo`, `microcontroller`, `imu`, `foot_switch` and `bus_adapter`.
- `spare`: `yes` for spares, which the test does not count.
- `reference_price`, `currency`, `tax_included`: the list price on the shop page named in `source`, per unit, on the
  date in `checked`. Shipping is not included. These are references for planning, not the prices paid.
- `second_source`: another shop or brand for the same part, where one is known. For the servo, Waveshare sells a
  7.4 V servo named ST3215 with the same torque rating; **that it is the same servo as the Feetech STS3215 is to be
  confirmed before buying it**, because the MuJoCo model uses the STS3215's datasheet values.
- The microswitch price is the estimate of the design record; the exact part and shop are not chosen yet.

Totals at the reference prices, without shipping: stage 1 about ¥15,000 plus about $17; stage 2 about ¥34,500
plus about $51.

## Not yet decided

- **Bench power supply model** (voltage and current range) and the **microswitch part number**: to be added to
  `hardware/bom.csv` once chosen.
- **Battery size, power wiring, cables and connectors** for the whole robot.
