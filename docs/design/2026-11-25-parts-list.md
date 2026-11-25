# Parts list for the first robot

Date: 2026-11-25. Issues: #23, #31 (both stages bought together; parts to run the robot). Decided by the maintainer; this record was written with an AI assistant.

The list itself is [`hardware/bom.csv`](../../hardware/bom.csv), one row per part. It follows the robot design record
([2026-11-24-robot-and-comparison.md](2026-11-24-robot-and-comparison.md)). `tests/test_parts_list.py` checks that
its quantities match the MuJoCo model.

## Decided (2026-11-25)

| Point | Decision | Reason |
|---|---|---|
| Order of purchase | **Stage 1**: one leg plus spares (3 servos and 1 spare, 1 microcontroller, 1 IMU, 1 servo bus adapter, 1 foot switch and 1 spare). **Stage 2**: the other three legs. **Changed on 2026-11-25 (#31): both stages are bought together, within a few days** | First: the first hardware measurements need only one leg, and the choice of microcontroller may change if inference does not fit in the 20 ms control period. Changed because the maintainer aims to show the robot walking with learned split control in December 2026. The risk accepted: if inference does not fit on the XIAO ESP32S3, four boards (about ¥5,800) may have to be replaced |
| IMU board | Adafruit LSM6DSOX (product 4438) | Same chip (LSM6DSOX) as Dyne's IMU record of 2026-11-21, so its analysis applies; I2C with a plug-in cable (STEMMA QT / Qwiic) |
| Servo bus interface | Waveshare Bus Servo Adapter (A), one per leg | The STS3215 sends and receives on one wire; this board already has the circuit that switches between the two, and passes the servo power through |
| Frame | Printed in PLA on a 3D printer available to the maintainer (Bambu Lab X2D); changed on 2026-11-25 from an outside printing service | The printer became available; only PLA is at hand. The frame is designed in #25 ([2026-11-25-first-leg-frame.md](2026-11-25-first-leg-frame.md)) |
| Where the list lives | In this public repository, with list prices and their sources | Anyone can rebuild the robot from it. Prices actually paid are kept private |
| Availability | Prefer parts that are unlikely to go out of stock; give a second source where one exists | Stock of the servo at one Japanese shop was seen both in stock and sold out on the same day |
| Power for stage 1 | A bench power supply with an adjustable current limit, set to 7.4 V: OWON SPE3103 (0-30 V, 0-10 A) | The servo's stall current is not clear from the translated datasheet (both 650 mA and 2.5 A appear); the limit keeps the first runs safe, and the same supply measures the current needed to choose the battery. 10 A covers twelve servos at the larger stall current reading only with the current limit on; walking draws far less than stall |
| Power on the robot (#31) | Two-cell lithium-polymer battery, 7.4 V 2200 mAh, two packs (Gens Ace 60C, XT60), and a two-cell balance charger (G-FORCE G2, 0.8 A, about 3 h per pack) | Matches the servos' 7.4 V rating directly. Two packs let one charge while the other runs. The capacity is a first choice, to be checked against the current measured on the bench |
| Microcontroller supply (#31) | One Pololu D24V10F5 step-down regulator (5 V, 1 A, input 5.1-36 V) per leg | Works over the whole range of a two-cell battery (8.4 V full, about 6.4 V empty); 1 A leaves room for the radio's current peaks. Modules found in Japanese shops needed at least 7.5-8 V in, or gave only 300 mA |
| Foot switch (#31) | Omron SS-5GLD subminiature microswitch with lever | Small enough for the foot (the shank ends in a 12 mm sphere); the lever gives a light touch. The shop notes that delivery can be slow, so it is ordered first |

## How to read `hardware/bom.csv`

- `stage`: 1 or 2, as above.
- `role`: what the part is for. The test counts `servo`, `microcontroller`, `imu`, `foot_switch`, `bus_adapter` and
  `regulator` against the model, and checks that `power_supply`, `battery` and `charger` are listed.
- `spare`: `yes` for spares, which the test does not count.
- `reference_price`, `currency`, `tax_included`: the list price on the shop page named in `source`, per unit, on the
  date in `checked`. Shipping is not included. These are references for planning, not the prices paid.
- `second_source`: another shop or brand for the same part, where one is known. For the servo, Waveshare sells a
  7.4 V servo named ST3215 with the same torque rating; **that it is the same servo as the Feetech STS3215 is to be
  confirmed before buying it**, because the MuJoCo model uses the STS3215's datasheet values.
- `small_parts` rows (screws, wires, connectors) are estimates without a shop. The servos come with horn screws
  and an accessory kit; M2 self-tapping screws for the frame are bought only if the kit has too few.

Totals at the reference prices, without shipping: stage 1 about ¥17,800 plus about $150 (of which the bench power
supply is $119); stage 2 about ¥38,700 plus about $140. Both stages: about ¥56,500 plus about $290, about ¥100,000
at about ¥150 per US dollar. The bench power supply can also be bought in Japan for about ¥9,500-12,000 (similar
30 V, 10 A supplies); either is fine as long as it has a current limit.

## Not yet decided

- **Power wiring** on the robot (how the battery feeds four servo bus adapters and four regulators), designed with
  the torso frame.
