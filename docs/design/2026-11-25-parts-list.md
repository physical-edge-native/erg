# Parts list for the first robot

Date: 2026-11-25, revised 2026-11-27. Issues: #23, #31 (both stages bought together; parts to run the robot), #60
(12 V class servos and the redesign). Decided by the maintainer; this record was written with an AI assistant.

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
| Power for stage 1 | A bench power supply with an adjustable current limit, set to 7.4 V: a 0-30 V, 0-10 A supply bought in Japan (any model with an adjustable current limit; for example OWON SPE3103) | The servo's stall current is not clear from the translated datasheet (both 650 mA and 2.5 A appear); the limit keeps the first runs safe, and the same supply measures the current needed to choose the battery. 10 A covers twelve servos at the larger stall current reading only with the current limit on; walking draws far less than stall |
| Power on the robot (#31) | Two-cell lithium-polymer battery, 7.4 V 2200 mAh, two packs (Gens Ace 60C, XT60), and a two-cell balance charger (G-FORCE G2, 0.8 A, about 3 h per pack) | Matches the servos' 7.4 V rating directly. Two packs let one charge while the other runs. The capacity is a first choice, to be checked against the current measured on the bench |
| Microcontroller supply (#31) | One Pololu D24V10F5 step-down regulator (5 V, 1 A, input 5.1-36 V) per leg | Works over the whole range of a two-cell battery (8.4 V full, about 6.4 V empty); 1 A leaves room for the radio's current peaks. Modules found in Japanese shops needed at least 7.5-8 V in, or gave only 300 mA |
| Foot switch (#31) | Omron SS-5GLD subminiature microswitch with lever | Small enough for the foot (the shank ends in a 12 mm sphere); the lever gives a light touch. The shop notes that delivery can be slow, so it is ordered first |

## Revised for the redesign (2026-11-27, #60)

The redesign ([2026-11-27-redesign.md](2026-11-27-redesign.md)) chose 12 V class servos and body L1; nothing is
ordered before this revision. The rows of the 2026-11-25 table above that name the STS3215, the 7.4 V supply, the
2-cell battery or its charger are replaced by these:

| Point | Revision | Reason |
|---|---|---|
| Servo | Waveshare ST3215, 12 V class (30 kg cm, 0.222 s per 60°, 6-12.6 V; Waveshare SKU 22414), 12 plus **2 spares** (1 before) | The redesign's choice. The seller's values are what the model uses (erg/design.py) and are checked on a real servo; one spare is for the bench identification of the servo, so that a broken one still leaves one spare |
| Second source of the servo | DFRobot ST3215 (SER0070) | Same name and rating; that it is the same servo is to be confirmed, as before |
| Battery | 3-cell lithium-polymer, 11.1 V 1300 mAh, two packs (G-FORCE Noir GFG910: 86 × 30 × 17 mm, 99 g), bought in Japan | 3 cells give 10.5-12.6 V between nearly empty and full, inside the servo's range and the servo bus adapter's labelled 9-12.6 V. The model's battery is an estimate of 120 g in a 35 × 75 × 25 mm box lying across the torso; this pack is lighter and fits the torso's inside width (about 98 mm), but is 86 mm long instead of 75 mm, so its box is checked in the torso's CAD. Bought in Japan because lithium batteries are hard to ship by air |
| Battery connector | The pack has a small (mini) connector; an adapter to XT60 is in the small parts | The robot's wiring keeps XT60 as before |
| Charger | G-FORCE G3 DUO (2-3 cells, two packs at once) | The 2-cell charger cannot charge 3-cell packs; two at once matches two packs |
| Bench power supply | Unchanged, set to 12 V with its current limit | It reaches 30 V |
| Microcontroller supply | Unchanged (Pololu D24V10F5, 5.1-36 V in) | Works over the 3-cell range |
| Servo bus adapter | Unchanged (Waveshare Bus Servo Adapter (A)) | Labelled for 9-12.6 V, the ST series' voltage |
| Push rods | M2 threaded rod and two ball links or rod ends per leg (estimate, one lot per stage) | The knee is driven by a push rod in body L1 |
| Foot caps | Rubber or silicone caps over the 12 mm foot sphere (estimate), or printed in TPU | Grip and some impact absorption; which one follows the test print |
| Foot force sensors (optional, stage `option`) | Round force-sensitive resistors, 0.3-10 N (Alpha MF01A-N-221-A01), 4 plus 2 spares | The chosen tier lists foot force sensors; the foot switch already gives contact, so they are bought only if the maintainer decides to. Not counted by the tests |

Prices of the new rows were checked on 2026-11-27 (column `checked`); the rows kept from 2026-11-25 keep that date.

## How to read `hardware/bom.csv`

- `stage`: 1 or 2, as above; `option` for parts not bought unless decided (not counted by the tests).
- `role`: what the part is for. The test counts `servo`, `microcontroller`, `imu`, `foot_switch`, `bus_adapter` and
  `regulator` against the model, and checks that `power_supply`, `battery` and `charger` are listed and that the
  battery's cell count suits the 12 V class servos.
- `spare`: `yes` for spares, which the test does not count.
- `reference_price`, `currency`, `tax_included`: the list price on the shop page named in `source`, per unit, on the
  date in `checked`. Shipping is not included. These are references for planning, not the prices paid.
- `second_source`: another shop or brand for the same part, where one is known. For the servo, DFRobot sells a 12 V
  servo named ST3215 with the same rating; **that it is the same servo is to be confirmed before buying it**, because
  the MuJoCo model uses the Waveshare listing's values.
- `small_parts` rows (screws, wires, connectors) are estimates without a shop. The servos come with horn screws
  and an accessory kit; M2 self-tapping screws for the frame are bought only if the kit has too few.

Totals at the reference prices, without shipping (revised 2026-11-27): stage 1 about ¥19,100 plus about $141; stage 2
about ¥23,400 plus about $291. Both stages: about ¥42,500 plus about $431, about ¥107,000 at about ¥150 per US dollar
(the redesign's estimate for the chosen tier was ¥105,000-125,000). The optional foot force sensors add about $24.
Before the revision (7.4 V servos): about ¥94,000. The bench power
supply is bought in Japan (maintainer's choice, 2026-11-25); its price is an estimate from listings of similar
30 V, 10 A supplies (about ¥9,500-12,000), the upper end is used.

## Not yet decided

- **Power wiring** on the robot (how the battery feeds four servo bus adapters and four regulators), designed with
  the torso frame.
