/*
 * Feetech STS serial bus protocol for the STS3215 servo (Issue #54). No ESP-IDF dependency: tested on the PC
 * (firmware/leg-esp32/host_test, tests/test_firmware_host.py).
 *
 * Packet: 0xFF 0xFF id length instruction parameters... checksum, with length = number of parameters + 2 and
 * checksum = ~(id + length + instruction + parameters) & 0xFF. Reply: 0xFF 0xFF id length error data... checksum.
 * Two-byte values are little-endian. Speed and load are sign-magnitude: bit 15 (speed) or bit 10 (load) is the sign.
 *
 * Register addresses from Feetech's STS memory table, to be checked on a real servo before relying on them.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#define STS_INST_PING 0x01
#define STS_INST_READ 0x02
#define STS_INST_WRITE 0x03
#define STS_INST_SYNC_WRITE 0x83
#define STS_BROADCAST_ID 0xFE

#define STS_ADDR_GOAL_POSITION 0x2A  /* goal position (2), goal time (2), goal speed (2) */
#define STS_ADDR_PRESENT_POSITION 0x38 /* present position (2), speed (2), load (2) */
#define STS_ADDR_TORQUE_LIMIT 0x30     /* torque limit (2, RAM): tenths of a percent of the maximum torque, 0..1000 */
#define STS_STEPS_PER_TURN 4096

/* Length of a packet with n parameters. */
#define STS_PACKET_LEN(n) ((n) + 6)

uint8_t sts_checksum(const uint8_t *packet, size_t len);

/* Each writes a packet into buf (of capacity cap) and returns its length, or 0 if it does not fit. */
size_t sts_ping(uint8_t *buf, size_t cap, uint8_t id);
size_t sts_read(uint8_t *buf, size_t cap, uint8_t id, uint8_t addr, uint8_t n);
/* Goal position (steps), with goal time and speed 0 (move as fast as allowed). */
size_t sts_write_goal(uint8_t *buf, size_t cap, uint8_t id, uint16_t position);
/* One packet setting the goal positions of n servos. */
size_t sts_sync_write_goal(uint8_t *buf, size_t cap, const uint8_t *ids, const uint16_t *positions, size_t n);
/* One packet setting a two-byte register at addr in n servos (broadcast: the servos do not reply). */
size_t sts_sync_write_u16(uint8_t *buf, size_t cap, uint8_t addr, const uint8_t *ids, const uint16_t *values, size_t n);

typedef struct {
    uint16_t position; /* steps, 0..4095 */
    int16_t speed;     /* steps per second, signed */
    int16_t load;      /* tenths of a percent of the maximum torque, signed, -1000..1000 */
    uint8_t error;     /* the servo's error bits */
} sts_state_t;

/* Parse the reply to sts_read(id, STS_ADDR_PRESENT_POSITION, 6). Returns 0, or a negative code:
 * -1 too short or bad header, -2 wrong id, -3 wrong length, -4 bad checksum. */
int sts_parse_state(const uint8_t *reply, size_t len, uint8_t id, sts_state_t *out);

int16_t sts_sign_magnitude(uint16_t raw, unsigned sign_bit);
