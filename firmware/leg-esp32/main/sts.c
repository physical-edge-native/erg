#include "sts.h"

uint8_t sts_checksum(const uint8_t *packet, size_t len)
{
    /* sum from the id to the last parameter (skips the two 0xFF and the checksum itself) */
    unsigned sum = 0;
    for (size_t i = 2; i + 1 < len; i++) sum += packet[i];
    return (uint8_t)(~sum & 0xFF);
}

static size_t finish(uint8_t *buf, size_t len)
{
    buf[len - 1] = sts_checksum(buf, len);
    return len;
}

static size_t header(uint8_t *buf, size_t cap, uint8_t id, uint8_t inst, size_t nparams)
{
    size_t len = STS_PACKET_LEN(nparams);
    if (len > cap || nparams + 2 > 255) return 0;
    buf[0] = 0xFF;
    buf[1] = 0xFF;
    buf[2] = id;
    buf[3] = (uint8_t)(nparams + 2);
    buf[4] = inst;
    return len;
}

size_t sts_ping(uint8_t *buf, size_t cap, uint8_t id)
{
    size_t len = header(buf, cap, id, STS_INST_PING, 0);
    return len ? finish(buf, len) : 0;
}

size_t sts_read(uint8_t *buf, size_t cap, uint8_t id, uint8_t addr, uint8_t n)
{
    size_t len = header(buf, cap, id, STS_INST_READ, 2);
    if (!len) return 0;
    buf[5] = addr;
    buf[6] = n;
    return finish(buf, len);
}

static void put_goal(uint8_t *p, uint16_t position)
{
    p[0] = (uint8_t)(position & 0xFF);
    p[1] = (uint8_t)(position >> 8);
    p[2] = p[3] = p[4] = p[5] = 0; /* goal time, goal speed */
}

size_t sts_write_goal(uint8_t *buf, size_t cap, uint8_t id, uint16_t position)
{
    size_t len = header(buf, cap, id, STS_INST_WRITE, 7);
    if (!len) return 0;
    buf[5] = STS_ADDR_GOAL_POSITION;
    put_goal(buf + 6, position);
    return finish(buf, len);
}

size_t sts_sync_write_goal(uint8_t *buf, size_t cap, const uint8_t *ids, const uint16_t *positions, size_t n)
{
    size_t len = header(buf, cap, STS_BROADCAST_ID, STS_INST_SYNC_WRITE, 2 + 7 * n);
    if (!len) return 0;
    buf[5] = STS_ADDR_GOAL_POSITION;
    buf[6] = 6;
    for (size_t i = 0; i < n; i++) {
        buf[7 + 7 * i] = ids[i];
        put_goal(buf + 8 + 7 * i, positions[i]);
    }
    return finish(buf, len);
}

size_t sts_sync_write_u16(uint8_t *buf, size_t cap, uint8_t addr, const uint8_t *ids, const uint16_t *values, size_t n)
{
    size_t len = header(buf, cap, STS_BROADCAST_ID, STS_INST_SYNC_WRITE, 2 + 3 * n);
    if (!len) return 0;
    buf[5] = addr;
    buf[6] = 2;
    for (size_t i = 0; i < n; i++) {
        buf[7 + 3 * i] = ids[i];
        buf[8 + 3 * i] = (uint8_t)(values[i] & 0xFF);
        buf[9 + 3 * i] = (uint8_t)(values[i] >> 8);
    }
    return finish(buf, len);
}

int16_t sts_sign_magnitude(uint16_t raw, unsigned sign_bit)
{
    uint16_t mag = raw & (uint16_t)((1u << sign_bit) - 1u);
    return (raw & (1u << sign_bit)) ? (int16_t)-(int16_t)mag : (int16_t)mag;
}

int sts_parse_state(const uint8_t *reply, size_t len, uint8_t id, sts_state_t *out)
{
    if (len < 6 || reply[0] != 0xFF || reply[1] != 0xFF) return -1;
    if (reply[2] != id) return -2;
    if (reply[3] != 8 || len != 12) return -3; /* error + 6 data + checksum */
    if (sts_checksum(reply, len) != reply[len - 1]) return -4;
    out->error = reply[4];
    out->position = (uint16_t)(reply[5] | (reply[6] << 8));
    out->speed = sts_sign_magnitude((uint16_t)(reply[7] | (reply[8] << 8)), 15);
    out->load = sts_sign_magnitude((uint16_t)(reply[9] | (reply[10] << 8)), 10);
    return 0;
}
