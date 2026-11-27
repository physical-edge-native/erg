/*
 * Runs the firmware's pure parts (sts.c, leg.c) on the PC for tests/test_firmware_host.py (Issue #54).
 * Reads one command per line on stdin, writes one line per command on stdout. Doubles are passed and printed as
 * the 16 hex digits of their IEEE 754 bits, so the comparison with Python is bit for bit.
 *
 *   ping ID | read ID ADDR N | goal ID POS | sync N ID POS ...   -> packet bytes in hex
 *   parse ID HEXBYTES                                           -> rc position speed load error
 *   angle ZERO SIGN POS | speed ZERO SIGN SPEED | load ZERO SIGN LOAD -> double
 *   goalangle ZERO SIGN LO HI ANGLE                             -> steps
 *   imu A0 A1 A2 S0 S1 S2 AX AY AZ GX GY GZ                     -> 6 doubles (acc, gyro)
 *   obs LEG CONTACT (ZERO SIGN POS SPEED LOAD) x3 ACC3 GYRO3 PHASE CMD3 -> OBS_SIZE doubles
 *   targets A0..A7 PHASE FORWARD                                -> 3 doubles
 *   clock PHASE F | base PHASE STRIDE | phasevals PHASE          -> 1, 3 or 2 doubles
 *   sincos X | atan2 Y X | acos C                               -> 2, 1 or 1 doubles
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../main/leg.h"
#include "../main/sts.h"
#include "../main/xmath.h"

static double rd(void)
{
    char s[32];
    if (scanf("%31s", s) != 1) exit(2);
    unsigned long long u = strtoull(s, NULL, 16);
    double d;
    memcpy(&d, &u, 8);
    return d;
}

static long ri(void)
{
    long v;
    if (scanf("%ld", &v) != 1) exit(2);
    return v;
}

static void pd(double d)
{
    unsigned long long u;
    memcpy(&u, &d, 8);
    printf(" %016llx", u);
}

static void pbytes(const uint8_t *b, size_t n)
{
    for (size_t i = 0; i < n; i++) printf("%02x", b[i]);
}

static leg_joint_cal_t cal(long zero, long sign)
{
    leg_joint_cal_t c = {(uint16_t)zero, (int8_t)sign, -10.0, 10.0};
    return c;
}

int main(void)
{
    char cmd[32];
    uint8_t buf[256];
    while (scanf("%31s", cmd) == 1) {
        if (!strcmp(cmd, "ping")) {
            pbytes(buf, sts_ping(buf, sizeof buf, (uint8_t)ri()));
        } else if (!strcmp(cmd, "read")) {
            long id = ri(), addr = ri(), n = ri();
            pbytes(buf, sts_read(buf, sizeof buf, (uint8_t)id, (uint8_t)addr, (uint8_t)n));
        } else if (!strcmp(cmd, "goal")) {
            long id = ri(), pos = ri();
            pbytes(buf, sts_write_goal(buf, sizeof buf, (uint8_t)id, (uint16_t)pos));
        } else if (!strcmp(cmd, "sync")) {
            long n = ri();
            uint8_t ids[16];
            uint16_t pos[16];
            for (long i = 0; i < n; i++) {
                ids[i] = (uint8_t)ri();
                pos[i] = (uint16_t)ri();
            }
            pbytes(buf, sts_sync_write_goal(buf, sizeof buf, ids, pos, (size_t)n));
        } else if (!strcmp(cmd, "parse")) {
            long id = ri();
            char hex[512];
            if (scanf("%511s", hex) != 1) return 2;
            size_t n = strlen(hex) / 2;
            for (size_t i = 0; i < n; i++) {
                unsigned v;
                sscanf(hex + 2 * i, "%2x", &v);
                buf[i] = (uint8_t)v;
            }
            sts_state_t s = {0};
            int rc = sts_parse_state(buf, n, (uint8_t)id, &s);
            printf("%d %u %d %d %u", rc, s.position, s.speed, s.load, s.error);
        } else if (!strcmp(cmd, "angle") || !strcmp(cmd, "speed") || !strcmp(cmd, "load")) {
            long zero = ri(), sign = ri(), v = ri();
            leg_joint_cal_t c = cal(zero, sign);
            double d = !strcmp(cmd, "angle") ? leg_joint_angle(&c, (uint16_t)v)
                       : !strcmp(cmd, "speed") ? leg_joint_speed(&c, (int16_t)v)
                                               : leg_joint_load(&c, (int16_t)v);
            pd(d);
        } else if (!strcmp(cmd, "goalangle")) {
            long zero = ri(), sign = ri();
            leg_joint_cal_t c = cal(zero, sign);
            c.lo = rd();
            c.hi = rd();
            printf("%u", leg_goal(&c, rd()));
        } else if (!strcmp(cmd, "imu")) {
            leg_imu_mount_t m;
            for (int i = 0; i < 3; i++) m.axis[i] = (uint8_t)ri();
            for (int i = 0; i < 3; i++) m.sign[i] = (int8_t)ri();
            int16_t a[3], g[3];
            for (int i = 0; i < 3; i++) a[i] = (int16_t)ri();
            for (int i = 0; i < 3; i++) g[i] = (int16_t)ri();
            double acc[3], gyro[3];
            leg_imu(&m, a, g, acc, gyro);
            for (int i = 0; i < 3; i++) pd(acc[i]);
            for (int i = 0; i < 3; i++) pd(gyro[i]);
        } else if (!strcmp(cmd, "obs")) {
            int leg = (int)ri();
            int contact = (int)ri();
            leg_joint_cal_t c[ACT_JOINTS];
            sts_state_t s[ACT_JOINTS];
            for (int j = 0; j < ACT_JOINTS; j++) {
                long zero = ri(), sign = ri();
                c[j] = cal(zero, sign);
                s[j].position = (uint16_t)ri();
                s[j].speed = (int16_t)ri();
                s[j].load = (int16_t)ri();
                s[j].error = 0;
            }
            double acc[3], gyro[3], command[3], obs[OBS_SIZE];
            for (int i = 0; i < 3; i++) acc[i] = rd();
            for (int i = 0; i < 3; i++) gyro[i] = rd();
            double phase = rd();
            for (int i = 0; i < 3; i++) command[i] = rd();
            leg_observation(obs, c, s, acc, gyro, contact != 0, phase, command, leg);
            for (int i = 0; i < OBS_SIZE; i++) pd(obs[i]);
        } else if (!strcmp(cmd, "targets")) {
            double a[ACT_SIZE_ALL], t[ACT_JOINTS];
            for (int j = 0; j < ACT_SIZE_ALL; j++) a[j] = rd();
            double phase = rd(), forward = rd();
            leg_targets(a, phase, forward, t);
            for (int j = 0; j < ACT_JOINTS; j++) pd(t[j]);
        } else if (!strcmp(cmd, "clock")) {
            double p = rd(), f = rd();
            pd(leg_clock(p, f));
        } else if (!strcmp(cmd, "base")) {
            double p = rd(), stride = rd(), out[ACT_JOINTS];
            leg_base(p, stride, out);
            for (int j = 0; j < ACT_JOINTS; j++) pd(out[j]);
        } else if (!strcmp(cmd, "phasevals")) {
            double c, sn;
            leg_phase_values(rd(), &c, &sn);
            pd(c);
            pd(sn);
        } else if (!strcmp(cmd, "sincos")) {
            double sn, c;
            xm_sincos(rd(), &sn, &c);
            pd(sn);
            pd(c);
        } else if (!strcmp(cmd, "atan2")) {
            double y = rd(), x = rd();
            pd(xm_atan2(y, x));
        } else if (!strcmp(cmd, "acos")) {
            pd(xm_acos(rd()));
        } else {
            return 3;
        }
        printf("\n");
        fflush(stdout);
    }
    return 0;
}
