/*
 * One leg's sensors to observation, and policy output to servo goals (Issue #54). No ESP-IDF dependency: tested on
 * the PC against the same definitions in Python (tests/test_firmware_host.py).
 *
 * All arithmetic is in double precision, in the order written here, so that the PC and the ESP32-S3 (which emulates
 * double precision in software, rounding as IEEE 754 requires) give the same bits.
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "obs_layout.h"
#include "sts.h"

/* Calibration of one joint, set on the device (not in the repository): the servo position that is joint angle 0,
 * and the direction (+1 or -1) from servo steps to the model's joint angle. */
typedef struct {
    uint16_t zero;
    int8_t sign;
    double lo, hi; /* joint range (rad), from the MuJoCo model */
} leg_joint_cal_t;

/* How the IMU board is mounted: torso axis i reads sign[i] * board axis axis[i]. */
typedef struct {
    uint8_t axis[3];
    int8_t sign[3];
} leg_imu_mount_t;

/* LSM6DSOX at +-4 g and +-500 degrees per second: 0.122 mg and 17.5 millidegrees per second per step. */
#define LEG_ACC_PER_STEP (0.122e-3 * 9.80665)              /* m/s^2 */
#define LEG_GYRO_PER_STEP (17.5e-3 * 3.141592653589793 / 180.0) /* rad/s */
#define LEG_RAD_PER_STEP (2.0 * 3.141592653589793 / STS_STEPS_PER_TURN)

double leg_joint_angle(const leg_joint_cal_t *c, uint16_t position);
double leg_joint_speed(const leg_joint_cal_t *c, int16_t speed);
double leg_joint_load(const leg_joint_cal_t *c, int16_t load); /* fraction of the maximum, in [-1, 1] */
/* Servo goal for a joint angle, clipped to the joint range, rounded to the nearest step. */
uint16_t leg_goal(const leg_joint_cal_t *c, double angle);

void leg_imu(const leg_imu_mount_t *m, const int16_t acc_raw[3], const int16_t gyro_raw[3], double acc[3],
             double gyro[3]);

/* The leg's gait clock (erg.env): the next phase from this one and the frequency correction (clipped to [-1, 1]). */
double leg_clock(double phase, double freq);
/* Cosine and sine of 2 pi phase (erg.env.Env.phase_values). */
void leg_phase_values(double phase, double *c, double *s);
/* The base gait's joint targets at a phase for a stride (erg.gait.leg_targets). */
void leg_base(double phase, double stride, double out[ACT_JOINTS]);

/* One leg's observation (OBS_SIZE values) with no messages received yet: every message slot reads zeros and the age
 * OBS_AGE_CAP, as in the training environment when nothing has arrived. */
void leg_observation(double obs[OBS_SIZE], const leg_joint_cal_t cal[ACT_JOINTS], const sts_state_t servo[ACT_JOINTS],
                     const double acc[3], const double gyro[3], bool contact, double phase, const double command[3],
                     int leg);

/* Joint targets (erg.ppo.Worker.step): the base gait at the phase, with the stride of the commanded forward speed,
 * + ACTION_SCALE * the policy's first ACT_JOINTS outputs. */
void leg_targets(const double action[ACT_SIZE_ALL], double phase, double forward, double targets[ACT_JOINTS]);

/* Knee torque limit near the straight end of the knee range (erg.knee_limit, Issue #81): the limit (N*m) at a knee
 * angle (rad), the torque limit register's value (tenths of a percent of the maximum torque, rounded down), and
 * whether to write a new value over the last one written (-1: none yet): at once when it lowers the limit, when it
 * raises it by KNEE_LIMIT_STEP or more, and when it returns to full. */
double leg_knee_torque_limit(double knee);
int leg_knee_limit_register(double knee);
bool leg_knee_limit_should_write(int last, int value);
