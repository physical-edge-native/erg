#include "leg.h"

#include <math.h>

#include "xmath.h"

double leg_joint_angle(const leg_joint_cal_t *c, uint16_t position)
{
    return (double)c->sign * ((double)((int)position - (int)c->zero) * LEG_RAD_PER_STEP);
}

double leg_joint_speed(const leg_joint_cal_t *c, int16_t speed)
{
    return (double)c->sign * ((double)speed * LEG_RAD_PER_STEP);
}

double leg_joint_load(const leg_joint_cal_t *c, int16_t load)
{
    return (double)c->sign * ((double)load / 1000.0);
}

uint16_t leg_goal(const leg_joint_cal_t *c, double angle)
{
    double a = angle < c->lo ? c->lo : (angle > c->hi ? c->hi : angle);
    double steps = (double)c->zero + (double)c->sign * (a / LEG_RAD_PER_STEP);
    double r = floor(steps + 0.5);
    if (r < 0.0) r = 0.0;
    if (r > STS_STEPS_PER_TURN - 1) r = STS_STEPS_PER_TURN - 1;
    return (uint16_t)r;
}

void leg_imu(const leg_imu_mount_t *m, const int16_t acc_raw[3], const int16_t gyro_raw[3], double acc[3],
             double gyro[3])
{
    for (int i = 0; i < 3; i++) {
        acc[i] = (double)m->sign[i] * ((double)acc_raw[m->axis[i]] * LEG_ACC_PER_STEP);
        gyro[i] = (double)m->sign[i] * ((double)gyro_raw[m->axis[i]] * LEG_GYRO_PER_STEP);
    }
}

double leg_clock(double phase, double freq)
{
    double f = freq < -1.0 ? -1.0 : (freq > 1.0 ? 1.0 : freq);
    return xm_mod1(phase + CLOCK_STEP * (1.0 + FREQ_RANGE * f));
}

void leg_phase_values(double phase, double *c, double *s)
{
    xm_sincos(2.0 * 3.141592653589793 * phase, s, c);
}

/* erg.gait.foot_path and leg_angles, operation for operation. */
void leg_base(double phase, double stride, double out[ACT_JOINTS])
{
    double s = xm_mod1(phase), x, z;
    if (s < 0.5) {
        double u = s / 0.5;
        x = stride / 2 - stride * u;
        z = -GAIT_STAND_DEPTH;
    } else {
        double u = (s - 0.5) / 0.5, sn, cs;
        xm_sincos(3.141592653589793 * u, &sn, &cs);
        x = -stride / 2 + stride * u;
        z = -GAIT_STAND_DEPTH + GAIT_STEP_HEIGHT * sn;
    }
    double r2 = x * x + z * z;
    double c = (r2 - GAIT_THIGH_SQ - GAIT_SHANK_SQ) / GAIT_TWO_THIGH_SHANK;
    double knee = -xm_acos(c < -1.0 ? -1.0 : (c > 1.0 ? 1.0 : c));
    double phi = xm_atan2(-x, -z);
    double sk, ck;
    xm_sincos(knee, &sk, &ck);
    double flex = phi - xm_atan2(GAIT_SHANK * sk, GAIT_THIGH + GAIT_SHANK * ck);
    out[0] = 0.0;
    out[1] = flex;
    out[2] = knee;
}

void leg_observation(double obs[OBS_SIZE], const leg_joint_cal_t cal[ACT_JOINTS], const sts_state_t servo[ACT_JOINTS],
                     const double acc[3], const double gyro[3], bool contact, double phase, const double command[3],
                     int leg)
{
    leg_phase_values(phase, &obs[OBS_PHASE], &obs[OBS_PHASE + 1]);
    for (int j = 0; j < ACT_JOINTS; j++) {
        obs[OBS_JOINT_POS + j] = leg_joint_angle(&cal[j], servo[j].position);
        obs[OBS_JOINT_VEL + j] = leg_joint_speed(&cal[j], servo[j].speed);
        obs[OBS_JOINT_LOAD + j] = leg_joint_load(&cal[j], servo[j].load);
    }
    for (int i = 0; i < 3; i++) {
        obs[OBS_ACC + i] = acc[i];
        obs[OBS_GYRO + i] = gyro[i];
        obs[OBS_COMMAND + i] = command[i];
    }
    obs[OBS_CONTACT] = contact ? 1.0 : 0.0;
    for (int i = 0; i < OBS_N_LEGS; i++) obs[OBS_LEG_ID + i] = (i == leg) ? 1.0 : 0.0;
    for (int s = 0; s < OBS_N_LEGS - 1; s++) {
        for (int v = 0; v < OBS_MSG_VALUES; v++) obs[OBS_SLOTS + s * OBS_MSG_SLOT + v] = 0.0;
        obs[OBS_SLOTS + s * OBS_MSG_SLOT + OBS_MSG_VALUES] = OBS_AGE_CAP;
    }
}

void leg_targets(const double action[ACT_SIZE_ALL], double phase, double forward, double targets[ACT_JOINTS])
{
    double base[ACT_JOINTS];
    leg_base(phase, forward * GAIT_PERIOD / 2, base);
    for (int j = 0; j < ACT_JOINTS; j++) targets[j] = base[j] + ACTION_SCALE * action[j];
}
