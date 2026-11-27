#include "base_ctrl.h"

#include <math.h>
#include <stddef.h>

#include "leg.h"
#include "xmath.h"

static const int NEIGHBOURS[OBS_N_LEGS][2] = BC_NEIGHBOURS;
static const double LEG_SX[OBS_N_LEGS] = BC_LEG_SX;
static const double LEG_SY[OBS_N_LEGS] = BC_LEG_SY;
static const double ADJ_SCALE[N_ADJUST] = ADJUST_SCALE;

static double clip(double v, double lo, double hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

static double xsin(double x)
{
    double s, c;
    xm_sincos(x, &s, &c);
    return s;
}

static double xcos(double x)
{
    double s, c;
    xm_sincos(x, &s, &c);
    return c;
}

void bc_init(bc_leg_t *c, int leg)
{
    bc_leg_t z = {0};
    *c = z;
    c->leg = leg;
}

void bc_policy_init(bc_policy_t *p)
{
    bc_policy_t z = {{0}, {0}};
    *p = z;
}

void bc_policy_new(bc_policy_t *p, const double raw[N_ADJUST])
{
    for (int i = 0; i < N_ADJUST; i++) {
        p->prev[i] = p->a[i];
        p->a[i] = clip(raw[i], -1.0, 1.0);
    }
}

/* erg.ppo.period_actions: prev + (a - prev) * ((k + 1) / POLICY_PERIODS) for all but the last period, then a. */
void bc_policy_period(const bc_policy_t *p, int k, bool interpolate, double out[N_ADJUST])
{
    if (!interpolate || k >= POLICY_PERIODS - 1) {
        for (int i = 0; i < N_ADJUST; i++) out[i] = p->a[i];
        return;
    }
    double f = (double)(k + 1) / (double)POLICY_PERIODS;
    for (int i = 0; i < N_ADJUST; i++) out[i] = p->prev[i] + (p->a[i] - p->prev[i]) * f;
}

void bc_adjust_from_action(const double a[N_ADJUST], bc_adjust_t *out)
{
    double v[N_ADJUST];
    for (int i = 0; i < N_ADJUST; i++) v[i] = clip(a[i], -1.0, 1.0) * ADJ_SCALE[i];
    out->freq = v[0];
    out->foot_dx = v[1];
    out->foot_dy = v[2];
    out->step_height = v[3];
    out->place_gain = v[4];
    out->att_gain = v[5];
}

/* erg.base_controller.foot_position */
static void foot_position(int leg, const double q[3], double out[3])
{
    double s1, c1, s12, c12, sa, ca;
    xm_sincos(q[1], &s1, &c1);
    xm_sincos(q[1] + q[2], &s12, &c12);
    double x = -BC_THIGH * s1 - BC_SHANK * s12;
    double zp = -BC_THIGH * c1 - BC_SHANK * c12;
    double lat = LEG_SY[leg] * BC_LEG_OFFSET;
    xm_sincos(q[0], &sa, &ca);
    out[0] = x;
    out[1] = lat * ca - zp * sa;
    out[2] = lat * sa + zp * ca;
}

/* erg.base_controller.leg_ik */
static void leg_ik(int leg, double x, double y, double z, double out[3])
{
    double lat = LEG_SY[leg] * BC_LEG_OFFSET;
    double d2 = y * y + z * z - lat * lat;
    double d = d2 > 1e-6 ? sqrt(d2) : 1e-3;
    out[0] = xm_atan2(z, y) - xm_atan2(-d, lat);
    leg_angles(x, -d, &out[1], &out[2]);
}

/* erg.base_controller._blend */
static double blend(double u)
{
    u = u < 0.0 ? 0.0 : (u > 1.0 ? 1.0 : u);
    return 0.5 - 0.5 * xcos(BC_PI * u);
}

/* erg.base_controller._glide */
static double glide(double u, double r)
{
    u = u < 0.0 ? 0.0 : (u > 1.0 ? 1.0 : u);
    if (r <= 0.0) return u;
    if (u > 0.5) return 1.0 - glide(1.0 - u, r);
    double p;
    if (u < r) {
        p = u / 2.0 - r / (2.0 * BC_PI) * xsin(BC_PI * u / r);
    } else {
        p = r / 2.0 + (u - r);
    }
    return p / (1.0 - r);
}

void bc_step(bc_leg_t *c, const double obs[OBS_SIZE], const bc_adjust_t *adj_in, bc_output_t *out)
{
    static const bc_adjust_t zero = {0};
    const bc_adjust_t *adj = adj_in ? adj_in : &zero;
    int leg = c->leg;
    double sx = LEG_SX[leg], sy = LEG_SY[leg];
    double hx = sx * BC_HIP_X, hy = sy * BC_HIP_Y;
    double lat = sy * BC_LEG_OFFSET;
    double vx_c = obs[OBS_COMMAND], vy_c = obs[OBS_COMMAND + 1], wz_c = obs[OBS_COMMAND + 2];

    /* Phase of the own clock, and anti-phase coupling to the neighbours (their phases advanced by the age). */
    double c_i = obs[OBS_PHASE], s_i = obs[OBS_PHASE + 1];
    double s = xm_atan2(s_i, c_i) / (2.0 * BC_PI);
    s = s < 0.0 ? s + 1.0 : s;
    double couple = 0.0;
    const double *shared[2];
    int n_shared = 0;
    for (int k = 0; k < 2; k++) {
        int nb = NEIGHBOURS[leg][k];
        const double *m = obs + OBS_SLOTS + (nb < leg ? nb : nb - 1) * OBS_MSG_SLOT;
        if (m[OBS_MSG_VALUES] >= OBS_AGE_CAP) continue;
        double sd, cd;
        xm_sincos(2.0 * BC_PI * BC_DT / BC_PERIOD * m[OBS_MSG_VALUES], &sd, &cd);
        double c_j0 = m[BC_M_PHASE], s_j0 = m[BC_M_PHASE + 1];
        double c_j = c_j0 * cd - s_j0 * sd, s_j = s_j0 * cd + c_j0 * sd;
        couple += -(s_j * c_i - c_j * s_i);
        shared[n_shared++] = m + BC_M_LEARNED;
    }
    double freq = clip(adj->freq + BC_COUPLING * couple, -1.0, 1.0);

    /* Velocity estimate and speed integral, averaged with the neighbours' (Python's sum() starts from 0). */
    double n = 1.0 + (double)n_shared;
    double sum[4] = {0.0, 0.0, 0.0, 0.0};
    for (int k = 0; k < n_shared; k++)
        for (int i = 0; i < 4; i++) sum[i] = sum[i] + shared[k][i];
    double vel[2] = {(c->vel[0] + sum[0]) / n, (c->vel[1] + sum[1]) / n};
    double integral[2] = {(c->integral[0] + sum[2]) / n, (c->integral[1] + sum[3]) / n};
    double foot[3];
    foot_position(leg, obs + OBS_JOINT_POS, foot);
    double fx = foot[0] + hx, fy = foot[1] + hy, fz = foot[2];
    double wx = obs[OBS_GYRO], wy = obs[OBS_GYRO + 1], wz = obs[OBS_GYRO + 2];
    bool contact = obs[OBS_CONTACT] > 0.5;
    if (contact && c->has_prev_foot) {
        double vx = -(fx - c->prev_foot[0]) / BC_DT - (wy * fz - wz * fy);
        double vy = -(fy - c->prev_foot[1]) / BC_DT - (wz * fx - wx * fz);
        double a = BC_DT / (BC_VEL_TAU + BC_DT);
        vel[0] = vel[0] + a * (vx - vel[0]);
        vel[1] = vel[1] + a * (vy - vel[1]);
    }
    c->has_prev_foot = contact;
    c->prev_foot[0] = fx;
    c->prev_foot[1] = fy;
    c->vel[0] = vel[0];
    c->vel[1] = vel[1];
    double ex = vx_c - vel[0], ey = vy_c - vel[1];
    c->integral[0] = clip(integral[0] + BC_SPEED_INTEGRAL * ex * BC_DT, -BC_INTEGRAL_LIMIT, BC_INTEGRAL_LIMIT);
    c->integral[1] = clip(integral[1] + BC_SPEED_INTEGRAL * ey * BC_DT, -BC_INTEGRAL_LIMIT, BC_INTEGRAL_LIMIT);

    /* Attitude from the IMU: complementary filter of roll and pitch. */
    double ax = obs[OBS_ACC], ay = obs[OBS_ACC + 1], az = obs[OBS_ACC + 2];
    c->roll = c->roll + wx * BC_DT;
    c->pitch = c->pitch + wy * BC_DT;
    double a2 = ax * ax + ay * ay + az * az;
    if ((1.0 - BC_ACC_TRUST) * BC_G * BC_G < a2 && a2 < (1.0 + BC_ACC_TRUST) * BC_G * BC_G) {
        double b = BC_DT / (BC_ATT_TAU + BC_DT);
        c->roll = c->roll + b * (xm_atan2(ay, az) - c->roll);
        c->pitch = c->pitch + b * (xm_atan2(-ax, az) - c->pitch);
    }
    double hip_x = hx, hip_y = hy + lat;
    double drop = hip_x * xsin(c->pitch) - hip_y * xsin(c->roll);
    double att = clip(BC_ATT_GAIN * (1.0 + adj->att_gain) * drop, -BC_MAX_ATT, BC_MAX_ATT);

    /* Stance speed of the foot for the commanded motion, with speed feedback; capture-point placement. */
    double turn = wz_c + BC_YAW_GAIN * (wz_c - wz);
    double hip_vx = vx_c + c->integral[0] - turn * hip_y;
    double hip_vy = vy_c + c->integral[1] + turn * hip_x;
    double t_stance = BC_STANCE * BC_PERIOD / (1.0 + FREQ_RANGE * freq);
    double k = BC_PLACE_GAIN * (1.0 + adj->place_gain);
    double tdx = clip(hip_vx * t_stance / 2.0 + k * (c->vel[0] - vx_c) + adj->foot_dx, -BC_MAX_REACH, BC_MAX_REACH);
    double tdy = clip(hip_vy * t_stance / 2.0 + k * (c->vel[1] - vy_c) + adj->foot_dy, -BC_MAX_REACH, BC_MAX_REACH);

    /* Phase changes: record touch-down and lift-off offsets. */
    if (c->has_prev_s) {
        if (s < BC_STANCE && BC_STANCE <= c->prev_s) {
            c->td[0] = c->last[0];
            c->td[1] = c->last[1];
        } else if (c->prev_s < BC_STANCE && BC_STANCE <= s) {
            c->lo[0] = c->last[0];
            c->lo[1] = c->last[1];
        }
    }
    c->has_prev_s = true;
    c->prev_s = s;

    double lift = 0.0, ox, oy;
    if (s < BC_STANCE) {
        double u = s / BC_STANCE;
        ox = c->td[0] - hip_vx * t_stance * u;
        oy = c->td[1] - hip_vy * t_stance * u;
    } else {
        double u = (s - BC_STANCE) / (1.0 - BC_STANCE);
        double g1 = glide(u / BC_RETRACT_START, BC_SWING_RAMP);
        double g2 = blend((u - BC_RETRACT_START) / (1.0 - BC_RETRACT_START));
        ox = c->lo[0] + (tdx + BC_RETRACT - c->lo[0]) * g1 - BC_RETRACT * g2;
        oy = c->lo[1] + (tdy - c->lo[1]) * g1;
        lift = (BC_STEP_HEIGHT + adj->step_height) * xsin(BC_PI * u);
    }
    c->last[0] = ox;
    c->last[1] = oy;

    double depth = BC_STAND_DEPTH + att;
    leg_ik(leg, BC_FOOT_X + ox, lat + oy, -depth + lift, out->targets);
    out->freq = freq;
    out->message[0] = c->vel[0];
    out->message[1] = c->vel[1];
    out->message[2] = c->integral[0];
    out->message[3] = c->integral[1];
}
