/*
 * The base controller of one leg (erg/base_controller.py, Issues #71 and #85) and the policy's adjustments of it
 * (erg.base_controller.adjust_from_action, Issue #82). No ESP-IDF dependency: tested on the PC against the Python
 * version bit for bit (tests/test_firmware_host.py).
 *
 * Every operation is the Python version's, in the same order, in double precision, with xmath.c for the sine,
 * cosine and arctangent; its constants are generated from Python into obs_layout.h (scripts/gen_firmware_layout.py),
 * so a retune of the Python controller only regenerates them.
 */
#pragma once

#include <stdbool.h>

#include "obs_layout.h"

/* erg.base_controller.Adjust: what the policy may change; all zero is the base controller alone. */
typedef struct {
    double freq, foot_dx, foot_dy, step_height, place_gain, att_gain;
} bc_adjust_t;

/* erg.base_controller.LegController's state. */
typedef struct {
    int leg; /* index in the order FL, FR, HL, HR */
    double vel[2], integral[2], roll, pitch;
    bool has_prev_foot;
    double prev_foot[2];
    bool has_prev_s;
    double prev_s;
    double td[2], lo[2], last[2];
} bc_leg_t;

/* erg.base_controller.LegOutput. */
typedef struct {
    double targets[ACT_JOINTS];
    double freq;
    double message[ACT_LEARNED]; /* the learned values the leg sends: velocity estimate x, y, speed integral x, y */
} bc_output_t;

/* The policy's actions over one policy step (erg.ppo.period_actions, Issue #95): the new action clipped to [-1, 1]
 * and the previous policy step's (zeros at start). */
typedef struct {
    double prev[N_ADJUST], a[N_ADJUST];
} bc_policy_t;

void bc_init(bc_leg_t *c, int leg);
void bc_policy_init(bc_policy_t *p);
/* A new policy output (raw, N_ADJUST values): the current action becomes the previous one. */
void bc_policy_new(bc_policy_t *p, const double raw[N_ADJUST]);
/* The action applied in control period k (0 .. POLICY_PERIODS - 1) of the policy step: with interpolate, a linear
 * ramp from the previous action that reaches the new one in the last period; otherwise the new one throughout. */
void bc_policy_period(const bc_policy_t *p, int k, bool interpolate, double out[N_ADJUST]);
/* The policy's N_ADJUST outputs, each clipped to [-1, 1] and scaled by ADJUST_SCALE. */
void bc_adjust_from_action(const double a[N_ADJUST], bc_adjust_t *out);
/* One control period, from the leg's observation row (OBS_SIZE values, as erg.base_controller.leg_input reads it:
 * only the two neighbours' message slots are used). adj may be NULL (the base controller alone). */
void bc_step(bc_leg_t *c, const double obs[OBS_SIZE], const bc_adjust_t *adj, bc_output_t *out);
