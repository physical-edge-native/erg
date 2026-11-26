#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/* Load Dyne's mlp-actor and the policy into it. Returns false (and the leg holds its standing posture) if either is
 * missing or refused. */
bool policy_init(const uint8_t *wasm, uint32_t wasm_len, const uint8_t *model, uint32_t model_len);
uint32_t policy_n_in(void);
uint32_t policy_n_out(void);
/* One evaluation: policy_n_in() inputs to policy_n_out() outputs. Returns 0, or a negative code from mlp-actor. */
int policy_eval(const double *in, double *out);
