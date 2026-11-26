/*
 * The leg's policy in Dyne's mlp-actor (WebAssembly), run by the WAMR interpreter as in Dyne's ESP32 firmware
 * (third_party/dyne/node-esp32). The .wasm and the policy (.dynm) are embedded in the firmware image when they were
 * made before building (scripts/make_firmware_files.py); without them the leg holds its standing posture.
 */
#include "policy.h"

#include <string.h>

#include "esp_log.h"
#include "wasm_export.h"

static const char *TAG = "policy";

static wasm_module_inst_t s_inst;
static wasm_exec_env_t s_env;
static wasm_function_inst_t f_load_buf, f_load_cap, f_model_load, f_in_buf, f_out_buf, f_eval, f_n_in, f_n_out;
static uint32_t s_n_in, s_n_out;

static uint32_t call0(wasm_function_inst_t f)
{
    wasm_val_t r[1];
    if (!wasm_runtime_call_wasm_a(s_env, f, 1, r, 0, NULL)) return 0;
    return (uint32_t)r[0].of.i32;
}

static int32_t call1(wasm_function_inst_t f, uint32_t arg)
{
    wasm_val_t a[1] = {{.kind = WASM_I32, .of.i32 = (int32_t)arg}};
    wasm_val_t r[1];
    if (!wasm_runtime_call_wasm_a(s_env, f, 1, r, 1, a)) return -1000;
    return r[0].of.i32;
}

static void *app(uint32_t offset)
{
    return wasm_runtime_addr_app_to_native(s_inst, offset);
}

bool policy_init(const uint8_t *wasm, uint32_t wasm_len, const uint8_t *model, uint32_t model_len)
{
    if (!wasm || !model || !wasm_len || !model_len) {
        ESP_LOGW(TAG, "no mlp-actor or no policy embedded: holding the standing posture");
        return false;
    }
    RuntimeInitArgs ia;
    memset(&ia, 0, sizeof(ia));
    ia.mem_alloc_type = Alloc_With_System_Allocator; /* large blocks go to PSRAM (CONFIG_SPIRAM_USE_MALLOC) */
    if (!wasm_runtime_full_init(&ia)) return false;
    char err[128];
    /* WAMR may modify the module bytes while loading: give it its own copy. */
    uint8_t *copy = malloc(wasm_len);
    if (!copy) return false;
    memcpy(copy, wasm, wasm_len);
    wasm_module_t mod = wasm_runtime_load(copy, wasm_len, err, sizeof(err));
    if (!mod) {
        ESP_LOGE(TAG, "load: %s", err);
        return false;
    }
    s_inst = wasm_runtime_instantiate(mod, 4 * 1024, 0, err, sizeof(err));
    if (!s_inst) {
        ESP_LOGE(TAG, "instantiate: %s", err);
        return false;
    }
    s_env = wasm_runtime_create_exec_env(s_inst, 16 * 1024);
    f_load_buf = wasm_runtime_lookup_function(s_inst, "dyne_load_buf");
    f_load_cap = wasm_runtime_lookup_function(s_inst, "dyne_load_cap");
    f_model_load = wasm_runtime_lookup_function(s_inst, "dyne_model_load");
    f_in_buf = wasm_runtime_lookup_function(s_inst, "dyne_in_buf");
    f_out_buf = wasm_runtime_lookup_function(s_inst, "dyne_out_buf");
    f_eval = wasm_runtime_lookup_function(s_inst, "dyne_model_eval");
    f_n_in = wasm_runtime_lookup_function(s_inst, "dyne_model_n_in");
    f_n_out = wasm_runtime_lookup_function(s_inst, "dyne_model_n_out");
    if (!s_env || !f_load_buf || !f_load_cap || !f_model_load || !f_in_buf || !f_out_buf || !f_eval || !f_n_in ||
        !f_n_out) {
        ESP_LOGE(TAG, "mlp-actor exports missing");
        return false;
    }
    if (model_len > call0(f_load_cap)) {
        ESP_LOGE(TAG, "policy too large: %u bytes", (unsigned)model_len);
        return false;
    }
    memcpy(app(call0(f_load_buf)), model, model_len);
    int32_t rc = call1(f_model_load, model_len);
    if (rc != 0) {
        ESP_LOGE(TAG, "dyne_model_load: %d", (int)rc);
        return false;
    }
    s_n_in = call0(f_n_in);
    s_n_out = call0(f_n_out);
    ESP_LOGI(TAG, "policy loaded: %u inputs, %u outputs", (unsigned)s_n_in, (unsigned)s_n_out);
    return true;
}

uint32_t policy_n_in(void) { return s_n_in; }
uint32_t policy_n_out(void) { return s_n_out; }

int policy_eval(const double *in, double *out)
{
    memcpy(app(call0(f_in_buf)), in, 8 * s_n_in);
    int32_t r = call1(f_eval, s_n_in);
    if (r != (int32_t)s_n_out) return r < 0 ? r : -100;
    memcpy(out, app(call0(f_out_buf)), 8 * s_n_out);
    return 0;
}
