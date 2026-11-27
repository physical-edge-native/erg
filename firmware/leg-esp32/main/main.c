/*
 * erg leg firmware for the Seeed XIAO ESP32-S3 (Issue #54): one leg on the bench.
 *
 * Every CONTROL_PERIOD_US (20 ms) the loop reads the leg's three STS3215 servos on their serial bus (through the
 * Waveshare Bus Servo Adapter (A)), the LSM6DSOX IMU on I2C and the foot switch on a GPIO, builds the leg's
 * observation in the order of the training environment (leg.c, obs_layout.h), runs the policy in Dyne's mlp-actor
 * (policy.c) every POLICY_PERIODS loops (25 Hz), and runs the leg's base controller (base_ctrl.c, Issue #85) with
 * the policy's adjustments, held in between; it sends the controller's three joint goals in one packet and advances
 * the leg's gait clock by the controller's frequency correction. Without a policy the adjustments are zero and the
 * base controller walks alone (stepping in place for a zero command). The controller's message values (velocity
 * estimate, speed integral) are kept for the neighbours; this stage has no links between legs yet, so the
 * controller hears no neighbour (its message slots read "nothing received"). After reading the
 * servos it sets the knee servo's torque limit from the measured knee angle (leg_knee_torque_limit, Issue #81), so
 * that the push rod's force stays under its allowable load near the straight end of the knee range.
 *
 * It prints, every REPORT_LOOPS loops, the time of each stage (servo bus, IMU, policy, goals) and of the whole loop,
 * and the jitter of the loop's start against its schedule, as "timing" lines for the design record.
 *
 * At start, if a check file is embedded (scripts/make_firmware_files.py), it runs the policy on the check's inputs
 * and prints whether the outputs equal, bit for bit, those Dyne's mlp-actor gave on the PC.
 *
 * Pins, servo ids, the leg's index and the joints' calibration are set with `idf.py menuconfig` ("erg leg"); they
 * live in the device's sdkconfig, which is not kept in the repository. No Wi-Fi is used in this stage.
 */
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "driver/gpio.h"
#include "driver/i2c_master.h"
#include "driver/uart.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "base_ctrl.h"
#include "leg.h"
#include "policy.h"
#include "sts.h"

static const char *TAG = "leg";

#define CONTROL_PERIOD_US 20000
#define REPORT_LOOPS 250
#define SERVO_UART UART_NUM_1
#define SERVO_REPLY_TIMEOUT_MS 4
#define IMU_ADDR CONFIG_ERG_IMU_ADDR
#define LSM6DSOX_WHO_AM_I 0x0F
#define LSM6DSOX_ID 0x6C
#define LSM6DSOX_CTRL1_XL 0x10
#define LSM6DSOX_CTRL2_G 0x11
#define LSM6DSOX_CTRL3_C 0x12
#define LSM6DSOX_OUTX_L_G 0x22 /* gyro x, y, z then accelerometer x, y, z, little-endian */

#if CONFIG_ERG_EMBED_POLICY
extern const uint8_t wasm_start[] asm("_binary_mlp_actor_wasm_start");
extern const uint8_t wasm_end[] asm("_binary_mlp_actor_wasm_end");
extern const uint8_t model_start[] asm("_binary_policy_dynm_start");
extern const uint8_t model_end[] asm("_binary_policy_dynm_end");
extern const uint8_t check_start[] asm("_binary_check_bin_start");
extern const uint8_t check_end[] asm("_binary_check_bin_end");
#endif

static const uint8_t SERVO_IDS[ACT_JOINTS] = {CONFIG_ERG_SERVO_ID_ABD, CONFIG_ERG_SERVO_ID_FLEX,
                                              CONFIG_ERG_SERVO_ID_KNEE};
/* Joint ranges of the MuJoCo model (models/quadruped.xml): abduction, hip flexion, knee. */
static leg_joint_cal_t s_cal[ACT_JOINTS] = {
    {CONFIG_ERG_ZERO_ABD, CONFIG_ERG_SIGN_ABD, -0.5, 0.5},
    {CONFIG_ERG_ZERO_FLEX, CONFIG_ERG_SIGN_FLEX, -1.6, 1.6},
    {CONFIG_ERG_ZERO_KNEE, CONFIG_ERG_SIGN_KNEE, -2.5, -0.2},
};
static const leg_imu_mount_t s_mount = {{0, 1, 2}, {1, 1, 1}}; /* to be set from how the board is mounted */

static i2c_master_dev_handle_t s_imu;

/* ---- timing statistics ---- */
typedef struct {
    int64_t sum, max;
} stat_t;

static void stat_add(stat_t *s, int64_t v)
{
    s->sum += v;
    if (v > s->max) s->max = v;
}

/* ---- servo bus ---- */
static void servo_bus_init(void)
{
    uart_config_t c = {
        .baud_rate = CONFIG_ERG_SERVO_BAUD,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    ESP_ERROR_CHECK(uart_driver_install(SERVO_UART, 512, 512, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_param_config(SERVO_UART, &c));
    ESP_ERROR_CHECK(uart_set_pin(SERVO_UART, CONFIG_ERG_SERVO_TX, CONFIG_ERG_SERVO_RX, UART_PIN_NO_CHANGE,
                                 UART_PIN_NO_CHANGE));
}

static int servo_read_state(uint8_t id, sts_state_t *s)
{
    uint8_t req[16], rep[16];
    size_t n = sts_read(req, sizeof req, id, STS_ADDR_PRESENT_POSITION, 6);
    uart_flush_input(SERVO_UART);
    uart_write_bytes(SERVO_UART, req, n);
    int got = uart_read_bytes(SERVO_UART, rep, 12, pdMS_TO_TICKS(SERVO_REPLY_TIMEOUT_MS));
    if (got <= 0) return -10;
    return sts_parse_state(rep, (size_t)got, id, s);
}

/* The knee servo's torque limit register (one broadcast packet, no reply), written only when the limit changes
 * (leg_knee_limit_should_write). Returns 1 if it wrote. */
static int s_knee_limit = -1;

static int servo_knee_limit(double knee)
{
    int v = leg_knee_limit_register(knee);
    if (!leg_knee_limit_should_write(s_knee_limit, v)) return 0;
    uint8_t pkt[16];
    uint16_t value = (uint16_t)v;
    size_t n = sts_sync_write_u16(pkt, sizeof pkt, STS_ADDR_TORQUE_LIMIT, &SERVO_IDS[2], &value, 1);
    uart_write_bytes(SERVO_UART, pkt, n);
    uart_wait_tx_done(SERVO_UART, pdMS_TO_TICKS(2));
    s_knee_limit = v;
    return 1;
}

static void servo_write_goals(const uint16_t goals[ACT_JOINTS])
{
    uint8_t pkt[64];
    size_t n = sts_sync_write_goal(pkt, sizeof pkt, SERVO_IDS, goals, ACT_JOINTS);
    uart_write_bytes(SERVO_UART, pkt, n);
    uart_wait_tx_done(SERVO_UART, pdMS_TO_TICKS(2));
}

/* ---- IMU ---- */
static esp_err_t imu_write(uint8_t reg, uint8_t v)
{
    uint8_t b[2] = {reg, v};
    return i2c_master_transmit(s_imu, b, 2, 10);
}

static bool imu_init(void)
{
    i2c_master_bus_config_t bc = {
        .i2c_port = I2C_NUM_0,
        .sda_io_num = CONFIG_ERG_I2C_SDA,
        .scl_io_num = CONFIG_ERG_I2C_SCL,
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,
    };
    i2c_master_bus_handle_t bus;
    ESP_ERROR_CHECK(i2c_new_master_bus(&bc, &bus));
    i2c_device_config_t dc = {.dev_addr_length = I2C_ADDR_BIT_LEN_7, .device_address = IMU_ADDR,
                              .scl_speed_hz = 400000};
    ESP_ERROR_CHECK(i2c_master_bus_add_device(bus, &dc, &s_imu));
    uint8_t reg = LSM6DSOX_WHO_AM_I, id = 0;
    if (i2c_master_transmit_receive(s_imu, &reg, 1, &id, 1, 10) != ESP_OK || id != LSM6DSOX_ID) {
        ESP_LOGE(TAG, "IMU not found (WHO_AM_I 0x%02x)", id);
        return false;
    }
    imu_write(LSM6DSOX_CTRL3_C, 0x44);  /* block data update, register address auto-increment */
    imu_write(LSM6DSOX_CTRL1_XL, 0x58); /* 208 Hz, +-4 g */
    imu_write(LSM6DSOX_CTRL2_G, 0x54);  /* 208 Hz, +-500 degrees per second */
    return true;
}

static bool imu_read(int16_t acc[3], int16_t gyro[3])
{
    uint8_t reg = LSM6DSOX_OUTX_L_G, b[12];
    if (i2c_master_transmit_receive(s_imu, &reg, 1, b, 12, 5) != ESP_OK) return false;
    for (int i = 0; i < 3; i++) {
        gyro[i] = (int16_t)(b[2 * i] | (b[2 * i + 1] << 8));
        acc[i] = (int16_t)(b[6 + 2 * i] | (b[7 + 2 * i] << 8));
    }
    return true;
}

#if CONFIG_ERG_EMBED_POLICY
/* ---- policy check: inputs and outputs computed on the PC (Dyne's mlp-actor in wasmtime) ---- */
static void policy_check(const uint8_t *buf, size_t len)
{
    /* check.bin: u32 rows, u32 n_in, u32 n_out, then rows x (n_in inputs, n_out outputs) as little-endian f64 */
    uint32_t h[3];
    if (len < 12) return;
    memcpy(h, buf, 12);
    if (h[1] != policy_n_in() || h[2] != policy_n_out() || len != 12 + 8ull * h[0] * (h[1] + h[2])) {
        ESP_LOGE(TAG, "check: file does not match the policy");
        return;
    }
    double in[128], out[64], want[64];
    int bad = 0;
    int64_t t_sum = 0, t_max = 0;
    for (uint32_t r = 0; r < h[0]; r++) {
        const uint8_t *row = buf + 12 + 8ull * r * (h[1] + h[2]);
        memcpy(in, row, 8 * h[1]);
        memcpy(want, row + 8 * h[1], 8 * h[2]);
        int64_t t0 = esp_timer_get_time();
        int rc = policy_eval(in, out);
        int64_t dt = esp_timer_get_time() - t0;
        t_sum += dt;
        if (dt > t_max) t_max = dt;
        if (rc != 0 || memcmp(out, want, 8 * h[2]) != 0) bad++;
    }
    printf("check: %s, %" PRIu32 " rows, %d differ; policy time mean %" PRId64 " us, max %" PRId64 " us\n",
           bad ? "MISMATCH" : "bit-exact", h[0], bad, t_sum / (int64_t)h[0], t_max);
}

#endif

void app_main(void)
{
    servo_bus_init();
    bool imu_ok = imu_init();
    gpio_config_t g = {.pin_bit_mask = 1ull << CONFIG_ERG_SWITCH_GPIO, .mode = GPIO_MODE_INPUT,
                       .pull_up_en = GPIO_PULLUP_ENABLE};
    ESP_ERROR_CHECK(gpio_config(&g));

    bool have_policy = false;
#if CONFIG_ERG_EMBED_POLICY
    have_policy = policy_init(wasm_start, (uint32_t)(wasm_end - wasm_start), model_start,
                              (uint32_t)(model_end - model_start));
    if (have_policy && (policy_n_in() != OBS_SIZE || policy_n_out() != N_ADJUST)) {
        ESP_LOGE(TAG, "policy has %u inputs and %u outputs; this firmware needs %d and %d", (unsigned)policy_n_in(),
                 (unsigned)policy_n_out(), OBS_SIZE, N_ADJUST);
        have_policy = false;
    }
    if (have_policy) policy_check(check_start, (size_t)(check_end - check_start));
#else
    ESP_LOGW(TAG, "built without a policy: the base controller alone");
#endif

    const double command[3] = {0.0, 0.0, 0.0};
    const double phase_start[OBS_N_LEGS] = GAIT_PHASE_START;
    double phase = phase_start[CONFIG_ERG_LEG_INDEX];
    double obs[OBS_SIZE], action[64] = {0};
    bc_leg_t ctrl;
    bc_init(&ctrl, CONFIG_ERG_LEG_INDEX);
    bc_adjust_t adj = {0};
    bc_output_t out;
    double learned[ACT_LEARNED] = {0}; /* the message values for the neighbours (erg.env clips them to [-1, 1]) */
    uint32_t period = 0;
    sts_state_t servo[ACT_JOINTS] = {0};
    int16_t acc_raw[3] = {0}, gyro_raw[3] = {0};
    double acc[3], gyro[3];
    stat_t st_bus = {0}, st_imu = {0}, st_pol = {0}, st_goal = {0}, st_loop = {0}, st_jit = {0};
    int bus_errors = 0, imu_errors = 0, loops = 0, knee_writes = 0;

    int64_t next = esp_timer_get_time() + CONTROL_PERIOD_US;
    for (;;) {
        int64_t now = esp_timer_get_time();
        if (next > now) vTaskDelay(pdMS_TO_TICKS((next - now) / 1000));
        while (esp_timer_get_time() < next) {
        } /* the last part of a millisecond */
        int64_t t0 = esp_timer_get_time();
        stat_add(&st_jit, t0 - next);
        next += CONTROL_PERIOD_US;

        bool knee_ok = true;
        for (int j = 0; j < ACT_JOINTS; j++) {
            if (servo_read_state(SERVO_IDS[j], &servo[j]) != 0) {
                bus_errors++;
                if (j == 2) knee_ok = false;
            }
        }
        /* the knee's torque limit from the measured knee angle (erg.knee_limit), before the new goals */
        if (knee_ok) knee_writes += servo_knee_limit(leg_joint_angle(&s_cal[2], servo[2].position));
        int64_t t1 = esp_timer_get_time();
        if (!imu_ok || !imu_read(acc_raw, gyro_raw)) imu_errors++;
        leg_imu(&s_mount, acc_raw, gyro_raw, acc, gyro);
        bool contact = gpio_get_level(CONFIG_ERG_SWITCH_GPIO) == 0; /* switch to ground, pulled up */
        leg_observation(obs, s_cal, servo, acc, gyro, contact, phase, command, CONFIG_ERG_LEG_INDEX);
        int64_t t2 = esp_timer_get_time();
        if (period % POLICY_PERIODS == 0) { /* the policy's adjustments, held for POLICY_PERIODS periods */
            if (!have_policy || policy_eval(obs, action) != 0) memset(action, 0, sizeof action);
            bc_adjust_from_action(action, &adj);
        }
        period++;
        bc_step(&ctrl, obs, &adj, &out);
        for (int i = 0; i < ACT_LEARNED; i++) {
            double v = out.message[i];
            learned[i] = v < -1.0 ? -1.0 : (v > 1.0 ? 1.0 : v);
        }
        int64_t t3 = esp_timer_get_time();
        uint16_t goals[ACT_JOINTS];
        for (int j = 0; j < ACT_JOINTS; j++) goals[j] = leg_goal(&s_cal[j], out.targets[j]);
        servo_write_goals(goals);
        phase = leg_clock(phase, out.freq);
        (void)learned;
        int64_t t4 = esp_timer_get_time();

        stat_add(&st_bus, t1 - t0);
        stat_add(&st_imu, t2 - t1);
        stat_add(&st_pol, t3 - t2);
        stat_add(&st_goal, t4 - t3);
        stat_add(&st_loop, t4 - t0);
        if (++loops == REPORT_LOOPS) {
            printf("timing (us, mean/max over %d loops): servo read %" PRId64 "/%" PRId64 ", imu+obs %" PRId64
                   "/%" PRId64 ", policy %" PRId64 "/%" PRId64 ", goals %" PRId64 "/%" PRId64 ", loop %" PRId64
                   "/%" PRId64 ", start jitter %" PRId64 "/%" PRId64 "; bus errors %d, imu errors %d, knee limit writes %d\n",
                   loops, st_bus.sum / loops, st_bus.max, st_imu.sum / loops, st_imu.max, st_pol.sum / loops,
                   st_pol.max, st_goal.sum / loops, st_goal.max, st_loop.sum / loops, st_loop.max,
                   st_jit.sum / loops, st_jit.max, bus_errors, imu_errors, knee_writes);
            memset(&st_bus, 0, sizeof st_bus);
            st_imu = st_pol = st_goal = st_loop = st_jit = st_bus;
            bus_errors = imu_errors = loops = knee_writes = 0;
        }
    }
}
