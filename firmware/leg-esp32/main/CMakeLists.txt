set(embed "")
if(CONFIG_ERG_EMBED_POLICY)
    # Made by scripts/make_firmware_files.py (not kept in the repository).
    set(embed "embed/mlp_actor.wasm" "embed/policy.dynm" "embed/check.bin")
endif()
idf_component_register(
    SRCS "main.c" "sts.c" "leg.c" "xmath.c" "policy.c"
    INCLUDE_DIRS "."
    EMBED_FILES ${embed}
    PRIV_REQUIRES esp_driver_uart esp_driver_i2c esp_driver_gpio esp_timer
)
# Round every floating-point operation separately (no fused multiply-add), as Python and WebAssembly do.
target_compile_options(${COMPONENT_LIB} PRIVATE -ffp-contract=off)
