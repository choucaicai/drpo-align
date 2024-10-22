



ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/gpu1.yaml  scripts/run_drpo.py  \
        recipes/xxxx.yaml

