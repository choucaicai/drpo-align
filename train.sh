



ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/multi_gpu.yaml  scripts/run_drpo.py  \
        recipes/xxx.yaml

