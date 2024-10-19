



ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/multi_gpu.yaml  scripts/run_drpo.py  \
        recipes/qwen1.5-0.5b/neuralsort/config_full_K8_neuralsort_tau1.0_lr5e-7_nohardloss_allrank.yaml

