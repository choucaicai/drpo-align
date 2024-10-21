



ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/gpu1.yaml  scripts/run_drpo.py  \
        recipes/qwen1.5-0.5b/qwen1.5-0.5-config_full_k8_diffndcg_lr5e-7_com+adam_prob_rank_0.2_adv2.yaml

