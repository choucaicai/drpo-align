#!/bin/bash
export CUDA_VISIBLE_DEVICES="0,"
# datasets="[hh]"
datasets="[hh_sample512]"
n_samples=512
eval_batch_size=4
config_path="eval_configs/vllm_config_t1.0"
mode="vllm_sample"
# mode="sample"
main_process_port=29501

# samples_dir="samples/method_comp_0.5"
samples_dir="samples/btdpo-ablations"

model_or_path_list=(
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-diffndcg/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-com+adam-prob-rank0.2-lr5e-7-adv2-3"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-diffndcg-ablation/step_models/qwen1.5-0.5-fastsort-k8-lr5e-7-ada-steps"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache/qwen1.5-0.5-hf-lambda-beta0.1-lr5e-7"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-base-hh-sft"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-diffndcg-ablation/qwen1.5-0.5-hf-drpo-k8-diffsort-lr5e-7-dposcore"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache/qwen1.5-0.5-hf-dpo-beta0.1-lr5e-7"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-extra/qwen1.5-0.5-hf-pirank-tau1.0-lr5e-7"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-extra/qwen1.5-0.5-fastsort-k8-lr5e-7"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-extra/qwen1.5-0.5-hf-pro-lr5e-7"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache/qwen1.5-0.5-listnet-beta0.1-lr5e-7"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache/qwen1.5-0.5-hf-drpo-btdpo-beta0.1-lr5e-7"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-extra/qwen1.5-0.5-hf-neuralsort-tau1.0-lr5e-7-nohard_loss-allrank"

    # # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-diffndcg-ablation/step_models/qwen1.5-0.5-hf-neuralsort-tau1.0-lr5e-7-ada-step"\
    # # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-diffndcg-ablation/step_models/qwen1.5-0.5-hf-pirank-tau1.0-lr5e-7-allrank-ada-step"

    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-diffndcg-ablation/qwen1.5-0.5-hf-drpo-k8-simple-diffndcg-adam_prob_rank0.2_-lr5e-7-square-ada"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-diffndcg-ablation/qwen1.5-0.5-hf-drpo-k8-simple-diffndcg-adam_prob_rank0.2_-lr5e-7-line_dis-ada"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-diffndcg-ablation/qwen1.5-0.5-hf-drpo-k8-simple-diffndcg-adam_prob_rank0.2_-lr5e-7-sqrtdis-ada"

    # # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-diffndcg-ablation/step_models/qwen1.5-0.5-listnet-k6-beta0.1-lr5e-7-ada-steps"
    # # old discounts
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-diffndcg-ablation/qwen1.5-0.5-hf-drpo-k8-simplediffndcg-dposcore-lr5e-7"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-diffndcg/qwen1.5-0.5-hf-drpo-k8-simple-diffndcg-dposcore-lr5e-7-noscale-line_dis"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-diffndcg/qwen1.5-0.5-hf-drpo-k8-simple-diffndcg-dposcore-lr5e-7-noscale-sqrtdis"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-diffndcg/qwen1.5-0.5-hf-drpo-k8-simple-diffndcg-dposcore-lr5e-7-noscale-square_dis"
    # # ablation
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-diffndcg/qwen1.5-0.5-hf-drpo-k8-diffsort-com+adam-prob-rank0.2-lr5e-7-adv2"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-diffndcg-ablation/qwen1.5-0.5-hf-drpo-k8-diffsort-lr5e-7-dposcore"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-K-exp/qwen1.5-0.5-hf-drpo-pldpo-beta0.1-lr5e-7"

    # exp-K
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-K-exp/qwen1.5-1.8-pldpo-k8-beta0.1-lr5e-7"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-K-exp/qwen1.5-0.5-hf-drpo-pldpo-k2-beta0.1-lr5e-7"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-K-exp/qwen1.5-0.5-hf-drpo-pldpo-k4-beta0.1-lr5e-7"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-K-exp/qwen1.5-0.5-hf-drpo-pldpo-k6-beta0.1-lr5e-7"

    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-K-exp/qwen1.5-0.5-hf-drpo-lambda-k2-beta0.1-lr5e-7"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-K-exp/qwen1.5-0.5-hf-drpo-lambda-K4-beta0.1-lr5e-7"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-K-exp/qwen1.5-0.5-hf-drpo-lambda-k6-beta0.1-lr5e-7"

    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-K-exp/qwen1.5-0.5-hf-drpo-btdpo-K2-beta0.1-lr5e-7"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-K-exp/qwen1.5-0.5-hf-drpo-btdpo-K4-beta0.1-lr5e-7"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-K-exp/qwen1.5-0.5-hf-drpo-btdpo-K6-beta0.1-lr5e-7"

    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache/qwen1.5-0.5-listnet-k2-beta0.1-lr5e-7"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache/qwen1.5-0.5-listnet-k4-beta0.1-lr5e-7"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache/qwen1.5-0.5-listnet-k6-beta0.1-lr5e-7"

    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-diff-exp/qwen1.5-0.5-fastsort-k2-lr5e-7"
    "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-diff-exp/qwen1.5-0.5-hf-drpo-k2-btdpo-lr5e-7-ada"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-diff-exp/qwen1.5-0.5-hf-drpo-k2-diffsort-lr5e-7-dposcore"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-diff-exp/qwen1.5-0.5-hf-drpo-k2-listnet-beta0.1-lr5e-7"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-diff-exp/qwen1.5-0.5-hf-k2-neuralsort-tau1.0-lr5e-7"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-diff-exp/qwen1.5-0.5-hf-k2-pirank-tau1.0-lr5e-7"
    "/media/mil/LLMfinetune/qwen1.5-1.8-hf-cache-diff-exp/qwen1.5-1.8-hf-drpo-k2-btdpo-lr5e-7-ada"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache/qwen1.5-0.5-hf-neuralsort-tau1.0-lr5e-7-nohard_loss-allrank-ad"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-extra/qwen1.5-0.5-hf-neuralsort-tau1.0-lr5e-7"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-extra/qwen1.5-0.5-hf-neuralsort-tau1.0-lr5e-7-nohard_loss"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-extra/qwen1.5-0.5-hf-neuralsort-tau1.0-lr5e-7-nohard_loss-allrank"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache/qwen1.5-0.5-hf-neuralsort-tau1.0-lr5e-7-nohard_loss-allrank-adv2"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache/qwen1.5-0.5-hf-neuralsort-tau1.0-lr5e-7-nohard_loss-allrank-run2"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-diffndcg/qwen1.5-0.5-hf-drpo-k8-simple-diffndcg-prob-rank0.2-lr5e-7"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-diffndcg/qwen1.5-0.5-hf-drpo-k8-simple-diffndcg-prob-rank0.2-lr5e-7-sim"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-diffndcg/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-com+adam-prob-rank0.2-lr5e-7-ada_coef0.5"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-diffndcg/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-com+adam-prob-rank0.2-lr5e-7-gamma0.99"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-diffndcg/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-com+adam-prob-rank0.2-lr5e-7-run2"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-diffndcg/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-com+adam-prob-rank0.2-lr5e-7-run3"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-diffndcg/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-com+adam-prob-rank0.2-lr5e-7"
    # "/media/mil/LLMfinetune/qwen1.5-1.8-hf-cache/qwen1.5-1.8-hf-drpo-btdpo-K2-beta0.1-lr5e-7"
    # "/media/mil/LLMfinetune/qwen1.5-1.8-hf-cache/qwen1.5-1.8-hf-pirank-ada0.1-lr5e-7/checkpoint-30000"
)



# accelerate launch --main_process_port 29501 eval.py --config-path=eval_configs/hh-qwen1.5-0.5b ++mode=sample ++datasets=[hh] ++n_samples=8 \
#     ++model.eval_batch_size=2 ++samples_dir=samples/extra_exp/drpo-ablation \
#     ++exp_name="qwen1.5-0.5-hf-drpo-btdpo-beta0.1-lr5e-7" \
#     ++model.name_or_path="/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache/qwen1.5-0.5-hf-drpo-btdpo-beta0.1-lr5e-7" 

# 打印所有参数
echo "Current configuration:"
echo "---------------------"
echo "Datasets: $datasets"
echo "Number of samples: $n_samples"
echo "Evaluation batch size: $eval_batch_size"
echo "Samples directory: $samples_dir"
echo "Config path: $config_path"
echo "Mode: $mode"
echo "Main process port: $main_process_port"
echo "Model paths:"
for path in "${model_or_path_list[@]}"; do
    echo "  - $path"
done
echo "---------------------"

# 遍历模型路径列表
for model_path in "${model_or_path_list[@]}"; do
    # 从模型路径中提取实验名称
    exp_name=$(basename "$model_path")
    
    echo "Evaluating model: $model_path"
    echo "Experiment name: $exp_name"

    # accelerate launch --main_process_port $main_process_port eval.py \
    #     --config-path="$config_path" \
    #     ++mode="$mode" \
    #     ++datasets="$datasets" \
    #     ++n_samples=$n_samples \
    #     ++model.eval_batch_size=$eval_batch_size \
    #     ++samples_dir="$samples_dir" \
    #     ++exp_name="$exp_name" \
    #     ++model.name_or_path="$model_path"
    if [ "$mode" = "vllm_sample" ]; then
        python eval.py \
            --config-path="$config_path" \
            ++mode="$mode" \
            ++datasets="$datasets" \
            ++n_samples=$n_samples \
            ++model.eval_batch_size=$eval_batch_size \
            ++samples_dir="$samples_dir" \
            ++exp_name="$exp_name" \
            ++model.name_or_path="$model_path"
    else
        accelerate launch --main_process_port $main_process_port eval.py \
            --config-path="$config_path" \
            ++mode="$mode" \
            ++datasets="$datasets" \
            ++n_samples=$n_samples \
            ++model.eval_batch_size=$eval_batch_size \
            ++samples_dir="$samples_dir" \
            ++exp_name="$exp_name" \
            ++model.name_or_path="$model_path"
    fi

    echo "Evaluation completed for $model_path "
    echo "Save to $samples_dir/$exp_name"
    echo "----------------------------------------"
done

echo "All evaluations completed."