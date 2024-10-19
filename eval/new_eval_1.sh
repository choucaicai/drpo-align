#!/bin/bash

export CUDA_VISIBLE_DEVICES="1,"
# datasets="[hh]"
datasets="[hh_sample512]"
n_samples=512
eval_batch_size=4
config_path="eval_configs/vllm_config_t1.0"
mode="vllm_sample"
# mode="sample"
main_process_port=29502

# 
samples_dir="samples/qwen1.5-1.8-ada-ablation"


model_or_path_list=(

    # "/media/mil/LLMfinetune/qwen1.5-1.8-hf-cache/qwen1.5-1.8-hf-drpo-k8-simdiffndcg-com+adam-prob-rank0.1-lr5e-7_v2"

    # " /media/mil/LLMfinetune/qwen1.5-1.8-base-hh-sft"
    # "/media/mil/LLMfinetune/qwen1.5-1.8b-cache/qwen1.5-1.8-hf-dpo-hh"
    # "/media/mil/LLMfinetune/qwen1.5-1.8b-cache/hf-qwen1.5-1.8-full-btdpo-k8-beta0.1-lr5e-7"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-K-exp/qwen1.5-1.8-pldpo-k8-beta0.1-lr5e-7"
    # "/media/mil/LLMfinetune/qwen1.5-1.8-hf-cache/qwen1.5-1.8-hf-pro-lr5e-7"
    # "/media/mil/LLMfinetune/qwen1.5-1.8b-cache/qwen1.5-1.8-full-lambda-k8-beta0.1-lr5e-7-xx"
    # "/media/mil/LLMfinetune/qwen1.5-1.8-hf-cache/qwen1.5-1.8-hf-pirank-tau1.0-lr5e-7"
    # "/media/mil/LLMfinetune/qwen1.5-1.8-hf-cache/qwen1.5-1.8-hf-drpo-listnet-beta0.1-lr5e-7"
    # "/media/mil/LLMfinetune/qwen1.5-1.8-hf-cache/qwen1.5-1.8-hf-neuralsort-tau1.0-lr5e-7-nohard_loss"
    # "/media/mil/LLMfinetune/qwen1.5-1.8-hf-cache/qwen1.5-1.8-fastsort-k8-lr5e-7"
    # diff sort methods
    
    # checkpoint 
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-diffndcg-ablation/step_models/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-ada-rank0.2-lr5e-7-step"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-diffndcg-ablation/step_models/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-ada-rank0.2-lr5e-7-step/checkpoint-5000"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-diffndcg-ablation/step_models/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-ada-rank0.2-lr5e-7-step/checkpoint-10000"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-diffndcg-ablation/step_models/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-ada-rank0.2-lr5e-7-step/checkpoint-15000"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-diffndcg-ablation/step_models/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-ada-rank0.2-lr5e-7-step/checkpoint-20000"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-diffndcg-ablation/step_models/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-ada-rank0.2-lr5e-7-step/checkpoint-25000"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-diffndcg-ablation/step_models/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-ada-rank0.2-lr5e-7-step/checkpoint-30000"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-diffndcg-ablation/step_models/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-ada-rank0.2-lr5e-7-step/checkpoint-35000"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-diffndcg-ablation/step_models/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-ada-rank0.2-lr5e-7-step/checkpoint-39994"
    # "/media/mil/LLMfinetune/qwen1.5-1.8-hf-cache-diff-exp/qwen1.5-1.8-hf-drpo-k2-btdpo-lr5e-7-ada"
    # "/media/mil/LLMfinetune/qwen1.5-1.8-hf-cache-diff-exp/qwen1.5-1.8-hf-drpo-k2-btdpo-lr5e-7-ada0.4"
    # "/media/mil/LLMfinetune/qwen1.5-1.8-hf-cache-diff-exp/qwen1.5-1.8-hf-drpo-k2-btdpo-lr5e-7-ada0.6/checkpoint-17000"
    # "/media/mil/LLMfinetune/qwen1.5-1.8-hf-cache-diff-exp/qwen1.5-1.8-hf-drpo-k2-btdpo-lr5e-7-ada0.6/checkpoint-22500"
    # "/media/mil/LLMfinetune/qwen1.5-1.8-hf-cache-diff-exp/qwen1.5-1.8-hf-drpo-k2-btdpo-lr5e-7-ada0.6/checkpoint-27500"
    # "/media/mil/LLMfinetune/qwen1.5-1.8-hf-cache-diff-exp/qwen1.5-1.8-hf-drpo-k2-btdpo-lr5e-7-ada0.6/checkpoint-35500"
    # "/media/mil/LLMfinetune/qwen1.5-1.8-hf-cache-diff-exp/qwen1.5-1.8-hf-drpo-k2-btdpo-lr5e-7-ada0.6"
    # "/media/mil/LLMfinetune/qwen1.5-1.8-hf-cache-diff-exp/qwen1.5-1.8-hf-drpo-k2-btdpo-lr5e-7-dposcore"
    # "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache/qwen1.5-0.5-hf-neuralndcg-lr5e-7"
    "/media/mil/LLMfinetune/qwen1.5-0.5-hf-cache-diff-exp/qwen1.5-0.5-hf-drpo-k2-btdpo-lr5e-7-dposcore-1016"
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