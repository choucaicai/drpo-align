
files=(

    # "samples/method_comp_0.5/qwen1.5-0.5-hf-drpo-pldpo-beta0.1-lr5e-7.json"
    # "samples/method_comp_1.8/hf-qwen1.5-1.8-full-btdpo-k8-beta0.1-lr5e-7.json"
    # "samples/method_comp_1.8/qwen1.5-1.8-fastsort-k8-lr5e-7.json"
    # "samples/method_comp_1.8/qwen1.5-1.8-full-lambda-k8-beta0.1-lr5e-7-xx.json"
    # "samples/method_comp_1.8/qwen1.5-1.8-hf-dpo-hh.json"
    # "samples/method_comp_1.8/qwen1.5-1.8-hf-drpo-listnet-beta0.1-lr5e-7.json"
    # "samples/method_comp_1.8/qwen1.5-1.8-hf-neuralsort-tau1.0-lr5e-7-nohard_loss.json"
    # "samples/method_comp_1.8/qwen1.5-1.8-hf-pirank-tau1.0-lr5e-7.json"
    # "samples/method_comp_1.8/qwen1.5-1.8-hf-pro-lr5e-7.json"
    # "samples/extra_exp/steps-ablation/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-ada-rank0.2-lr5e-7-steps/checkpoint-39994.json"
    
    # "samples/method_comp_0.5/qwen1.5-0.5-fastsort-k8-lr5e-7-ada-steps.json"
    # "samples/method_comp_0.5/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-com+adam-prob-rank0.2-lr5e-7-adv2-3.json"

    # "samples/method_comp_0.5/qwen1.5-0.5-hf-drpo-btdpo-beta0.1-lr5e-7.json"
    # "samples/method_comp_0.5/qwen1.5-0.5-hf-lambda-beta0.1-lr5e-7.json"
    # "samples/method_comp_0.5/qwen1.5-0.5-fastsort-k8-lr5e-7.json"
    # "samples/method_comp_0.5/qwen1.5-0.5-hf-drpo-pldpo-beta0.1-lr5e-7.json"
    # "samples/method_comp_0.5/qwen1.5-0.5-hf-neuralsort-tau1.0-lr5e-7-nohard_loss-allrank.json"
    # "samples/method_comp_0.5/qwen1.5-0.5-hf-pirank-tau1.0-lr5e-7.json"
    # "samples/method_comp_0.5/qwen1.5-0.5-hf-pro-lr5e-7.json"
    # "samples/base-sft/qwen1.5-0.5-base-hh-sft.json"
    # "samples/extra_exp/drpo-ablationt/qwen1.5-0.5-hf-drpo-k8-diffsort-com+adam-prob-rank0.2-lr5e-7-adv2.json"
    # "samples/extra_exp/drpo-ablationt/qwen1.5-0.5-hf-drpo-k8-diffsort-lr5e-7-dposcore.json"
    # "samples/method_comp_0.5/qwen1.5-0.5-listnet-beta0.1-lr5e-7.json"

    # "samples/method_comp_0.5/qwen1.5-0.5-hf-dpo-beta0.1-lr5e-7.json"
    

    # "samples/base-sft/qwen1.5-1.8-base-hh-sft.json"
    # "samples/method_comp_1.8/qwen1.5-1.8-hf-drpo-k8-simdiffndcg-com+adam-prob-rank0.1-lr5e-7_v2.json"
    

    # "samples/method_comp_1.8/hf-qwen1.5-1.8-full-btdpo-k8-beta0.1-lr5e-7.json"
    # "samples/method_comp_1.8/qwen1.5-1.8-fastsort-k8-lr5e-7.json"
    # "samples/method_comp_1.8/qwen1.5-1.8-full-lambda-k8-beta0.1-lr5e-7-xx.json"
    # "samples/method_comp_1.8/qwen1.5-1.8-hf-dpo-hh.json"
    # "samples/method_comp_1.8/qwen1.5-1.8-hf-drpo-k8-simdiffndcg-com+adam-prob-rank0.1-lr5e-7_v2.json"
    # "samples/method_comp_1.8/qwen1.5-1.8-hf-drpo-listnet-beta0.1-lr5e-7.json"
    # "samples/method_comp_1.8/qwen1.5-1.8-hf-neuralsort-tau1.0-lr5e-7-nohard_loss.json"
    # "samples/method_comp_1.8/qwen1.5-1.8-hf-pirank-tau1.0-lr5e-7.json"
    # "samples/method_comp_1.8/qwen1.5-1.8-hf-pro-lr5e-7.json"
    # "samples/method_comp_1.8/qwen1.5-1.8-pldpo-k8-beta0.1-lr5e-7.json"
    # "samples/main_method_0.5b/qwen1.5-0.5-fastsort-k8-lr5e-7-ada-steps.json"
    # "samples/main_method_0.5b/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-com+adam-prob-rank0.2-lr5e-7-adv2-3.json"
    # "samples/main_method_0.5b/qwen1.5-0.5-hf-lambda-beta0.1-lr5e-7.json"
    
)
# compare result
judge_bot="gpt-4-1106-preview"
result_dir="gpt-results/qwen-drpo-main-noasync-uni-gpt-4-1106-ms"

mkdir -p "$result_dir"
echo "Result Dir: $result_dir"

for file in "${files[@]}"
do
    base_name=$(basename "$file" .json)
    result_file="${result_dir}/${base_name}-with_chosen.json"
    echo "-------------------------------------------------"
    echo "Strting Compare: $file"
    
    # python tools/combine_policy_sft.py --samples_file "$file"  --sft_file "$sft_file" --output_path "$output_file"
    python tools/compare.py -f "$file"  -mc 512 -bk chosen -ck policy --r "$result_file" --judge "$judge_bot"
    # python tools/compare_async.py -f "$file"  -mc 512 -bk chosen -ck policy --r "$result_file"

    echo "Completing Compare, Result save to : $result_file"
    echo "-------------------------------------------------"

done