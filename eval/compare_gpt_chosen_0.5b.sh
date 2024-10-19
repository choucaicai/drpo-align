
files=(
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/main_method_0.5b/qwen1.5-0.5-fastsort-k8-lr5e-7-ada-steps.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/main_method_0.5b/qwen1.5-0.5-fastsort-k8-lr5e-7.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/main_method_0.5b/qwen1.5-0.5-hf-dpo-beta0.1-lr5e-7.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/main_method_0.5b/qwen1.5-0.5-hf-drpo-btdpo-beta0.1-lr5e-7.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/main_method_0.5b/qwen1.5-0.5-hf-drpo-k8-diffsort-com+adam-prob-rank0.2-lr5e-7-adv2.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/main_method_0.5b/qwen1.5-0.5-hf-drpo-k8-diffsort-lr5e-7-dposcore.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/main_method_0.5b/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-com+adam-prob-rank0.2-lr5e-7-adv2-3.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/main_method_0.5b/qwen1.5-0.5-hf-drpo-k8-simple-diffndcg-adam_prob_rank0.2_-lr5e-7-line_dis-ada.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/main_method_0.5b/qwen1.5-0.5-hf-drpo-k8-simple-diffndcg-adam_prob_rank0.2_-lr5e-7-sqrtdis-ada.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/main_method_0.5b/qwen1.5-0.5-hf-drpo-k8-simple-diffndcg-adam_prob_rank0.2_-lr5e-7-square-ada.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/main_method_0.5b/qwen1.5-0.5-hf-drpo-k8-simple-diffndcg-dposcore-lr5e-7-noscale-line_dis.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/main_method_0.5b/qwen1.5-0.5-hf-drpo-k8-simple-diffndcg-dposcore-lr5e-7-noscale-sqrtdis.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/main_method_0.5b/qwen1.5-0.5-hf-drpo-k8-simple-diffndcg-dposcore-lr5e-7-noscale-square_dis.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/main_method_0.5b/qwen1.5-0.5-hf-drpo-k8-simplediffndcg-dposcore-lr5e-7.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/main_method_0.5b/qwen1.5-0.5-hf-lambda-beta0.1-lr5e-7.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/main_method_0.5b/qwen1.5-0.5-hf-neuralsort-tau1.0-lr5e-7-nohard_loss-allrank.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/main_method_0.5b/qwen1.5-0.5-hf-pirank-tau1.0-lr5e-7-allrank-ada-step.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/main_method_0.5b/qwen1.5-0.5-hf-pirank-tau1.0-lr5e-7.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/main_method_0.5b/qwen1.5-0.5-hf-pro-lr5e-7.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/main_method_0.5b/qwen1.5-0.5-listnet-beta0.1-lr5e-7.json"

    "/home/mil/MIL201/CC/alignment-handbook/eval/samples/qwen1.5-0.5-exp-K/qwen1.5-0.5-hf-drpo-btdpo-K2-beta0.1-lr5e-7.json"
    "/home/mil/MIL201/CC/alignment-handbook/eval/samples/qwen1.5-0.5-exp-K/qwen1.5-0.5-hf-drpo-btdpo-K4-beta0.1-lr5e-7.json"
    "/home/mil/MIL201/CC/alignment-handbook/eval/samples/qwen1.5-0.5-exp-K/qwen1.5-0.5-hf-drpo-btdpo-K6-beta0.1-lr5e-7.json"
    "/home/mil/MIL201/CC/alignment-handbook/eval/samples/qwen1.5-0.5-exp-K/qwen1.5-0.5-hf-drpo-lambda-k2-beta0.1-lr5e-7.json"
    "/home/mil/MIL201/CC/alignment-handbook/eval/samples/qwen1.5-0.5-exp-K/qwen1.5-0.5-hf-drpo-lambda-k6-beta0.1-lr5e-7.json"
    "/home/mil/MIL201/CC/alignment-handbook/eval/samples/qwen1.5-0.5-exp-K/qwen1.5-0.5-hf-drpo-pldpo-k2-beta0.1-lr5e-7.json"
    "/home/mil/MIL201/CC/alignment-handbook/eval/samples/qwen1.5-0.5-exp-K/qwen1.5-0.5-hf-drpo-pldpo-k4-beta0.1-lr5e-7.json"
    "/home/mil/MIL201/CC/alignment-handbook/eval/samples/qwen1.5-0.5-exp-K/qwen1.5-0.5-hf-drpo-pldpo-k6-beta0.1-lr5e-7.json"
    "/home/mil/MIL201/CC/alignment-handbook/eval/samples/qwen1.5-0.5-exp-K/qwen1.5-0.5-listnet-k2-beta0.1-lr5e-7.json"
    "/home/mil/MIL201/CC/alignment-handbook/eval/samples/qwen1.5-0.5-exp-K/qwen1.5-0.5-listnet-k4-beta0.1-lr5e-7.json"
    "/home/mil/MIL201/CC/alignment-handbook/eval/samples/qwen1.5-0.5-exp-K/qwen1.5-0.5-listnet-k6-beta0.1-lr5e-7.json"
    "/home/mil/MIL201/CC/alignment-handbook/eval/samples/qwen1.5-0.5-exp-K/qwen1.5-1.8-pldpo-k8-beta0.1-lr5e-7.json"
)

# compare result
judge_bot="gpt-4o-2024-08-06"
result_dir="gpt-results/qwen-exp-k-noasync-gpt-4o-2024-08-06-ms"

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