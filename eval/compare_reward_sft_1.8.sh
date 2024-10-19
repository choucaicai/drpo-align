
export CUDA_VISIBLE_DEVICES="1,"

files=(

    "/home/mil/MIL201/CC/alignment-handbook/eval/samples/main_method_1.8b/hf-qwen1.5-1.8-full-btdpo-k8-beta0.1-lr5e-7.json"
    "/home/mil/MIL201/CC/alignment-handbook/eval/samples/main_method_1.8b/qwen1.5-1.8-fastsort-k8-lr5e-7.json"
    "/home/mil/MIL201/CC/alignment-handbook/eval/samples/main_method_1.8b/qwen1.5-1.8-full-lambda-k8-beta0.1-lr5e-7-xx.json"
    "/home/mil/MIL201/CC/alignment-handbook/eval/samples/main_method_1.8b/qwen1.5-1.8-hf-dpo-hh.json"
    "/home/mil/MIL201/CC/alignment-handbook/eval/samples/main_method_1.8b/qwen1.5-1.8-hf-drpo-k8-simdiffndcg-com+adam-prob-rank0.1-lr5e-7_v2.json"
    "/home/mil/MIL201/CC/alignment-handbook/eval/samples/main_method_1.8b/qwen1.5-1.8-hf-drpo-listnet-beta0.1-lr5e-7.json"
    "/home/mil/MIL201/CC/alignment-handbook/eval/samples/main_method_1.8b/qwen1.5-1.8-hf-neuralsort-tau1.0-lr5e-7-nohard_loss.json"
    "/home/mil/MIL201/CC/alignment-handbook/eval/samples/main_method_1.8b/qwen1.5-1.8-hf-pirank-tau1.0-lr5e-7.json"
    "/home/mil/MIL201/CC/alignment-handbook/eval/samples/main_method_1.8b/qwen1.5-1.8-hf-pro-lr5e-7.json"
    "/home/mil/MIL201/CC/alignment-handbook/eval/samples/main_method_1.8b/qwen1.5-1.8-pldpo-k8-beta0.1-lr5e-7.json"
)
# 
result_dir="rewards_eval/drpo-main-ms-sft"
# 
# sft_file="samples/qwen1.5-1.8b-sft-512/qwen1.5-1.8-base-sft.json"
sft_file="samples/base-sft/qwen1.5-1.8-base-hh-sft.json"

# compare result
# result_dir="rewards_eval/extra-exp"
mkdir -p "$result_dir"

echo "SFT files: $sft_file"
echo "Result Dir: $result_dir"
echo "========================="

for file in "${files[@]}"
do
    # 执行 Python 命令
    base_name=$(basename "$file" .json)
    output_result="${result_dir}/${base_name}-sft/"

    echo "Compare: $file"
    echo "Result folder: $output_result"

    # python tools/combine_policy_sft.py --samples_file "$file"  --sft_file "$sft_file" --output_path "$output_file"
    python tools/compare_reward_sft.py --samples_file "$file"  --sft_file "$sft_file" --output_path "$output_result"
done

