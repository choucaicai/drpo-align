
export CUDA_VISIBLE_DEVICES="0,"

files=(
    # "samples/extra_exp/drpo-ablationt0.8/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-com+adam-prob-rank0.2-lr5e-7-adv2-3.json"

    # drpo ablation
    # "samples/extra_exp/drpo-ablationt/qwen1.5-0.5-hf-drpo-k8-diffsort-lr5e-7-dposcore.json"
    # "samples/extra_exp/drpo-ablationt/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-com+adam-prob-rank0.2-lr5e-7-adv2-3.json"
    # "samples/extra_exp/drpo-ablationt/qwen1.5-0.5-hf-drpo-k8-simplediffndcg-dposcore-lr5e-7.json"
    # "samples/extra_exp/drpo-ablationt/qwen1.5-0.5-hf-drpo-k8-diffsort-com+adam-prob-rank0.2-lr5e-7-adv2.json"

    # discounts
    # "samples/extra_exp/discounts-ablation/qwen1.5-0.5-hf-drpo-k8-simple-diffndcg-adam_prob_rank0.2_-lr5e-7-line_dis-ada.json"
    # "samples/extra_exp/discounts-ablation/qwen1.5-0.5-hf-drpo-k8-simple-diffndcg-adam_prob_rank0.2_-lr5e-7-sqrtdis-ada.json"

    # steps
    # "samples/extra_exp/steps-ablation/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-ada-rank0.2-lr5e-7-steps/checkpoint-5000.json"
    # "samples/extra_exp/steps-ablation/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-ada-rank0.2-lr5e-7-steps/checkpoint-10000.json"
    # "samples/extra_exp/steps-ablation/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-ada-rank0.2-lr5e-7-steps/checkpoint-15000.json"
    # "samples/extra_exp/steps-ablation/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-ada-rank0.2-lr5e-7-steps/checkpoint-20000.json"
    # "samples/extra_exp/steps-ablation/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-ada-rank0.2-lr5e-7-steps/checkpoint-25000.json"
    # "samples/extra_exp/steps-ablation/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-ada-rank0.2-lr5e-7-steps/checkpoint-30000.json"
    # "samples/extra_exp/steps-ablation/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-ada-rank0.2-lr5e-7-steps/checkpoint-35000.json"
    # "samples/extra_exp/steps-ablation/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-ada-rank0.2-lr5e-7-steps/checkpoint-39994.json"
    
    # "samples/base-sft/qwen1.5-0.5-base-hh-sft.json"

    # "samples/extra_exp/steps-ablation/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-ada-rank0.2-lr5e-7-steps/checkpoint-39994.json"
    # "samples/extra_exp/diff-sorts/qwen1.5-0.5-fastsort-k8-lr5e-7.json"
    # "samples/extra_exp/diff-sorts/qwen1.5-0.5-hf-drpo-btdpo-beta0.1-lr5e-7.json"
    # "samples/extra_exp/diff-sorts/qwen1.5-0.5-hf-lambda-beta0.1-lr5e-7.json"
    # "samples/extra_exp/diff-sorts/qwen1.5-0.5-hf-neuralsort-tau1.0-lr5e-7-nohard_loss-allrank.json"
    # "samples/extra_exp/diff-sorts/qwen1.5-0.5-hf-pirank-tau1.0-lr5e-7.json"
    # "samples/extra_exp/diff-sorts/qwen1.5-0.5-hf-pro-lr5e-7.json"
    # "samples/extra_exp/diff-sorts/qwen1.5-0.5-listnet-beta0.1-lr5e-7.json"
    


    # "samples/method_comp_1.8/hf-qwen1.5-1.8-full-btdpo-k8-beta0.1-lr5e-7.json"
    # "samples/method_comp_1.8/qwen1.5-1.8-fastsort-k8-lr5e-7.json"
    # "samples/method_comp_1.8/qwen1.5-1.8-full-lambda-k8-beta0.1-lr5e-7-xx.json"
    # "samples/method_comp_1.8/qwen1.5-1.8-hf-dpo-hh.json"
    # "samples/method_comp_1.8/qwen1.5-1.8-hf-drpo-k8-simdiffndcg-com+adam-prob-rank0.1-lr5e-7_v2.json"
    # "samples/method_comp_1.8/qwen1.5-1.8-hf-drpo-listnet-beta0.1-lr5e-7.json"
    # "samples/method_comp_1.8/qwen1.5-1.8-hf-neuralsort-tau1.0-lr5e-7-nohard_loss.json"
    # "samples/method_comp_1.8/qwen1.5-1.8-hf-pirank-tau1.0-lr5e-7.json"
    # "samples/method_comp_1.8/qwen1.5-1.8-hf-pro-lr5e-7.json"

    # "samples/extra_exp/discounts_ablation/qwen1.5-0.5-hf-drpo-k8-simple-diffndcg-dposcore-lr5e-7-noscale-line_dis.json"
    # "samples/extra_exp/discounts_ablation/qwen1.5-0.5-hf-drpo-k8-simple-diffndcg-dposcore-lr5e-7-noscale-sqrtdis.json"
    # "samples/extra_exp/discounts_ablation/qwen1.5-0.5-hf-drpo-k8-simple-diffndcg-dposcore-lr5e-7-noscale-square_dis.json"
    # "samples/extra_exp/discounts_ablation/qwen1.5-0.5-hf-drpo-k8-simple-diffndcg-dposcore-lr5e-7-noscale.json"
    
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/extra_exp/discounts-ablation/qwen1.5-0.5-hf-drpo-k8-simple-diffndcg-adam_prob_rank0.2_-lr5e-7-line_dis-ada.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/extra_exp/discounts-ablation/qwen1.5-0.5-hf-drpo-k8-simple-diffndcg-adam_prob_rank0.2_-lr5e-7-sqrtdis-ada.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/extra_exp/discounts-ablation/qwen1.5-0.5-hf-drpo-k8-simple-diffndcg-adam_prob_rank0.2_-lr5e-7-square-ada.json"
    # "samples/method_comp_0.5/qwen1.5-0.5-hf-drpo-pldpo-beta0.1-lr5e-7.json"
    # "samples/method_comp_1.8/qwen1.5-1.8-pldpo-k8-beta0.1-lr5e-7.json"

    # "samples/method_comp_0.5/qwen1.5-0.5-fastsort-k8-lr5e-7.json"
    # "samples/method_comp_0.5/qwen1.5-0.5-hf-dpo-beta0.1-lr5e-7.json"
    # "samples/method_comp_0.5/qwen1.5-0.5-hf-drpo-btdpo-beta0.1-lr5e-7.json"
    # "samples/method_comp_0.5/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-com+adam-prob-rank0.2-lr5e-7-adv2-3.json"
    # "samples/method_comp_0.5/qwen1.5-0.5-hf-drpo-pldpo-beta0.1-lr5e-7.json"
    # "samples/method_comp_0.5/qwen1.5-0.5-hf-lambda-beta0.1-lr5e-7.json"
    # "samples/method_comp_0.5/qwen1.5-0.5-hf-neuralsort-tau1.0-lr5e-7-nohard_loss-allrank.json"
    # "samples/method_comp_0.5/qwen1.5-0.5-hf-pirank-tau1.0-lr5e-7.json"
    # "samples/method_comp_0.5/qwen1.5-0.5-hf-pro-lr5e-7.json"
    # "samples/method_comp_0.5/qwen1.5-0.5-listnet-beta0.1-lr5e-7.json"

    # "samples/method_comp_0.5/qwen1.5-0.5-hf-neuralsort-tau1.0-lr5e-7-ada-step.json"
    # "samples/method_comp_0.5/qwen1.5-0.5-fastsort-k8-lr5e-7-ada-steps.json"
    # "samples/method_comp_0.5/qwen1.5-0.5-hf-pirank-tau1.0-lr5e-7-allrank-ada-step.json"
    # "samples/method_comp_0.5/qwen1.5-0.5-listnet-k6-beta0.1-lr5e-7-ada-steps.json"
    # "samples/extra_exp/discounts-ablation/qwen1.5-0.5-hf-drpo-k8-simple-diffndcg-adam_prob_rank0.2_-lr5e-7-line_dis-ada.json"
    # "samples/extra_exp/discounts-ablation/qwen1.5-0.5-hf-drpo-k8-simple-diffndcg-adam_prob_rank0.2_-lr5e-7-sqrtdis-ada.json"
    # "samples/extra_exp/discounts-ablation/qwen1.5-0.5-hf-drpo-k8-simple-diffndcg-adam_prob_rank0.2_-lr5e-7-square-ada.json"


    # test
    # "samples/extra_exp_test/drpo-ablation-t1/qwen1.5-0.5-fastsort-k8-lr5e-7-ada-steps.json"
    # "samples/extra_exp_test/drpo-ablation-t1/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-com+adam-prob-rank0.2-lr5e-7-adv2-3.json"
    # "samples/extra_exp_test/drpo-ablation-t1/qwen1.5-0.5-hf-pirank-tau1.0-lr5e-7-allrank-ada-step.json"
    # "samples/extra_exp_test/drpo-ablation-t1/qwen1.5-0.5-hf-drpo-k8-simple-diffndcg-adam_prob_rank0.2_-lr5e-7-square-ada.json"
    
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
    # "samples/extra_exp/drpo-ablationt/qwen1.5-0.5-hf-drpo-k8-diffsort-com+adam-prob-rank0.2-lr5e-7-adv2.json"
    # "samples/extra_exp/drpo-ablationt/qwen1.5-0.5-hf-drpo-k8-diffsort-lr5e-7-dposcore.json"
    # "samples/method_vllm_temp_t1.0-batch4/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-com+adam-prob-rank0.2-lr5e-7-adv2-3.json"
    # "samples/main_method_0.5b/qwen1.5-0.5-fastsort-k8-lr5e-7-ada-steps.json"
    # "samples/main_method_0.5b/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-com+adam-prob-rank0.2-lr5e-7-adv2-3.json"
    # "samples/main_method_0.5b/qwen1.5-0.5-hf-lambda-beta0.1-lr5e-7.json"
    # "samples/main_method_0.5b/qwen1.5-0.5-hf-pirank-tau1.0-lr5e-7-allrank-ada-step.json"


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
    
    # "samples/main_method_0.5b/qwen1.5-0.5-hf-drpo-pldpo-beta0.1-lr5e-7.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/qwen1.5-0.5-exp-K/qwen1.5-0.5-hf-drpo-btdpo-K2-beta0.1-lr5e-7.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/qwen1.5-0.5-exp-K/qwen1.5-0.5-hf-drpo-btdpo-K4-beta0.1-lr5e-7.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/qwen1.5-0.5-exp-K/qwen1.5-0.5-hf-drpo-btdpo-K6-beta0.1-lr5e-7.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/qwen1.5-0.5-exp-K/qwen1.5-0.5-hf-drpo-lambda-k2-beta0.1-lr5e-7.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/qwen1.5-0.5-exp-K/qwen1.5-0.5-hf-drpo-lambda-k6-beta0.1-lr5e-7.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/qwen1.5-0.5-exp-K/qwen1.5-0.5-hf-drpo-pldpo-k2-beta0.1-lr5e-7.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/qwen1.5-0.5-exp-K/qwen1.5-0.5-hf-drpo-pldpo-k4-beta0.1-lr5e-7.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/qwen1.5-0.5-exp-K/qwen1.5-0.5-hf-drpo-pldpo-k6-beta0.1-lr5e-7.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/qwen1.5-0.5-exp-K/qwen1.5-0.5-listnet-k2-beta0.1-lr5e-7.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/qwen1.5-0.5-exp-K/qwen1.5-0.5-listnet-k4-beta0.1-lr5e-7.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/qwen1.5-0.5-exp-K/qwen1.5-0.5-listnet-k6-beta0.1-lr5e-7.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/qwen1.5-0.5-exp-K/qwen1.5-1.8-pldpo-k8-beta0.1-lr5e-7.json"
    # "samples/qwen1.5-0.5-exp-K2/qwen1.5-0.5-fastsort-k2-lr5e-7.json"
    # "samples/qwen1.5-0.5-exp-K2/qwen1.5-0.5-hf-drpo-k2-btdpo-lr5e-7-ada.json"
    # "samples/qwen1.5-0.5-exp-K2/qwen1.5-0.5-hf-drpo-k2-diffsort-lr5e-7-dposcore.json"
    # "samples/qwen1.5-0.5-exp-K2/qwen1.5-0.5-hf-drpo-k2-listnet-beta0.1-lr5e-7.json"
    # "samples/qwen1.5-0.5-exp-K2/qwen1.5-0.5-hf-k2-neuralsort-tau1.0-lr5e-7.json"
    # "samples/qwen1.5-0.5-exp-K2/qwen1.5-0.5-hf-k2-pirank-tau1.0-lr5e-7.json"

    # "samples/main_method_0.5b/qwen1.5-0.5-hf-neuralsort-tau1.0-lr5e-7-nohard_loss-allrank-ad.json"
    # "samples/qwen1.5-0.5-neural-sort/qwen1.5-0.5-hf-neuralsort-tau1.0-lr5e-7-nohard_loss-allrank.json"
    # "samples/qwen1.5-0.5-neural-sort/qwen1.5-0.5-hf-neuralsort-tau1.0-lr5e-7-nohard_loss.json"
    # "samples/qwen1.5-0.5-neural-sort/qwen1.5-0.5-hf-neuralsort-tau1.0-lr5e-7.json"
    # "samples/qwen1.5-0.5-exp-K/qwen1.5-0.5-hf-drpo-lambda-K4-beta0.1-lr5e-7.json"
    # "samples/qwen1.5-0.5-drpo-steps/checkpoint-5000.json"
    # "samples/qwen1.5-0.5-drpo-steps/checkpoint-10000.json"
    # "samples/qwen1.5-0.5-drpo-steps/checkpoint-15000.json"
    # "samples/qwen1.5-0.5-drpo-steps/checkpoint-20000.json"
    # "samples/qwen1.5-0.5-drpo-steps/checkpoint-25000.json"
    # "samples/qwen1.5-0.5-drpo-steps/checkpoint-30000.json"
    # "samples/qwen1.5-0.5-drpo-steps/checkpoint-35000.json"
    # "samples/qwen1.5-0.5-drpo-steps/checkpoint-39994.json"
    # "samples/qwen1.5-0.5-drpo-steps/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-ada-rank0.2-lr5e-7-step.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/qwen1.5-0.5-ARP-ablation/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-com+adam-prob-rank0.2-lr5e-7-ada_coef0.5.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/qwen1.5-0.5-ARP-ablation/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-com+adam-prob-rank0.2-lr5e-7-gamma0.99.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/qwen1.5-0.5-ARP-ablation/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-com+adam-prob-rank0.2-lr5e-7-run2.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/qwen1.5-0.5-ARP-ablation/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-com+adam-prob-rank0.2-lr5e-7-run3.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/qwen1.5-0.5-ARP-ablation/qwen1.5-0.5-hf-drpo-k8-simdiffndcg-com+adam-prob-rank0.2-lr5e-7.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/qwen1.5-0.5-ARP-ablation/qwen1.5-0.5-hf-drpo-k8-simple-diffndcg-prob-rank0.2-lr5e-7-sim.json"
    # "/home/mil/MIL201/CC/alignment-handbook/eval/samples/qwen1.5-0.5-ARP-ablation/qwen1.5-0.5-hf-drpo-k8-simple-diffndcg-prob-rank0.2-lr5e-7.json"
    # "samples/qwen1.5-0.5-exp-K2/qwen1.5-1.8-hf-drpo-btdpo-K2-beta0.1-lr5e-7.json"
    # "samples/qwen1.5-1.8-ada-ablation/qwen1.5-1.8-hf-drpo-k2-btdpo-lr5e-7-ada0.4.json"
    # "samples/qwen1.5-1.8-ada-ablation/qwen1.5-1.8-hf-drpo-k2-btdpo-lr5e-7-ada0.6.json"
    # "samples/main_method_1.8b/qwen1.5-1.8-hf-dpo-hh.json"
    # "samples/qwen1.5-1.8-ada-ablation/checkpoint-27500.json"
    # "samples/qwen1.5-1.8-ada-ablation/qwen1.5-1.8-hf-drpo-k2-btdpo-lr5e-7-dposcore.json"
    # "samples/qwen1.5-1.8-ada-ablation/qwen1.5-0.5-hf-neuralndcg-lr5e-7.json"
    "samples/qwen1.5-1.8-ada-ablation/qwen1.5-0.5-hf-drpo-k2-btdpo-lr5e-7-dposcore-1016.json"
)
# 
result_dir="rewards_eval/qwen1.5-1.8-K2"
# 
# sft_file="samples/qwen1.5-1.8b-sft-512/qwen1.5-1.8-base-sft.json"
sft_file="samples/base-sft/qwen1.5-0.5-base-hh-sft.json"

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

