<div align="center">
<h1>Rationale Matters: Learning Transferable Rubrics via Proxy-Guided Critique for VLM Reward Models</h1>


<!-- Badges -->
<a><img 
     src="https://img.shields.io/badge/Qwen-Applications-4433FF?style=for-the-badge&logo=data:image/png;base64,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" 
     alt="Github"></a> 
<a href="https://arxiv.org/abs/2603.16600"><img src="https://img.shields.io/badge/arXiv-2603.16600-b31b1b.svg?style=for-the-badge" alt="arXiv"></a> 
<a href="https://github.com/Qwen-Applications/Proxy-GRM"><img src="https://img.shields.io/badge/Github-Code-black?style=for-the-badge&logo=github" alt="Github"></a> 
<a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg?style=for-the-badge" alt="License"></a>


<p align="center">
  <i><b> <img src="https://img.alicdn.com/imgextra/i2/O1CN01FPcQDy1WTPjPX6IH9_!!6000000002789-2-tps-96-96.png" width="16px"  style="vertical-align: middle;"> Qwen Large Model Application Team, Alibaba</b></i>
</p>

<p align="center">
  <img src="figure/teaser_proxy_grm.png" alt="Teaser Figure" width="90%"/>
</p>
<p align="center">
  <b>Figure 1.</b> Teaser Figure: Rationale Matters: Learning Transferable Rubrics via Proxy-Guided Critique for VLM Reward Models.
</p>


<p align="center">
  <img src="figure/framework.png" alt="Proxy-GRM framework" width="90%"/>
</p>
<p align="center">
  <b>Figure 2.</b> Overview of the Proxy-GRM framework, Training data distillation, Proxy model training and Proxy-GRM training.
</p>

</div>

## ⚙️ 1. Setup and Training
### Dataset Preparation
Download the following datasets: [LLava-Critic-113k](https://huggingface.co/datasets/lmms-lab/llava-critic-113k), [RLAIF-V](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset), [RLHF-V](https://huggingface.co/datasets/llamafactory/RLHF-V) and  [MMIF-23K](https://huggingface.co/datasets/ChrisDing1105/MMIF-23k). Then, distill all datasets using the Qwen3-VL-235B-A22B-Instruct model with the following prompt.

![Distillation Prompt](./figure/distillation_prompt.png 'Distillation Prompt')


### For SFT Stage
We use the [**ms-swift**](https://swift.readthedocs.io/en/latest/GetStarted/Quick-start.html) training framework for the SFT stage; therefore, the training datasets shoule be converted to the [**ms-swift multimodal**](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html#multimodal) data format.
#### Environment
```bash
pip install -r requirements_sft.txt
```
#### Training
```bash
cd Proxy-GRM/scripts/sft

# swanlab
export swanlab_token='Your swanlab api key.'
export swanlab_project='Your swanlab project name.'
export swanlab_mode='Your swanlab mode.'
export SWANLAB_LOG_DIR='Your swanlab local log dir.'

export MASTER_PORT=

export gpus=0,1,2,3
IFS=',' read -ra gpu_array <<< "$gpus"

export gpu_count=${#gpu_array[@]}

export data_path='Path to your training file.'
export model_path='Path to your training model.'

export freeze_vit=false
export freeze_llm=false
export freeze_aligner=false

export swanlab_exp_name='Your swanlab experiment name.'

export output_path='Your saved model path.'

bash train_scripts.sh
```

### For RL Stage
We use the **Verl** training framework for the RL stage; therefore, the training datasets shoule be converted to the parquet data format.
#### Environment
```bash
conda create -n verl_env python=3.12 -y
conda activate verl_env

cd Proxy-GRM/verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh

pip install --no-deps -e .
```
#### RL training

##### Start the proxy model
After launching the proxy model, add its **IP address** and **port** to the **RL training script**.
```bash 
# 1. get ip address of the proxy model
hostname -I

# 2. start
cd Proxy-GRM/proxy
python proxy.py --port port --model_id model_id
```

##### VERL training
Set the ip address and port of the proxy model in the script below.
```bash
cd Proxy-GRM/scripts/rl
export train_file='Path to your training file.'
export test_file='Path to your testing file.'

export model_path='Path to your training model'
# for swanlab
export project_name='Your swanlab project name.'
export experiment_name='Your swanlab experiment name.'
export SWANLAB_MODE='Your swanlab mode.'
export SWANLAB_API_KEY='Your swanlab api key.'
export SWANLAB_LOG_DIR='Your swanlab local log dir'

export save_path='Path to your saved model.'

export REWARD_PATH=Proxy-GRM/verl/verl/utils/reward_score/grm_reward_with_instruct.py
export REWARD_FUNC_NAME=compute_score
export epochs=1
# for proxy model
export ip='IP address of the proxy model.'
export api_port='Port of the proxy model.'
export use_api='1'
export use_format='1'
export gpus=0,1,2,3,4,5,6,7

export batch_size=128

bash scripts.sh
```


## 🚀 Evaluation
Download the following evluation datasets: [VL-RewardBench](https://huggingface.co/datasets/MMInstruction/VL-RewardBench), [Multimodal RewardBench](https://huggingface.co/datasets/syhuggingface/multimodal_rewardbench) and [MM-RLHF-RewardBench](https://huggingface.co/datasets/yifanzhang114/MM-RLHF-RewardBench). Then, prepare all evaluation data in a **JSONL** file using the following format.
```json
{"id": "LRVInstruction-000000007746", "prompt": "The dog has one floppy ear and one straight ear.", "chosen": "There is no existence of the dog's ear shape in the image description.", "rejected": "The dog is brown, white, and black in color with a blue background on its face. It lies down next to the mirror, looking at itself from above or below it. Its ears are both floppy and straight.", "image": "Path to LRVInstruction-000000007746 image file", "chosen_reason": "", "rejected_reason": ""}
```
### Evaluation on VL-RewardBench
```bash
cd Proxy-GRM/eval
# step 1: generate the evaluation result jsonl
python vl_reward_bench.py --model-path your_model_path --question_file path_to_the_vl_rewardbench_evalution_jsonl_file --answers_file path_to_the_saved_jsonl_file

# step 2: calculate the evaluation metric
python get_acc_vl_reward_bench.py --file_path path_to_the_saved_jsonl_file
```

### Evaluation on Multimodal RewardBench
```bash
cd Proxy-GRM/eval
# step 1: generate the evaluation result jsonl
python multimodal_reward_bench.py --model-path your_model_path --question_file path_to_the_multimodal_rewad_bench_evalution_jsonl_file --answers_file path_to_the_saved_jsonl_file

# step 2: calculate the evaluation metric
python get_acc_multimodal_reward_bench.py --file_path path_to_the_saved_jsonl_file
```

### Evaluation on MM-RLHF-RewardBench
```bash
cd Proxy-GRM/eval
# step 1: generate the evaluation result jsonl
python mmrlhf_reward_bench.py --model-path your_model_path --question_file path_to_the_mm_rlhf_rewardbench_evalution_jsonl_file --answers_file path_to_the_saved_jsonl_file

# step 2: calculate the evaluation metric
python get_acc_mmrlhf_reward_bench.py --file_path path_to_the_saved_jsonl_file
```


## 📁 Repository Structure

```
Proxy-GRM/
├── README.md
├── requirements_sft.txt
├── verl/                                   # verl framework
├── eval/
│   ├── vl_reward_bench.py                  # VL-RewardBench Evaluation
│   ├── get_acc_vl_reward_bench.py          # get accuracy of VL-RewardBench Evaluation
│   │
│   ├── multimodal_reward_bench.py          # MultiModal-Reward Bench Evaluation
│   ├── get_acc_multimodal_reward_bench.py  # get accuracy of  MultiModal-Reward Bench Evaluation
│   │
│   ├── mmrlhf_reward_bench.py              # MM-RLHF-Reward Bench Evaluation
│   ├── get_acc_vl_reward_bench.py          # get accuracy of MM-RLHF-Reward Bench Evaluation
│   │
│
└── proxy/
    ├── proxy.py                            # proxy model
```

## 🙏 Acknowledgements

We build on and thank the open-source communities behind QwenVL, verl, vLLM, and the benchmark datasets (VL-RewardBench, Multimodal RewardBench and MM-RLHF-RewardeBench).

## 📜 Citation

If you find our work useful, please consider citing:

```bibtex
@misc{qiu2026rationalematterslearningtransferable,
      title={Rationale Matters: Learning Transferable Rubrics via Proxy-Guided Critique for VLM Reward Models}, 
      author={Weijie Qiu and Dai Guan and Junxin Wang and Zhihang Li and Yongbo Gai and Mengyu Zhou and Erchao Zhao and Xiaoxi Jiang and Guanjun Jiang},
      year={2026},
      eprint={2603.16600},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.16600}, 
}
```
