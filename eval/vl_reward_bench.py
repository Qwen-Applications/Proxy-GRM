import argparse
import os
import json
from tqdm import tqdm
import re
import random
import torch

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


system_prompt = '''You are provided with an image and a question for this image. Please act as an impartial judge and evaluate the quality of the responses provided by two AI Chatbots to the Client's question displayed below.\n\n
### Evaluation Process:\n
1. **Develop a Multi-Dimensional Rubric**:\n
   Generate evaluation criteria (rubric) tailored to the image, Client's question and context, enclosed in <rubric>...</rubric> tags.\n

2. **Assign Weights**: Distribute weights (totaling 100%) to all rubric items based on their relative importance to the specific task.\n

3. **Justify and Compare**: Inside <rubric>, include a <justify> section. Then, compare both responses based on these criteria in the <eval> section.\n\n

Important Notes:\n
- Be objective and base your evaluation only on the visual evidence in the image, question and the content of the responses.\n
- Do not let response order, length, or Chatbot names affect your judgment.\n\n
- You need to choose which response is better for the given question and provide a detailed reason enclosed within <eval> and </eval> tags. Conclude with a single numeric choice enclosed within <answer> and </answer> tags:\n Output \"1\" if Response 1 is better.\n Output \"2\" if Response 2 is better.\n\n


### Output Format:\n
Your output must follow this format:\n\n

<rubric>\n
  [List each rubric item with its description and assigned weight]\n
  <justify> \n
    Explain why these criteria (including the mandatory and custom ones) and weights were chosen based on the specific context of the image and question.\n
  </justify>\n
</rubric>\n\n

<eval>\n
  [Provide a detailed comparison]\n
  - Use <quote_1> or <summary_1> for Chatbot 1's performance.\n
  - Use <quote_2> or <summary_2> for Chatbot 2's performance.\n
  - Compare how each chatbot met or failed each specific rubric item.\n
</eval>\n\n

<answer>1/2</answer>
'''


def make_conv(prompt, chosen, rejected, has_image=True):
    prompt_template = (
        "<image>\n[Client Question]\n{question}\n\n[The Start of Chatbot 1's Response]\n{answer_1}\n[The End of Chatbot 1's Response]\n\n"
        "[The Start of Chatbot 2's Response]\n{answer_2}\n[The End of Chatbot 2's Response]"
    )
    formatted_prompt = system_prompt + prompt_template.format(
        question=prompt, answer_1=chosen, answer_2=rejected
    )
    return formatted_prompt


def parse_answer(critic_text: str):
    content_match = re.search(r"<answer>(.*?)</answer>", critic_text, re.DOTALL)
    if content_match:
        ans = content_match.group(1).strip()
        if "1" in ans and "2" not in ans:
            return "1"
        if "2" in ans and "1" not in ans:
            return "2"
    last_match = re.findall(r"[12]", critic_text)
    return last_match[-1] if last_match else None


@torch.inference_mode()
def eval_model(args):
    # device / dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # generation params
    gen_kwargs = dict(
        max_new_tokens=4096,
        do_sample=False,
        temperature=0.0,
    )

    questions = []
    with open(args.question_file, "r", encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line))

    if os.path.exists(args.answers_file):
        os.remove(args.answers_file)
    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    ans_file = open(args.answers_file, "a+", encoding="utf-8")

    for i, line in tqdm(enumerate(questions), total=len(questions)):
        video_file = line.get("video", None)
        image_file = line.get("image", None)

        chosen, rejected = line["chosen"], line["rejected"]

        final_reverse = False
        if random.random() >= 0.5:
            final_reverse = True
            chosen, rejected = rejected, chosen

        qs = make_conv(line["prompt"], chosen, rejected, has_image=bool(video_file or image_file))

        file_mm = image_file if image_file is not None else video_file
        messages = [
            {
                "role": "user",
                "content": [
                    ({"type": "image", "image": file_mm} if image_file is not None else {"type": "video", "video": file_mm}),
                    {"type": "text", "text": qs},
                ],
            }
        ]

        # chat template -> text
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # vision preprocessing
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

        # processor 
        inputs = processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt",
        )

        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        if video_kwargs:
            gen_kwargs_use = dict(gen_kwargs)
            gen_kwargs_use.update(video_kwargs)
        else:
            gen_kwargs_use = gen_kwargs

        output_ids = model.generate(**inputs, **gen_kwargs_use)

        input_len = inputs["input_ids"].shape[1]
        gen_ids = output_ids[0][input_len:]

        critic = processor.decode(gen_ids, skip_special_tokens=True).strip()

        student_answer = parse_answer(critic)
        if student_answer == "1":
            rewards = [1, 0]
        elif student_answer == "2":
            rewards = [0, 1]
        else:
            rewards = [0, 0]

        line_out = dict(line)
        line_out["qs"] = qs
        line_out["rewards"] = rewards
        line_out["critic"] = critic
        line_out["reverse"] = final_reverse

        ans_file.write(json.dumps(line_out, ensure_ascii=False) + "\n")
        ans_file.flush()

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--question-file", type=str, default="jsonl file")
    parser.add_argument("--answers-file", type=str, default="jsonl file")
    args = parser.parse_args()


    if os.path.exists(args.answers_file):
        with open(args.answers_file, "r", encoding="utf-8") as reader1, open(args.question_file, "r", encoding="utf-8") as reader2:
            lines1 = reader1.readlines()
            lines2 = reader2.readlines()
            if len(lines1) != len(lines2):
                exit()

    eval_model(args)
