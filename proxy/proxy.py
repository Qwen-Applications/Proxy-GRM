import json
import os
import asyncio
import uuid
from argparse import ArgumentParser
from multiprocessing import Process, Queue

from fastapi import FastAPI, Request
import uvicorn
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import functools
import re

app = FastAPI()

# Global variable definitions
engine = None
processor = None
sampling_params = None

def get_prompt(data):
    prompt = '''
    You are a highly precise evaluation assistant.\n Your Task:\n
    1. Carefully analyze the image and the question.\n
    2. Evaluate both responses strictly according to the provided Evaluation Rubric.\n
    3. Compare the strengths and weaknesses of both responses and provide a detailed reason enclosed within <think> and 

</think>
 tags.\n
    4. Decide which response is better based on the rubric with a single numeric choice enclosed within <answer> and </answer> tags:\n Output \"1\" if Response 1 is better.\n Output \"2\" if Response 2 is better.\n\n\n

    ### Data for Evaluation:\n
    <image>\n**User Question:** {question}\n
    **Response 1:** {response_1}\n
    **Response 2:** {response_2}\n
    **Evaluation Rubric:** {rubric}\n
    **Formatting Requirements:**\n<think>Your detailed reason goes here</think><answer>1/2</answer>
    '''
    return prompt.format(
        question=data.get('question', ''), 
        response_1=data.get('chosen', ''), 
        response_2=data.get('rejected', ''), 
        rubric=data.get('rubric', '')
    )

@app.post("/get_reward_mllm")
async def get_reward(request: Request):
    # Get request data
    data = await request.json()
    
    qs = get_prompt(data)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": data['image'], 
                },
                {"type": "text", "text": qs},
            ],
        }
    ]

    loop = asyncio.get_event_loop()
    # Use run_in_executor to avoid blocking the event loop
    text = await loop.run_in_executor(
        None, 
        functools.partial(processor.apply_chat_template, messages, tokenize=False, add_generation_prompt=True)
    )
    image_inputs, video_inputs, video_kwargs = await loop.run_in_executor(
        None,
        functools.partial(process_vision_info, messages, return_video_kwargs=True)
    )
        
    mm_data = {}
    if image_inputs:
        mm_data['image'] = image_inputs
    if video_inputs:
        mm_data['video'] = video_inputs

    llm_inputs = {
        'prompt': text,
        'multi_modal_data': mm_data,
        **video_kwargs
    }

    request_id = str(uuid.uuid4())
    results_generator = engine.generate(llm_inputs, sampling_params, request_id)
        
    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    critic = final_output.outputs[0].text

    reward = -1.0
    content_match = re.search(r"<answer>(.*?)</answer>", critic, re.DOTALL)
    if content_match:
        student_answer = content_match.group(1).strip()
    else:
        last_match = re.findall(r"[12]", critic)
        student_answer = last_match[-1] if last_match else None
    if student_answer:
        if '1' in student_answer and '1' in str(data.get('answer', '')):
            if '2' in student_answer:
                reward = -1.0
            else:
                reward = 1.0
        elif '2' in student_answer and '2' in str(data.get('answer', '')):
            if '1' in student_answer:
                reward = -1.0
            else:
                        reward = 1.0
        else:
            reward = -1.0
    else:
        reward = -1.0
            
    print('==========')
    print('critic')
    print(critic)
    print('reward')
    print(reward)
    print('==========')
    return {"reward": reward}


def verify_worker(input_queue, output_queue):
    """ Keep the original verification logic unchanged """
    while True:
        item = input_queue.get()
        if item is None: break
        content, sols, problem = item
        reward = 1.0 if content else 0.0
        output_queue.put(reward)

models = {
    3: 'Path to Qwen2.5-VL-3B-Instruct',
    7: 'Path to Qwen2.5-VL-7B-Instruct'
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_id", type=int, default=32)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20540)
    parser.add_argument("--tp", type=int, default=4)
    args = parser.parse_args()

    args.model_path = models[args.model_id]
    print(args.model_path)

    # Initialize the vLLM async engine
    engine_args = AsyncEngineArgs(
        model=args.model_path,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=0.9,  # Leave a bit of headroom for the system and other overhead
        trust_remote_code=True,
        max_model_len=32000,  # Adjust according to GPU memory
        enable_prefix_caching=True,
        # disable_log_requests=True,
        max_num_seqs=64,
        limit_mm_per_prompt={"image": 1, "video": 1},
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    # Initialize the Processor
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Set generation parameters
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=16384  # Evaluation outputs are usually short; limiting tokens improves speed
    )

    # Start the background math verification process
    input_queue = Queue()
    output_queue = Queue()
    p = Process(target=verify_worker, args=(input_queue, output_queue))
    p.start()

    # Start the FastAPI service
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    
    p.terminate()