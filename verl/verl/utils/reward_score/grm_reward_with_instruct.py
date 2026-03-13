import requests
import re
import os
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


api_port = os.getenv('api_port', '20240')

POOL_SIZE = int(os.environ.get("HTTP_POOL_SIZE", "100"))

ip = os.getenv('ip', '0.0.0.0')

use_api = int(os.getenv('use_api', '0'))
use_format = int(os.getenv('use_format', '0'))
use_fb = int(os.getenv('use_fb', '0'))

REMOTE_RM_URL = os.environ.get("REMOTE_RM_URL", f"http://{ip}:{api_port}/get_reward_mllm")


def create_session():
    s = requests.Session()
    
    retries = Retry(
        total=3, 
        backoff_factor=0.5, 
        status_forcelist=[500, 502, 503, 504],
        raise_on_status=False 
    )
    
    adapter = HTTPAdapter(
        pool_connections=POOL_SIZE, 
        pool_maxsize=POOL_SIZE, 
        max_retries=retries
    )
    
    s.mount('http://', adapter)
    s.mount('https://', adapter)
    return s

session = create_session()

def extract_characters_regex(text):
    return re.sub(r'[^a-zA-Z0-9]', '', text)

def verify_math(solution_str, ground_truth):
    """
    """
    try:
        sol_match = re.search(r"<answer>(.*?)</answer>", ground_truth, re.DOTALL)
        gt_content = sol_match.group(1).strip() if sol_match else ground_truth.strip()

        content_match = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
        st_content = content_match.group(1).strip() if content_match else solution_str.strip()
        
        student_answer = extract_characters_regex(st_content)
        ground_truth_clean = extract_characters_regex(gt_content)
        
        if '1' in student_answer and '1' in ground_truth_clean:
            if '2' in student_answer:
                return -1.0
            else:
                return 1.0
        elif '2' in student_answer and '2' in ground_truth_clean:
            if '1' in student_answer:
                return -1.0
            else:
                return 1.0
        return -1.0
    except Exception:
        return -1.0

def verify_format(content):
    """
    Verify if the string meets the format requirements:
    - Must start with <think> and end with </answer>
    - Must contain exactly one pair of <think>...</think> and <answer>...</answer> tags
    - No extra characters allowed between </think> and <answer> tags
    """
    content = content.replace('\n', '')
    format_pattern = r"^<think>.*?</think><summary>.*?</summary><answer>.*?</answer>$"
    think_count = content.count("<think>")
    answer_count = content.count("<answer>")
    summary_count = content.count('<summary>')
    normal_format_reward = bool(re.match(format_pattern, content, re.DOTALL)) and think_count == 1 and answer_count == 1 and summary_count == 1

    format_pattern = r"^<rubric>.*?</rubric><eval>.*?</eval><answer>.*?</answer>$"
    rubric_count = content.count("<rubric")
    eval_count = content.count("<eval>")
    answer_count = content.count("<answer>")
    double_rubric_reward = bool(re.match(format_pattern, content, re.DOTALL)) and rubric_count == 1 and answer_count == 1 and eval_count == 1

    return normal_format_reward or double_rubric_reward

def instruct_verify(solution_str, ground_truth, extra_info):

    if not extra_info or not isinstance(extra_info, dict):
        return 0.0
    rubric = solution_str.split('</rubric>')[0] + '</rubric>'
    if '<justify>' in rubric:
        rubric = rubric.split('<justify>')[0] + '</rubric>'
    
    data = {
        'question': extra_info.get('question', ''),
        'chosen': extra_info.get('chosen', ''),
        'rejected': extra_info.get('rejected', ''),
        'image': extra_info.get('image', ''),
        'answer': extra_info.get('answer', ''),
        'rubric': rubric
    }

    try:
        response = session.post(REMOTE_RM_URL, json=data, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            return float(result.get('reward', 0.0))
        else:
            return 0.0
            
    except Exception as e:
        return 0.0

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    acc_reward = verify_math(solution_str, ground_truth)
    
    if use_api == 1:
        instruct_reward = instruct_verify(solution_str, ground_truth, extra_info)
        if use_fb == 1:
            if instruct_reward == -1 and acc_reward == 1:
                acc_reward = -1
            elif instruct_reward == 1 and acc_reward == -1:
                acc_reward = -1
                instruct_reward = -1
        elif use_fb == 2:
            if instruct_reward == -1 and acc_reward == 1:
                acc_reward = -1
            instruct_reward = 0
        elif use_fb == 3:
            if acc_reward == 1:
                if instruct_reward == 1:
                    acc_reward = 1.5
                else:
                    acc_reward = 0.5
            else:
                acc_reward = -1
        elif use_fb == 4:
            if acc_reward == 1:
                if instruct_reward == 1:
                    acc_reward = 1.5
                else:
                    acc_reward = 0.5
            else:
                if instruct_reward == 1:
                    acc_reward = -1.5
                else:
                    acc_reward = -1
        elif use_fb == 5:
            if acc_reward == 1:
                if instruct_reward == 1:
                    acc_reward = 1.5
                else:
                    acc_reward = 0.5
            else:
                if instruct_reward == 1:
                    acc_reward = -1.5
                else:
                    acc_reward = -2
        elif use_fb == 6:
            if acc_reward == 1:
                if instruct_reward == 1:
                    acc_reward = 1.0
                else:
                    acc_reward = 0.5
            else:
                acc_reward = -1
        elif use_fb == 7:
            if acc_reward == 1:
                if instruct_reward == 1:
                    acc_reward = 1.5
                else:
                    acc_reward = 0.5
            else:
                acc_reward = -1
            instruct_reward = 0
        elif use_fb == 8:
            if acc_reward == 1:
                if instruct_reward == 1:
                    acc_reward = 1.5
                else:
                    acc_reward = 0.5
            else:
                if instruct_reward == 1:
                    acc_reward = -1.5
                else:
                    acc_reward = -1
            instruct_reward = 0
        elif use_fb == 9:
            if acc_reward == 1:
                if instruct_reward == 1:
                    acc_reward = 1.5
                else:
                    acc_reward = 0.5
            else:
                if instruct_reward == 1:
                    acc_reward = -1.5
                else:
                    acc_reward = -2
            instruct_reward = 0
        elif use_fb == 10:
            if acc_reward == 1:
                if instruct_reward == 1:
                    acc_reward = 1.0
                else:
                    acc_reward = 0.5
            else:
                acc_reward = -1
            instruct_reward = 0
    if use_format == 1:
        format_reward = verify_format(solution_str)
    else:
        format_reward = 0
    
    if use_api:
        return {
            'score': acc_reward + instruct_reward + 0.5 * format_reward,
            'accuracy_reward': acc_reward,
            'instruct_reward': instruct_reward,
            'format_reward': 0.5 * format_reward
        }
    else:
        return {
            'score': acc_reward + 0.5 * format_reward,
            'accuracy_reward': acc_reward,
            'format_reward': 0.5 * format_reward
        }
