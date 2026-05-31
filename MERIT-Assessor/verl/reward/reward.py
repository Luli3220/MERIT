import json
import re
import sys
import os
from pathlib import Path
from typing import Any
import time
from openai import OpenAI

# Import prompts locally
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
from prompts import judge_system_prompt, judge_user_prompt


API_KEY = os.getenv("MERIT_JUDGE_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("MERIT_JUDGE_BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("MERIT_JUDGE_MODEL", "deepseek-reasoner")
TIMEOUT_S = float(os.getenv("MERIT_JUDGE_TIMEOUT_S", "240.0"))
TEMPERATURE = float(os.getenv("MERIT_JUDGE_TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.getenv("MERIT_JUDGE_MAX_TOKENS", str(1024 * 6)))


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _fill_template(template: str, mapping: dict) -> str:
    result = template
    for key, value in mapping.items():
        result = result.replace(f"{{{{{key}}}}}", _as_text(value))
    return result


def _call_qwen_chat_completion(messages: list[dict[str, str]]) -> str | None:
    if not API_KEY:
        print("Judge API key is not configured. Set MERIT_JUDGE_API_KEY or DEEPSEEK_API_KEY.")
        return None

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=TIMEOUT_S)
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            extra_body={
            "chat_template_kwargs": {"enable_thinking": True},
        }, 
        )
        content = response.choices[0].message.content
        return _as_text(content).strip()
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        return None



def remove_thinking(text: str) -> str:
    pattern = r"<think>.*?(?:</think>|<\[PLHD21_never_used_51bce0c785ca2f68081bfa7d91973934\]>)"
    return re.sub(pattern, "", text, flags=re.DOTALL).strip()

def extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return json.JSONDecoder().raw_decode(text)[0]
    except json.JSONDecodeError:
        pass
    pattern = r"```json\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass
    return {}

def _safe_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        v = x.strip().lower()
        if v in {"true", "1", "yes"}:
            return True
        if v in {"false", "0", "no"}:
            return False
    return False

def _coerce_rubric_breakdown(x: Any) -> dict:
    result = {}
    if isinstance(x, dict):
        for k, v in x.items():
            result[str(k)] = _safe_bool(v)
    return result

def _call_with_retry(messages: list[dict[str, str]], max_retries: int = 3, delay_s: float = 2.0) -> str | None:
    for _ in range(max_retries):
        content = _call_qwen_chat_completion(messages)
        if content:
            return content
        time.sleep(delay_s)
    return None

def _try_binary_label(ground_truth: Any, extra_info: dict | None) -> int | None:
    if isinstance(extra_info, dict) and "label" in extra_info:
        try:
            return 1 if int(extra_info["label"]) == 1 else 0
        except (ValueError, TypeError):
            pass
    if isinstance(ground_truth, bool):
        return 1 if ground_truth else 0
    if isinstance(ground_truth, (int, float)):
        return 1 if int(ground_truth) == 1 else 0
    if isinstance(ground_truth, str):
        v = ground_truth.strip().lower()
        if v in {"1", "true", "yes"}:
            return 1
        if v in {"0", "false", "no"}:
            return 0
    return None

def compute_score(data_source: str, solution_str: str, ground_truth: Any, extra_info: dict) -> float:
    binary_label = _try_binary_label(ground_truth, extra_info)
    if binary_label is not None:
        return compute_val_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=binary_label,
            extra_info=extra_info,
        )

    # Extract context from extra_info
    paper_title = extra_info.get("paper_title", "")
    paper_abstract = extra_info.get("paper_abstract", "")
    paper_introduction = extra_info.get("paper_introduction", "")
    candidate_history_list = extra_info.get("candidate_history_list", "")
    
    # In training, ground_truth is the rubric
    rubric_data = ground_truth

    # If ground_truth is the whole item dict (with "rubric" key), extract the rubric list
    if isinstance(rubric_data, dict) and "rubric" in rubric_data:
        rubric_data = rubric_data["rubric"]

    # ground_truth is the rubric list (or dict)
    rubrics_json = _as_text(rubric_data)
    
    # Clean the solution string by removing thinking process
    clean_solution = remove_thinking(solution_str)
    
    # Fill the user prompt template
    mapping = {
        "paper_title": paper_title,
        "paper_abstract": paper_abstract,
        "paper_introduction": paper_introduction,
        "candidate_history_list": candidate_history_list,
        "rubrics_json": rubrics_json,
        "actor_output_text": clean_solution,
    }
    user_content = _fill_template(judge_user_prompt, mapping)

    messages = [
        {"role": "system", "content": judge_system_prompt},
        {"role": "user", "content": user_content},
    ]

    # No try-except as requested
    content = _call_with_retry(messages)
    
    if not content:
        return {
            "score": 0.0,
            "rubric_score": 0.0,
            "logical_score": 0.0,
            "eval_mode_judge": 1.0,
            "eval_mode_label": 0.0,
        }

    # Extract JUDGE_ANALYSIS and JSON_RESULT
    if "[JSON_RESULT]" in content:
        parts = content.split("[JSON_RESULT]")
        json_part = parts[1].strip()
        result = extract_json(json_part)
    else:
        # Fallback if tags missing but maybe JSON is there
        result = extract_json(content)
    
    if not result:
        return {
            "score": 0.0,
            "rubric_score": 0.0,
            "logical_score": 0.0,
            "eval_mode_judge": 1.0,
            "eval_mode_label": 0.0,
        }
    rubric_breakdown = _coerce_rubric_breakdown(result.get("rubric_breakdown", {}))
    logical = _safe_bool(result.get("logical", False))

    # Calculate Rubric Score
    total_score = 0.0
    total_weight = 0.0
    
    # Parse rubric_data to get weights
    rubric_list = rubric_data if isinstance(rubric_data, list) else []
    if isinstance(rubric_data, str):
        try:
            rubric_list = json.loads(rubric_data)
        except:
            rubric_list = []
            
    # Create a map of title -> weight
    weight_map = {}
    for item in rubric_list:
        if isinstance(item, list):
             for subitem in item:
                 if isinstance(subitem, dict):
                    try:
                        weight_map[subitem.get("title")] = int(subitem.get("weight", 0))
                    except (ValueError, TypeError):
                        weight_map[subitem.get("title")] = 0
        elif isinstance(item, dict):
            try:
                weight_map[item.get("title")] = int(item.get("weight", 0))
            except (ValueError, TypeError):
                weight_map[item.get("title")] = 0



    # Sum weighted scores
    for title, passed in rubric_breakdown.items():
        weight = weight_map.get(title, 0)
        if passed:
            total_score += weight
            
    # Normalize: divide by sum of positive weights (max possible score)
    max_possible_score = sum(w for w in weight_map.values() if w > 0)
    
    if max_possible_score > 0:
        normalized_rubric_score = total_score / max_possible_score
    else:
        normalized_rubric_score = 0.0
    
    logical_reward = 1.0 if logical else 0.0

    final_score = normalized_rubric_score * logical_reward
    
    result_dict = {
        "score": float(final_score),
        "rubric_score": float(normalized_rubric_score),
        "logical_score": float(logical_reward),
        "eval_mode_judge": 1.0,
        "eval_mode_label": 0.0,
    }

    return result_dict


def compute_val_score(data_source: str, solution_str: str, ground_truth: Any, extra_info: dict) -> dict:
    pred = 0
    try:
        match = re.search(r'\[FINAL_LABEL\]\s*(?:<)?(\d+)(?:>)?', solution_str, re.IGNORECASE)
        if match:
            pred = int(match.group(1))
        else:
            matches = re.findall(r'\d+', solution_str)
            if matches:
                pred = int(matches[-1])
        
        # Ensure pred is 0 or 1
        if pred != 0 and pred != 1:
            pred = 0 
    except:
        pred = 0

    label = 0
    try:
        label = int(ground_truth)
    except (ValueError, TypeError):
        label = 0
    
    # Ensure label is binary
    if label != 1:
        label = 0

    # 3. Compute metrics
    acc = 1.0 if pred == label else 0.0
    tp = 1.0 if (pred == 1 and label == 1) else 0.0
    fp = 1.0 if (pred == 1 and label == 0) else 0.0
    fn = 1.0 if (pred == 0 and label == 1) else 0.0
    tn = 1.0 if (pred == 0 and label == 0) else 0.0

    return {
        "score": acc, 
        "acc": acc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pred": float(pred),
        "label": float(label),
        "eval_mode_judge": 0.0,
        "eval_mode_label": 1.0,
    }
