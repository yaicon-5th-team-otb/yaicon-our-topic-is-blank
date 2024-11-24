import os
import json
import random 
import requests
from dataclasses import dataclass

@dataclass
class Usage:
    """Mock usage statistics for compatibility with existing code"""
    input_tokens: int = 0
    output_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

class OllamaClient:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.messages = Messages(self)
        
    def chat(self):
        return ChatCompletions(self)
    
class Messages:
    def __init__(self, client):
        self.client = client
        
    def create(self, model, messages, max_tokens=100, temperature=1.0):
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"[SYSTEM] {content}\n"
            elif role == "user":
                prompt += f"[USER] {content}\n"
            elif role == "assistant":
                prompt += f"[ASSISTANT] {content}\n"

        response = requests.post(
            f"{self.client.base_url}/api/generate",
            json={
                "model": "ollama2:latest",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                }
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            input_chars = len(prompt)
            output_chars = len(result['response'])
            
            usage = Usage(
                input_tokens=input_chars // 4,
                output_tokens=output_chars // 4,
                prompt_tokens=input_chars // 4,
                completion_tokens=output_chars // 4
            )
            
            return MessageResponse(
                content=[MessageContent(text=result['response'])],
                usage=usage
            )
        else:
            raise Exception(f"Ollama API error: {response.text}")

class ChatCompletions:
    def __init__(self, client):
        self.client = client
    
    def create(self, model, messages, max_completion_tokens=100, temperature=1.0, seed=None, response_format=None):
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"[SYSTEM] {content}\n"
            elif role == "user":
                prompt += f"[USER] {content}\n"
            elif role == "assistant":
                prompt += f"[ASSISTANT] {content}\n"

        if response_format and response_format.get("type") == "json_object":
            prompt += "\nRespond only with a valid JSON object."

        response = requests.post(
            f"{self.client.base_url}/api/generate",
            json={
                "model": "ollama2:latest ",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_completion_tokens
                }
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            input_chars = len(prompt)
            output_chars = len(result['response'])
            print(result)
            usage = Usage(
                input_tokens=input_chars // 4,
                output_tokens=output_chars // 4,
                prompt_tokens=input_chars // 4,
                completion_tokens=output_chars // 4
            )
            
            return CompletionResponse(
                choices=[Choice(message=Message(content=result['response'].strip()))],
                usage=usage
            )
        else:
            raise Exception(f"Ollama API error: {response.text}")

class MessageContent:
    def __init__(self, text):
        self.text = text

class MessageResponse:
    def __init__(self, content, usage):
        self.content = content
        self.usage = usage

class Message:
    def __init__(self, content):
        self.content = content

class Choice:
    def __init__(self, message):
        self.message = message

class CompletionResponse:
    def __init__(self, choices, usage):
        self.choices = choices
        self.usage = usage

def calc_price(model, usage):
    """Mock pricing function that provides reasonable estimates"""
    if "llama" in model.lower():
        return (0.5 * usage.input_tokens + 1.0 * usage.output_tokens) / 1000000.0
    if "claude" in model.lower():
        return (3.0 * usage.input_tokens + 15.0 * usage.output_tokens) / 1000000.0
    if model == "gpt-4o":
        return (2.5 * usage.prompt_tokens + 10.0 * usage.completion_tokens) / 1000000.0
    return None

def call_api(client, model, prompt_messages, temperature=1.0, max_tokens=100, seed=2024, json_output=False):
    """Main API function that maintains the same interface"""
    if json_output:
        prompt = prompt_messages[0]["content"] + " Directly output the JSON dict with no additional text. Make sure to follow the exact same JSON format as shown in the examples."
        prompt_messages = [{"role": "user", "content": prompt}]
        response_format = {"type": "json_object"}
    else:
        response_format = {"type": "text"}
    
    if hasattr(client, 'messages'):  # Claude-style API
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=prompt_messages
        )
        cost = calc_price(model, message.usage)
        response = message.content[0].text
    else:  # GPT-style API
        completion = client.chat.completions.create(
            model=model,
            messages=prompt_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            response_format=response_format
        )
        cost = calc_price(model, completion.usage)
        response = completion.choices[0].message.content.strip()
    
    return response, cost

def call_api_claude(client, model, prompt_messages, temperature=1.0, max_tokens=100):
    """Dedicated function for Claude-style API calls"""
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=prompt_messages
    )
    cost = calc_price(model, message.usage)
    response = message.content[0].text
    return response, cost

def cache_output(output, file_name):
    """Cache output to file"""
    if file_name.endswith(".txt"):
        with open(file_name, "w") as f:
            f.write(output)
    elif file_name.endswith(".json"):
        with open(file_name, "w") as f:
            json.dump(output, f, indent=4)
    return 

def print_idea_json(filename):
    """Print formatted idea JSON from file"""
    with open(filename, "r") as f:
        idea_json = json.load(f)
    idea = idea_json["final_plan_json"]
    name = idea_json["idea_name"]
    print(name)
    for k,v in idea.items():
        if len(v) > 5:
            print('- ' + k)
            print(v.strip() + '\n')

def format_plan_json(experiment_plan_json, indent_level=0, skip_test_cases=True, skip_fallback=True):
    """Format experiment plan JSON with proper indentation"""
    try:
        if isinstance(experiment_plan_json, str):
            return experiment_plan_json
        
        output_str = ""
        indent = "  " * indent_level
        for k, v in experiment_plan_json.items():
            if k == "score":
                continue
            if skip_test_cases and k == "Test Case Examples":
                continue
            if skip_fallback and k == "Fallback Plan":
                continue
            if isinstance(v, (str, int, float)):  
                output_str += f"{indent}{k}: {v}\n"
            elif isinstance(v, list):
                output_str += f"{indent}{k}:\n"
                for item in v:
                    if isinstance(item, dict):
                        output_str += format_plan_json(item, indent_level + 1)
                    else:
                        output_str += f"{indent}  - {item}\n"
            elif isinstance(v, dict):
                output_str += f"{indent}{k}:\n"
                output_str += format_plan_json(v, indent_level + 1)
        return output_str
    except Exception as e:
        print("Error in formatting experiment plan json: ", e)
        return ""

def shuffle_dict_and_convert_to_string(input_dict, n=20):
    """Shuffle dictionary items and convert to JSON string"""
    items = list(input_dict.items())
    random.shuffle(items)
    items = items[:n]
    shuffled_dict = dict(items)
    return json.dumps(shuffled_dict, indent=4)

def clean_code_output(code_output):
    """Clean code output by removing markdown formatting"""
    code_output = code_output.strip()
    if code_output.startswith("```python"):
        code_output = code_output[len("```python"):].strip()
    if code_output.endswith("```"):
        code_output = code_output[:-len("```")].strip()
    return code_output

def concat_reviews(paper_json):
    """Concatenate paper reviews into a single string"""
    review_str = ""
    meta_review = paper_json["meta_review"]
    all_reviews = paper_json["reviews"]

    review_str += "Meta Review:\n" + meta_review + "\n\n"
    for idx, review in enumerate(all_reviews):
        review_str += f"Reviewer #{idx+1}:\n\n"
        for key, value in review.items():
            if key in ["summary", "soundness", "contribution", "strengths", "weaknesse", "questions", "rating", "confidence"]:
                review_str += f"{key}: {value['value']}\n"
        review_str += "\n"

    return review_str

def avg_score(scores):
    """Calculate average score"""
    scores = [int(s[0]) for s in scores]
    return sum(scores) / len(scores)

def max_score(scores):
    """Calculate maximum score"""
    scores = [int(s[0]) for s in scores]
    return max(scores)

def min_score(scores):
    """Calculate minimum score"""
    scores = [int(s[0]) for s in scores]
    return min(scores)