import torch, json, os, re, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import classification_report
from tqdm import tqdm

SYSTEM_PROMPT = (
    "You are a fact-checking assistant. You are given a news claim along with two generated justifications: "
    "a 'True Justification' and a 'False Justification'. Do not consider any other evidence. Your task is to compare these "
    "justifications and determine the overall veracity of the claim based solely on them.\n\n"
    "Output your answer in exactly the following format:\n"
    "Claim Veracity: [label]\n\n"
    "Answer with one word: pants-fire, false, barely-true, half-true, mostly-true, or true"
)

def construct_prompt(instruction, input_text):
    return f"{SYSTEM_PROMPT}\n\n{instruction}\n\n{input_text}"

def clean_response(response):
    response = response.lower().strip()
    response = re.sub(r'<\|.*?\|>', '', response)
    if response.startswith("claim veracity:"):
        label = response.split("claim veracity:")[1].strip().split()[0]
        if label in ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]:
            return label
    for label in ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]:
        if label in response:
            return label
    return "false"

def evaluate_model(config):
    base_model_path = config["base_model_path"]
    adapter_path = config["adapter_path"]
    test_data_path = config["test_data_path"]
    output_dir = config["output_dir"]
    max_new_tokens = config.get("max_new_tokens", 32)
    temperature = config.get("temperature", 0.1)
    truncate_len = config.get("truncate_len", 2048)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    predictions, true_labels, all_outputs = [], [], []
    for item in tqdm(test_data):
        try:
            prompt = construct_prompt(item['instruction'], item['input'])
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=truncate_len).to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=False)
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            predicted_label = clean_response(response)

            all_outputs.append({
                "instruction": item['instruction'],
                "input": item['input'],
                "true_label": item['output'].lower(),
                "predicted_label": predicted_label,
                "model_response": response,
                "prompt": prompt
            })
            predictions.append(predicted_label)
            true_labels.append(item['output'].lower())
        except Exception as e:
            predictions.append("false")
            true_labels.append(item['output'].lower())

    report = classification_report(true_labels, predictions, digits=4)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "predictions_with_responses.json"), 'w') as f:
        json.dump(all_outputs, f, indent=2)
    with open(os.path.join(output_dir, "metrics.txt"), 'w') as f:
        f.write(report)

    print("\nClassification Report:")
    print(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    evaluate_model(config)

