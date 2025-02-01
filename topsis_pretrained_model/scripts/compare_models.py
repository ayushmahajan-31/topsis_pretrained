import torch
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from math import exp, sqrt
import numpy as np
import matplotlib.pyplot as plt

# Device setup for GPU acceleration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model names and paths
model_names = {
    "DialoGPT-medium": "microsoft/DialoGPT-medium",
    "GPT-2": "gpt2",
    "BlenderBot-small": "facebook/blenderbot-400M-distill"
}

# Load models & tokenizers
models = {name: AutoModelForCausalLM.from_pretrained(path).to(device).eval() for name, path in model_names.items()}
tokenizers = {name: AutoTokenizer.from_pretrained(path) for name, path in model_names.items()}

# Set padding tokens to avoid warnings
for tokenizer in tokenizers.values():
    tokenizer.pad_token = tokenizer.eos_token

# Load test data
with open("../data/test_conversations.json", "r") as f:
    test_data = json.load(f)

# Function to generate responses
def generate_response(model, tokenizer, user_input, max_length=50):
    inputs = tokenizer.encode(user_input, return_tensors="pt").to(device)
    attention_mask = torch.ones(inputs.shape, dtype=torch.long).to(device)
    outputs = model.generate(inputs, max_length=max_length, pad_token_id=tokenizer.eos_token_id, attention_mask=attention_mask)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Generate responses
generated_responses = {}
for model_name, model in models.items():
    print(f"Generating responses for {model_name}...")
    generated_responses[model_name] = [
        {
            "user_input": item["user_input"],
            "generated_response": generate_response(model, tokenizers[model_name], item["user_input"]),
            "reference_response": item.get("reference_response", "")
        }
        for item in test_data if "user_input" in item
    ]

# Load evaluation metrics
bleu_metric = evaluate.load("sacrebleu")
rouge_metric = evaluate.load("rouge")

# Function to evaluate metrics
def evaluate_metrics(model_name, responses):
    print(f"\nEvaluating metrics for {model_name}...")
    references = [[item["reference_response"]] for item in responses]
    predictions = [item["generated_response"] for item in responses]

    # BLEU Score
    bleu = bleu_metric.compute(predictions=predictions, references=references)

    # ROUGE Score
    rouge = rouge_metric.compute(predictions=predictions, references=references)

    # Perplexity Calculation
    perplexities = []
    for response in predictions:
        if not response.strip():
            perplexities.append(float('inf'))  # Handle empty responses
            continue

        inputs = tokenizers[model_name].encode(response, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = models[model_name](inputs, labels=inputs)
            loss = outputs.loss
            perplexities.append(exp(loss.item()))

    avg_perplexity = sum(perplexities) / len(perplexities)

    return {"BLEU": bleu["score"], "ROUGE-L": rouge["rougeL"], "Perplexity": avg_perplexity}

# Evaluate all models
evaluation_results = {model_name: evaluate_metrics(model_name, responses)
                      for model_name, responses in generated_responses.items()}

# Extract metrics for TOPSIS
model_names = list(evaluation_results.keys())
bleu_scores = np.array([evaluation_results[name]["BLEU"] for name in model_names])
rouge_scores = np.array([evaluation_results[name]["ROUGE-L"] for name in model_names])
perplexities = np.array([evaluation_results[name]["Perplexity"] for name in model_names])

# Combine metrics into a decision matrix
decision_matrix = np.vstack((bleu_scores, rouge_scores, perplexities)).T

# Normalize the decision matrix
norm_matrix = decision_matrix / np.sqrt((decision_matrix**2).sum(axis=0))

# Define weights (equal weights for BLEU, ROUGE-L, and Perplexity)
weights = np.array([0.33, 0.33, 0.34])

# Weighted normalized decision matrix
weighted_matrix = norm_matrix * weights

# Identify ideal and anti-ideal solutions
ideal_solution = np.max(weighted_matrix, axis=0)  # Maximizing BLEU & ROUGE-L
anti_ideal_solution = np.min(weighted_matrix, axis=0)  # Minimizing Perplexity

# Calculate the distance to the ideal and anti-ideal solutions
distance_to_ideal = np.sqrt(((weighted_matrix - ideal_solution)**2).sum(axis=1))
distance_to_anti_ideal = np.sqrt(((weighted_matrix - anti_ideal_solution)**2).sum(axis=1))

# Calculate TOPSIS scores
topsis_scores = distance_to_anti_ideal / (distance_to_ideal + distance_to_anti_ideal)

# Determine the best model
best_model_index = np.argmax(topsis_scores)
best_model = model_names[best_model_index]

# Print results
print("\nEvaluation Results:")
for model_name, metrics in evaluation_results.items():
    print(f"{model_name}: {metrics}")

print("\nTOPSIS Scores:")
for i, model_name in enumerate(model_names):
    print(f"{model_name}: {topsis_scores[i]:.4f}")

print(f"\nBest Model according to TOPSIS: {best_model}")

# Plot metrics
x = range(len(model_names))
width = 0.25

plt.figure(figsize=(10, 6))
plt.bar(x, bleu_scores / max(bleu_scores), width, label="BLEU (Normalized)", color="blue")
plt.bar([p + width for p in x], rouge_scores / max(rouge_scores), width, label="ROUGE-L (Normalized)", color="orange")
plt.bar([p + 2 * width for p in x], 1 - (perplexities / max(perplexities)), width, label="Perplexity (Inverted & Normalized)", color="green")

plt.xlabel("Models")
plt.ylabel("Normalized Scores")
plt.title("Comparison of Conversational Models with TOPSIS")
plt.xticks([p + width for p in x], model_names)
plt.legend()
plt.tight_layout()
plt.savefig("../results/model_comparison_with_topsis.png")
plt.show()
