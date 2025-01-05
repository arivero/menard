from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import Dataset
import random
from peft import get_peft_model, LoraConfig, TaskType

# Add device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize teacher model normally in full precision
model_name = "meta-llama/Llama-2-7b-hf"
teacher_tokenizer = AutoTokenizer.from_pretrained(model_name)
teacher_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"  # This will handle CUDA allocation efficiently
)

# Set padding token for the tokenizer
if teacher_tokenizer.pad_token is None:
    teacher_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # Resize token embeddings for the model to account for the new token
    teacher_model.resize_token_embeddings(len(teacher_tokenizer))

# Configure LoRA to only train the adapters
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    modules_to_save=None  # Don't save any full modules
)

# Create student model using LoRA - this only creates adapter weights
student_model = get_peft_model(teacher_model, lora_config)

# Print only the trainable parameters (should be much smaller)
print("Trainable parameters for LoRA adapters:")
student_model.print_trainable_parameters()

# Define system prompt and tasks
SYSTEM_PROMPT = """You are a helpful AI assistant that provides clear, accurate, and concise responses.
Always format code properly and explain technical concepts clearly."""

tasks = [
    "Explain how a binary search works.",
    "What is the difference between a list and tuple in Python?",
    "How does garbage collection work in Python?",
    "Explain the concept of decorators in Python.",
]

def generate_response(model, tokenizer, prompt, max_length=512):
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        padding='max_length',
        truncation=True, 
        max_length=max_length // 2  # Reduce input length to leave room for generation
    ).to(device)  # Move inputs to GPU
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_length // 2,  # Allow generation of new tokens up to half max_length
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Create training dataset
def create_training_examples():
    examples = []
    for task in tasks:
        full_prompt = f"{SYSTEM_PROMPT}\n\nTask: {task}"
        # Get teacher's response
        teacher_response = generate_response(teacher_model, teacher_tokenizer, full_prompt)
        examples.append({
            "prompt": full_prompt,
            "response": teacher_response
        })
    return examples

# Create training dataset
training_data = create_training_examples()
dataset = Dataset.from_list(training_data)

# Training configuration
training_args = {
    "learning_rate": 1e-5,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
}

# Fine-tune student model
def train_student():
    # Initialize training
    student_model.train()
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=training_args["learning_rate"])

    for epoch in range(training_args["num_train_epochs"]):
        for batch in dataset:
            # Prepare input and move to GPU
            inputs = teacher_tokenizer(batch["prompt"], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            labels = teacher_tokenizer(batch["response"], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

            # Forward pass
            outputs = student_model(**inputs, labels=labels.input_ids)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            optimizer.zero_grad()
            
            print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    print("Starting training...")
    train_student()
    
    # Test student model
    test_prompt = f"{SYSTEM_PROMPT}\n\nTask: Explain what is recursion in programming?"
    student_response = generate_response(student_model, teacher_tokenizer, test_prompt)
    print("\nTest Response from Student:")
    print(student_response)