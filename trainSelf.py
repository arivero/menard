from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import Dataset
import random
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn.functional as F

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
        print(f"Teacher prompt: \n{full_prompt}")
        teacher_response = generate_response(teacher_model, teacher_tokenizer, full_prompt)
        print (f"Teacher response: \n{teacher_response}")
        print("=====================================")
        examples.append({
            "prompt": full_prompt,
            "student_prompt": f"\n\nTask: {task}",
            "response": teacher_response,
            "combined": f"{full_prompt}\n\n{teacher_response}",
            "combined_student": f"\n\nTask: {task}\n\n{teacher_response}"
        })
    return examples

# Create training dataset
training_data = create_training_examples()
dataset = Dataset.from_list(training_data)

print("Training dataset:")
print(dataset)
for row in dataset:
    print(row)

# Training configuration
training_args = {
    "learning_rate": 1e-5,
    "num_train_epochs": 3000,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
}

def compute_kl_loss(teacher_logits, student_logits):
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    student_logs = F.log_softmax(student_logits, dim=-1)
    return F.kl_div(student_logs, teacher_probs, reduction='batchmean')

# Fine-tune student model
def train_student():
    student_model.train()
    teacher_model.eval()
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=training_args["learning_rate"])
    max_length = 512  # Define max sequence length

    for epoch in range(training_args["num_train_epochs"]):
        for batch in dataset:
            # Concatenate prompt and response
            encoded = teacher_tokenizer(
                batch["combined"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt"
            )#.to(device)

            labels = encoded.input_ids.clone() #   detach().clone()
            # Compute lengths of prompts for masking
            prompt_enc = teacher_tokenizer(
                batch["prompt"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt"
            ) #no need to move to device
            prompt_lengths = (prompt_enc.input_ids != teacher_tokenizer.pad_token_id).sum(dim=1)
            # Set prompt tokens to -100
            for i, length in enumerate(prompt_lengths):
                labels[i, :length] = -100


            #concatenate student prompt and response:
            student_encoded = teacher_tokenizer(
                batch["combined_student"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt"
            )#.to(device)

            #remove the prompt tokens from the labels for the student
            student_labels = student_encoded.input_ids.clone()
            student_prompt_enc = teacher_tokenizer(
                batch["student_prompt"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt"
            )
            student_prompt_lengths = (student_prompt_enc.input_ids != teacher_tokenizer.pad_token_id).sum(dim=1)
            for i, length in enumerate(student_prompt_lengths):
                student_labels[i, :length] = -100

            # Get teacher logits
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=encoded.input_ids,
                    attention_mask=encoded.attention_mask
                )


            # Get student outputs and compute combined loss
            student_outputs = student_model(
                input_ids=student_encoded.input_ids,
                attention_mask=student_encoded.attention_mask,
                labels=student_labels
            )

            # Align logits for KL divergence. This is complex because the lengths of the prompts can be different,
            # Our approache is goint to be to shift both student and teacher logits to the left, removing the prompt
            # tokens, and then truncating the longer of the two to the length of the shorter one.
            total_length = teacher_outputs.logits.shape[1]
            teacher_logits = teacher_outputs.logits.clone()
            student_logits = student_outputs.logits.clone()
            # Shift logits left to remove prompt tokens
            for i, length in enumerate(prompt_lengths):
                teacher_logits[i, :total_length-length] = teacher_outputs.logits[i, length:total_length]
                teacher_logits[i, total_length-length:] = 32000  # Zero-pad the end

            for i, length in enumerate(student_prompt_lengths):
                student_logits[i, :total_length-length] = student_outputs.logits[i, length:total_length]
                student_logits[i, total_length-length:] = 32000  # Zero-pad the end
            #truncate the longer of the two to the length of the shorter one:
            kl_loss = compute_kl_loss(teacher_logits, student_logits)

            total_loss = student_outputs.loss + kl_loss * 0.5  # Weighted combination
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"Epoch: {epoch}, Loss: {total_loss.item()}")

            if epoch % 100 == 0:
                # print the encodings of the prompt and expected response, for the teacher and student
                print(f"Prompt: {batch['prompt']}")
                print(f"Student Prompt: {batch['student_prompt']}")
                print(f"Response: {batch['response']}")
                print(f"Combined: {batch['combined']}")
                print(f"Combined Student: {batch['combined_student']}")
                #print the respective tokenized encodings:
                print(f"Encoded: {encoded}")
                print(f"Student Encoded: {student_encoded}")
                #and the labels:
                print(f"Labels: {labels}")
                print(f"Student Labels: {student_labels}")


                # Print outputs of teacher and student, and calculate the level of similarity
                teacher_output = teacher_tokenizer.decode(torch.argmax(teacher_outputs.logits, dim=-1)[0][:30], skip_special_tokens=True)
                student_output = teacher_tokenizer.decode(torch.argmax(student_outputs.logits, dim=-1)[0][:30], skip_special_tokens=True)
                print(f"Epoch: {epoch}, Teacher Output: {teacher_output}")
                print(f"Epoch: {epoch}, Student Output: {student_output}")
                #print(f"Epoch: {epoch}, Similarity: {teacher_output == student_output}")
                #from Levenshtein import distance
                #print(f"Epoch: {epoch}, Levenshtein Distance: {distance(teacher_output, student_output)}")
                #print all relevant shapes:
                print(f"Teacher logits shape: {teacher_logits.shape}")
                print(f"Student logits shape: {student_logits.shape}")
                print(f"Prompt lengths shape: {prompt_lengths.shape}")
                print(f"Student Prompt lengths shape: {student_prompt_lengths.shape}")
                print(f"Labels shape: {labels.shape}")
                print(f"Student Labels shape: {student_labels.shape}")
                #also outputs logits shapes before truncation:
                print(f"Teacher outputs logits shape: {teacher_outputs.logits.shape}")
                print(f"Student outputs logits shape: {student_outputs.logits.shape}")
                #detokenize student and teacher logits after shifting and truncating:
                print(f"Teacher logits: {teacher_tokenizer.decode(torch.argmax(teacher_logits, dim=-1)[0][:30], skip_special_tokens=True)}")
                print(f"Student logits: {teacher_tokenizer.decode(torch.argmax(student_logits, dim=-1)[0][:30], skip_special_tokens=True)}")
                #print the prompt lengths:
                print(f"Prompt lengths: {prompt_lengths}")
                print(f"Student Prompt lengths: {student_prompt_lengths}")
                #print the 30 first teacher and student argmax logits:
                print(f"Teacher argmax logits: {torch.argmax(teacher_logits, dim=-1)[0][:30]}")
                print(f"Student argmax logits: {torch.argmax(student_logits, dim=-1)[0][:30]}")




if __name__ == "__main__":
    print("Starting training...")
    train_student()

    # Test student model
    test_prompt = f"{SYSTEM_PROMPT}\n\nTask: Explain what is recursion in programming?"
    student_response = generate_response(student_model, teacher_tokenizer, test_prompt)
    print("\nTest Response from Student:")
    print(student_response)