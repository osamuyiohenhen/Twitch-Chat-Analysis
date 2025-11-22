import multiprocessing
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset

# ==========================================
# 1. CONFIGURATION
# ==========================================
model_name = "./twitch-roberta-v1"
input_file = "twitch_chat.csv" 
output_dir = "./twitch-roberta-v2"
block_size = 128 

# ==========================================
# 2. GLOBAL INITIALIZATION (THE FIX)
# ==========================================
# The tokenizer MUST be loaded here so worker processes can see it.
print("Loading Tokenizer globally...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def tokenize_function(examples):
    # Now this works because 'tokenizer' is global
    return tokenizer(examples["text"])

def group_texts(examples):
    # Concatenate all texts in this batch
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    # Drop the small remainder at the end
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
        
    # Split by chunks of block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Load the Model ONLY inside main (to save VRAM)
    print(f"Loading model: {model_name}...")
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    print("Loading CSV...")
    # Ensure you have the header fixed or use column_names=["text"]
    dataset = load_dataset("csv", data_files=input_file)

    print("Tokenizing data...")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,             
        remove_columns=["text"] 
    )

    print("Grouping texts into blocks...")
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=4
    )
    
    print(f"Original messages: {len(dataset['train'])}")
    print(f"Grouped blocks: {len(lm_datasets['train'])}")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=15,             
        per_device_train_batch_size=4,  
        gradient_accumulation_steps=4, 
        learning_rate=5e-5,
        save_total_limit=2,
        weight_decay=0.01,
        fp16=True,                      
        save_steps=500,
        logging_steps=50,
        report_to="none"                
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        data_collator=data_collator,
    )

    print("Starting training... (Fans will spin up now)")
    trainer.train()

    print("Saving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"DONE! Model saved to {output_dir}")