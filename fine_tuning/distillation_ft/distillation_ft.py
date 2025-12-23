import gc
import torch
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from fine_tuning.model_utils.eval_utils import *
from fine_tuning.model_utils.preprocessing import * 

class distillation_ft:
    def __init__(
        self,
        model_name,
        training_dataset,
        testing_dataset,
        train_batch_size,
        eval_batch_size,
        gradient_accumulation_steps,
        num_epochs,
        learning_rate,
        max_input_length,
        max_target_length,
        lora_r=64,
        lora_alpha=64,
        lora_dropout=0.05,
        lora_target_modules=None,  # if None, good defaults for LLaMA
        lora_bias="none",          # "none" | "lora_only" | "all"
    ):
        # Configs
        self.model_name = model_name
        self.training_dataset = training_dataset
        self.testing_dataset = testing_dataset
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # LoRA configs
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules
        self.lora_bias = lora_bias

        # Setup
        self.tokenizer = None
        self.train_loader = None
        self.val_loader = None
        self.model = None
        self.optimizer = None

    def preprocess_dataset(self):
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)\
        
        # Ensure padding is on the left for generation
        self.tokenizer.padding_side = "left" 
        
        # Llama 3 specific padding fix
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Call the external preprocessing function
        self.train_loader, self.val_loader = preprocess_dataset(
            tokenizer=self.tokenizer,
            training_dataset=self.training_dataset,
            testing_dataset=self.testing_dataset,
            max_input_length=self.max_input_length,
            max_target_length=self.max_target_length,
            train_batch_size=self.train_batch_size,
            eval_batch_size=self.eval_batch_size,
        )

    def init_model(self):

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ) 

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Enable gradient checkpointing to save memory
        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False

        # Llama 3 target modules
        if self.lora_target_modules is None:
            target_modules = ["q_proj", "v_proj"]
        else:
            target_modules = self.lora_target_modules

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias=self.lora_bias,
            target_modules=target_modules,
        )

        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

        # Optimizer over only trainable (LoRA) params
        trainable_params = (p for p in self.model.parameters() if p.requires_grad)
        self.optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate)

    def train(self):
        self.model.train()
        global_step = 0
        
        print(f"Starting training on {self.device}...")
        
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            epoch_steps = 0
            
            for step, batch in enumerate(self.train_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    labels=labels
                )
                loss = outputs.loss            
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                is_last_step = (step + 1) == len(self.train_loader)
                if (step + 1) % self.gradient_accumulation_steps == 0 or is_last_step:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    epoch_steps += 1

                running_loss += loss.item() * self.gradient_accumulation_steps
                
                if (step + 1) % 10 == 0: # Log every 10 steps (since dataset is small)
                    avg_loss = running_loss / (step + 1)
                    print(f"Epoch {epoch+1}/{self.num_epochs} | Step {step+1} | Loss: {avg_loss:.4f}")
            
            epoch_loss = running_loss / len(self.train_loader)
            print(f"Epoch {epoch+1} Complete | Avg Loss: {epoch_loss:.4f} | Steps: {epoch_steps}")

        # Save Final LoRA adapters
        adapter_dir = "./llama3-70b-instruct-police-questions-lora-gemini-vlm"
        print(f"Saving adapter to {adapter_dir}...")
        self.model.save_pretrained(adapter_dir)
        self.tokenizer.save_pretrained(adapter_dir)

    def evaluate(self):
        if self.val_loader is None:
            print("Validation loader not initialized. Run preprocess_dataset() first.")
            return
        
        model_device = next(self.model.parameters()).device
        print("Evaluating...")
        
        # Increased max_gen_length to 256 to allow for 4 full questions
        token_f1, bert_score_mean = evaluate_model(
            self.model,
            self.val_loader,
            model_device,
            self.tokenizer,
            max_gen_length=256, 
            show_samples=5,
        )
        
        return token_f1, bert_score_mean
    
    def cleanup(self):
        # Clears GPU memory for the next tuning trial
        if self.model:
            del self.model
        if self.optimizer:
            del self.optimizer
        if self.tokenizer:
            del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()
