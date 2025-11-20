from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
from typing import Optional
from diversity_metrics import *
from thefuzz import fuzz
from transformers import TrainerCallback, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import torch
import random
from diversity_gen import ExampleSummarizer, DataCreator
from dspy.adapters import ChatAdapter
from dspy.utils.exceptions import AdapterParseError
import json


class DynamicPromptDataGenCallback(TrainerCallback):
    def __init__(self, trainer: GRPOTrainer):
        super().__init__()
        self.trainer = trainer
        self.data_summaries = {}
        
    def on_step_start(self, args, state, control, **kwargs):
        dataset = self.trainer.train_dataset
        # Update prompts in dataset to include summary
        if state.global_step % 10 == 0:  # Every 50 steps
            if len(data_storage.generated_data):
                summary = self._generate_data_summary(
                    data_storage.generated_data,
                    data_storage.seen_examples
                )
                # Store for next round
                data_storage.data_summary = summary
                self.data_summaries[state.global_step] = summary
            else:
                self.data_summaries[state.global_step] = data_storage.data_summary
        self._update_prompts_with_summary(data_storage.data_summary)
        print(data_storage.data_summary)
        
        return control
    
    def _generate_data_summary(self, generated: set, seen: set) -> str:
        """Generate a summary of data collected so far"""
        model = self.trainer.model
        tokenizer = self.trainer.processing_class

        # Create summary prompt
        all_data = list(generated) + list(seen)
        random.shuffle(all_data)

        # Use chat message format for instruct model
        summary_messages = chat_adapter.format(
            ExampleSummarizer,
            demos=[],
            inputs={
                "example_list": all_data[:20],
                "curr_summary": data_storage.data_summary
            }
        )
        #         summary_messages = [
        #             {
        #                 "role": "user",
        #                 "content": f"""Given a list of example data points for a dataset, provide a brief summary of these examples. If there are no examples, your summary should be "No data has been generated yet". Be comprehensive in your summary but additionally concise. The summary should be at most 3 sentences.

        # Data points:
        # {chr(10).join(all_data[:50])}

        # Please provide your summary:"""
        #             }
        #         ]

        # Generate summary using chat template
        inputs = tokenizer.apply_chat_template(
            summary_messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=256,
                do_sample=False,  # Use greedy decoding for summary
                pad_token_id=tokenizer.eos_token_id
            )

        summary = tokenizer.decode(
            outputs[0][inputs.shape[1]:],
            skip_special_tokens=True
        )
        
        try:
            summary = chat_adapter.parse(ExampleSummarizer, summary)["summary"]
        except AdapterParseError:
            summary = data_storage.data_summary
        print(summary)
        return summary.strip()
    
    def _update_prompts_with_summary(self, summary: str):
        """Update dataset prompts to include current data summary"""
        dataset = self.trainer.train_dataset

        def add_summary_to_prompt(example):
            # Handle chat message format
            if isinstance(example["prompt"], list):
                # Already in chat format - update the last user message
                prompt_messages = example["prompt"].copy()
                if "<SUMMARY>" in prompt_messages[-1]["content"]:
                    prompt_messages[-1]["content"] = prompt_messages[-1]["content"].replace(
                        "<SUMMARY>", summary
                    )
                example["prompt"] = prompt_messages
            else:
                # String format - replace directly
                original_prompt = example["prompt"]
                enhanced_prompt = original_prompt.replace("<SUMMARY>", summary)
                example["prompt"] = enhanced_prompt

            return example

        # Update dataset
        self.trainer.train_dataset = dataset.map(add_summary_to_prompt)


class DataShareSingleton():
    def __init__(self):
        self.generated_data = []
        self.seen_examples = []
        self.data_summary = "No data has been generated yet."
    
data_storage = DataShareSingleton()

def set_singleton(generated_data, seen_examples, data_summary):
    global data_storage
    data_storage.generated_data = generated_data
    data_storage.seen_examples = seen_examples
    data_storage.data_summary = data_summary

def diversity_metric(gold_examples: list[str], pred: list[str]):
    computed_dc_score = dc_score(data_storage.seen_examples + data_storage.generated_data)
    computed_style_cos_score = style_cosine_sim(gold_examples, pred)
    all_brevity_penalty = []
    for g in gold_examples:
        for p in pred:
            brevity = brevity_penalty(len(p.split()), len(g.split()))
            all_brevity_penalty.append(brevity)
    brevity_score = sum(all_brevity_penalty) / len(all_brevity_penalty)
    # overall_score = computed_dc_score - computed_cos_score + computed_neg_cos_sim + computed_style_cos_score + brevity_score
    overall_score = computed_dc_score + computed_style_cos_score + brevity_score
    return overall_score

def diversity_reward(completions: list[list[dict[str, str]]], golden_examples: list[str], pii_integration: Optional[list[str]], **kwargs) -> list[float | None]:
    """
    Sum between the diversity-driven metric score and the pii integration (computed by fuzzy match)
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, gold, pii in zip(contents, golden_examples, pii_integration):
        gold = gold.split("|||")
        data_storage.seen_examples.extend(gold)
        data_storage.seen_examples = list(set(data_storage.seen_examples))
        try:
            gen_data = chat_adapter.parse(DataCreator, content)["generated_instances"]
        except AdapterParseError:
            rewards.append(0)
            continue
        if len(gen_data) == 0:
            rewards.append(0)
            continue
        data_storage.generated_data.extend(gen_data)
        print(gen_data)
        # content = content.split("|||")
        if pii:
            reward = diversity_metric(gold, gen_data) + fuzz.partial_token_set_ratio(pii, content) / 100.0
        else:
            reward = diversity_metric(gold, gen_data)
        rewards.append(reward)

    return rewards


def load_model_with_peft(
    model_name: str,
    use_lora: bool = True,
    use_4bit: bool = True,
    use_8bit: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[list[str]] = None,
):
    """
    Load model with PEFT (LoRA) for efficient fine-tuning

    Args:
        model_name: HuggingFace model name or path
        use_lora: Whether to use LoRA
        use_4bit: Use 4-bit quantization (requires bitsandbytes)
        use_8bit: Use 8-bit quantization (requires bitsandbytes)
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling parameter
        lora_dropout: LoRA dropout rate
        lora_target_modules: Which modules to apply LoRA to (None = auto-detect)

    Returns:
        model: The prepared model (quantized + LoRA if configured)
        tokenizer: The tokenizer
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure quantization
    quantization_config = None
    if use_4bit:
        print("Loading model with 4-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif use_8bit:
        print("Loading model with 8-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    # Load model
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16 if quantization_config else "auto",
    )

    # Prepare model for k-bit training if using quantization
    if quantization_config is not None:
        print("Preparing model for k-bit training...")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True
        )
        # Enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    # Apply LoRA
    if use_lora:
        print("Applying LoRA configuration...")

        # Auto-detect target modules if not specified
        if lora_target_modules is None:
            # Common target modules for different model architectures
            if "qwen" in model_name.lower():
                lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            elif "llama" in model_name.lower():
                lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            elif "mistral" in model_name.lower():
                lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            elif "phi" in model_name.lower():
                lora_target_modules = ["q_proj", "k_proj", "v_proj", "dense"]
            else:
                # Default fallback
                lora_target_modules = ["q_proj", "v_proj"]

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Fix dtype mismatch issues - ensure lm_head is in correct dtype
    # This fixes the "expected scalar type Float but found BFloat16" error
    if hasattr(model, 'lm_head'):
        if quantization_config is not None:
            # For quantized models, keep lm_head in float32 for stability
            model.lm_head = model.lm_head.to(torch.float32)

    return model, tokenizer


if __name__ == "__main__":
    # Configuration flags
    USE_PEFT = True  # Set to False to disable PEFT/LoRA
    USE_4BIT = True  # Use 4-bit quantization (reduces memory by ~4x)
    USE_8BIT = False  # Use 8-bit quantization (reduces memory by ~2x)

    # LoRA configuration
    LORA_R = 16  # LoRA rank (higher = more parameters, better but slower)
    LORA_ALPHA = 32  # LoRA alpha (typically 2x rank)
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = None  # Auto-detect, or specify: ["q_proj", "v_proj", ...]

    # Load datasets
    train_json = json.load(open("training_data.json"))
    test_json = json.load(open("test_data.json"))
    train_dataset = Dataset.from_dict(train_json)
    test_dataset = Dataset.from_dict(test_json)

    # GRPO configuration
    grpo_config = GRPOConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # Effective batch size = 2 * 4 = 8
        learning_rate=1e-6,  # Lower LR for LoRA fine-tuning
        num_train_epochs=3,
        do_eval=False,
        do_train=True,
        output_dir="trained_qwen_peft/",
        overwrite_output_dir=True,
        torch_empty_cache_steps=50,
        save_strategy="steps",
        save_steps=100,
        logging_steps=10,
        # Mixed precision training
        # NOTE: With quantization, use fp16 instead of bf16 to avoid dtype mismatches
        bf16=False,  # Disable bf16 when using quantization
        fp16=True if USE_4BIT or USE_8BIT else False,
        # Gradient checkpointing for memory efficiency
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    chat_adapter = ChatAdapter()

    # Load model with PEFT
    if USE_PEFT:
        print("=" * 60)
        print("Loading model with PEFT (LoRA) for efficient training")
        print("=" * 60)
        model, tokenizer = load_model_with_peft(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            use_lora=True,
            use_4bit=USE_4BIT,
            use_8bit=USE_8BIT,
            lora_r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            lora_target_modules=LORA_TARGET_MODULES,
        )

        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=diversity_reward,
            train_dataset=train_dataset,
            args=grpo_config
        )
    else:
        # Standard training without PEFT
        print("=" * 60)
        print("Loading model without PEFT (full fine-tuning)")
        print("=" * 60)
        trainer = GRPOTrainer(
            model="Qwen/Qwen2.5-1.5B-Instruct",
            reward_funcs=diversity_reward,
            train_dataset=train_dataset,
            args=grpo_config
        )

    # Add dynamic callback
    dynamic_callback = DynamicPromptDataGenCallback(trainer=trainer)
    trainer.add_callback(dynamic_callback)

    # Train
    print("=" * 60)
    print("Starting training...")
    print("=" * 60)
    trainer.train()

    # Save model
    print("=" * 60)
    print("Saving model...")
    print("=" * 60)
    if USE_PEFT:
        # Save LoRA adapters
        trainer.model.save_pretrained("trained_qwen_peft/lora_adapters")
        tokenizer.save_pretrained("trained_qwen_peft/lora_adapters")
        print(f"LoRA adapters saved to trained_qwen_peft/lora_adapters/")
        print("To load: model = AutoModelForCausalLM.from_pretrained('base_model')")
        print("         model = PeftModel.from_pretrained(model, 'trained_qwen_peft/lora_adapters')")
    else:
        trainer.save_model("trained_qwen_peft/full_model")
        print(f"Full model saved to trained_qwen_peft/full_model/")