"""
PPO-based RLHF Training Script for Small LLMs
Fine-tune models like Llama-3.2-1B-Instruct using Proximal Policy Optimization

Requirements:
    pip install transformers trl peft accelerate bitsandbytes torch datasets
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from typing import List, Dict, Optional, Callable, Union
import os
from dataclasses import dataclass
import re


@dataclass
class RLHFConfig:
    """Configuration for RLHF training"""

    # Model settings
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"

    # Reward settings - can use either a reward model OR a custom reward function
    use_reward_model: bool = False  # Set to True to use a reward model
    reward_model_name: Optional[str] = None  # Only needed if use_reward_model=True

    # Training settings
    learning_rate: float = 1.41e-5
    batch_size: int = 4
    mini_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    ppo_epochs: int = 4
    max_steps: int = 1000

    # Generation settings
    max_new_tokens: int = 4000
    min_new_tokens: int = 32
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95

    # PPO hyperparameters
    init_kl_coef: float = 0.2
    target_kl: float = 6.0
    gamma: float = 1.0
    lam: float = 0.95
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.1

    # LoRA settings
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None

    # Quantization
    use_8bit: bool = False
    use_4bit: bool = True

    # Other settings
    output_dir: str = "./ppo_rlhf_output"
    log_with: str = "wandb"  # or "tensorboard"
    seed: int = 42

    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


class RLHFTrainer:
    """Trainer for PPO-based RLHF fine-tuning"""

    def __init__(
        self,
        config: RLHFConfig,
        reward_fn: Optional[Callable[[List[str], List[str]], List[float]]] = None,
    ):
        """
        Initialize RLHF trainer

        Args:
            config: Training configuration
            reward_fn: Optional custom reward function that takes (prompts, responses)
                      and returns a list of reward scores. If None and use_reward_model=False,
                      a default reward function will be used.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set random seed
        torch.manual_seed(config.seed)

        # Initialize components
        self.tokenizer = None
        self.model = None
        self.ref_model = None
        self.reward_model = None
        self.reward_tokenizer = None
        self.ppo_trainer = None

        # Set reward function
        if reward_fn is not None:
            self.reward_fn = reward_fn
        elif not config.use_reward_model:
            # Use default reward function if none provided
            self.reward_fn = self._default_reward_function
        else:
            self.reward_fn = None  # Will use reward model

    def setup_models(self):
        """Initialize policy model, reference model, and reward model"""
        print("Loading tokenizer and models...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Quantization config
        quantization_config = None
        if self.config.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.config.use_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        # Prepare for k-bit training if using quantization
        if quantization_config is not None:
            model = prepare_model_for_kbit_training(model)

        # Apply LoRA
        if self.config.use_lora:
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.lora_target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        # Wrap with value head for PPO
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

        # Reference model (frozen copy for KL divergence)
        # For efficiency, we can use the same model in eval mode
        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.config.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        # Load reward model only if configured to use one
        if self.config.use_reward_model:
            if self.config.reward_model_name is None:
                raise ValueError(
                    "reward_model_name must be specified when use_reward_model=True"
                )
            print("Loading reward model...")
            self.reward_tokenizer = AutoTokenizer.from_pretrained(
                self.config.reward_model_name
            )
            self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                self.config.reward_model_name,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            self.reward_model.eval()
        else:
            print("Using custom reward function (no reward model)")

        print("Models loaded successfully!")

    def setup_ppo_trainer(self):
        """Initialize PPO trainer"""
        ppo_config = PPOConfig(
            model_name=self.config.model_name,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            mini_batch_size=self.config.mini_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            ppo_epochs=self.config.ppo_epochs,
            init_kl_coef=self.config.init_kl_coef,
            target=self.config.target_kl,
            gamma=self.config.gamma,
            lam=self.config.lam,
            cliprange=self.config.cliprange,
            cliprange_value=self.config.cliprange_value,
            vf_coef=self.config.vf_coef,
            log_with=self.config.log_with,
            seed=self.config.seed,
        )

        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
        )

    def prepare_dataset(self, dataset_name: str = "Anthropic/hh-rlhf"):
        """
        Load and prepare dataset for RLHF training

        Args:
            dataset_name: HuggingFace dataset name or path to custom dataset
        """
        print(f"Loading dataset: {dataset_name}")

        # Load dataset
        dataset = load_dataset(dataset_name, split="train")

        # Filter and format dataset
        # Assuming dataset has 'chosen' field or 'prompt' field
        def tokenize_and_format(examples):
            # Customize this based on your dataset structure
            if "prompt" in examples:
                prompts = examples["prompt"]
            elif "chosen" in examples:
                # Extract prompt from chosen response
                prompts = [text.split("Assistant:")[0] + "Assistant:"
                          for text in examples["chosen"]]
            else:
                raise ValueError("Dataset must contain 'prompt' or 'chosen' field")

            return {"query": prompts}

        dataset = dataset.map(tokenize_and_format, batched=True)
        dataset = dataset.filter(lambda x: len(x["query"]) > 0)

        return dataset

    def _default_reward_function(
        self, prompts: List[str], responses: List[str]
    ) -> List[float]:
        """
        Default reward function that encourages helpful, harmless, and honest responses

        This is a simple heuristic-based reward function. You should replace this
        with your own domain-specific reward function.

        Args:
            prompts: List of input prompts
            responses: List of model responses

        Returns:
            List of reward scores
        """
        rewards = []
        for prompt, response in zip(prompts, responses):
            reward = 0.0

            # Length reward (prefer responses between 50-200 chars)
            response_len = len(response)
            if 50 <= response_len <= 200:
                reward += 1.0
            elif response_len < 50:
                reward += response_len / 50.0
            else:
                reward += max(0, 2.0 - response_len / 200.0)

            # Penalize very short responses
            if response_len < 10:
                reward -= 2.0

            # Penalize responses with excessive repetition
            words = response.lower().split()
            if len(words) > 0:
                unique_ratio = len(set(words)) / len(words)
                reward += unique_ratio

            # Reward proper sentence structure (ending with punctuation)
            if response.strip() and response.strip()[-1] in ".!?":
                reward += 0.5

            # Penalize toxic patterns (very simple check)
            toxic_patterns = ["hate", "kill", "stupid", "idiot", "shut up"]
            if any(pattern in response.lower() for pattern in toxic_patterns):
                reward -= 2.0

            rewards.append(reward)

        return rewards

    def compute_reward(self, prompts: List[str], responses: List[str]) -> List[float]:
        """
        Compute rewards for generated responses using either reward model or custom function

        Args:
            prompts: List of input prompts
            responses: List of model responses

        Returns:
            List of reward scores
        """
        if self.config.use_reward_model:
            # Use reward model
            texts = [prompt + response for prompt, response in zip(prompts, responses)]

            # Tokenize
            inputs = self.reward_tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.reward_model.device)

            # Get reward scores
            with torch.no_grad():
                outputs = self.reward_model(**inputs)
                rewards = outputs.logits.squeeze(-1).cpu().tolist()

            return rewards
        else:
            # Use custom reward function
            return self.reward_fn(prompts, responses)

    def generate_responses(self, queries: List[str]) -> Dict[str, List]:
        """
        Generate responses for a batch of queries

        Args:
            queries: List of input prompts

        Returns:
            Dictionary containing response tokens and text
        """
        # Tokenize queries
        query_tensors = [
            self.tokenizer.encode(query, return_tensors="pt")[0].to(self.device)
            for query in queries
        ]

        # Generate responses
        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "min_new_tokens": self.config.min_new_tokens,
            "temperature": self.config.temperature,
            "top_k": self.config.top_k,
            "top_p": self.config.top_p,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        response_tensors = []
        for query_tensor in query_tensors:
            response = self.ppo_trainer.generate(
                query_tensor.unsqueeze(0),
                **generation_kwargs
            )
            response_tensors.append(response.squeeze()[len(query_tensor):])

        # Decode responses
        response_texts = [
            self.tokenizer.decode(r, skip_special_tokens=True)
            for r in response_tensors
        ]

        return {
            "response_tensors": response_tensors,
            "response_texts": response_texts,
        }

    def train(
        self,
        dataset_name: str = "Anthropic/hh-rlhf",
        num_samples: Optional[int] = None,
    ):
        """
        Main training loop

        Args:
            dataset_name: Name of dataset to use
            num_samples: Optional limit on number of samples to use
        """
        # Setup
        self.setup_models()
        self.setup_ppo_trainer()
        dataset = self.prepare_dataset(dataset_name)

        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))

        print(f"Starting PPO training on {len(dataset)} samples...")
        print(f"Output directory: {self.config.output_dir}")

        # Training loop
        for step, batch in enumerate(self.ppo_trainer.dataloader):
            if step >= self.config.max_steps:
                break

            # Get queries
            queries = [item["query"] for item in batch]

            # Generate responses
            outputs = self.generate_responses(queries)
            response_tensors = outputs["response_tensors"]
            response_texts = outputs["response_texts"]

            # Compute rewards
            rewards = self.compute_reward(queries, response_texts)
            rewards = [torch.tensor(r) for r in rewards]

            # Get query tensors
            query_tensors = [
                self.tokenizer.encode(query, return_tensors="pt")[0].to(self.device)
                for query in queries
            ]

            # Run PPO step
            stats = self.ppo_trainer.step(query_tensors, response_tensors, rewards)

            # Log statistics
            if step % 10 == 0:
                print(f"\nStep {step}/{self.config.max_steps}")
                print(f"Mean reward: {stats['ppo/mean_scores']:.4f}")
                print(f"Mean KL: {stats.get('objective/kl', 0):.4f}")
                print(f"Policy loss: {stats.get('ppo/policy/loss', 0):.4f}")
                print(f"Value loss: {stats.get('ppo/value/loss', 0):.4f}")

                # Show example
                if len(response_texts) > 0:
                    print(f"\nExample generation:")
                    print(f"Query: {queries[0][:100]}...")
                    print(f"Response: {response_texts[0][:200]}...")
                    print(f"Reward: {rewards[0]:.4f}")

            # Save checkpoint
            if step % 100 == 0 and step > 0:
                save_path = os.path.join(self.config.output_dir, f"checkpoint-{step}")
                self.ppo_trainer.save_pretrained(save_path)
                print(f"Saved checkpoint to {save_path}")

        # Save final model
        final_path = os.path.join(self.config.output_dir, "final_model")
        self.ppo_trainer.save_pretrained(final_path)
        print(f"\nTraining complete! Model saved to {final_path}")

    def save_model(self, output_path: str):
        """Save the trained model"""
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print(f"Model saved to {output_path}")


def custom_reward_function(prompts: List[str], responses: List[str]) -> List[float]:
    """
    Example custom reward function

    This example rewards concise, informative responses and penalizes
    overly long or repetitive ones. Customize this for your specific use case.

    Args:
        prompts: List of input prompts
        responses: List of model responses

    Returns:
        List of reward scores
    """
    rewards = []
    for prompt, response in zip(prompts, responses):
        reward = 0.0

        # Base reward for completing the response
        reward += 1.0

        # Length-based reward (prefer 20-150 chars)
        response_len = len(response)
        if 20 <= response_len <= 150:
            reward += 2.0
        elif response_len < 20:
            reward += response_len / 20.0 - 1.0
        else:
            reward -= (response_len - 150) / 100.0

        # Reward diversity (penalize repetition)
        words = response.lower().split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            reward += unique_ratio * 2.0

        # Reward proper formatting
        if response.strip():
            # Starts with capital letter
            if response[0].isupper():
                reward += 0.5
            # Ends with punctuation
            if response.strip()[-1] in ".!?":
                reward += 0.5

        # Domain-specific rewards (customize these!)
        # Example: Reward helpful phrases
        helpful_phrases = ["here", "you can", "to help", "let me"]
        if any(phrase in response.lower() for phrase in helpful_phrases):
            reward += 1.0

        # Example: Penalize refusals (if you want the model to be more helpful)
        refusal_phrases = ["i cannot", "i can't", "unable to", "not able"]
        if any(phrase in response.lower() for phrase in refusal_phrases):
            reward -= 1.0

        rewards.append(reward)

    return rewards




def main():
    """Example usage"""

    # Example 1: Using custom reward function (default)
    print("=" * 60)
    print("Example 1: Using custom reward function")
    print("=" * 60)

    config = RLHFConfig(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        use_reward_model=False,  # Use custom reward function
        learning_rate=1.41e-5,
        batch_size=4,
        mini_batch_size=1,
        max_steps=1000,
        output_dir="./llama_ppo_rlhf_custom",
        use_lora=True,
        use_4bit=True,
        log_with="tensorboard",
    )

    # Initialize trainer with custom reward function
    trainer = RLHFTrainer(config, reward_fn=custom_reward_function)

    # Train
    trainer.train(
        dataset_name="Anthropic/hh-rlhf",
        num_samples=100,  # Use subset for testing
    )

    # Example 2: Using reward model
    # Uncomment to use a reward model instead
    """
    print("=" * 60)
    print("Example 2: Using reward model")
    print("=" * 60)

    config_with_model = RLHFConfig(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        use_reward_model=True,
        reward_model_name="OpenAssistant/reward-model-deberta-v3-large-v2",
        learning_rate=1.41e-5,
        batch_size=4,
        mini_batch_size=1,
        max_steps=1000,
        output_dir="./llama_ppo_rlhf_model",
        use_lora=True,
        use_4bit=True,
        log_with="tensorboard",
    )

    trainer_with_model = RLHFTrainer(config_with_model)
    trainer_with_model.train(
        dataset_name="Anthropic/hh-rlhf",
        num_samples=100,
    )
    """

    # Example 3: Using different custom reward functions
    # Uncomment to try length-based reward
    """
    config_length = RLHFConfig(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        use_reward_model=False,
        output_dir="./llama_ppo_rlhf_length",
    )
    trainer_length = RLHFTrainer(config_length, reward_fn=length_based_reward)
    trainer_length.train(dataset_name="Anthropic/hh-rlhf", num_samples=100)
    """

    # Example 4: Using brevity penalty reward
    """
    config_bp = RLHFConfig(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        use_reward_model=False,
        output_dir="./llama_ppo_rlhf_brevity",
    )
    trainer_bp = RLHFTrainer(config_bp, reward_fn=brevity_penalty_reward)
    trainer_bp.train(dataset_name="Anthropic/hh-rlhf", num_samples=100)
    """


if __name__ == "__main__":
    main()
