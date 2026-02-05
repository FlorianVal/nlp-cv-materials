import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

# 1. Load GPT-2 Model
print("Loading pre-trained model...")
config = AutoConfig.from_pretrained('google/gemma-3-1b-it')
model = AutoModel.from_pretrained('google/gemma-3-1b-it', config=config)
print("model loaded.")
print("-" * 50)

# 2. Inspect Original Model Layers
print(model)
print("Original Model Layers:")
for name, module in model.named_modules():
    if name:  # Avoid printing the top-level model itself
        print(f"- {name}: {type(module).__name__}")
print("-" * 50)


# 3. Define LoRA Layer
class LoraLayer(nn.Module):
    """Implements the LoRA layer logic."""
    def __init__(self, in_features, out_features, rank, alpha):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank
        self.rank = rank

        # Initialize LoRA parameters
        nn.init.uniform_(self.lora_A, a=0.01)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Original forward pass is handled by the wrapped layer
        # This computes the LoRA adjustment: B * A * x
        # Note: The order is B then A, matching the paper's notation delta_W = B @ A
        # Input x has shape (batch_size, seq_len, in_features)
        # lora_A @ x needs careful handling of dimensions
        # We want (batch, seq, out) = (batch, seq, in) @ (in, rank) @ (rank, out) * scale
        # Let's adjust x shape for matmul: (batch * seq, in)
        original_shape = x.shape
        x_reshaped = x.reshape(-1, original_shape[-1])  # (batch * seq, in_features)

        # Compute LoRA adjustment
        # lora_A: (rank, in_features)
        # lora_B: (out_features, rank)
        # x_reshaped: (batch * seq, in_features)
        # Adjustment: (x_reshaped @ lora_A.T) @ lora_B.T * scaling
        #             (batch*seq, in) @ (in, rank) -> (batch*seq, rank)
        #             (batch*seq, rank) @ (rank, out) -> (batch*seq, out)
        lora_adjustment = (x_reshaped @ self.lora_A.T) @ self.lora_B.T * self.scaling

        # Reshape back to original sequence dimension
        lora_adjustment = lora_adjustment.reshape(original_shape[0], original_shape[1], -1)  # (batch, seq, out_features)
        return lora_adjustment


# 4. Define LoRA Integration Wrapper for Linear Layers
class LoraLinear(nn.Module):
    """Wraps an existing nn.Linear layer and adds LoRA adaptation."""
    def __init__(self, linear_layer, rank, alpha):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank
        self.alpha = alpha

        # Keep the original linear layer
        self.linear = linear_layer
        # Freeze the original weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        # Add the LoRA layer
        self.lora_layer = LoraLayer(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=self.rank,
            alpha=self.alpha
        )

    def forward(self, x):
        # Original output
        original_output = self.linear(x)
        # LoRA adjustment
        lora_adjustment = self.lora_layer(x)
        # Combine
        return original_output + lora_adjustment

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, alpha={self.alpha}'


# 5. Function to Apply LoRA to Target Layers
def apply_lora_to_gpt2(model, rank, alpha, target_module_names=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]):
    """
    Applies LoRA wrappers to specified Linear layers within the GPT-2 model.

    Args:
        model: The GPT-2 model instance.
        rank (int): The rank for the LoRA decomposition.
        alpha (int): The scaling factor for LoRA.
        target_module_names (list): List of names for Linear layers to target.
                                    Defaults usually target attention projections.
    """
    print(f"\nApplying LoRA with rank={rank}, alpha={alpha} to modules containing: {target_module_names}")
    for name, module in model.named_modules():
        # Split name to check the last part (actual layer name)
        parts = name.split('.')
        if len(parts) > 0:
            module_name = parts[-1]
            # Check if it's a Linear layer and its name matches targets
            if isinstance(module, nn.Linear) and any(target_name in module_name for target_name in target_module_names):
                # Find the parent module to replace the child
                parent_name = '.'.join(parts[:-1])
                parent_module = model.get_submodule(parent_name)

                # Create the LoraLinear wrapper
                lora_wrapped_layer = LoraLinear(module, rank=rank, alpha=alpha)

                # Replace the original linear layer with the wrapped one
                setattr(parent_module, module_name, lora_wrapped_layer)
                print(f"  - Replaced '{name}' with LoraLinear")

    print("LoRA application complete.")
    print("-" * 50)


# 6. Instantiate & Apply LoRA
LORA_RANK = 8  # Example rank
LORA_ALPHA = 16  # Example alpha (often rank * 2)

# Reload the model to ensure we start fresh before applying LoRA
model = AutoModel.from_pretrained('google/gemma-3-1b-it', config=config)
apply_lora_to_gpt2(model, rank=LORA_RANK, alpha=LORA_ALPHA)

# 7. Print Only LoRA Parameters
print("LoRA Parameters (Trainable):")
total_lora_params = 0
total_params = 0
for name, param in model.named_parameters():
    total_params += param.numel()
    if param.requires_grad:  # LoRA parameters are the only ones requiring grad now
        print(f"- {name}: {param.shape} ({param.numel()} parameters)")
        total_lora_params += param.numel()

print("-" * 50)
print(f"Total parameters in original model: {model.num_parameters()}")  # Transformers helper
print(f"Total parameters in modified model: {total_params}")  # Manual count includes LoRA
print(f"Total trainable LoRA parameters added: {total_lora_params}")
print(f"Percentage of trainable parameters: {total_lora_params / total_params * 100:.4f}%")
print("-" * 50)

# Example of how to use the modified model (optional)
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model(**inputs)
# print("\nExample Output Shape:", outputs.last_hidden_state.shape)
