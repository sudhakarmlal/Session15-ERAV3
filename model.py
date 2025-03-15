import torch
import torch.nn as nn
import math

# RMSNorm is a normalization technique that normalizes the input by dividing by the square root of the variance plus a small number to prevent division by zero
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5): # the number of features/dimensions/embeddings in the input, eps is a small number to prevent division by zero
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size)) # weight is a learnable parameter that scales the input
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).sqrt() + self.eps # compute the norm of the input
        return x / norm * self.weight # normalize the input by dividing by the norm and scale it by the weight parameter


# RotaryEmbedding is a technique that rotates the input by a learnable angle
class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim

    def forward(self, seq_len, device):
        # Create position embeddings
        t = torch.arange(seq_len, device=device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        
        # Create rotation matrices [seq_len, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # [1, seq_len, 1, dim]
        cos = emb.cos().view(1, seq_len, 1, self.dim)
        sin = emb.sin().view(1, seq_len, 1, self.dim)
        
        return cos, sin

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def apply_rotary_pos_emb(self, x, cos, sin):
        # Expand cos/sin to match batch size and heads
        cos = cos.expand(x.shape[0], -1, x.shape[2], -1)
        sin = sin.expand(x.shape[0], -1, x.shape[2], -1)
        
        return (x * cos) + (self.rotate_half(x) * sin)

# This code is commented as new LlamaMLP to be created as per Mixture of Experts implementation
# class LlamaMLP(nn.Module):
#     def __init__(self, dim, hidden_dim):
#         super().__init__()
#         self.gate_proj = nn.Linear(dim, hidden_dim, bias=False) # create the gate projection layer with the input dimension and the hidden dimension
#         self.up_proj = nn.Linear(dim, hidden_dim, bias=False) # create the up projection layer with the input dimension and the hidden dimension
#         self.down_proj = nn.Linear(hidden_dim, dim, bias=False) # create the down projection layer with the hidden dimension and the output dimension
#         self.act_fn = nn.SiLU() # create the activation function

#     def forward(self, x):
#         gated = self.gate_proj(x) # apply the gate projection to the input
#         hidden = self.up_proj(x) # apply the up projection to the input
#         return self.down_proj(self.act_fn(gated * hidden)) # apply the activation function to the gated and hidden values and then apply the down projection

class LlamaMLP(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts, num_shared_experts, top_k):
        super().__init__()
        self.moe = DeepSeekMoE(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            top_k=top_k
        )
    def forward(self, x):
        return self.moe(x)

class DeepSeekExpertLayer(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim,bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.act_fn = nn.SiLU()
    
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class DeepSeekMoE(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts, num_shared_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_experts - num_shared_experts
        self.top_k = top_k
        self.dim = dim

        # Shared experts
        self.shared_experts = nn.ModuleList([
            DeepSeekExpertLayer(dim, hidden_dim)
            for _ in range(self.num_shared_experts)
        ])
        
        # Routed Experts
        self.routed_experts = nn.ModuleList([
            DeepSeekExpertLayer(dim, hidden_dim)
            for _ in range(self.num_routed_experts)
        ])

        # Routed Components
        self.router = nn.Linear(dim, self.num_routed_experts, bias=False)
        self.routing_bias = nn.Parameter(torch.zeros(self.num_routed_experts))
    
    def forward(self, x):
        batch_size, seq_len, dim = x.shape

        shared_output = sum(expert(x) for expert in self.shared_experts)
        if self.num_shared_experts > 1:
            shared_output = shared_output / self.num_shared_experts

        # calculating routing scores
        routing_logits = self.router(x) * self.routing_bias

        # get top-k experts per token
        routing_probs = torch.sigmoid(routing_logits)
        scores, indices = torch.topk(routing_probs, self.top_k, dim=-1)

        #normalize the top k scores
        scores = scores / scores.sum(dim=-1, keepdim=True)

        # Process through selected experts
        combined_output = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_indices = indices[..., k]
            expert_scores = scores[..., k:k+1]

            #process each expert
            for i in range(self.num_routed_experts):
                mask = (expert_indices == i)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.routed_experts[i](expert_input)
                    combined_output[mask] += expert_output * expert_scores[mask]
        
        #combine shared and routed outputs
        final_output = shared_output + combined_output
        return final_output
    
    def update_bias_terms(self, expert_load):
        target_load = 1.0 / self.num_routed_experts
        load_diff = expert_load - target_load

        update_rate = 0.1 * torch.abs(load_diff)

        self.routing_bias.data -= update_rate * load_diff

class LlamaAttention(nn.Module):
    def __init__(self, dim, num_heads, compress_ratio):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.latent_dim = dim // compress_ratio
        
        # Decomposed projections for latent attention
        self.q_proj_d = nn.Linear(dim, self.latent_dim, bias=False)  # Down projection for Q
        self.kv_proj_d = nn.Linear(dim, self.latent_dim, bias=False)  # Down projection for K,V
        
        half_head_dim = self.head_dim // 2
        # Up projections from latent space
        self.q_proj_u = nn.Linear(self.latent_dim, num_heads * half_head_dim, bias=False)
        self.k_proj_u = nn.Linear(self.latent_dim, num_heads * half_head_dim, bias=False)
        self.v_proj_u = nn.Linear(self.latent_dim, dim, bias=False)
        
        # Rotary components
        self.rotary_emb = LlamaRotaryEmbedding(dim=half_head_dim)
        
        # Output projection
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        half_head_dim = self.head_dim // 2
        
        # Project to latent space
        q_latent = self.q_proj_d(x)
        kv_latent = self.kv_proj_d(x)
        
        # Project up from latent space
        q = self.q_proj_u(q_latent)
        k = self.k_proj_u(kv_latent)
        v = self.v_proj_u(kv_latent)
        
        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_heads, half_head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, half_head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply rotary embeddings
        cos, sin = self.rotary_emb(seq_len, x.device)
        q = self.rotary_emb.apply_rotary_pos_emb(q, cos, sin)
        k = self.rotary_emb.apply_rotary_pos_emb(k, cos, sin)
        
        # Prepare for attention
        q = q.transpose(1, 2)  # [batch, heads, seq, half_head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention with scaled dot product
        scale = 1 / math.sqrt(half_head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous()  # [batch, seq, heads, head_dim]
        out = out.reshape(batch_size, seq_len, self.dim)
        
        return self.o_proj(out)


        # previous working code
        # q = self.q_proj(x)
        # k = self.k_proj(x)
        # v = self.v_proj(x)

        # # Split heads
        # q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # [batch_size, num_heads, seq_len, head_dim]
        # k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # # Scaled dot-product attention
        # scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # attention = torch.softmax(scores, dim=-1)
        # context = torch.matmul(attention, v)

        # # Combine heads
        # context = context.transpose(1, 2).reshape(batch_size, seq_len, dim)
        # return self.o_proj(context)

class LlamaDecoderLayer(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads, compress_ratio, num_experts, num_shared_experts, top_k):
        super().__init__()
        self.self_attn = LlamaAttention(dim, num_heads, compress_ratio=3)
        self.mlp = LlamaMLP(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            top_k=top_k)
        self.input_layernorm = LlamaRMSNorm(dim)
        self.post_attention_layernorm = LlamaRMSNorm(dim)

    def forward(self, x):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        x = x + residual

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = x + residual
        return x


class LlamaModel(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, hidden_dim, num_heads):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(dim, hidden_dim, num_heads, compress_ratio=3, num_experts=8, num_shared_experts=1, top_k=2) for _ in range(num_layers)
        ])
        self.norm = LlamaRMSNorm(dim)
        #self.rotary_emb = LlamaRotaryEmbedding(dim)

    def forward(self, x):
        x = self.embed_tokens(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class LlamaForCausalLM(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, hidden_dim, num_heads):
        super().__init__()
        self.model = LlamaModel(vocab_size, dim, num_layers, hidden_dim, num_heads)
        self.num_heads = num_heads
        # Share weights between embedding and lm_head
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        # Tie weights
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, x):
        x = self.model(x)
        return self.lm_head(x)

def get_model(tokenizer):
    vocab_size = tokenizer.vocab_size  # Use actual tokenizer vocab size
    return LlamaForCausalLM(
        vocab_size=vocab_size,
        dim=576,
        num_layers=30,
        hidden_dim=1536,
        num_heads=9
    )
