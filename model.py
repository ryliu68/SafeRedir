import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import MethodType


class LoRADelta(nn.Module):
    def __init__(self, dim=768, rank=8):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return self.up(self.down(x)) * self.scale


class MaskPredictor(nn.Module):
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)
        self.predictor = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        mask_logits = self.predictor(attn_out).squeeze(-1)
        return torch.sigmoid(mask_logits)


class AdaptiveAlpha(nn.Module):
    def __init__(self, prompt_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(prompt_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, prompt_embed):
        return self.net(prompt_embed)


class PromptCrossAttentionEnhanced(nn.Module):
    def __init__(self, joint_dim, prompt_dim):
        super().__init__()
        self.query = nn.Linear(joint_dim, prompt_dim)
        self.key = nn.Linear(prompt_dim, prompt_dim)
        self.value = nn.Linear(prompt_dim, prompt_dim)
        self.norm = nn.LayerNorm(prompt_dim)

    def forward(self, joint_feat, prompt_embed):
        q = self.query(joint_feat).unsqueeze(1)
        k = self.key(prompt_embed)
        v = self.value(prompt_embed)
        attn = torch.softmax((q @ k.transpose(-2, -1)) / (768 ** 0.5), dim=-1)
        out = attn @ v
        return self.norm(out.squeeze(1) + v.mean(dim=1))


class MultiScalePromptAttention(nn.Module):
    def __init__(self, joint_dim, prompt_dim, n_heads=4):
        super().__init__()
        self.heads = nn.ModuleList([
            PromptCrossAttentionEnhanced(joint_dim, prompt_dim)
            for _ in range(n_heads)
        ])
        self.fuse = nn.Sequential(
            nn.Linear(n_heads * prompt_dim, prompt_dim),
            nn.LayerNorm(prompt_dim)
        )

    def forward(self, joint_feat, prompt_embed):
        heads_out = [head(joint_feat, prompt_embed) for head in self.heads]
        concat = torch.cat(heads_out, dim=-1)
        return self.fuse(concat)


def get_timestep_embedding(timesteps, embedding_dim=320, max_period=10000):
    if timesteps.ndim == 2:
        timesteps = timesteps.view(-1)
    half = embedding_dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if embedding_dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = nn.Conv2d(in_channels, out_channels, 1, stride=stride) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return F.relu(self.conv(x) + self.downsample(x))


class SafeRedir(nn.Module):
    def __init__(self, latent_dim=4, t_dim=320, seq_len=77, prompt_dim=768,
                 rank=16, prompt_dropout=0.0, token_dropout=0.0, use_alpha=True):
        super().__init__()
        self.seq_len = seq_len
        self.prompt_dim = prompt_dim
        self.t_dim = t_dim
        self.prompt_dropout = prompt_dropout
        self.token_dropout = token_dropout
        self.use_alpha = use_alpha
        self.joint_feat_dim = 512 + 64

        self.latent_encoder = nn.Sequential(
            ResidualSEBlock(latent_dim, 64),
            ResidualSEBlock(64, 128),
            ResidualSEBlock(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.t_encoder = nn.Sequential(
            nn.Linear(t_dim, 64),
            nn.SiLU(),
            nn.LayerNorm(64)
        )

        self.cross_attn = MultiScalePromptAttention(joint_dim=self.joint_feat_dim, prompt_dim=prompt_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, prompt_dim))

        self.classifier = nn.Sequential(
            nn.Linear(self.joint_feat_dim + prompt_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )

        self.token_proj = nn.Sequential(
            nn.Linear(self.joint_feat_dim + prompt_dim, 512),
            nn.ReLU(),
            nn.Linear(512, prompt_dim)
        )

        self.delta_head = LoRADelta(dim=prompt_dim, rank=rank)
        self.gate_linear = nn.Linear(prompt_dim, 1)
        self.mask_predictor = MaskPredictor(prompt_dim, n_heads=4)

        if self.use_alpha:
            self.alpha_predictor = AdaptiveAlpha(prompt_dim)

    def _apply_prompt_dropout(self, prompt_embed):
        if self.prompt_dropout > 0:
            B, T, D = prompt_embed.shape
            drop_mask = (torch.rand(B, T, device=prompt_embed.device) > self.prompt_dropout).float()
            prompt_embed = prompt_embed * drop_mask.unsqueeze(-1)
        return prompt_embed

    def forward(self, latent, prompt_embed, t_int, token_mask=None):
        if prompt_embed.ndim == 4 and prompt_embed.shape[1] == 2:
            prompt_embed = prompt_embed[:, 1]
        elif prompt_embed.ndim == 3 and prompt_embed.shape[0] == 2 * latent.shape[0]:
            prompt_embed = prompt_embed[latent.shape[0]:]

        prompt_embed = self._apply_prompt_dropout(prompt_embed)
        latent_feat = self.latent_encoder(latent)
        t_embed = get_timestep_embedding(t_int, embedding_dim=self.t_dim)
        t_feat = self.t_encoder(t_embed)
        joint_feat = torch.cat([latent_feat, t_feat], dim=1)

        cross_attn_feat = self.cross_attn(joint_feat, prompt_embed)
        pooled_prompt = prompt_embed.mean(dim=1)
        classifier_input = torch.cat([joint_feat, pooled_prompt, cross_attn_feat], dim=1)
        logits = self.classifier(classifier_input)

        token_input = joint_feat.unsqueeze(1).repeat(1, self.seq_len, 1)
        cross_feat = cross_attn_feat.unsqueeze(1).repeat(1, self.seq_len, 1)
        concat_feat = torch.cat([token_input, cross_feat], dim=-1)
        delta_base = self.token_proj(concat_feat)
        delta = self.delta_head(delta_base)

        enriched_embed = prompt_embed + self.pos_embed
        gate = torch.sigmoid(self.gate_linear(enriched_embed))
        delta = delta * gate

        pred_mask = self.mask_predictor(prompt_embed)
        if token_mask is not None:
            delta = delta * token_mask.unsqueeze(-1)
        else:
            delta = delta * pred_mask.unsqueeze(-1)

        if self.use_alpha:
            alpha = self.alpha_predictor(prompt_embed)
        else:
            alpha = None

        return logits, delta, pred_mask, alpha


def register_guided_redirector_hooks(pipe, redirector, replace_steps=1, alpha_scale=2.0, mask_again=False):
    state = {
        "prompt_embed": None,
        "original_embed": None,
        "device": pipe.device,
        "countdown": 0
    }

    if not hasattr(pipe, "_original_encode_prompt"):
        pipe._original_encode_prompt = pipe._encode_prompt

    def embed_hook(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance,
                   negative_prompt=None, prompt_embeds=None, negative_prompt_embeds=None, **kwargs):
        if prompt_embeds is not None:
            state["prompt_embed"] = prompt_embeds
            state["original_embed"] = prompt_embeds.clone()
            return prompt_embeds
        prompt_embed = self._original_encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, **kwargs
        )
        state["prompt_embed"] = prompt_embed
        state["original_embed"] = prompt_embed.clone()
        return prompt_embed

    pipe._encode_prompt = MethodType(embed_hook, pipe)

    if not hasattr(pipe.unet, "_original_forward"):
        pipe.unet._original_forward = pipe.unet.forward

    def unet_forward_hook(self, sample, timestep, encoder_hidden_states=None, **kwargs):
        if state["prompt_embed"] is not None:
            encoder_hidden_states = state["prompt_embed"]
        return self._original_forward(sample, timestep, encoder_hidden_states=encoder_hidden_states, **kwargs)

    pipe.unet.forward = MethodType(unet_forward_hook, pipe.unet)

    if not hasattr(pipe.scheduler, "_original_step"):
        pipe.scheduler._original_step = pipe.scheduler.step

    def hooked_step(self, model_output, timestep, sample, **kwargs):
        t_tensor = torch.tensor([timestep], dtype=torch.float32, device=state["device"])
        latent = sample.detach().clone()

        if state["countdown"] > 0:
            state["countdown"] -= 1
        else:
            with torch.no_grad():
                logits, delta, pred_mask, alpha_pred = redirector(latent, state["prompt_embed"], t_tensor)
                pred = logits.argmax(dim=1).item()

            if pred == 1:
                delta_normalized = F.normalize(delta, dim=-1)
                norm_target = state["original_embed"].norm(dim=-1, keepdim=True)

                if mask_again:
                    pred_mask = pred_mask.unsqueeze(-1)
                    alpha_pred = alpha_pred if alpha_pred is not None else torch.ones_like(pred_mask)
                    delta_final = pred_mask * alpha_pred * delta_normalized * norm_target
                else:
                    delta_final = alpha_pred * delta_normalized * norm_target

                real_delta = alpha_scale * delta_final
                state["prompt_embed"] = state["original_embed"] + real_delta
                state["countdown"] = replace_steps

        return self._original_step(model_output, timestep, sample, **kwargs)

    pipe.scheduler.step = MethodType(hooked_step, pipe.scheduler)
