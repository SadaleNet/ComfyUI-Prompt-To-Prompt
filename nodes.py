from .third_party.prompt_to_prompt import *

import torch
from torch import einsum
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
from comfy.ldm.modules.attention import optimized_attention, _ATTN_PRECISION
import functools
import re
import nodes
from comfy.ldm.modules.attention import SpatialTransformer
import folder_paths
import os
import shutil

def get_tokens_first_match(clip, prompt, search_phrase):
    if len(search_phrase) == 0:
        return []

    word_position = prompt.find(search_phrase)
    if word_position == -1:
        raise Exception(f"Local blend: The word '{search_phrase}' not found in the prompt.")

    token_position = 1 # +1 because of of the startoftext token
    word_start_positions = [0] + [i.span()[1] for i in list(re.finditer(r'[ ]+', prompt))] + [len(prompt)]
    for i in range(len(word_start_positions)-1):
        s = word_start_positions[i]
        e = word_start_positions[i+1]
        if s == word_position:
            break
        token_position += get_clip_token_length(clip, prompt[s:e])
    token_index = token_position
    token_length = get_clip_token_length(clip, search_phrase)

    return list(range(token_index, token_index+token_length))


# from comfy/ldm/modules/attention.py
# but modified to return attention scores as well as output
# also modified to accept optional sim replacement function and reweight maps
def attention_basic_with_sim(q, k, v, heads, sim_replace_func=None, reweight_map=None, mask=None):
    b, _, dim_head = q.shape
    dim_head //= heads
    scale = dim_head ** -0.5

    h = heads
    q, k, v = map(
        lambda t: t.unsqueeze(3)
        .reshape(b, -1, heads, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b * heads, -1, dim_head)
        .contiguous(),
        (q, k, v),
    )

    # force cast to fp32 to avoid overflowing
    if _ATTN_PRECISION =="fp32":
        sim = einsum('b i d, b j d -> b i j', q.float(), k.float()) * scale
    else:
        sim = einsum('b i d, b j d -> b i j', q, k) * scale

    del q, k

    if mask is not None:
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h=h)
        sim.masked_fill_(~mask, max_neg_value)

    # attention, what we cannot get enough of
    sim = sim.softmax(dim=-1)

    if sim_replace_func is not None:
        sim = sim_replace_func(sim)

    if reweight_map is not None:
        sim = sim * reweight_map[None, :]

    out = einsum('b i j, b j d -> b i d', sim.to(v.dtype), v)
    out = (
        out.unsqueeze(0)
        .reshape(b, heads, -1, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b, -1, heads * dim_head)
    )
    return (out, sim)

def search_transformer_blocks_inner(name, i, module):
    ret = []
    for a,layer in enumerate(module):
        if isinstance(layer, SpatialTransformer):
            for b,inner_layer in enumerate(layer.get_submodule('transformer_blocks')):
                ret.append((name, i, b,)) # a is always 1 and is useless for the purpose of identification of the layer
    return ret

def search_transformer_blocks(model):
    key_list = []
    for i, module in enumerate(model.model.get_submodule('diffusion_model.input_blocks')):
        key_list += search_transformer_blocks_inner("input", i, module)

    key_list += search_transformer_blocks_inner("middle", 0, model.model.get_submodule('diffusion_model.middle_block'))

    for i, module in enumerate(model.model.get_submodule('diffusion_model.output_blocks')):
        key_list += search_transformer_blocks_inner("output", i, module)
    return key_list

def attach_attention_functions(model, attn1_func, attn2_func):
    for i in search_transformer_blocks(model):
        model.set_model_attn1_replace(attn1_func, i[0], i[1], i[2])
        model.set_model_attn2_replace(attn2_func, i[0], i[1], i[2])

class Attn2SimStorage:
    def __init__(self):
        self.storage = {}
        self.storage_previous = {}
        self.storage_counter = {}
    def normalize(self, block, weight):
        self.storage[block] = self.storage[block] * weight / self.storage_counter[block]
        self.storage_counter[block] = weight
    def insert(self, block, sim):
        if block not in self.storage:
            self.storage[block] = sim
            self.storage_counter[block] = 1.0
        else:
            self.storage[block] += sim
            self.storage_counter[block] += 1.0
        self.storage_previous[block] = sim
    def get_average_attention(self, block):
        return (self.storage[block] / self.storage_counter[block]).cpu()
    def get_previous(self):
        return self.storage_previous

def process_attn1(q, k, v, extra_options, out_storage, out_replace):
    replacement_index = (*extra_options['block'], extra_options['block_index'])
    if out_replace is not None:
        out = out_replace[replacement_index]
    else:
        out = optimized_attention(q, k, v, heads=extra_options["n_heads"])
    if out_storage is not None:
        out_storage[replacement_index] = out
    return out

def process_attn2(q, k, v, extra_options, sim_storage, sim_storage_key_list, out_storage, sim_replace, replacement_mapper, reweight_map, refinement_sim):
    replacement_index = (*extra_options['block'], extra_options['block_index'])
    if sim_replace is not None:
        # FIXME: The extra_options["cond_or_uncond"] condition handling doesn't work here!
        sim_temp = torch.einsum('hpw,bwn->bhpn', sim_replace[replacement_index], replacement_mapper).sum(0)
        (out, sim) = attention_basic_with_sim(q, k, v, heads=extra_options["n_heads"], sim_replace_func=lambda x: sim_temp, reweight_map=reweight_map)
        if sim_storage is not None:
            if sim_storage_key_list is None:
                sim_storage.insert(replacement_index, sim)
                sim_storage.normalize(replacement_index, 1.0)
            elif replacement_index in sim_storage_key_list:
                sim_storage.insert(None, sim)
    elif refinement_sim is not None:
        # FIXME: The extra_options["cond_or_uncond"] condition handling doesn't work here!
        base = refinement_sim[0][replacement_index]
        mapper = refinement_sim[1]
        alpha = refinement_sim[2]
        (out, sim) = attention_basic_with_sim(q, k, v, heads=extra_options["n_heads"], sim_replace_func=lambda sim: (base[:, :, mapper].permute(2, 0, 1, 3) * alpha + sim * (1 - alpha)).sum(0), reweight_map=reweight_map)
    else:
        if 1 in extra_options["cond_or_uncond"]:
            (out, sim) = attention_basic_with_sim(q, k, v, heads=extra_options["n_heads"], reweight_map=reweight_map)
            if sim_storage is not None:
                if sim_storage_key_list is None:
                    sim_storage.insert(replacement_index, sim)
                elif replacement_index in sim_storage_key_list:
                    sim_storage.insert(None, sim)
        else:
            out = optimized_attention(q, k, v, heads=extra_options["n_heads"])
            sim = None
    if out_storage is not None:
        out_storage[replacement_index] = out
    return out

def dual_ksampler(parent, model, seed, steps, cfg, sampler_name, scheduler, positive_p2p, negative, latent, self_attention_step, cross_attention_step, local_blend_threshold, local_blend_layers, preview_enabled, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    positive = positive_p2p["positive"]
    positive2 = positive_p2p["positive2"]
    replacement_mapper = positive_p2p["replacement_mapper"].to(device="cuda:0")
    local_blend_tokens = positive_p2p["local_blend_tokens"]
    local_blend_tokens2 = positive_p2p["local_blend_tokens2"]
    refinement_mapper = positive_p2p["refinement_mapper"].to(device="cuda:0") if positive_p2p["refinement_mapper"] is not None else None
    refinement_alpha = positive_p2p["refinement_alpha"].to(device="cuda:0") if positive_p2p["refinement_alpha"] is not None else None

    reweight_map = positive_p2p["reweight_map"].to(device="cuda:0") if positive_p2p["reweight_map"] is not None else None

    latent_image = latent["samples"]
    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    samples = latent_image
    samples2 = latent_image.detach().clone()

    attention_storage_attn1 = {}
    original_attn2_sim_storage = Attn2SimStorage()
    replacement_attn2_sim_storage = Attn2SimStorage()

    local_blend_key_list = [(i.split(',')[0], int(i.split(',')[1]), int(i.split(',')[2])) for i in local_blend_layers.split(' ')]

    for step in range(steps-1):
        attach_attention_functions(model,
            functools.partial(process_attn1, out_storage=attention_storage_attn1, out_replace=None),
            functools.partial(process_attn2, sim_storage=original_attn2_sim_storage, sim_storage_key_list=None, out_storage=None, sim_replace=None, replacement_mapper=None, reweight_map=None, refinement_sim=None)
        )
        samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, samples,
                                  denoise=denoise, disable_noise=disable_noise, start_step=step, last_step=step+1,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=None, disable_pbar=disable_pbar, seed=seed)

        attach_attention_functions(model,
            functools.partial(process_attn1, out_storage=None, out_replace=attention_storage_attn1 if step < self_attention_step else None),
            functools.partial(process_attn2,
                sim_storage=replacement_attn2_sim_storage,
                sim_storage_key_list=local_blend_key_list,
                out_storage=None,
                sim_replace=original_attn2_sim_storage.get_previous() if step < cross_attention_step else None,
                replacement_mapper=replacement_mapper,
                reweight_map=None if step < cross_attention_step else reweight_map,
                refinement_sim=None if step < cross_attention_step or refinement_mapper is None else (original_attn2_sim_storage.get_previous(), refinement_mapper, refinement_alpha),
            )
        )

        samples2 = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive2, negative, samples2,
                                  denoise=denoise, disable_noise=disable_noise, start_step=step, last_step=step+1,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=None, disable_pbar=disable_pbar, seed=seed)

        if len(local_blend_tokens) > 0 or len(local_blend_tokens2) > 0:
            samples2 = local_blend(samples, samples2, original_attn2_sim_storage, replacement_attn2_sim_storage,
                    local_blend_key_list, local_blend_tokens, local_blend_tokens2, local_blend_threshold)

        # The preview image looks burnt. Not sure on why.
        if preview_enabled:
            callback(step, samples2.to('cuda:0'), None, steps)

        # Disable adding noise except for the first step
        disable_noise = True
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")

    out = latent.copy()
    out["samples"] = samples
    out2 = latent.copy()
    out2["samples"] = samples2

    return (out, out2, )

class CLIPTextEncodePromptToPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
        "mode": (["word swap", "refinement"], ),
        "text": ("STRING", {"multiline": True}),
        "text2": ("STRING", {"multiline": True}),
        "local_blend": ("STRING", {"multiline": True}),
        "local_blend2": ("STRING", {"multiline": True}),
        "reweight_words": ("STRING", {"multiline": True}),
        "reweight_value": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.1, "round": 0.01}),
        "clip": ("CLIP", )},
    }
    RETURN_TYPES = ("P2PCONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "conditioning"

    def encode(self, clip, mode, text, text2, local_blend, local_blend2, reweight_words, reweight_value):
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)

        refinement_mapper = None
        refinement_alpha = None
        if mode == "word swap":
            replacement_mapper = get_replacement_mapper(clip, text, text2)
        else:
            refinement_mapper, refinement_alpha = get_refinement_mapper(clip, text, text2)
            replacement_mapper = get_replacement_mapper(clip, text, text) # Uses direct mapping. #0 maps to #0, #1 maps to #1 and so on.

        reweight_tokens = get_tokens_first_match(clip, text2, reweight_words)
        reweight_map = get_equalizer(reweight_tokens, reweight_value) if len(reweight_tokens) > 0 else None

        tokens2 = clip.tokenize(text2)
        cond2, pooled2 = clip.encode_from_tokens(tokens2, return_pooled=True)

        local_blend_tokens = get_tokens_first_match(clip, text, local_blend)
        local_blend_tokens2 = get_tokens_first_match(clip, text2, local_blend2)

        return ({
            "positive": [[cond, {"pooled_output": pooled}]],
            "positive2": [[cond2, {"pooled_output": pooled2}]],
            "replacement_mapper": replacement_mapper,
            "local_blend_tokens": local_blend_tokens,
            "local_blend_tokens2": local_blend_tokens2,
            "reweight_map": reweight_map,
            "refinement_mapper": refinement_mapper,
            "refinement_alpha": refinement_alpha
        }, )


class LocalBlendLayerPresetPromptToPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
        "mode": (["sd1.5", "sdxl"], ),
        },
    }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "output"

    CATEGORY = "conditioning"

    def output(self, mode):
        if mode == "sd1.5":
            return ("input,7,0 input,8,0 output,3,0", )
        elif mode == "sdxl":
            return ("input,8,3 middle,0,3", )

class KSamplerPromptToPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive_p2p": ("P2PCONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "self_attention_step": ("INT", {"default": 4, "min": 0, "max": 10000}),
                    "cross_attention_step": ("INT", {"default": 10, "min": 0, "max": 10000}),
                    "local_blend_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.001}),
                    "local_blend_layers": ("STRING", {"multiline": True}),
                    "preview": (["enabled", "disabled"], ),
                    "latent_image": ("LATENT", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     }
                }

    RETURN_TYPES = ("LATENT","LATENT",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive_p2p, negative, self_attention_step, cross_attention_step, local_blend_threshold, local_blend_layers, latent_image, preview, denoise=1.0):
        return dual_ksampler(self, model, seed, steps, cfg, sampler_name, scheduler, positive_p2p, negative, latent_image, self_attention_step, cross_attention_step, local_blend_threshold, local_blend_layers, preview == "enabled", denoise=denoise)

class KSamplerPromptToPromptAttentionMapLogger:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "output_name": ("STRING", {"default": "out"}),
                     }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, output_name):
        output_prefix = os.path.join(self.output_dir, f"attn_{output_name.replace('/', '')}")
        if os.path.exists(output_prefix):
            shutil.rmtree(output_prefix)
        os.makedirs(output_prefix, exist_ok=True)

        attn2_storage = Attn2SimStorage()
        key_list = search_transformer_blocks(model)
        for i in key_list:
            model.set_model_attn2_replace(
                functools.partial(process_attn2, sim_storage=attn2_storage, sim_storage_key_list=None, out_storage=None, sim_replace=None, replacement_mapper=None, reweight_map=None, refinement_sim=None), i[0], i[1], i[2]
            )

        ret = nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)
        output_attn2_map(attn2_storage, latent_image["samples"].shape[2:], key_list, output_prefix)
        return ret
