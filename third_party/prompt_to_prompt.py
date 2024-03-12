import torch
import os
import torch.nn.functional as nnf
from PIL import Image
import numpy
import math

def get_clip_token_length(clip, prompt):
    return clip.tokenize(prompt)['l'][0].index((49407, 1.0))-1 # -1 for removal of endoftext tokens


def split_prompt_by_tokens(clip, prompt):
    ret = []
    for i in clip.tokenizer.untokenize(clip.tokenize(prompt)["l"][0]):
        value = i[1]
        if value.endswith('</w>'):
            value = value[:-4]
        ret.append(value)
        if value == '<|endoftext|>':
            break
    return ret

def get_word_inds(text: str, word_place: int, clip):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = split_prompt_by_tokens(clip, text)[1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return torch.tensor(out)


def get_replacement_mapper_(x: str, y: str, clip, max_len=77):
    words_x = x.split(' ')
    words_y = y.split(' ')
    if len(words_x) != len(words_y):
        raise ValueError(f"attention replacement edit can only be applied on prompts with the same length"
                         f" but prompt A has {len(words_x)} words and prompt B has {len(words_y)} words.")
    inds_replace = [i for i in range(len(words_y)) if words_y[i] != words_x[i]]
    inds_source = [get_word_inds(x, i, clip) for i in inds_replace]
    inds_target = [get_word_inds(y, i, clip) for i in inds_replace]
    mapper = torch.zeros((max_len, max_len))
    i = j = 0
    cur_inds = 0
    while i < max_len and j < max_len:
        if cur_inds < len(inds_source) and inds_source[cur_inds][0] == i:
            inds_source_, inds_target_ = inds_source[cur_inds], inds_target[cur_inds]
            if len(inds_source_) == len(inds_target_):
                mapper[inds_source_, inds_target_] = 1
            else:
                ratio = 1 / len(inds_target_)
                for i_t in inds_target_:
                    mapper[inds_source_, i_t] = ratio
            cur_inds += 1
            i += len(inds_source_)
            j += len(inds_target_)
        elif cur_inds < len(inds_source):
            mapper[i, j] = 1
            i += 1
            j += 1
        else:
            mapper[j, j] = 1
            i += 1
            j += 1

    return mapper.float()

def get_replacement_mapper(clip, prompt, prompt2, max_len=77):
    mappers = [get_replacement_mapper_(prompt, prompt2, clip, max_len)]
    return torch.stack(mappers)

def get_refinement_mapper(clip, prompt, prompt2):
    max_len = 77
    mapper = torch.zeros(1, max_len, dtype=torch.int64)
    alpha = torch.ones(1, max_len)
    token = clip.tokenize(prompt)['l'][0]
    token2 = clip.tokenize(prompt2)['l'][0]

    for i in range(max_len):
        mapper[0, i] = i

    i = 0
    j = 0
    while j < max_len:
        if token[i] == token2[j]:
            mapper[0, j] = i
            alpha[0, j] = 1.0
            i += 1
            j += 1
        else:
            mapper[0, j] = -1
            alpha[0, j] = 0.0
            j += 1
        if token2[j][0] == 49407: # EOS condition
            mapper[0, j] = i
            alpha[0, j] = 1.0
            break
    return mapper, alpha


def get_equalizer(tokens, value):
    values = (value, )
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)
    for ind in tokens:
        equalizer[:, ind] = values
    return equalizer


def get_dimensions_by_aspect_ratio(latent_size, tensor_dimension):
    ret = list(latent_size)
    while ret[0] * ret[1] != tensor_dimension and (ret[0] != 1 or ret[1] != 1):
        ret[0] = math.ceil(ret[0]/2)
        ret[1] = math.ceil(ret[1]/2)
    return tuple(ret)

def obtain_mask(sim, key_list, latent_size, alpha_layers, num_words, threshold):
    k = 1

    height, width = get_dimensions_by_aspect_ratio(latent_size, sim.get_average_attention(key_list[0]).shape[1])
    maps = torch.zeros(sim.get_average_attention(key_list[0]).shape[0], height, width, num_words)
    for key in key_list:
        item = sim.get_average_attention(key)
        maps += item.reshape(-1, height, width, item.shape[-1])
    maps /= len(key_list)
    maps = [item.reshape(alpha_layers.shape[0], -1, 1, height, width, num_words) for item in maps]
    maps = torch.cat(maps, dim=1)
    maps = (maps * alpha_layers).sum(-1).mean(1)
    mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
    mask = nnf.interpolate(mask, size=(latent_size))
    mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
    return mask.ge(threshold)

def local_blend(x_original, x_replace, sim_original, sim_replace, key_list, token_list_original, token_list_replace, threshold):
    MAX_NUM_WORDS = 77
    alpha_layers = torch.zeros(1, 1, 1, 1, 1, MAX_NUM_WORDS)
    for i in token_list_original:
        alpha_layers[:, :, :, :, :, i] = 1

    alpha_layers2 = torch.zeros(1, 1, 1, 1, 1, MAX_NUM_WORDS)
    for i in token_list_replace:
        alpha_layers2[:, :, :, :, :, i] = 1

    mask = torch.logical_or(
        obtain_mask(sim_original, key_list, x_original.shape[2:], alpha_layers, MAX_NUM_WORDS, threshold),
        obtain_mask(sim_replace, [None], x_replace.shape[2:], alpha_layers2, MAX_NUM_WORDS, threshold)
    ).float()

    return (1.0-mask) * x_original + mask * x_replace

def output_attn2_map(attn2_storage, dimensions, key_list, output_folder):
	cross_map = {}
	for key in key_list:
		item = attn2_storage.get_average_attention(key)
		height, width = get_dimensions_by_aspect_ratio(dimensions, item.shape[1])
		cross_map[key] = item.reshape(-1, height, width, item.shape[-1])
		cross_map[key] = cross_map[key].sum(0) / cross_map[key].shape[0]

	for i in range(77):
		image_max_for_the_token = float('-inf')
		for key in key_list:
			image_max_for_the_token = max(image_max_for_the_token, cross_map[key][:, :, i].max())
		for key in key_list:
			image = 255 * cross_map[key][:, :, i] / image_max_for_the_token
			image = image.unsqueeze(-1).expand(*image.shape, 3)
			image = image.numpy().astype(numpy.uint8)
			Image.fromarray(image).save(os.path.join(output_folder, f"{i}_{key[0]}_{key[1]}_{key[2]}.png"))
