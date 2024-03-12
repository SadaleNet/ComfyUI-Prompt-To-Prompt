from .nodes import *

NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodePromptToPrompt": CLIPTextEncodePromptToPrompt,
    "KSamplerPromptToPrompt": KSamplerPromptToPrompt,
    "LocalBlendLayerPresetPromptToPrompt": LocalBlendLayerPresetPromptToPrompt,
    "KSamplerPromptToPromptAttentionMapLogger": KSamplerPromptToPromptAttentionMapLogger,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTextEncodePromptToPrompt": "CLIPTextEncodeP2P",
    "KSamplerPromptToPrompt": "KSamplerP2P",
    "LocalBlendLayerPresetPromptToPrompt": "LocalBlendPresets",
    "KSamplerPromptToPromptAttentionMapLogger": "KSamplerAttn2Log"
}
