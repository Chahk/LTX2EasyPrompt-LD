"""
LTX2 Inference Server Configuration Node

This node provides a reusable configuration for OpenAI-compatible inference servers.
Wire it to LTX2PromptArchitect and/or LTX2VisionDescribe nodes to use remote
inference instead of local transformer models.

Benefits:
- Single source of truth for server settings
- Reusable across multiple nodes - same config works for both text and vision
- Clean separation: connected = server mode, disconnected = local mode
- Simple: Just 3 parameters (url, model_name, api_key)
"""


class LTX2InferenceServerConfig:
    """
    Configuration node for OpenAI-compatible inference servers.

    Outputs a configuration dict that can be wired to LTX2PromptArchitect
    and LTX2VisionDescribe nodes to enable inference server mode.

    The same model_name is used intelligently:
    - When connected to LTX2PromptArchitect: used as text generation model
    - When connected to LTX2VisionDescribe: used as vision model
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {
                    "default": "http://localhost:8000/v1",
                    "multiline": False,
                    "placeholder": "e.g. http://192.168.1.100:8000/v1",
                    "tooltip": "OpenAI-compatible API endpoint URL. Include /v1 at the end for most servers (vLLM, text-generation-webui, LocalAI, etc.)"
                }),
                "model_name": ("STRING", {
                    "default": "mlabonne/NeuralDaredevil-8B-abliterated",
                    "multiline": False,
                    "placeholder": "Model name on server",
                    "tooltip": "The model identifier on the inference server. Used for both text generation (when connected to Prompt node) and vision (when connected to Vision node)."
                }),
                "api_key": ("STRING", {
                    "default": "not-needed",
                    "multiline": False,
                    "placeholder": "API key (usually not needed for local servers)",
                    "tooltip": "API key for the inference server. Most local servers don't require authentication - leave as 'not-needed'."
                }),
            },
        }

    RETURN_TYPES = ("SERVER_CONFIG",)
    RETURN_NAMES = ("server_config",)
    FUNCTION = "create_config"
    CATEGORY = "LTX2"
    OUTPUT_NODE = False

    def create_config(self, url, model_name, api_key):
        """
        Creates a configuration dictionary for inference server settings.

        This dict is passed to other nodes to enable inference server mode.
        The same model_name is interpreted contextually by the receiving node.
        """
        config = {
            "url": url.strip(),
            "model_name": model_name.strip(),
            "api_key": api_key.strip(),
        }

        print(f"[LTX2 ServerConfig] Created config:")
        print(f"[LTX2 ServerConfig]   URL: {config['url']}")
        print(f"[LTX2 ServerConfig]   Model: {config['model_name']}")

        return (config,)


# ── ComfyUI Registration ──────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "LTX2InferenceServerConfig": LTX2InferenceServerConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX2InferenceServerConfig": "LTX2 Inference Server Config",
}
