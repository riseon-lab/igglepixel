import diffusers
import inspect

kwargs = inspect.signature(diffusers.loaders.lora_pipeline.LoraLoaderMixin.load_lora_weights).parameters
print(kwargs.keys())
