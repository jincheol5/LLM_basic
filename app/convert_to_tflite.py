import os
from litert_torch.generative.examples.gemma3 import gemma3
from litert_torch.generative.utilities import converter
from litert_torch.generative.utilities.export_config import ExportConfig
from litert_torch.generative.layers import kv_cache
from modules import DataUtils

model_path=os.path.join(DataUtils.get_base_path(),"gemma-270m-it")
pytorch_model=gemma3.build_model_270m(model_path)
export_config=ExportConfig()
export_config.kvcache_layout=kv_cache.KV_LAYOUT_TRANSPOSED
export_config.mask_as_input=True

output_path=os.path.join("..","data","tflite","gemma-270m-it")
converter.convert_to_tflite(
    pytorch_model,
    output_path=output_path,
    output_name_prefix="gemma-270m-it",
    prefill_seq_len=2048,
    kv_cache_max_len=4096,
    quantize="dynamic_int8",
    export_config=export_config,
)