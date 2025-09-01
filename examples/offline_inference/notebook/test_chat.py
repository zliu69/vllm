import torch
from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser
from vllm.sampling_params import SamplingParams
import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
os.environ["CUDA_HOME"] = "/usr/local/cuda"


MODEL_PATH = "/sharedata/zimoliu/ckpts/jamba_60B_128k_v6_1node_pp8_ep1_official_ckpt38000/hf"
SAMPLING_KWARGS = dict(
    max_tokens=128,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
)



# ========== 2. 加载模型 ==========
print(">>> 正在加载模型，请稍候...")
llm = LLM(
    model=MODEL_PATH,
    # pipeline_parallel_size=8,
    # 如有其它 EngineArgs，可在此追加，例如：
    # tensor_parallel_size=1,
    # gpu_memory_utilization=0.8,
    # 关键：关闭 flashinfer，用原生 top-k/top-p
    # enforce_eager=True,
    # 或者
    # disable_flashinfer=True,
)
sampling_params = SamplingParams(**SAMPLING_KWARGS)
print(">>> 模型已就绪，输入 q 退出对话 <<<")




def chat_loop():
    while True:
        try:
            user_input = input("").strip()          # 去掉提示符
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break
        if user_input.lower() == "q":
            print("再见！")
            break
        if not user_input:
            continue

        # 打印用户输入
        print(f"\n你：{user_input}")

        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input},
        ]
        outputs = llm.generate(prompts=user_input, sampling_params=sampling_params)
        # outputs = llm.chat([conversation], sampling_params, use_tqdm=False)
        assistant_reply = outputs[0].outputs[0].text.strip()
        print(f"助手：{assistant_reply}")



# 运行对话
chat_loop()
del llm
torch.cuda.empty_cache()
torch.cuda.synchronize()