"""GGUF 转换 + Q4_K_M 量化脚本

在 Google Colab 上训练完成后运行此脚本：
1. 克隆 llama.cpp
2. 将合并后的 HF 模型转为 GGUF
3. 量化为 Q4_K_M
4. 生成 Ollama Modelfile

产出：
- qwen25-15b-tcm-q4_k_m.gguf  (约 1GB, 可下载)
- Modelfile (用于 ollama create)
"""
import os
import subprocess
import sys
from pathlib import Path

# ==================== 配置 ====================
MERGED_DIR = "./output_merged"
GGUF_OUTPUT = "./qwen25-15b-tcm-f16.gguf"
QUANT_OUTPUT = "./qwen25-15b-tcm-q4_k_m.gguf"
LLAMA_CPP_DIR = "./llama.cpp"
QUANT_TYPE = "q4_k_m"

# Ollama Modelfile 模板
MODELFILE_TEMPLATE = '''FROM ./qwen25-15b-tcm-q4_k_m.gguf

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""

SYSTEM """你是一位中医老师，擅长用通俗易懂的方式讲解中医经典知识。请用口语化的讲解风格，像老师讲课一样回答问题。引用经典原文时标注条文编号。解释方剂时列出完整组成。不提供具体诊疗建议。如果检索结果中没有相关信息，请如实说明。"""

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.8
PARAMETER top_k 20
'''


def run(cmd: str, check: bool = True):
    """运行 shell 命令"""
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout[:2000])
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[:2000]}")
        if check:
            sys.exit(1)
    return result


def main():
    print("=" * 60)
    print("GGUF 转换 + Q4_K_M 量化")
    print("=" * 60)

    # 1. 检查合并模型是否存在
    if not os.path.exists(MERGED_DIR):
        print(f"错误: 合并模型目录不存在: {MERGED_DIR}")
        print("请先运行 train_lora.py 完成训练和合并。")
        sys.exit(1)

    # 2. 克隆 llama.cpp
    print("\n[1/4] 克隆 llama.cpp...")
    if not os.path.exists(LLAMA_CPP_DIR):
        run(f"git clone https://github.com/ggerganov/llama.cpp.git {LLAMA_CPP_DIR}")
    else:
        print("llama.cpp 已存在，跳过克隆")

    # 3. 安装依赖
    print("\n[2/4] 安装转换依赖...")
    run(f"pip install -r {LLAMA_CPP_DIR}/requirements/requirements-convert_hf_to_gguf.txt")

    # 4. 转换为 GGUF (F16)
    print("\n[3/4] 转换为 GGUF (F16)...")
    convert_script = f"{LLAMA_CPP_DIR}/convert_hf_to_gguf.py"
    run(f"python {convert_script} {MERGED_DIR} --outfile {GGUF_OUTPUT} --outtype f16")

    if not os.path.exists(GGUF_OUTPUT):
        print(f"错误: GGUF 文件未生成: {GGUF_OUTPUT}")
        sys.exit(1)

    f16_size = os.path.getsize(GGUF_OUTPUT) / 1024**3
    print(f"F16 GGUF 大小: {f16_size:.2f} GB")

    # 5. 量化为 Q4_K_M
    print(f"\n[4/4] 量化为 {QUANT_TYPE}...")

    # 编译 llama-quantize
    print("编译 llama-quantize...")
    run(f"cd {LLAMA_CPP_DIR} && make llama-quantize")

    # 运行量化
    quant_bin = f"{LLAMA_CPP_DIR}/llama-quantize"
    run(f"{quant_bin} {GGUF_OUTPUT} {QUANT_OUTPUT} {QUANT_TYPE}")

    if not os.path.exists(QUANT_OUTPUT):
        print(f"错误: 量化文件未生成: {QUANT_OUTPUT}")
        sys.exit(1)

    q4_size = os.path.getsize(QUANT_OUTPUT) / 1024**3
    print(f"Q4_K_M GGUF 大小: {q4_size:.2f} GB")

    # 6. 生成 Modelfile
    modelfile_path = "./Modelfile"
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(MODELFILE_TEMPLATE)
    print(f"\nModelfile 已生成: {modelfile_path}")

    # 7. 汇总
    print("\n" + "=" * 60)
    print("转换完成！")
    print(f"  GGUF (Q4_K_M): {QUANT_OUTPUT} ({q4_size:.2f} GB)")
    print(f"  Modelfile: {modelfile_path}")
    print("=" * 60)
    print("\n下一步操作（在本地机器上）：")
    print(f"  1. 下载 {QUANT_OUTPUT} 和 {modelfile_path}")
    print(f"  2. 将两个文件放在同一目录")
    print(f"  3. 运行: ollama create qwen25-15b-tcm -f Modelfile")
    print(f"  4. 验证: ollama run qwen25-15b-tcm '什么是太阳病'")
    print(f"  5. 更新 RAG 管道模型名: RAGPipeline(model='qwen25-15b-tcm')")


if __name__ == "__main__":
    main()
