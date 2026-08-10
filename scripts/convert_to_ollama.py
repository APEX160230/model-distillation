"""GGUF 转换脚本 — HF 模型 → GGUF → Q4_K_M 量化 → Ollama 导入

使用 llama.cpp 的 convert_hf_to_gguf.py 进行转换。
需要在本机安装 llama.cpp (或通过 pip 安装 llama-cpp-python)。

用法：
    # 1. 克隆 llama.cpp
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp && make

    # 2. 运行转换
    python scripts/convert_to_ollama.py

产出：
    models/qwen25-15b-tcm-q4_k_m.gguf  — Q4_K_M 量化模型
    models/Modelfile                   — Ollama Modelfile
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MERGED_DIR = PROJECT_ROOT / "output_merged"
GGUF_DIR = PROJECT_ROOT / "models"
LLAMA_CPP_DIR = PROJECT_ROOT / "llama.cpp"

# GGUF 输出文件名
GGUF_F32 = "qwen25-15b-tcm-f32.gguf"
GGUF_Q4 = "qwen25-15b-tcm-q4_k_m.gguf"


def check_llama_cpp() -> bool:
    """检查 llama.cpp 是否已安装"""
    convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    return convert_script.exists()


def clone_llama_cpp():
    """克隆 llama.cpp"""
    if LLAMA_CPP_DIR.exists():
        print(f"llama.cpp 已存在: {LLAMA_CPP_DIR}")
        return

    print("克隆 llama.cpp...")
    subprocess.run(
        ["git", "clone", "https://github.com/ggerganov/llama.cpp", str(LLAMA_CPP_DIR)],
        check=True,
    )
    print("安装 Python 依赖...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r",
         str(LLAMA_CPP_DIR / "requirements.txt")],
        check=True,
    )


def convert_to_gguf():
    """将 HF 模型转换为 GGUF 格式 (f32)"""
    gguf_path = GGUF_DIR / GGUF_F32
    if gguf_path.exists():
        print(f"GGUF 文件已存在: {gguf_path}")
        return gguf_path

    convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        print(f"ERROR: convert_hf_to_gguf.py 不存在: {convert_script}")
        sys.exit(1)

    print(f"转换 HF → GGUF (f32)...")
    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable, str(convert_script),
            str(MERGED_DIR),
            "--outfile", str(gguf_path),
            "--outtype", "f32",
        ],
        check=True,
    )
    print(f"GGUF 文件: {gguf_path} ({gguf_path.stat().st_size / 1024**3:.1f}GB)")
    return gguf_path


def quantize_gguf(gguf_f32_path: Path):
    """量化为 Q4_K_M"""
    gguf_q4_path = GGUF_DIR / GGUF_Q4
    if gguf_q4_path.exists():
        print(f"量化文件已存在: {gguf_q4_path}")
        return gguf_q4_path

    # 查找 llama-quantize 可执行文件
    quantize_bin = None
    for name in ["llama-quantize", "quantize", "llama-quantize.exe"]:
        path = LLAMA_CPP_DIR / name
        if path.exists():
            quantize_bin = path
            break
        path = LLAMA_CPP_DIR / "build" / "bin" / name
        if path.exists():
            quantize_bin = path
            break

    if quantize_bin is None:
        print("WARNING: llama-quantize 未找到，跳过量化")
        print("你需要手动编译 llama.cpp 并运行:")
        print(f"  ./llama-quantize {gguf_f32_path} {gguf_q4_path} q4_k_m")
        return gguf_f32_path

    print(f"量化 Q4_K_M...")
    subprocess.run(
        [str(quantize_bin), str(gguf_f32_path), str(gguf_q4_path), "q4_k_m"],
        check=True,
    )
    print(f"量化文件: {gguf_q4_path} ({gguf_q4_path.stat().st_size / 1024**3:.1f}GB)")
    return gguf_q4_path


def create_modelfile(gguf_path: Path):
    """创建 Ollama Modelfile"""
    modelfile_path = GGUF_DIR / "Modelfile"
    content = f'''FROM {gguf_path.name}

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_predict 512
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"

SYSTEM """你是一位中医老师，擅长用通俗易懂的方式讲解中医经典知识。请用口语化的讲解风格，像老师讲课一样回答问题。引用经典原文时标注条文编号。解释方剂时列出完整组成。不提供具体诊疗建议。如果检索结果中没有相关信息，请如实说明。"""

TEMPLATE """{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ range .Messages }}}}<|im_start|>{{{{ .Role }}}}
{{{{ .Content }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
"""
'''
    modelfile_path.write_text(content, encoding="utf-8")
    print(f"Modelfile: {modelfile_path}")
    return modelfile_path


def import_to_ollama(gguf_path: Path, modelfile_path: Path):
    """导入到 Ollama"""
    model_name = "qwen25-15b-tcm"
    print(f"导入到 Ollama: {model_name}")
    subprocess.run(
        ["ollama", "create", model_name, "-f", str(modelfile_path)],
        check=True,
        cwd=str(GGUF_DIR),
    )
    print(f"Ollama 模型已创建: {model_name}")
    print("验证: ollama list")
    return model_name


def main():
    print("=" * 60)
    print("GGUF 转换: HF 模型 → GGUF → Q4_K_M → Ollama")
    print("=" * 60)

    if not MERGED_DIR.exists():
        print(f"ERROR: 合并模型不存在: {MERGED_DIR}")
        print("请先运行训练: python scripts/train_lora_cpu.py")
        sys.exit(1)

    # 1. 检查/安装 llama.cpp
    if not check_llama_cpp():
        print("\n[1/5] 安装 llama.cpp...")
        clone_llama_cpp()
    else:
        print(f"\n[1/5] llama.cpp 已安装: {LLAMA_CPP_DIR}")

    # 2. 转换为 GGUF
    print("\n[2/5] 转换 HF → GGUF...")
    gguf_f32 = convert_to_gguf()

    # 3. 量化
    print("\n[3/5] 量化 Q4_K_M...")
    gguf_q4 = quantize_gguf(gguf_f32)

    # 4. 创建 Modelfile
    print("\n[4/5] 创建 Modelfile...")
    modelfile = create_modelfile(gguf_q4)

    # 5. 导入 Ollama
    print("\n[5/5] 导入 Ollama...")
    try:
        model_name = import_to_ollama(gguf_q4, modelfile)
        print(f"\n完成！Ollama 模型: {model_name}")
        print(f"使用方法: ollama run {model_name}")
    except Exception as e:
        print(f"\nOllama 导入失败: {e}")
        print("你可以手动导入:")
        print(f"  cd {GGUF_DIR}")
        print(f"  ollama create qwen25-15b-tcm -f Modelfile")

    print("\n" + "=" * 60)
    print("全部完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
