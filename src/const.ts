export const modelNames = [
    "onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX", // fp16
    "onnx-community/Llama-3.2-3B-Instruct-onnx-web", // q4f16
    "onnx-community/Phi-3.5-mini-instruct-onnx-web", //q4f16
    "Xenova/Phi-3-mini-4k-instruct_fp16", // q4
    "Xenova/Phi-3-mini-4k-instruct", // q4
    "schmuell/DeepSeek-R1-Distill-Qwen-1.5B-onnx", // q4f16
] as const;

export const dtypes = [
    "auto", // Auto-detect based on environment
    "fp32",
    "fp16",
    "q8",
    "int8",
    "uint8",
    "q4",
    "bnb4",
    "q4f16", // fp16 model with int4 block weight quantization
] as const;

export const devices = [
    "auto", // Auto-detect based on device and environment
    "gpu", // Auto-detect GPU
    "cpu", // CPU
    "wasm", // WebAssembly
    "webgpu", // WebGPU
    "cuda", // CUDA
    "dml", // DirectML
    "webnn", // WebNN (default)
    "webnn-npu", // WebNN NPU
    "webnn-gpu", // WebNN GPU
    "webnn-cpu", // WebNN CPU
] as const;