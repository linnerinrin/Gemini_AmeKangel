from transformers import MT5ForConditionalGeneration, T5Tokenizer

model_name = "models/mt5-small"  # 或本地路径
model = MT5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# 关键：加上 paraphrase 前缀
input_text = "paraphrase: 这家餐厅的味道非常好，服务也很周到。"

inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)

outputs = model.generate(
    **inputs,
    max_length=128,
    num_return_sequences=3,    # 生成3个不同版本
    do_sample=True,            # 启用采样
    temperature=0.8,           # 控制创意度
    top_p=0.9                  # 核采样
)

results = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
for i, r in enumerate(results):
    print(f"{i+1}. {r}")