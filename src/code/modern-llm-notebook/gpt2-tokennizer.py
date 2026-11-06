# 加载真实 GPT-2 tokenizer（byte-level BPE）
try:
    import tiktoken

    real_tokenizer_name = "gpt2"
    real_tokenizer = tiktoken.get_encoding(real_tokenizer_name)
    print(f"真实 tokenizer: {real_tokenizer_name}")
    print(f"词表大小: {real_tokenizer.n_vocab}")
except Exception as e:
    real_tokenizer = None
    print("未能加载 tiktoken 的 GPT-2 tokenizer")
    print(f"原因: {e}")
    print("安装并缓存 tiktoken 后，这个 cell 会打印真实 tokenizer 结果。")


def show_real_tokenization(text):
    """打印真实 tokenizer 如何把文本切成 token 和 ID"""
    ids = real_tokenizer.encode(text)
    tokens = [real_tokenizer.decode([tok_id]) for tok_id in ids]

    print(f"文本: {text!r}")
    print(f"token 数: {len(ids)}")
    for i, (tok_id, token) in enumerate(zip(ids, tokens)):
        visible = token.replace("\n", "\\n").replace("\t", "\\t")
        token_bytes = list(real_tokenizer.decode_single_token_bytes(tok_id))
        print(f"  {i:02d} | id={tok_id:<6} | token={visible!r} | bytes={token_bytes}")
    return ids, tokens


if real_tokenizer is not None:
    show_real_tokenization("the cat sat on the mat")
