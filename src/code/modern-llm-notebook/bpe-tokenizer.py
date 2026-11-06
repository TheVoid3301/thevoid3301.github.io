class BPETokenizer:
    """
    BPE Tokenizer 完整实现
    
    三个核心功能：
    1. train()  — 从语料学习 merge rules
    2. encode() — 文本 → token ID 列表
    3. decode() — token ID 列表 → 文本
    """
    
    def __init__(self):
        # merge_rules: 按顺序记录每次合并的 pair
        # 这是 BPE 最核心的数据结构，encode 时按这个顺序贪心应用
        self.merge_rules = []
        # 最终词表：初始字符 + 每一步 merge 学到的新 token
        self.vocab = set()
        self.stoi = {}
        self.itos = {}
    
    def train(self, texts, num_merges=15, verbose=True):
        """
        BPE 训练
        
        参数:
            texts: 语料列表
            num_merges: 合并次数
            verbose: 是否打印每一步
        """
        # Step 0: 初始化为字符级
        token_lists = [list(text) for text in texts]
        
        # 收集语料中所有不重复的字符，作为初始词表
        base_vocab = set()
        for tokens in token_lists:
            for c in tokens:
                base_vocab.add(c)
        learned_vocab = set(base_vocab)
        
        if verbose:
            print(f"{'='*60}")
            print(f"BPE 训练开始！初始状态: 每条句子按字符拆分")
            print(f"{'='*60}")
            print(f"初始字符集大小: {len(base_vocab)}")
            print()
        
        for step in range(num_merges):
            # Step 1: 统计所有 pair 的频率
            pairs = {}
            for tokens in token_lists:
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    if pair not in pairs:
                        pairs[pair] = 0
                    pairs[pair] += 1
            
            if not pairs:
                break
            
            # Step 2: 找最高频 pair
            best_pair = max(pairs, key=pairs.get)
            best_count = pairs[best_pair]
            
            # Step 3: 记录 merge rule
            self.merge_rules.append(best_pair)
            
            # Step 4: 合并
            a, b = best_pair
            new_token = a + b
            
            new_token_lists = []
            for tokens in token_lists:
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                        new_tokens.append(new_token)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                new_token_lists.append(new_tokens)
            
            token_lists = new_token_lists
            
            learned_vocab.add(new_token)
            
            if verbose:
                print(f"Step {step+1:2d}: merge {best_pair} → '{new_token}'  (出现 {best_count} 次) | 当前词表大小: {len(learned_vocab)}")
        
        # 建立最终词表
        # 注意：词表不能只保留最终语料里还出现的 token。
        # encode 遇到没见过的组合时，需要能退回到初始字符。
        self.vocab = learned_vocab
        sorted_vocab = sorted(list(self.vocab))
        self.stoi = {t: i for i, t in enumerate(sorted_vocab)}
        self.itos = {i: t for i, t in enumerate(sorted_vocab)}
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"训练完成！")
            print(f"  - 合并次数: {len(self.merge_rules)}")
            print(f"  - 最终词表大小: {len(self.vocab)} 个 token")
            print(f"  - Merge rules: {self.merge_rules}")
            print(f"{'='*60}")
        
        return token_lists

    # 给 BPETokenizer 加上 encode 方法
    def bpe_encode(self, text):
        """
        BPE 编码：文本 → token ID 列表
        
        核心逻辑：按训练时的 merge rules 顺序，贪心地合并
        """
        # Step 1: 拆成字符
        tokens = list(text)
        
        # Step 2: 依次应用每条 merge rule
        for (a, b) in self.merge_rules:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(a + b)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        # Step 3: 每个 token 去词表查 ID。
        # 如果找不到，说明输入里有训练时没见过的字符；不能静默跳过。
        ids = []
        for token in tokens:
            if token not in self.stoi:
                raise KeyError(token)
            ids.append(self.stoi[token])
        return ids