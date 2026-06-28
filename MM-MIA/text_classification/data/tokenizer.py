"""tokenizer.py — BPE tokenizer builder (HuggingFace tokenizers)."""

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders

SP = {"pad": "[PAD]", "unk": "[UNK]", "cls": "[CLS]", "sep": "[SEP]", "mask": "[MASK]"}


def build_tokenizer(texts, vocab_size=8192, max_len=256):
    tok = Tokenizer(models.BPE(unk_token=SP["unk"]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.train_from_iterator(texts, trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=list(SP.values()), min_frequency=2))
    cid, sid = tok.token_to_id(SP["cls"]), tok.token_to_id(SP["sep"])
    tok.post_processor = processors.TemplateProcessing(
        single=f"{SP['cls']} $A {SP['sep']}", pair=f"{SP['cls']} $A {SP['sep']} $B {SP['sep']}",
        special_tokens=[(SP["cls"], cid), (SP["sep"], sid)])
    tok.decoder = decoders.ByteLevel()
    tok.enable_padding(pad_id=tok.token_to_id(SP["pad"]), pad_token=SP["pad"])
    tok.enable_truncation(max_length=max_len)
    return tok
