# hf_tokenizer.py
from typing import List, Union, Optional, Tuple, Dict
from transformers import PreTrainedTokenizer
from .tokenizers import ProformaTokenizer

class HFProformaTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        proforma_tokenizer: ProformaTokenizer,
        **kwargs
    ):
        self.backend = proforma_tokenizer
        vocab = self.backend.get_vocab()
        self.vocab = {tok: i for i, tok in enumerate(vocab)}
        self.ids_to_tokens = {i: tok for tok, i in self.vocab.items()}

        super().__init__(
            pad_token="[PAD]",
            unk_token="[UNK]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
            **kwargs
        )

    @classmethod
    def from_proforma_tokenizer(cls, tokenizer: ProformaTokenizer) -> "HFProformaTokenizer":
        return cls(tokenizer)

    def _tokenize(self, text: str) -> List[str]:
        return self.backend.tokenize(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return " ".join(tokens)

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab

    def build_inputs_with_special_tokens(self, token_ids: List[int]) -> List[int]:
        return [self.vocab[self.cls_token]] + token_ids + [self.vocab[self.sep_token]]

    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        import os, json
        name = "vocab.json" if filename_prefix is None else f"{filename_prefix}-vocab.json"
        path = os.path.join(save_directory, name)
        with open(path, "w") as f:
            json.dump(self.vocab, f, indent=2)
        return (path,)

    def __call__(
        self,
        text: Union[str, List[str]],
        return_attention_mask: bool = True,
        padding: Union[bool, str] = "longest",
        truncation: bool = False,
        return_tensors: Optional[str] = None,
        **kwargs
    ):
        is_batched = isinstance(text, list)

        if not is_batched:
            tokens = self.backend.tokenize(text)
            ids = self.backend.encode(tokens)
            mask = [1] * len(ids)
            return {
                "input_ids": [ids],
                "attention_mask": [mask] if return_attention_mask else None,
                "tokens": [tokens],
            }

        token_batches = self.backend.tokenize_batch(text)
        ids, masks = self.backend.encode_batch(token_batches, pad=(padding is not False))
        return {
            "input_ids": ids,
            "attention_mask": masks if return_attention_mask else None,
            "tokens": token_batches,
        }
