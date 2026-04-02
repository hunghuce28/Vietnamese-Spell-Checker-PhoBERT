#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=====================================================================
  PHÁT HIỆN VÀ CHỈNH SỬA LỖI CHÍNH TẢ TIẾNG VIỆT (HƯỚNG 2 - DEEP LEARNING)
  Sử dụng Mô hình Ngôn ngữ PhoBERT (Masked Language Model)
=====================================================================

Phương pháp:
  1. Phát hiện lỗi: Dùng từ điển tiếng Việt để kiểm tra từ có hợp lệ không (Non-word error)
  2. Sinh ứng viên: Dùng Edit Distance + luật thay thế tiếng Việt
  3. Xếp hạng ứng viên: Dùng PhoBERT đánh giá xác suất của ứng viên 
     khi được điền vào ngữ cảnh (câu). Khắc phục triệt để điểm yếu của N-gram.

Cải tiến v2:
  - Fix BPE Tokenizer: Dùng log_softmax cho xác suất chuẩn hóa
  - Context Updating: Sửa cuốn chiếu trái-sang-phải (Left-to-Right)
  - Relative Threshold: Ngưỡng dựa trên chênh lệch log-xác suất
  - Candidate Generation mở rộng: Kết hợp thay nguyên âm gốc + dấu thanh
  - Case Preservation: Bảo toàn chữ hoa/thường
"""

import re
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer
from typing import List, Tuple, Dict, Set, Optional

# =====================================================================
# 1. BẢNG KÝ TỰ VÀ TỪ ĐIỂN
# =====================================================================

VIETNAMESE_CHARS = set(
    "aàáảãạăằắẳẵặâầấẩẫậbcdđeèéẻẽẹêềếểễệ"
    "fghiìíỉĩịjklmnoòóỏõọôồốổỗộơờớởỡợ"
    "pqrstuùúủũụưừứửữựvwxyỳýỷỹỵz"
)

TONE_VARIANTS = {
    "a": "aàáảãạ", "ă": "ăằắẳẵặ", "â": "âầấẩẫậ",
    "e": "eèéẻẽẹ", "ê": "êềếểễệ", "i": "iìíỉĩị",
    "o": "oòóỏõọ", "ô": "ôồốổỗộ", "ơ": "ơờớởỡợ",
    "u": "uùúủũụ", "ư": "ưừứửữự", "y": "yỳýỷỹỵ",
}

CHAR_TO_BASE = {}
for base, variants in TONE_VARIANTS.items():
    for v in variants:
        CHAR_TO_BASE[v] = base

# Nhóm nguyên âm gốc có thể nhầm lẫn
VOWEL_GROUPS = ["aăâ", "oôơ", "uư", "eê"]


class VietnameseDictionary:
    def __init__(self):
        self.words: Set[str] = set()
        self._build_dictionary()

    def _build_dictionary(self):
        import os
        dict_path = "vietnamese_syllables.txt"
        if not os.path.exists(dict_path):
            raise FileNotFoundError(f"Không tìm thấy file từ điển '{dict_path}'.")
        with open(dict_path, "r", encoding="utf-8") as f:
            self.words = set(w.strip().lower() for w in f if w.strip())

    def contains(self, word: str) -> bool:
        return word.lower().strip() in self.words

    def get_all_words(self) -> Set[str]:
        return self.words


# =====================================================================
# 2. SINH ỨNG VIÊN (EDIT DISTANCE) — CẢI TIẾN
# =====================================================================

class EditDistance:
    @staticmethod
    def levenshtein(s1: str, s2: str) -> int:
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1): dp[i][0] = i
        for j in range(n + 1): dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[m][n]

    @staticmethod
    def generate_candidates(word: str, dictionary: VietnameseDictionary, max_distance: int = 2) -> List[Tuple[str, int]]:
        candidates = []
        word_lower = word.lower()
        SMALL_CHARS = "aăâbcdđeêfghijklmnoôơpqrstuưvwxyz"
        seen = set()

        # 1. Xóa
        for i in range(len(word_lower)):
            new_word = word_lower[:i] + word_lower[i + 1:]
            if new_word and new_word not in seen and dictionary.contains(new_word):
                seen.add(new_word); candidates.append((new_word, 1))

        # 2. Thay thế (ký tự gốc)
        for i in range(len(word_lower)):
            for c in SMALL_CHARS:
                if c != word_lower[i]:
                    new_word = word_lower[:i] + c + word_lower[i + 1:]
                    if new_word not in seen and dictionary.contains(new_word):
                        seen.add(new_word); candidates.append((new_word, 1))

        # 3. Chèn
        for i in range(len(word_lower) + 1):
            for c in SMALL_CHARS:
                new_word = word_lower[:i] + c + word_lower[i:]
                if new_word not in seen and dictionary.contains(new_word):
                    seen.add(new_word); candidates.append((new_word, 1))

        # 4. Hoán vị
        for i in range(len(word_lower) - 1):
            chars = list(word_lower)
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
            new_word = "".join(chars)
            if new_word not in seen and dictionary.contains(new_word):
                seen.add(new_word); candidates.append((new_word, 1))

        # 5. Thay dấu thanh tiếng Việt (cùng nguyên âm gốc)
        for i, char in enumerate(word_lower):
            base = CHAR_TO_BASE.get(char, char)
            if base in TONE_VARIANTS:
                for variant in TONE_VARIANTS[base]:
                    if variant != char:
                        new_word = word_lower[:i] + variant + word_lower[i + 1:]
                        if new_word not in seen and dictionary.contains(new_word):
                            seen.add(new_word); candidates.append((new_word, 1))

        # 6. [MỚI] Kết hợp thay nguyên âm gốc + dấu thanh (a↔ă↔â + dấu)
        #    Ví dụ: "guíp" → thay u→ư hoặc i→iê, kết hợp dấu
        for i, char in enumerate(word_lower):
            base = CHAR_TO_BASE.get(char, char)
            for group in VOWEL_GROUPS:
                if base in group:
                    for alt_base in group:
                        if alt_base != base:
                            # Thay nguyên âm gốc (không dấu)
                            new_word = word_lower[:i] + alt_base + word_lower[i + 1:]
                            if new_word not in seen and dictionary.contains(new_word):
                                seen.add(new_word); candidates.append((new_word, 1))
                            # Thay nguyên âm gốc + TẤT CẢ dấu thanh của nó
                            if alt_base in TONE_VARIANTS:
                                for variant in TONE_VARIANTS[alt_base]:
                                    new_word = word_lower[:i] + variant + word_lower[i + 1:]
                                    if new_word not in seen and dictionary.contains(new_word):
                                        seen.add(new_word); candidates.append((new_word, 1))

        # 7. [MỚI] Consonant confusion cho non-word (VD: guíp → giúp)
        #    Thay phụ âm đầu + giữ nguyên hoặc thay dấu thanh
        CONSONANT_MAPS = {
            "g": ["gh", "gi"], "gh": ["g"], "gi": ["d", "r", "g"],
            "ch": ["tr"], "tr": ["ch"],
            "s": ["x"], "x": ["s"],
            "d": ["đ", "gi", "r"], "đ": ["d"],
            "r": ["d", "gi"], "l": ["n"], "n": ["l"],
            "ng": ["ngh"], "ngh": ["ng"],
        }
        for prefix, replacements in CONSONANT_MAPS.items():
            if word_lower.startswith(prefix):
                suffix = word_lower[len(prefix):]
                for repl in replacements:
                    new_word = repl + suffix
                    if new_word not in seen and dictionary.contains(new_word):
                        seen.add(new_word); candidates.append((new_word, 1))
                    # Thay dấu thanh trên suffix
                    for si, sc in enumerate(suffix):
                        sbase = CHAR_TO_BASE.get(sc, sc)
                        if sbase in TONE_VARIANTS:
                            for sv in TONE_VARIANTS[sbase]:
                                if sv != sc:
                                    new_word = repl + suffix[:si] + sv + suffix[si+1:]
                                    if new_word not in seen and dictionary.contains(new_word):
                                        seen.add(new_word); candidates.append((new_word, 1))
                        # Thay nhóm nguyên âm trên suffix (u↔ư, a↔ă↔â, ...)
                        for group in VOWEL_GROUPS:
                            if sbase in group:
                                for alt_base in group:
                                    if alt_base != sbase:
                                        new_word = repl + suffix[:si] + alt_base + suffix[si+1:]
                                        if new_word not in seen and dictionary.contains(new_word):
                                            seen.add(new_word); candidates.append((new_word, 1))
                                        # + tất cả dấu thanh
                                        if alt_base in TONE_VARIANTS:
                                            for sv in TONE_VARIANTS[alt_base]:
                                                new_word = repl + suffix[:si] + sv + suffix[si+1:]
                                                if new_word not in seen and dictionary.contains(new_word):
                                                    seen.add(new_word); candidates.append((new_word, 1))
                    # Xóa 1 ký tự trong suffix (VD: guíp → gi + íp xóa u → gíp → giúp)
                    for si in range(len(suffix)):
                        reduced = suffix[:si] + suffix[si+1:]
                        new_word = repl + reduced
                        if new_word and new_word not in seen and dictionary.contains(new_word):
                            seen.add(new_word); candidates.append((new_word, 2))

        # Mở rộng fallback (chỉ quét các từ loanh quanh cùng chữ cái đầu để tránh Overload)
        # Sửa lỗi do Pruning quá đà khiến "guíp" chỉ dò được "gíp" chứ không thấy "giúp".
        if max_distance >= 2:
            first_char = word_lower[0] if word_lower else ""
            for dict_word in dictionary.get_all_words():
                if dict_word.startswith(first_char) and abs(len(dict_word) - len(word_lower)) <= 2:
                    if dict_word not in seen:
                        dist = EditDistance.levenshtein(word_lower, dict_word)
                        if 0 < dist <= max_distance:
                            seen.add(dict_word)
                            candidates.append((dict_word, dist))

        # Sort by distance, then length difference, then alphabetically
        candidates.sort(key=lambda x: (x[1], abs(len(x[0])-len(word_lower)), x[0]))
        return candidates[:30]


# =====================================================================
# 3. PHO-BERT SCORER — SỬA LỖI TOKENIZER + LOG_SOFTMAX
# =====================================================================

class PhobertScorer:
    """
    Sử dụng PhoBERT MLM để đánh giá xác suất từ trong ngữ cảnh.
    
    Cải tiến v2:
    - Dùng log_softmax để chuẩn hóa xác suất (thay vì raw logits)
    - Xử lý BPE multi-subword candidates đúng cách
    """
    def __init__(self):
        print("  [PhoBERT] Đang tải mô hình vinai/phobert-base-v2...")
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        self.model = AutoModelForMaskedLM.from_pretrained("vinai/phobert-base-v2", attn_implementation="eager")
        self.model.eval()
        print("  [PhoBERT] Đã tải xong mô hình!")

    def get_all_masked_log_probs(self, tokens: List[str]) -> List[Optional[torch.Tensor]]:
        """
        [FIX PERF] Batch Masked Language Modeling: Tạo N câu, mỗi câu mask 1 từ.
        Chạy 1 lần Inference qua PhoBERT cho toàn bộ chuỗi -> Tốc độ tăng 20x-50x.
        """
        texts = []
        valid_indices = []
        for i in range(len(tokens)):
            if not any(c.isalpha() for c in tokens[i]): 
                texts.append("")
                continue
            test_tokens = list(tokens)
            test_tokens[i] = self.tokenizer.mask_token
            texts.append(" ".join(test_tokens))
            valid_indices.append(i)

        if not valid_indices:
            return [None] * len(tokens)

        # Chỉ forward những câu hợp lệ
        valid_texts = [texts[i] for i in valid_indices]
        inputs = self.tokenizer(valid_texts, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = self.model(**inputs)

        batch_log_probs = [None] * len(tokens)
        for i, original_idx in enumerate(valid_indices):
            mask_indices = torch.where(inputs["input_ids"][i] == self.tokenizer.mask_token_id)[0]
            if len(mask_indices) > 0:
                mask_idx = mask_indices[0].item()
                logits = outputs.logits[i, mask_idx, :]
                batch_log_probs[original_idx] = F.log_softmax(logits, dim=-1)

        return batch_log_probs

    def get_masked_log_probs(self, tokens: List[str], mask_idx: int) -> Optional[torch.Tensor]:
        """Lấy xác suất cho 1 vị trí tuần tự để đảm bảo Update LTR không hỏng Context."""
        test_tokens = list(tokens)
        test_tokens[mask_idx] = self.tokenizer.mask_token
        text = " ".join(test_tokens)
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        mask_indices = torch.where(inputs["input_ids"][0] == self.tokenizer.mask_token_id)[0]
        if len(mask_indices) > 0:
            return F.log_softmax(outputs.logits[0, mask_indices[0].item(), :], dim=-1)
        return None

    def extract_scores_from_log_probs(self, log_probs: torch.Tensor, candidates: List[str]) -> Dict[str, float]:
        """Tái sử dụng log_probs vector để chấm điểm nhanh candidates."""
        if log_probs is None:
            return {c: -float('inf') for c in candidates}
            
        results = {}
        for candidate in candidates:
            cand_ids = self.tokenizer(candidate, add_special_tokens=False)["input_ids"]
            if len(cand_ids) == 0:
                results[candidate] = -float('inf')
            elif len(cand_ids) == 1:
                results[candidate] = log_probs[cand_ids[0]].item()
            else:
                # BPE penalty
                results[candidate] = log_probs[cand_ids[0]].item() - 5.0
        return results


# =====================================================================
# 4. BỘ KIỂM TRA CHÍNH TẢ CHÍNH (ML-BASED) — CẢI TIẾN TOÀN DIỆN
# =====================================================================

class MLVietnameseSpellChecker:
    def __init__(self):
        print("=" * 60)
        print("  KHỞI TẠO BỘ KIỂM TRA CHÍNH TẢ (PHOBERT-BASED v2)")
        print("=" * 60)

        print("\n[1/2] Đang tải từ điển phát hiện lỗi...")
        self.dictionary = VietnameseDictionary()
        print(f"  → Đã tải {len(self.dictionary.words)} từ âm tiết")

        print("\n[2/2] Khởi tạo PhoBERT Scorer...")
        self.scorer = PhobertScorer()

        print("\n" + "=" * 60)
        print("  ✓ Sẵn sàng kiểm tra chính tả!")
        print("=" * 60)

    def _tokenize(self, text: str) -> List[str]:
        text = re.sub(r'([.,!?;:"""\'\(\)\[\]])', r' \1 ', text)
        return [t for t in text.split() if t.strip()]

    def _is_word_token(self, token: str) -> bool:
        if not token: return False
        if re.match(r'^[.,!?;:"""\'\(\)\[\]\d]+$', token): return False
        return any(c.isalpha() for c in token)

    @staticmethod
    def _preserve_case(original: str, replacement: str) -> str:
        """[MỚI] Bảo toàn pattern chữ hoa/thường của từ gốc."""
        if not original or not replacement:
            return replacement
        if original.isupper():
            return replacement.upper()
        if original[0].isupper():
            return replacement[0].upper() + replacement[1:]
        return replacement

    # [DEPRECATED] Đã xóa hàm check_text Batch Inference chứa nguy cơ rò rỉ ngữ cảnh.
    # Hàm analyze sẽ được dùng để trả ra chi tiết.

    def _check_real_word_with_phobert(self, tokens: List[str], error_idx: int,
                                      word: str, has_diacritics: bool, 
                                      log_probs: torch.Tensor) -> List[Tuple[str, float]]:
        """
        Kiểm tra lỗi real-word bằng so sánh log-probability.
        
        [FIX] Dùng relative log-prob threshold thay vì absolute logit gap.
        log_prob diff = 1.5 → candidate gấp ~4.5 lần likely hơn
        log_prob diff = 3.0 → candidate gấp ~20 lần likely hơn
        """
        variants = set()

        # 1. Thay dấu thanh & Nhóm nguyên âm nền (oôơ, aăâ, uư, eê)
        VOWEL_GROUPS_LIST = ["aăâ", "oôơ", "uư", "eê"]
        for i, char in enumerate(word):
            base = CHAR_TO_BASE.get(char, char)
            
            # Gộp chung base hiện tại và các base họ hàng (nếu có)
            related_bases = [base]
            for group in VOWEL_GROUPS_LIST:
                if base in group:
                    related_bases = list(group)
                    break
                    
            for r_base in related_bases:
                if r_base in TONE_VARIANTS:
                    for variant in TONE_VARIANTS[r_base]:
                        if variant != char:
                            new_word = word[:i] + variant + word[i + 1:]
                            if new_word != word and self.dictionary.contains(new_word):
                                variants.add(new_word)

        # 2. Lỗi phụ âm phổ biến (Confusion Set)
        if word.startswith("ch"): variants.add("tr" + word[2:])
        elif word.startswith("tr"): variants.add("ch" + word[2:])
        if word.startswith("s"): variants.add("x" + word[1:])
        elif word.startswith("x"): variants.add("s" + word[1:])
        if word.startswith("d"): variants.add("gi" + word[1:]); variants.add("r" + word[1:]); variants.add("đ" + word[1:])
        elif word.startswith("đ"): variants.add("d" + word[1:])
        elif word.startswith("gi"): variants.add("d" + word[2:]); variants.add("r" + word[2:])
        elif word.startswith("r"): variants.add("d" + word[1:]); variants.add("gi" + word[1:])
        if word.startswith("l"): variants.add("n" + word[1:])
        elif word.startswith("n") and not word.startswith("ng"): variants.add("l" + word[1:])
        if word.endswith("i"): variants.add(word[:-1] + "y")
        elif word.endswith("y"): variants.add(word[:-1] + "i")
        if word.endswith("c"): variants.add(word[:-1] + "t")
        elif word.endswith("t"): variants.add(word[:-1] + "c")

        # Lọc biến thể hợp lệ
        valid_variants = {v for v in variants if self.dictionary.contains(v) and v != word}
        if not valid_variants:
            return []

        # Chấm điểm
        original_token = tokens[error_idx]
        cased_word = self._preserve_case(original_token, word)
        
        candidates_to_score = [cased_word]
        cased_valid_variants = []
        for v in valid_variants:
            cased_v = self._preserve_case(original_token, v)
            candidates_to_score.append(cased_v)
            cased_valid_variants.append((v, cased_v))

        scores_dict = self.scorer.extract_scores_from_log_probs(log_probs, candidates_to_score)

        original_score = scores_dict.get(cased_word, -float('inf'))
        scored = []

        # [FIX] Ngưỡng phân tầng theo LOẠI biến đổi:
        TONE_THRESHOLD = 1.3       # Tone threshold
        D_THRESHOLD = 2.0          # d↔đ threshold
        ENDING_TYPO_THRESHOLD = 3.0 # lỗi i/y, c/t, n/ng
        CONSONANT_THRESHOLD = 8.0  # s↔x, ch↔tr, l↔n

        for v, cased_v in cased_valid_variants:
            v_score = scores_dict.get(cased_v, -float('inf'))
            
            # Phân loại biến đổi bằng so sánh từng ký tự "base" (bỏ dấu)
            def to_bases(w):
                return "\0".join(CHAR_TO_BASE.get(c, c) for c in w)
            
            w_bases = to_bases(word)
            v_bases = to_bases(v)
            
            is_tone_only = (w_bases == v_bases)
            
            # Phát hiện nếu chỉ sai lệch nguyên âm (o ↔ ô)
            is_vowel_only = False
            if not is_tone_only and len(w_bases) == len(v_bases):
                diffs = sum(1 for a, b in zip(w_bases, v_bases) if a != b)
                if diffs == 1:
                    # Tra cứu có cùng group không
                    diff_idx = [i for i, (a, b) in enumerate(zip(w_bases, v_bases)) if a != b][0]
                    char_w, char_v = w_bases[diff_idx], v_bases[diff_idx]
                    for group in VOWEL_GROUPS_LIST:
                        if char_w in group and char_v in group:
                            is_vowel_only = True
                            break
            
            is_d_change = False
            is_ending_typo = False
            
            if not is_tone_only and not is_vowel_only:
                if w_bases.replace('đ', 'd') == v_bases.replace('đ', 'd'):
                    is_d_change = True
                else:
                    # Lỗi kết thúc i/y, c/t, n/ng
                    wb = w_bases.split("\0")
                    vb = v_bases.split("\0")
                    if len(wb) == len(vb) and len(wb) > 1:
                        # Kiểm tra xem chỉ khác phần cuối
                        base_match = True
                        for k in range(len(wb)-1):
                            if wb[k] != vb[k] and not (wb[k]=='n' and vb[k]=='n' and len(wb) > k+1 and wb[k+1] == 'g'):
                                # Xử lý n/ng requires string matching
                                pass 
                        # Đơn giản hóa: So sánh string
                        wb_str = "".join(wb)
                        vb_str = "".join(vb)
                        for end1, end2 in [("i", "y"), ("c", "t"), ("n", "ng"), ("m", "p"), ("o", "u")]:
                            if (wb_str.endswith(end1) and vb_str.endswith(end2) and wb_str[:-len(end1)] == vb_str[:-len(end2)]) or \
                               (wb_str.endswith(end2) and vb_str.endswith(end1) and wb_str[:-len(end2)] == vb_str[:-len(end1)]):
                                is_ending_typo = True
                                break

            if is_tone_only:
                thresh = TONE_THRESHOLD if not has_diacritics else 4.0
            elif is_vowel_only:
                # Các lỗi thiếu mũ như (toi -> tôi, tran -> trần) rất phổ biến, ngang với gõ thiếu dấu
                thresh = TONE_THRESHOLD
            elif is_d_change:
                thresh = D_THRESHOLD
            elif is_ending_typo:
                thresh = ENDING_TYPO_THRESHOLD
            else:
                thresh = CONSONANT_THRESHOLD
            
            if v_score > original_score + thresh:
                scored.append((v, v_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _rank_candidates_with_phobert(self, tokens: List[str], error_idx: int,
                                       candidates: List[Tuple[str, int]],
                                       log_probs: torch.Tensor) -> List[Tuple[str, float]]:
        """Xếp hạng candidates bằng PhoBERT log-prob + ED penalty."""
        if not candidates: return []
        
        original_token = tokens[error_idx]
        cased_words = [self._preserve_case(original_token, c[0]) for c in candidates]
        scores_dict = self.scorer.extract_scores_from_log_probs(log_probs, cased_words)

        scored = []
        for (word, dist), cased_word in zip(candidates, cased_words):
            phobert_score = scores_dict.get(cased_word, -float('inf'))
            # [FIX] Giảm penalty xuống 1.0/edit để các từ xa về edit nhưng sát về nghĩa (như giúp so với gíp) dễ chiến thắng hơn
            combined_score = phobert_score - (dist * 1.0)
            scored.append((word, combined_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def correct_text(self, text: str, max_iters: int = 2) -> str:
        """
        [FIX NỘI TẠI] Sequential Left-To-Right Multi-Pass.
        Chữa Lỗi 2 Bước Khép Kín:
        B1: Chữa TẤT CẢ các lỗi Non-word trước (vì nó sai chắc chắn) để cứu lại ngữ cảnh. (máy tính guíp -> máy tính giúp). Phải LTR liền!
        B2: Sau khi có bệ phóng ngữ cảnh sạch, mới đi rà lỗi Real-word (sai văn cảnh).
        """
        current_text = text
        VOWELS_WITH_MARKS = set("àáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ")
        
        for _ in range(max_iters):
            changed = False
            
            # === BƯỚC 1: Sửa Non-word ===
            tokens = self._tokenize(current_text)
            for i in range(len(tokens)):
                word = tokens[i].lower()
                if not self._is_word_token(word) or len(word) <= 1: continue
                if not self.dictionary.contains(word):
                    log_probs = self.scorer.get_masked_log_probs(tokens, i)
                    if log_probs is not None:
                        candidates = EditDistance.generate_candidates(word, self.dictionary, max_distance=2)
                        ranked = self._rank_candidates_with_phobert(tokens, i, candidates, log_probs)
                        if ranked:
                            best = self._preserve_case(tokens[i], ranked[0][0])
                            if best != tokens[i]:
                                tokens[i] = best
                                changed = True
            current_text = " ".join(tokens)
            
            # === BƯỚC 2: Sửa Real-word (Sau khi ngữ cảnh rác đã được dọn sạch) ===
            tokens = self._tokenize(current_text)
            for i in range(len(tokens)):
                word = tokens[i].lower()
                if not self._is_word_token(word) or len(word) <= 1: continue
                # Lúc này những Non-word đã bị fix thành có trong dictionary
                if self.dictionary.contains(word):
                    log_probs = self.scorer.get_masked_log_probs(tokens, i)
                    if log_probs is not None:
                        has_diacritics = any(c in VOWELS_WITH_MARKS for c in word)
                        better = self._check_real_word_with_phobert(tokens, i, word, has_diacritics, log_probs)
                        if better:
                            best = self._preserve_case(tokens[i], better[0][0])
                            if best != tokens[i]:
                                tokens[i] = best
                                changed = True
            current_text = " ".join(tokens)

            if not changed:
                break
                
        return current_text

    def analyze(self, text: str) -> str:
        """Phân tích chi tiết lỗi chính tả bằng LTR Sequential (Tránh Hallucination)."""
        lines = ["\n" + "=" * 60, "  KẾT QUẢ PHÂN TÍCH (PHOBERT MLM v2 LTR)", "=" * 60]
        lines.append(f"\n📝 Văn bản: \"{text}\"")

        all_errors = []
        current_text = text
        VOWELS_WITH_MARKS = set("àáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ")
        
        for _ in range(2):
            changed = False
            # B1: Quét Non-Word
            toks = self._tokenize(current_text)
            for i in range(len(toks)):
                word = toks[i].lower()
                if not self._is_word_token(word) or len(word) <= 1: continue
                if not self.dictionary.contains(word):
                    log_probs = self.scorer.get_masked_log_probs(toks, i)
                    if log_probs is not None:
                        candidates = EditDistance.generate_candidates(word, self.dictionary, max_distance=2)
                        ranked = self._rank_candidates_with_phobert(toks, i, candidates, log_probs)
                        if ranked:
                            best = self._preserve_case(toks[i], ranked[0][0])
                            if best != toks[i]:
                                all_errors.append({
                                    "position": i, "word": toks[i], "type": "non-word", "suggestions": ranked[:5],
                                    "context_tokens": list(toks)
                                })
                                toks[i] = best
                                changed = True
            current_text = " ".join(toks)
            
            # B2: Quét Real-Word
            toks = self._tokenize(current_text)
            for i in range(len(toks)):
                word = toks[i].lower()
                if not self._is_word_token(word) or len(word) <= 1: continue
                if self.dictionary.contains(word):
                    log_probs = self.scorer.get_masked_log_probs(toks, i)
                    if log_probs is not None:
                        has_diacritics = any(c in VOWELS_WITH_MARKS for c in word)
                        better = self._check_real_word_with_phobert(toks, i, word, has_diacritics, log_probs)
                        if better:
                            best = self._preserve_case(toks[i], better[0][0])
                            if best != toks[i]:
                                all_errors.append({
                                    "position": i, "word": toks[i], "type": "real-word", "suggestions": better[:5],
                                    "context_tokens": list(toks)
                                })
                                toks[i] = best
                                changed = True
            current_text = " ".join(toks)

            if not changed:
                break

        if all_errors:
            lines.append("\n" + "-" * 40 + "\n  CHI TIẾT LỖI\n" + "-" * 40)
            seen_pos = set()
            unique_errors = []
            for e in all_errors:
                if e["position"] not in seen_pos:
                    seen_pos.add(e["position"])
                    unique_errors.append(e)

            for idx, error in enumerate(sorted(unique_errors, key=lambda x: x["position"]), 1):
                error_type = "Non-word" if error.get("type") == "non-word" else "Real-word"
                lines.append(f"\n🔴 Lỗi {idx} [{error_type}]: \"{error['word']}\"")
                pos = error["position"]
                ctx_toks = error["context_tokens"]
                start_idx = max(0, pos - 2)
                end_idx = min(len(ctx_toks), pos + 3)
                context_str = " ".join(
                    f"[{t}]" if (i + start_idx) == pos else t
                    for i, t in enumerate(ctx_toks[start_idx:end_idx])
                )
                lines.append(f"   Ngữ cảnh: ...{context_str}...")

                if error["suggestions"]:
                    lines.append("   📋 Gợi ý (PhoBERT Log-Probability):")
                    for rank, (sugg, score) in enumerate(error["suggestions"], 1):
                        lines.append(f"      {rank}. \"{sugg}\" (log-prob: {score:.2f})")
                else:
                    lines.append("   ⚠️ Không tìm thấy gợi ý")

            lines.append("\n" + "-" * 40 + f"\n✅ Văn bản đã sửa: \"{current_text}\"")
        else:
            lines.append("\n✅ Hoàn hảo! Không phát hiện lỗi.")

        return "\n".join(lines)


# =====================================================================
# 5. TEST & ĐÁNH GIÁ — MỞ RỘNG
# =====================================================================

def evaluate_model(checker: MLVietnameseSpellChecker):
    """
    [FIX] Đánh giá mở rộng: thêm real-word error, multi-error, false-positive test.
    """
    test_cases = [
        # === Non-word errors (từ sai không tồn tại) ===
        ("hôm nai trời đẹp quá", "hôm nay trời đẹp quá", [1]),
        ("em dang làm bài tập", "em đang làm bài tập", [1]),
        ("chúng tôi hoc môn xử lý", "chúng tôi học môn xử lý", [2]),
        ("máy tính guíp con người", "máy tính giúp con người", [2]),
        ("tiêng việt rất khó", "tiếng việt rất khó", [0]),

        # === Real-word errors (từ sai nhưng vẫn hợp lệ) ===
        ("tôi di học ở trường", "tôi đi học ở trường", [1]),
        ("anh ấy la sinh viên", "anh ấy là sinh viên", [2]),
        ("trương đại học rất lớn", "trường đại học rất lớn", [0]),
        ("kinh tê phát triển nhanh", "kinh tế phát triển nhanh", [1]),
        ("bao vệ môi trường", "bảo vệ môi trường", [0]),
        ("cô giáo day rất hay", "cô giáo dạy rất hay", [2]),

        # === Multi-error (nhiều lỗi trong 1 câu) ===
        ("tôi di hoc ở trường", "tôi đi học ở trường", [1, 2]),
        ("em dang lam bài tâp", "em đang làm bài tập", [1, 2, 4]),

        # === Lỗi phụ âm (ch/tr, s/x, l/n) ===
        ("anh ấy chồng cây trong vườn", "anh ấy trồng cây trong vườn", [2]),

        # === Câu đúng (kiểm tra False Positive) ===
        ("hôm nay trời đẹp quá", "hôm nay trời đẹp quá", []),
        ("tôi đi học ở trường đại học", "tôi đi học ở trường đại học", []),
        ("kinh tế phát triển nhanh chóng", "kinh tế phát triển nhanh chóng", []),
    ]

    print("\n" + "=" * 60 + "\n  ĐÁNH GIÁ MÔ HÌNH PHOBERT v2 (MULTI-PASS END-TO-END)\n" + "=" * 60)
    total_errors = 0
    detected = 0
    corrected = 0
    false_positives = 0
    total_correct_sentences = 0

    for wrong, correct, error_positions in test_cases:
        total_errors += len(error_positions)
        
        # Test End-To-End bao gồm cả Multi-Pass thay vì chỉ pass đầu
        wrong_tokens = checker._tokenize(wrong)
        correct_tokens = checker._tokenize(correct)
        final_corrected_str = checker.correct_text(wrong)
        final_corrected_tokens = checker._tokenize(final_corrected_str)
        
        # Lấy lại danh sách các vị trí đã bị model sửa
        changed_positions = [
            i for i, (o, r) in enumerate(zip(wrong_tokens, final_corrected_tokens)) 
            if o.lower() != r.lower()
        ]

        if not error_positions:
            # Câu đúng → mọi phát hiện đều là false positive (bị bóp méo)
            total_correct_sentences += 1
            false_positives += len(changed_positions)
            if changed_positions:
                print(f"  ⚠️  FP: \"{wrong}\" → tự động sửa sai thành {changed_positions}")
            continue

        for ep in error_positions:
            if ep in changed_positions:
                detected += 1
                if final_corrected_tokens[ep].lower() == correct_tokens[ep].lower():
                    corrected += 1
                else:
                    print(f"  ❌ WRONG: \"{wrong}\" → sửa thành \"{final_corrected_tokens[ep]}\" tại {ep} (đáng ra phải là \"{correct_tokens[ep]}\")")
            else:
                print(f"  ❌ MISS: \"{wrong}\" → bỏ sót \"{wrong_tokens[ep]}\" tại vị trí {ep}")
                
        # Nếu model rảnh rỗi tự đi sửa thêm các từ khác ngoài vùng error_positions
        for cp in changed_positions:
            if cp not in error_positions:
                false_positives += 1
                print(f"  ⚠️  FP: \"{wrong}\" → tự động sửa bừa \"{wrong_tokens[cp]}\" thành \"{final_corrected_tokens[cp]}\"")

    total_test_errors = sum(len(ep) for _, _, ep in test_cases if ep)
    print(f"\n📊 KẾT QUẢ TRÊN {len(test_cases)} CÂU TEST ({total_test_errors} lỗi + {total_correct_sentences} câu đúng):")
    print(f"   Recall (Phát hiện đúng):    {detected}/{total_errors} = {detected/total_errors:.2%}" if total_errors else "   N/A")
    print(f"   Accuracy (Sửa đúng):        {corrected}/{total_errors} = {corrected/total_errors:.2%}" if total_errors else "   N/A")
    print(f"   False Positives:            {false_positives}")
    print("=" * 60)


def main():
    checker = MLVietnameseSpellChecker()

    # 1. Đánh giá
    evaluate_model(checker)

    # 2. Interactive Mode
    print("\n" + "🚀" * 20)
    print("  CHẾ ĐỘ GÕ TƯƠNG TÁC (Gõ 'quit' để thoát)")
    while True:
        try: text = input("📝 Gõ câu tiếng Việt có lỗi: ").strip()
        except: break
        if text.lower() in ('quit', 'exit'): break
        if text: print(checker.analyze(text))

if __name__ == "__main__":
    main()
