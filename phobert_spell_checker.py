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
"""

import re
import torch
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

class VietnameseDictionary:
    def __init__(self):
        self.words: Set[str] = set()
        self._build_dictionary()

    def _build_dictionary(self):
        import os
        dict_path = "vietnamese_syllables.txt"
        
        if not os.path.exists(dict_path):
            raise FileNotFoundError(f"Không tìm thấy file từ điển '{dict_path}'. Vui lòng để file này cùng thư mục với script.")
            
        with open(dict_path, "r", encoding="utf-8") as f:
            self.words = set(w.strip().lower() for w in f if w.strip())

    def contains(self, word: str) -> bool:
        return word.lower().strip() in self.words

    def get_all_words(self) -> Set[str]:
        return self.words


# =====================================================================
# 2. SINH ỨNG VIÊN (EDIT DISTANCE)
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

        # 2. Thay thế
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

        # 5. Thay dấu thanh tiếng Việt
        for i, char in enumerate(word_lower):
            base = CHAR_TO_BASE.get(char, char)
            if base in TONE_VARIANTS:
                for variant in TONE_VARIANTS[base]:
                    if variant != char:
                        new_word = word_lower[:i] + variant + word_lower[i + 1:]
                        if new_word not in seen and dictionary.contains(new_word):
                            seen.add(new_word); candidates.append((new_word, 1))

        # Scan fallback
        if len(candidates) < 3 and max_distance >= 2:
            count = 0
            for dict_word in dictionary.get_all_words():
                if abs(len(dict_word) - len(word_lower)) <= 1:
                    dist = EditDistance.levenshtein(word_lower, dict_word)
                    if 0 < dist <= max_distance and dict_word not in seen:
                        candidates.append((dict_word, dist))
                        count += 1
                        if count >= 20: break

        candidates.sort(key=lambda x: x[1])
        return candidates[:10]


# =====================================================================
# 3. PHO-BERT SCORER (THAY THẾ N-GRAM)
# =====================================================================

class PhobertScorer:
    """
    Sử dụng PhoBERT (Masked Language Model) để đánh giá xác suất của từ trong ngữ cảnh.
    Mô hình được huấn luyện trên 20GB văn bản tiếng Việt.
    """
    def __init__(self):
        print("  [PhoBERT] Đang tải mô hình vinai/phobert-base-v2 từ Hugging Face...")
        print("  [PhoBERT] Có thể mất vài phút nếu chạy lần đầu...")
        # Sử dụng tokenizer và model của PhoBERT
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        self.model = AutoModelForMaskedLM.from_pretrained("vinai/phobert-base-v2", attn_implementation="eager")
        self.model.eval() # Chế độ suy luận
        print("  [PhoBERT] Đã tải xong mô hình!")

    def score_candidates(self, tokens: List[str], error_idx: int, candidates: List[str]) -> Dict[str, float]:
        """
        Đánh giá điểm của nhiều candidates cùng một lúc chỉ với 1 lần chạy PhoBERT.
        Bằng cách mask vị trí lỗi và bắt PhoBERT dự đoán lô.
        """
        # Tạo câu với token mask ở vị trí lỗi
        test_tokens = list(tokens)
        test_tokens[error_idx] = self.tokenizer.mask_token # "<mask>"
        text = " ".join(test_tokens)
        
        inputs = self.tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Tìm vị trí của mask token trong input
        mask_token_indices = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]
        if len(mask_token_indices) == 0:
            return {c: -float('inf') for c in candidates}
        
        mask_idx = mask_token_indices[0].item()
        logits = outputs.logits[0, mask_idx, :] # Lấy logits của các từ tại vị trí mask
        
        results = {}
        for candidate in candidates:
            candidate_input = self.tokenizer(candidate, add_special_tokens=False)["input_ids"]
            if len(candidate_input) == 0:
                results[candidate] = -float('inf')
            elif len(candidate_input) == 1:
                results[candidate] = logits[candidate_input[0]].item()
            else:
                results[candidate] = sum(logits[tok].item() for tok in candidate_input) / len(candidate_input)
                
        return results


# =====================================================================
# 4. BỘ KIỂM TRA CHÍNH TẢ CHÍNH (ML-BASED)
# =====================================================================

class MLVietnameseSpellChecker:
    def __init__(self):
        print("=" * 60)
        print("  KHỞI TẠO BỘ KIỂM TRA CHÍNH TẢ (PHOBERT-BASED)")
        print("=" * 60)

        print("\n[1/2] Đang tải từ điển phát hiện lỗi...")
        self.dictionary = VietnameseDictionary()
        print(f"  → Đã tải {len(self.dictionary.words)} từ âm tiết")

        print("\n[2/2] Khởi tạo PhoBERT Scorer...")
        self.scorer = PhobertScorer()

        print("\n" + "=" * 60)
        print("  ✓ Sẵn sàng kiểm tra chính tả siêu việt!")
        print("=" * 60)

    def _tokenize(self, text: str) -> List[str]:
        text = re.sub(r'([.,!?;:"""\'\(\)\[\]])', r' \1 ', text)
        return [t for t in text.split() if t.strip()]

    def _is_word_token(self, token: str) -> bool:
        if not token: return False
        if re.match(r'^[.,!?;:"""\'\(\)\[\]\d]+$', token): return False
        return any(c.isalpha() for c in token)

    def check_text(self, text: str) -> List[Dict]:
        tokens = self._tokenize(text)
        errors = []

        for i, token in enumerate(tokens):
            if not self._is_word_token(token):
                continue

            word = token.lower()
            if len(word) <= 1:
                continue

            # === LỖI NON-WORD: từ không có trong từ điển ===
            if not self.dictionary.contains(word):
                candidates = EditDistance.generate_candidates(word, self.dictionary, max_distance=2)
                ranked = self._rank_candidates_with_phobert(tokens, i, candidates)
                
                if ranked:
                    # Cập nhật ngữ cảnh ngay lập tức để PhoBERT chấm chính xác cho từ tiếp theo
                    tokens[i] = ranked[0][0]
                    
                errors.append({
                    "position": i,
                    "word": token,
                    "suggestions": ranked[:5]
                })
            else:
                has_diacritics = any(c in VIETNAMESE_CHARS and c not in "abcdefghijklmnopqrstuvwxyz" for c in word)
                better = self._check_real_word_with_phobert(tokens, i, word, has_diacritics)
                if better:
                    tokens[i] = better[0][0]
                    errors.append({
                        "position": i,
                        "word": token,
                        "suggestions": better[:5]
                    })

        return errors

    def _check_real_word_with_phobert(self, tokens: List[str], error_idx: int, word: str, has_diacritics: bool) -> List[Tuple[str, float]]:
        variants = set()
        
        # 1. Thay thanh điệu và nguyên âm (Áp dụng cho mọi từ để sửa sai dấu: ví dụ 'tê' -> 'tế')
        for i, char in enumerate(word):
            base = CHAR_TO_BASE.get(char, char)
            if base in TONE_VARIANTS:
                for variant in TONE_VARIANTS[base]:
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
        elif word.startswith("n"): variants.add("l" + word[1:])
        if word.endswith("i"): variants.add(word[:-1] + "y")
        elif word.endswith("y"): variants.add(word[:-1] + "i")
        if word.endswith("c"): variants.add(word[:-1] + "t")
        elif word.endswith("t"): variants.add(word[:-1] + "c")
        
        # Lọc biến thể có thật
        valid_variants = {v for v in variants if self.dictionary.contains(v)}

        if not valid_variants:
            return []

        # Chấm điểm toàn bộ candidate
        candidates_to_score = [word] + list(valid_variants)
        scores_dict = self.scorer.score_candidates(tokens, error_idx, candidates_to_score)

        original_score = scores_dict[word]
        scored = []
        
        # Ngưỡng chọn: Lỗi không dấu dễ thay hơn nên threshold thấp, lỗi có dấu (nhầm s/x, ch/tr) cần threshold cao
        threshold = 1.0 if not has_diacritics else 2.0
        
        for v in valid_variants:
            v_score = scores_dict[v]
            if v_score > original_score + threshold:
                scored.append((v, v_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _rank_candidates_with_phobert(self, tokens: List[str], error_idx: int, candidates: List[Tuple[str, int]]) -> List[Tuple[str, float]]:
        if not candidates: return []
        word_list = [c[0] for c in candidates]
        scores_dict = self.scorer.score_candidates(tokens, error_idx, word_list)
        
        scored = []
        for word, dist in candidates:
            phobert_score = scores_dict[word]
            # Trừ điểm những từ có khoảng cách chỉnh sửa quá lớn (phạt 1.5 điểm logit cho mỗi Edit Distance)
            combined_score = phobert_score - (dist * 1.5)
            scored.append((word, combined_score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def correct_text(self, text: str) -> str:
        # Cứ gọi check_text, nó sẽ trả về lỗi. Nhưng ta có thể tự sửa bằng cách thay thế trực tiếp
        errors = self.check_text(text)
        tokens = self._tokenize(text)
        
        for error in errors:
            pos = error["position"]
            if error["suggestions"]:
                tokens[pos] = error["suggestions"][0][0]
                
        return " ".join(tokens)

    def analyze(self, text: str) -> str:
        errors = self.check_text(text)
        tokens = self._tokenize(text)

        lines = ["\n" + "=" * 60, "  KẾT QUẢ PHÂN TÍCH (PHOBERT MLM)", "=" * 60]
        lines.append(f"\n📝 Văn bản: \"{text}\"")

        if errors:
            lines.append("\n" + "-" * 40 + "\n  CHI TIẾT LỖI\n" + "-" * 40)
            for idx, error in enumerate(errors, 1):
                lines.append(f"\n🔴 Lỗi {idx}: \"{error['word']}\"")
                pos = error["position"]
                context_str = " ".join(f"[{t}]" if i == pos else t for i, t in enumerate(tokens[max(0, pos-2):min(len(tokens), pos+3)]))
                lines.append(f"   Ngữ cảnh: ...{context_str}...")

                if error["suggestions"]:
                    lines.append("   📋 Gợi ý (xếp hạng bởi PhoBERT Logits):")
                    for rank, (sugg, score) in enumerate(error["suggestions"], 1):
                        lines.append(f"      {rank}. \"{sugg}\" (điểm tự tin: {score:.2f})")
                else:
                    lines.append("   ⚠️ Không tìm thấy gợi ý")

            lines.append("\n" + "-" * 40 + f"\n✅ Văn bản đã sửa: \"{self.correct_text(text)}\"")
        else:
            lines.append("\n✅ Hoàn hảo! Không phát hiện lỗi.")

        return "\n".join(lines)


# =====================================================================
# 5. TEST & DEMO
# =====================================================================

def evaluate_model(checker: MLVietnameseSpellChecker):
    test_cases = [
        ("tôi di học ở trường", "tôi đi học ở trường", [1]),
        ("hôm nai trời đẹp quá", "hôm nay trời đẹp quá", [1]),
        ("em dang làm bài tập", "em đang làm bài tập", [1]),
        ("chúng tôi hoc môn xử lý", "chúng tôi học môn xử lý", [2]),
        ("anh ấy la sinh viên", "anh ấy là sinh viên", [2]),
        ("máy tính guíp con người", "máy tính giúp con người", [2]),
        ("trương đại học rất lớn", "trường đại học rất lớn", [0]),
        ("tiêng việt rất khó", "tiếng việt rất khó", [0]),
        ("kinh tê phát triển nhanh", "kinh tế phát triển nhanh", [1]),
        ("bao vệ môi trường", "bảo vệ môi trường", [0]),
    ]
    
    print("\n" + "=" * 60 + "\n  ĐÁNH GIÁ MÔ HÌNH PHOBERT\n" + "=" * 60)
    total, detected, corrected = 0, 0, 0

    for wrong, correct, error_positions in test_cases:
        total += len(error_positions)
        errors = checker.check_text(wrong)
        correct_tokens = checker._tokenize(correct)

        for error in errors:
            pos = error["position"]
            if pos in error_positions:
                detected += 1
                if error["suggestions"] and error["suggestions"][0][0] == correct_tokens[pos].lower():
                    corrected += 1

    print(f"\n📊 KẾT QUẢ TRÊN {len(test_cases)} CÂU TEST:")
    print(f"   Recall (Phát hiện đúng): {detected/total if total else 0:.2%}")
    print(f"   Accuracy (Sửa đúng Tóp 1):  {corrected/total if total else 0:.2%}")
    print("=" * 60)


def main():
    checker = MLVietnameseSpellChecker()
    
    # 1. Chạy đánh giá (Evaluations on Test set)
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
