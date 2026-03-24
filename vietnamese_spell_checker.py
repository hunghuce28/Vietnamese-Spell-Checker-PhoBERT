#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=====================================================================
  PHÁT HIỆN VÀ CHỈNH SỬA LỖI CHÍNH TẢ TIẾNG VIỆT
  Sử dụng Mô hình Ngôn ngữ Thống kê (N-gram)
=====================================================================
  Môn: Xử lý Ngôn ngữ Tự nhiên (NLP)
=====================================================================

Phương pháp:
  1. Phát hiện lỗi: Dùng từ điển tiếng Việt để kiểm tra từ có hợp lệ không
  2. Sinh ứng viên: Dùng khoảng cách chỉnh sửa (Edit Distance) + 
     thay thế ký tự đặc trưng tiếng Việt
  3. Xếp hạng ứng viên: Dùng mô hình N-gram (bigram) với Laplace smoothing
     để chọn từ phù hợp nhất dựa trên ngữ cảnh
"""

import re
import math
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Set, Optional


# =====================================================================
# 1. BẢNG KÝ TỰ TIẾNG VIỆT
# =====================================================================

# Các nguyên âm tiếng Việt (có dấu)
VIETNAMESE_VOWELS = set(
    "aàáảãạăằắẳẵặâầấẩẫậeèéẻẽẹêềếểễệ"
    "iìíỉĩịoòóỏõọôồốổỗộơờớởỡợ"
    "uùúủũụưừứửữựyỳýỷỹỵ"
)

# Bảng chữ cái tiếng Việt đầy đủ (thường)
VIETNAMESE_CHARS = set(
    "aàáảãạăằắẳẵặâầấẩẫậbcdđeèéẻẽẹêềếểễệ"
    "fghiìíỉĩịjklmnoòóỏõọôồốổỗộơờớởỡợ"
    "pqrstuùúủũụưừứửữựvwxyỳýỷỹỵz"
)

# Các cặp phụ âm dễ nhầm lẫn trong tiếng Việt
CONFUSED_CONSONANTS = {
    "s": ["x"],
    "x": ["s"],
    "ch": ["tr"],
    "tr": ["ch"],
    "d": ["gi", "r"],
    "gi": ["d", "r"],
    "r": ["d", "gi"],
    "l": ["n"],
    "n": ["l"],
    "c": ["k", "q"],
    "k": ["c", "q"],
    "ng": ["ngh"],
    "ngh": ["ng"],
    "g": ["gh"],
    "gh": ["g"],
}

# Các nhóm nguyên âm dễ nhầm (dấu thanh)
TONE_VARIANTS = {
    "a": "aàáảãạ", "ă": "ăằắẳẵặ", "â": "âầấẩẫậ",
    "e": "eèéẻẽẹ", "ê": "êềếểễệ",
    "i": "iìíỉĩị",
    "o": "oòóỏõọ", "ô": "ôồốổỗộ", "ơ": "ơờớởỡợ",
    "u": "uùúủũụ", "ư": "ưừứửữự",
    "y": "yỳýỷỹỵ",
}

# Ánh xạ ngược: từ ký tự có dấu → nhóm gốc
CHAR_TO_BASE = {}
for base, variants in TONE_VARIANTS.items():
    for v in variants:
        CHAR_TO_BASE[v] = base


# =====================================================================
# 2. TỪ ĐIỂN TIẾNG VIỆT
# =====================================================================

class VietnameseDictionary:
    """
    Từ điển tiếng Việt chứa các âm tiết hợp lệ.
    Tiếng Việt viết tách âm tiết bằng dấu cách,
    nên mỗi đơn vị kiểm tra là một âm tiết.
    """

    def __init__(self):
        self.words: Set[str] = set()
        self._build_dictionary()

    def _build_dictionary(self):
        """Xây dựng từ điển từ danh sách âm tiết."""
        import os
        dict_path = "vietnamese_syllables.txt"
        
        if not os.path.exists(dict_path):
            raise FileNotFoundError(f"Không tìm thấy file từ điển '{dict_path}'.")
            
        with open(dict_path, "r", encoding="utf-8") as f:
            self.words = set(w.strip().lower() for w in f if w.strip())

    def contains(self, word: str) -> bool:
        """Kiểm tra từ có trong từ điển không."""
        return word.lower().strip() in self.words

    def get_all_words(self) -> Set[str]:
        """Trả về toàn bộ từ trong từ điển."""
        return self.words


# =====================================================================
# 3. MÔ HÌNH NGÔN NGỮ N-GRAM
# =====================================================================

class NGramModel:
    """
    Mô hình ngôn ngữ N-gram cho tiếng Việt.
    
    Sử dụng bigram (n=2) với Laplace smoothing.
    P(w_i | w_{i-1}) = (count(w_{i-1}, w_i) + 1) / (count(w_{i-1}) + V)
    
    Trong đó:
    - count(w_{i-1}, w_i): số lần bigram xuất hiện
    - count(w_{i-1}): số lần unigram xuất hiện
    - V: kích thước từ vựng
    """

    def __init__(self, n: int = 2):
        self.n = n
        # Đếm bigram: bigram_counts[w1][w2] = số lần w1 w2 xuất hiện liên tiếp
        self.bigram_counts: Dict[str, Counter] = defaultdict(Counter)
        # Đếm unigram: unigram_counts[w] = số lần w xuất hiện
        self.unigram_counts: Counter = Counter()
        # Kích thước từ vựng
        self.vocab_size: int = 0
        # Tổng số token
        self.total_tokens: int = 0

    def train(self, corpus: List[List[str]]):
        """
        Huấn luyện mô hình từ corpus.
        
        Args:
            corpus: Danh sách các câu, mỗi câu là danh sách từ.
                    Ví dụ: [["tôi", "đi", "học"], ["hôm", "nay", "trời", "đẹp"]]
        """
        for sentence in corpus:
            # Thêm token đặc biệt <s> và </s>
            tokens = ["<s>"] + [w.lower() for w in sentence] + ["</s>"]

            # Đếm unigram
            for token in tokens:
                self.unigram_counts[token] += 1

            # Đếm bigram
            for i in range(len(tokens) - 1):
                self.bigram_counts[tokens[i]][tokens[i + 1]] += 1

        self.vocab_size = len(self.unigram_counts)
        self.total_tokens = sum(self.unigram_counts.values())
        print(f"  [N-gram] Đã huấn luyện: {self.vocab_size} từ vựng, "
              f"{self.total_tokens} token, {len(corpus)} câu")

    def bigram_probability(self, word: str, prev_word: str) -> float:
        """
        Tính xác suất bigram P(word | prev_word) với Laplace smoothing.
        
        Args:
            word: Từ hiện tại
            prev_word: Từ trước đó
            
        Returns:
            Xác suất bigram đã được smoothing
        """
        count_bigram = self.bigram_counts[prev_word][word]
        count_prev = self.unigram_counts[prev_word]
        # Laplace smoothing (add-1)
        return (count_bigram + 1) / (count_prev + self.vocab_size)

    def unigram_probability(self, word: str) -> float:
        """
        Tính xác suất unigram P(word) với Laplace smoothing.
        """
        count = self.unigram_counts[word]
        return (count + 1) / (self.total_tokens + self.vocab_size)

    def sentence_log_probability(self, words: List[str]) -> float:
        """
        Tính log xác suất của cả câu dùng bigram model.
        log P(w1, w2, ..., wn) = Σ log P(wi | wi-1)
        """
        tokens = ["<s>"] + [w.lower() for w in words] + ["</s>"]
        log_prob = 0.0
        for i in range(1, len(tokens)):
            prob = self.bigram_probability(tokens[i], tokens[i - 1])
            log_prob += math.log(prob) if prob > 0 else float("-inf")
        return log_prob

    def score_word_in_context(self, word: str, prev_word: str, next_word: str = None) -> float:
        """
        Tính điểm của một từ trong ngữ cảnh (dùng cả bigram trái và phải).
        
        Score = λ * log P(word | prev) + (1-λ) * log P(next | word)
        """
        lambda_weight = 0.6  # Trọng số cho ngữ cảnh bên trái

        # Bigram trái: P(word | prev_word)
        left_prob = self.bigram_probability(word, prev_word)
        left_log = math.log(left_prob) if left_prob > 0 else -20

        if next_word:
            # Bigram phải: P(next_word | word)
            right_prob = self.bigram_probability(next_word, word)
            right_log = math.log(right_prob) if right_prob > 0 else -20
            return lambda_weight * left_log + (1 - lambda_weight) * right_log
        else:
            return left_log


# =====================================================================
# 4. KHOẢNG CÁCH CHỈNH SỬA (EDIT DISTANCE)
# =====================================================================

class EditDistance:
    """
    Tính khoảng cách chỉnh sửa (Levenshtein Distance)
    và sinh ứng viên sửa lỗi cho tiếng Việt.
    """

    @staticmethod
    def levenshtein(s1: str, s2: str) -> int:
        """
        Tính khoảng cách Levenshtein giữa 2 chuỗi.
        
        Sử dụng quy hoạch động (Dynamic Programming):
        dp[i][j] = khoảng cách chỉnh sửa tối thiểu giữa s1[:i] và s2[:j]
        """
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],       # Xóa
                        dp[i][j - 1],       # Chèn
                        dp[i - 1][j - 1]    # Thay thế
                    )
        return dp[m][n]

    @staticmethod
    def generate_candidates(word: str, dictionary: VietnameseDictionary,
                           max_distance: int = 2) -> List[Tuple[str, int]]:
        """
        Sinh các ứng viên sửa lỗi từ từ điển.
        
        Sử dụng 4 phép biến đổi cơ bản + thay thế đặc trưng tiếng Việt:
        1. Xóa (Deletion): bỏ 1 ký tự
        2. Chèn (Insertion): thêm 1 ký tự  
        3. Thay thế (Substitution): đổi 1 ký tự
        4. Hoán vị (Transposition): đổi chỗ 2 ký tự liền kề
        5. Thay dấu thanh tiếng Việt
        
        Args:
            word: Từ cần tìm ứng viên sửa lỗi
            dictionary: Từ điển tiếng Việt
            max_distance: Khoảng cách chỉnh sửa tối đa
            
        Returns:
            Danh sách (ứng_viên, khoảng_cách)
        """
        candidates = []
        word_lower = word.lower()

        # Dùng bộ ký tự NHỎ (chỉ chữ cái gốc) cho chèn/thay thế
        # → Giảm từ ~100 ký tự xuống ~29 → nhanh hơn nhiều lần
        SMALL_CHARS = "aăâbcdđeêfghijklmnoôơpqrstuưvwxyz"

        # Tập hợp các từ đã sinh (tránh trùng)
        seen = set()

        # --- Phép biến đổi khoảng cách 1 ---

        # 1. Xóa 1 ký tự
        for i in range(len(word_lower)):
            new_word = word_lower[:i] + word_lower[i + 1:]
            if new_word and new_word not in seen:
                seen.add(new_word)
                if dictionary.contains(new_word):
                    candidates.append((new_word, 1))

        # 2. Thay thế 1 ký tự (dùng bộ ký tự nhỏ)
        for i in range(len(word_lower)):
            for c in SMALL_CHARS:
                if c != word_lower[i]:
                    new_word = word_lower[:i] + c + word_lower[i + 1:]
                    if new_word not in seen:
                        seen.add(new_word)
                        if dictionary.contains(new_word):
                            candidates.append((new_word, 1))

        # 3. Chèn 1 ký tự (dùng bộ ký tự nhỏ)
        for i in range(len(word_lower) + 1):
            for c in SMALL_CHARS:
                new_word = word_lower[:i] + c + word_lower[i:]
                if new_word not in seen:
                    seen.add(new_word)
                    if dictionary.contains(new_word):
                        candidates.append((new_word, 1))

        # 4. Hoán vị 2 ký tự liền kề
        for i in range(len(word_lower) - 1):
            chars = list(word_lower)
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
            new_word = "".join(chars)
            if new_word not in seen:
                seen.add(new_word)
                if dictionary.contains(new_word):
                    candidates.append((new_word, 1))

        # 5. Thay dấu thanh (đặc trưng tiếng Việt) — rất quan trọng!
        for i, char in enumerate(word_lower):
            base = CHAR_TO_BASE.get(char, char)
            if base in TONE_VARIANTS:
                for variant in TONE_VARIANTS[base]:
                    if variant != char:
                        new_word = word_lower[:i] + variant + word_lower[i + 1:]
                        if new_word not in seen:
                            seen.add(new_word)
                            if dictionary.contains(new_word):
                                candidates.append((new_word, 1))

        # 6. Thay thế nhóm nguyên âm gốc (a↔ă↔â, o↔ô↔ơ, u↔ư, e↔ê)
        VOWEL_GROUPS = [
            "aăâ", "oôơ", "uư", "eê"
        ]
        for i, char in enumerate(word_lower):
            base = CHAR_TO_BASE.get(char, char)
            for group in VOWEL_GROUPS:
                if base in group:
                    for alt_base in group:
                        if alt_base != base:
                            new_word = word_lower[:i] + alt_base + word_lower[i + 1:]
                            if new_word not in seen:
                                seen.add(new_word)
                                if dictionary.contains(new_word):
                                    candidates.append((new_word, 1))

        # Nếu chưa tìm đủ ứng viên, quét từ điển (giới hạn để tránh chậm)
        if len(candidates) < 3 and max_distance >= 2:
            count = 0
            for dict_word in dictionary.get_all_words():
                # Chỉ xét từ có độ dài gần giống
                if abs(len(dict_word) - len(word_lower)) <= 1:
                    dist = EditDistance.levenshtein(word_lower, dict_word)
                    if 0 < dist <= max_distance and dict_word not in seen:
                        candidates.append((dict_word, dist))
                        count += 1
                        if count >= 20:  # Giới hạn để không bị chậm
                            break

        # Sắp xếp theo khoảng cách
        candidates.sort(key=lambda x: x[1])
        return candidates[:10]  # Trả về tối đa 10 ứng viên tốt nhất


# =====================================================================
# 5. BỘ KIỂM TRA CHÍNH TẢ CHÍNH
# =====================================================================

class VietnameseSpellChecker:
    """
    Bộ kiểm tra và sửa lỗi chính tả tiếng Việt.
    
    Kết hợp:
    - Từ điển: phát hiện từ sai
    - Edit Distance: sinh ứng viên sửa lỗi
    - N-gram: xếp hạng ứng viên theo ngữ cảnh
    """

    def __init__(self):
        print("=" * 60)
        print("  KHỞI TẠO BỘ KIỂM TRA CHÍNH TẢ TIẾNG VIỆT")
        print("=" * 60)

        print("\n[1/3] Đang tải từ điển...")
        self.dictionary = VietnameseDictionary()
        print(f"  → Đã tải {len(self.dictionary.words)} từ")

        print("\n[2/3] Đang khởi tạo mô hình N-gram...")
        self.ngram_model = NGramModel(n=2)

        print("\n[3/3] Đang huấn luyện mô hình từ corpus...")
        self._train_model()

        print("\n" + "=" * 60)
        print("  ✓ Sẵn sàng kiểm tra chính tả!")
        print("=" * 60)

    def _train_model(self):
        """Huấn luyện mô hình N-gram từ corpus tiếng Việt mẫu."""
        # Corpus tiếng Việt mẫu - các câu phổ biến thuộc nhiều chủ đề
        corpus_text = [
            "tôi đi học ở trường đại học",
            "hôm nay trời đẹp quá",
            "anh ấy là sinh viên giỏi",
            "chúng tôi đang học môn xử lý ngôn ngữ tự nhiên",
            "tiếng việt là ngôn ngữ có thanh điệu",
            "mô hình ngôn ngữ thống kê rất quan trọng",
            "phát hiện lỗi chính tả là bài toán khó",
            "chúng ta cần xử lý văn bản tiếng việt",
            "em đang làm bài tập về nhà",
            "cô giáo dạy rất hay và dễ hiểu",
            "trường đại học có nhiều sinh viên",
            "máy tính giúp con người xử lý thông tin",
            "công nghệ thông tin phát triển rất nhanh",
            "trí tuệ nhân tạo đang thay đổi thế giới",
            "chúng tôi nghiên cứu thuật toán mới",
            "bài báo khoa học được công bố năm nay",
            "dữ liệu lớn giúp cải thiện mô hình",
            "thành phố hồ chí minh là thành phố lớn nhất",
            "hà nội là thủ đô của việt nam",
            "đất nước việt nam rất đẹp",
            "người việt nam yêu hòa bình",
            "kinh tế việt nam đang phát triển mạnh",
            "giáo dục là quốc sách hàng đầu",
            "sức khỏe là vốn quý nhất",
            "gia đình là nền tảng của xã hội",
            "trẻ em là tương lai của đất nước",
            "bảo vệ môi trường là trách nhiệm của mọi người",
            "học sinh cần chăm chỉ học tập",
            "sinh viên cần có kỹ năng mềm",
            "thầy giáo và cô giáo rất quan trọng",
            "tôi thích đọc sách và viết văn",
            "anh ấy chơi bóng đá rất giỏi",
            "chị ấy nấu ăn rất ngon",
            "bố mẹ tôi làm việc rất chăm chỉ",
            "ông bà tôi sống ở nông thôn",
            "thời tiết hôm nay rất nóng",
            "mùa xuân là mùa đẹp nhất",
            "tết nguyên đán là ngày lễ lớn nhất",
            "chúng tôi đi du lịch đà nẵng",
            "biển đà nẵng rất đẹp và sạch",
            "con đường này rất rộng và đẹp",
            "thành phố có nhiều tòa nhà cao tầng",
            "xe máy là phương tiện phổ biến nhất",
            "tôi muốn mua một chiếc xe mới",
            "giá cả thị trường đang tăng cao",
            "người dân sống ở đây rất thân thiện",
            "văn hóa việt nam rất đa dạng",
            "ẩm thực việt nam nổi tiếng thế giới",
            "phở là món ăn truyền thống",
            "bánh mì sài gòn rất ngon",
            "tôi đi làm bằng xe buýt mỗi ngày",
            "công ty này có rất nhiều nhân viên",
            "doanh nghiệp cần đổi mới sáng tạo",
            "chính phủ đã ban hành chính sách mới",
            "pháp luật bảo vệ quyền lợi người dân",
            "bác sĩ khám bệnh cho bệnh nhân",
            "bệnh viện này rất hiện đại",
            "thuốc men cần được sử dụng đúng cách",
            "thể dục thể thao giúp tăng cường sức khỏe",
            "âm nhạc mang lại niềm vui cho mọi người",
            "phim ảnh là hình thức giải trí phổ biến",
            "internet kết nối mọi người trên thế giới",
            "điện thoại thông minh rất tiện lợi",
            "phần mềm máy tính ngày càng phát triển",
            "lập trình là kỹ năng quan trọng",
            "thuật toán giúp giải quyết vấn đề phức tạp",
            "xử lý ngôn ngữ tự nhiên là lĩnh vực thú vị",
            "mô hình thống kê được sử dụng rộng rãi",
            "khoảng cách chỉnh sửa giúp tìm từ tương tự",
            "ngữ pháp tiếng việt có nhiều đặc điểm riêng",
            "từ vựng tiếng việt rất phong phú",
            "chính tả tiếng việt cần được viết đúng",
            "lỗi chính tả làm giảm chất lượng văn bản",
            "kiểm tra chính tả tự động rất hữu ích",
            "mô hình ngôn ngữ giúp hiểu ngữ cảnh",
            "xác suất thống kê là nền tảng của mô hình",
            "dữ liệu huấn luyện cần đa dạng và chính xác",
            "đánh giá mô hình bằng độ chính xác",
            "kết quả thực nghiệm cho thấy hiệu quả tốt",
            "nghiên cứu khoa học đòi hỏi kiên trì",
            "tôi yêu việt nam",
            "việt nam đất nước tươi đẹp",
            "hà nội mùa thu lá vàng rơi",
            "sông hồng chảy qua thành phố hà nội",
            "chợ bến thành ở thành phố hồ chí minh",
            "huế là cố đô của việt nam",
            "đà lạt là thành phố ngàn hoa",
            "nha trang có bãi biển đẹp",
            "phú quốc là hòn đảo lớn nhất",
            "cần thơ là thành phố miền tây",
            "hải phòng là thành phố cảng",
            "quảng ninh có vịnh hạ long",
            "tôi rất thích ăn phở bò",
            "cà phê việt nam rất ngon",
            "áo dài là trang phục truyền thống",
            "nhạc trịnh rất hay và sâu lắng",
            "việt nam có bốn mùa rõ rệt",
            "mùa hè rất nóng và oi bức",
            "mùa đông lạnh và hanh khô",
            "mùa thu se lạnh và lãng mạn",
            "mưa phùn là đặc trưng miền bắc",
        ]

        # Tokenize: tách câu thành danh sách từ
        corpus = [sentence.split() for sentence in corpus_text]
        self.ngram_model.train(corpus)

    def _tokenize(self, text: str) -> List[str]:
        """
        Tách văn bản thành danh sách token (từ/âm tiết).
        Giữ nguyên dấu câu riêng biệt.
        """
        # Tách dấu câu khỏi từ
        text = re.sub(r'([.,!?;:"""\'\(\)\[\]])', r' \1 ', text)
        tokens = text.split()
        return [t for t in tokens if t.strip()]

    def _is_word_token(self, token: str) -> bool:
        """Kiểm tra token có phải là từ (không phải dấu câu, số)."""
        if not token:
            return False
        # Bỏ qua dấu câu và số
        if re.match(r'^[.,!?;:"""\'\(\)\[\]\d]+$', token):
            return False
        # Kiểm tra có ít nhất 1 ký tự chữ
        return any(c.isalpha() for c in token)

    def check_text(self, text: str) -> List[Dict]:
        """
        Kiểm tra lỗi chính tả trong văn bản.
        
        Bao gồm:
        - Lỗi non-word: từ không có trong từ điển
        - Lỗi real-word: từ hợp lệ nhưng có thể sai dấu (ví dụ: "di" → "đi")
        
        Args:
            text: Văn bản cần kiểm tra
            
        Returns:
            Danh sách các lỗi tìm thấy
        """
        tokens = self._tokenize(text)
        errors = []

        for i, token in enumerate(tokens):
            if not self._is_word_token(token):
                continue

            word = token.lower()
            
            # Bỏ qua từ 1 ký tự (thường là viết tắt)
            if len(word) <= 1:
                continue

            prev_word = tokens[i - 1].lower() if i > 0 else "<s>"
            next_word = tokens[i + 1].lower() if i < len(tokens) - 1 else None

            # === LỖI NON-WORD: từ không có trong từ điển ===
            if not self.dictionary.contains(word):
                candidates = EditDistance.generate_candidates(
                    word, self.dictionary, max_distance=2
                )
                ranked = self._rank_candidates(candidates, prev_word, next_word)
                errors.append({
                    "position": i,
                    "word": token,
                    "suggestions": ranked[:5]
                })
            else:
                # === LỖI REAL-WORD: chỉ kiểm tra từ KHÔNG CÓ DẤU ===
                # Từ đã có dấu tiếng Việt thì ít khả năng bị sai
                has_diacritics = any(c in VIETNAMESE_CHARS and c not in "abcdefghijklmnopqrstuvwxyz" for c in word)
                if not has_diacritics and len(word) <= 6:
                    better = self._check_real_word(word, prev_word, next_word)
                    if better:
                        errors.append({
                            "position": i,
                            "word": token,
                            "suggestions": better[:5]
                        })

        return errors

    def _check_real_word(self, word: str, prev_word: str, 
                         next_word: str = None) -> List[Tuple[str, float]]:
        """
        Kiểm tra lỗi real-word: từ hợp lệ nhưng có thể thiếu dấu.
        
        Ví dụ: "di" hợp lệ nhưng trong ngữ cảnh "tôi di học"
        thì "đi" có xác suất cao hơn nhiều.
        
        Chỉ gợi ý sửa nếu biến thể có dấu có điểm N-gram
        CAO HƠN ĐÁNG KỂ so với từ gốc (threshold > 1.5).
        """
        variants = set()
        
        # 1. Thay dấu thanh cho từng ký tự
        for i, char in enumerate(word):
            base = CHAR_TO_BASE.get(char, char)
            if base in TONE_VARIANTS:
                for variant in TONE_VARIANTS[base]:
                    if variant != char:
                        new_word = word[:i] + variant + word[i + 1:]
                        if new_word != word and self.dictionary.contains(new_word):
                            variants.add(new_word)

        # 2. Thay nhóm nguyên âm (a↔ă↔â, o↔ô↔ơ, u↔ư, e↔ê)
        VOWEL_GROUPS = ["aăâ", "oôơ", "uư", "eê"]
        for i, char in enumerate(word):
            base = CHAR_TO_BASE.get(char, char)
            for group in VOWEL_GROUPS:
                if base in group:
                    for alt_base in group:
                        if alt_base != base:
                            new_word = word[:i] + alt_base + word[i + 1:]
                            if new_word != word and self.dictionary.contains(new_word):
                                variants.add(new_word)

        # 3. Thay phụ âm đầu dễ nhầm (d↔đ, s↔x, ch↔tr, gi↔d, n↔l)
        CONSONANT_PAIRS = {
            "d": ["đ"], "đ": ["d"],
            "s": ["x"], "x": ["s"],
            "n": ["l"], "l": ["n"],
        }
        if word[0] in CONSONANT_PAIRS:
            for alt in CONSONANT_PAIRS[word[0]]:
                new_word = alt + word[1:]
                if new_word != word and self.dictionary.contains(new_word):
                    variants.add(new_word)
        # ch↔tr (2 ký tự đầu)
        if word[:2] == "ch" and len(word) > 2:
            new_word = "tr" + word[2:]
            if self.dictionary.contains(new_word):
                variants.add(new_word)
        elif word[:2] == "tr" and len(word) > 2:
            new_word = "ch" + word[2:]
            if self.dictionary.contains(new_word):
                variants.add(new_word)
        # gi↔d
        if word[:2] == "gi" and len(word) > 2:
            new_word = "d" + word[2:]
            if self.dictionary.contains(new_word):
                variants.add(new_word)
        elif word[0] == "d" and len(word) > 1:
            new_word = "gi" + word[1:]
            if self.dictionary.contains(new_word):
                variants.add(new_word)

        if not variants:
            return []

        # Tính điểm N-gram cho từ gốc
        original_score = self.ngram_model.score_word_in_context(
            word, prev_word, next_word
        )
        original_unigram = self.ngram_model.unigram_probability(word)

        # Tính điểm cho từng biến thể
        scored = []
        for v in variants:
            v_ngram = self.ngram_model.score_word_in_context(v, prev_word, next_word)
            v_unigram_prob = self.ngram_model.unigram_probability(v)
            v_score = 0.5 * v_ngram + 0.3 * math.log(v_unigram_prob) + 0.2 * (-1)

            # Gợi ý sửa nếu:
            # - Biến thể xuất hiện thường xuyên hơn ≥3 lần (ví dụ: "đi" >> "di")
            # - HOẶC bigram score tốt hơn đáng kể
            freq_ratio = v_unigram_prob / max(original_unigram, 1e-10)
            if freq_ratio >= 3.0:
                scored.append((v, v_score))
            elif v_ngram > original_score + 0.5:
                scored.append((v, v_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _rank_candidates(self, candidates: List[Tuple[str, int]],
                         prev_word: str, next_word: str = None) -> List[Tuple[str, float]]:
        """
        Xếp hạng ứng viên sửa lỗi dựa trên:
        1. Điểm N-gram (ngữ cảnh) - trọng số 0.5
        2. Điểm unigram (tần suất từ) - trọng số 0.3
        3. Khoảng cách chỉnh sửa (càng nhỏ càng tốt) - trọng số 0.2
        
        Args:
            candidates: Danh sách (ứng_viên, khoảng_cách)
            prev_word: Từ trước đó
            next_word: Từ sau đó
            
        Returns:
            Danh sách (ứng_viên, điểm_tổng_hợp) đã sắp xếp
        """
        if not candidates:
            return []

        scored = []
        for word, edit_dist in candidates:
            # Điểm N-gram (ngữ cảnh bigram)
            ngram_score = self.ngram_model.score_word_in_context(
                word, prev_word, next_word
            )

            # Điểm unigram (tần suất từ trong corpus)
            # Từ xuất hiện nhiều hơn trong corpus → điểm cao hơn
            unigram_prob = self.ngram_model.unigram_probability(word)
            unigram_score = math.log(unigram_prob) if unigram_prob > 0 else -20

            # Điểm edit distance (chuẩn hóa, càng nhỏ càng tốt)
            edit_score = -edit_dist  # Âm vì khoảng cách nhỏ = tốt

            # Điểm tổng hợp: kết hợp 3 yếu tố
            combined_score = 0.5 * ngram_score + 0.3 * unigram_score + 0.2 * edit_score
            scored.append((word, combined_score))

        # Sắp xếp giảm dần theo điểm
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def correct_text(self, text: str) -> str:
        """
        Tự động sửa lỗi chính tả trong văn bản.
        Chọn ứng viên có điểm cao nhất cho mỗi từ sai.
        
        Args:
            text: Văn bản cần sửa
            
        Returns:
            Văn bản đã được sửa lỗi
        """
        tokens = self._tokenize(text)
        errors = self.check_text(text)

        # Tạo mapping vị trí → từ sửa
        corrections = {}
        for error in errors:
            if error["suggestions"]:
                corrections[error["position"]] = error["suggestions"][0][0]

        # Thay thế từ sai bằng từ đúng
        result = []
        for i, token in enumerate(tokens):
            if i in corrections:
                result.append(corrections[i])
            else:
                result.append(token)

        return " ".join(result)

    def analyze(self, text: str) -> str:
        """
        Phân tích chi tiết lỗi chính tả trong văn bản.
        Trả về báo cáo dạng chuỗi.
        """
        errors = self.check_text(text)
        tokens = self._tokenize(text)

        lines = []
        lines.append("\n" + "=" * 60)
        lines.append("  KẾT QUẢ PHÂN TÍCH CHÍNH TẢ")
        lines.append("=" * 60)
        lines.append(f"\n📝 Văn bản gốc: \"{text}\"")
        lines.append(f"📊 Tổng số từ: {len([t for t in tokens if self._is_word_token(t)])}")
        lines.append(f"❌ Số lỗi phát hiện: {len(errors)}")

        if errors:
            lines.append("\n" + "-" * 40)
            lines.append("  CHI TIẾT CÁC LỖI")
            lines.append("-" * 40)

            for idx, error in enumerate(errors, 1):
                lines.append(f"\n🔴 Lỗi {idx}: \"{error['word']}\" (vị trí {error['position'] + 1})")

                # Ngữ cảnh
                pos = error["position"]
                context_start = max(0, pos - 2)
                context_end = min(len(tokens), pos + 3)
                context = tokens[context_start:context_end]
                context_str = " ".join(
                    f"[{t}]" if i + context_start == pos else t
                    for i, t in enumerate(context)
                )
                lines.append(f"   Ngữ cảnh: ...{context_str}...")

                if error["suggestions"]:
                    lines.append("   📋 Gợi ý sửa:")
                    for rank, (suggestion, score) in enumerate(error["suggestions"], 1):
                        lines.append(f"      {rank}. \"{suggestion}\" (điểm: {score:.4f})")
                else:
                    lines.append("   ⚠️  Không tìm thấy gợi ý phù hợp")

            # Văn bản đã sửa
            corrected = self.correct_text(text)
            lines.append("\n" + "-" * 40)
            lines.append(f"✅ Văn bản đã sửa: \"{corrected}\"")
        else:
            lines.append("\n✅ Không phát hiện lỗi chính tả!")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


# =====================================================================
# 6. ĐÁNH GIÁ MÔ HÌNH
# =====================================================================

def evaluate_model(checker: VietnameseSpellChecker):
    """
    Đánh giá hiệu quả của mô hình trên tập test.
    
    Metrics:
    - Accuracy: Tỷ lệ phát hiện đúng từ sai
    - Precision: Tỷ lệ từ được phát hiện sai thực sự sai
    - Recall: Tỷ lệ từ sai được phát hiện
    - Correction Accuracy: Tỷ lệ sửa đúng (top-1)
    """
    # Tập test: (câu_sai, câu_đúng, vị_trí_lỗi)
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

    print("\n" + "=" * 60)
    print("  ĐÁNH GIÁ MÔ HÌNH")
    print("=" * 60)

    total_errors = 0
    detected_correctly = 0
    corrected_correctly = 0

    for wrong_text, correct_text, error_positions in test_cases:
        total_errors += len(error_positions)
        errors = checker.check_text(wrong_text)
        
        wrong_tokens = checker._tokenize(wrong_text)
        correct_tokens = checker._tokenize(correct_text)

        for error in errors:
            pos = error["position"]
            if pos in error_positions:
                detected_correctly += 1
                # Kiểm tra sửa đúng
                if error["suggestions"] and pos < len(correct_tokens):
                    top_suggestion = error["suggestions"][0][0]
                    if top_suggestion == correct_tokens[pos].lower():
                        corrected_correctly += 1

    detection_recall = detected_correctly / total_errors if total_errors > 0 else 0
    correction_accuracy = corrected_correctly / total_errors if total_errors > 0 else 0

    print(f"\n📊 Kết quả đánh giá trên {len(test_cases)} câu test:")
    print(f"   Tổng số lỗi trong test set: {total_errors}")
    print(f"   Số lỗi phát hiện đúng:      {detected_correctly}")
    print(f"   Số lỗi sửa đúng (top-1):    {corrected_correctly}")
    print(f"\n   📈 Detection Recall:      {detection_recall:.2%}")
    print(f"   📈 Correction Accuracy:   {correction_accuracy:.2%}")
    print("=" * 60)


# =====================================================================
# 7. GIAO DIỆN TƯƠNG TÁC
# =====================================================================

def interactive_mode(checker: VietnameseSpellChecker):
    """Chế độ tương tác: người dùng nhập văn bản để kiểm tra."""
    print("\n" + "🇻🇳" * 20)
    print("\n  KIỂM TRA CHÍNH TẢ TIẾNG VIỆT - CHẾ ĐỘ TƯƠNG TÁC")
    print(f"\n{'🇻🇳' * 20}")
    print("\n  Nhập văn bản tiếng Việt để kiểm tra chính tả.")
    print("  Gõ 'demo' để xem các ví dụ minh họa.")
    print("  Gõ 'eval' để đánh giá mô hình.")
    print("  Gõ 'quit' hoặc 'exit' để thoát.\n")

    while True:
        try:
            text = input("📝 Nhập văn bản: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Tạm biệt!")
            break

        if not text:
            continue

        if text.lower() in ("quit", "exit", "thoát"):
            print("\n👋 Tạm biệt!")
            break

        if text.lower() == "demo":
            run_demo(checker)
            continue

        if text.lower() == "eval":
            evaluate_model(checker)
            continue

        # Phân tích và hiển thị kết quả
        result = checker.analyze(text)
        print(result)


def run_demo(checker: VietnameseSpellChecker):
    """Chạy demo với các ví dụ minh họa."""
    demo_texts = [
        "Tôi di học ở trương đại hoc",
        "Hôm nai trơi đep quá",
        "Chúng tôi dang hoc môn xử lý ngôn ngư tự nhiên",
        "Máy tín giúp con ngươi xử lý thông tin nhan chóng",
        "Tiêng Viêt là ngon ngữ có thanh điệu",
        "Em dang làm bài tâp về nhà",
        "Bao vệ môi trương là trách nihệm của mọi ngươi",
        "Kinh tê Viêt Nam dang phát triên mạnh",
    ]

    print("\n" + "🌟" * 20)
    print("\n  DEMO: KIỂM TRA CHÍNH TẢ TIẾNG VIỆT")
    print(f"\n{'🌟' * 20}")

    for i, text in enumerate(demo_texts, 1):
        print(f"\n{'━' * 60}")
        print(f"  VÍ DỤ {i}")
        print(f"{'━' * 60}")
        result = checker.analyze(text)
        print(result)

    print(f"\n{'━' * 60}")
    print("  KẾT THÚC DEMO")
    print(f"{'━' * 60}\n")


# =====================================================================
# 8. HÀM CHÍNH
# =====================================================================

def main():
    """Hàm chính - điểm bắt đầu chương trình."""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║   PHÁT HIỆN VÀ CHỈNH SỬA LỖI CHÍNH TẢ TIẾNG VIỆT      ║
    ║   Sử dụng Mô hình Ngôn ngữ Thống kê (N-gram)           ║
    ║                                                          ║
    ║   Môn: Xử lý Ngôn ngữ Tự nhiên (NLP)                   ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    # Khởi tạo bộ kiểm tra
    checker = VietnameseSpellChecker()

    # Chạy demo trước
    print("\n" + "─" * 60)
    print("  Chạy demo minh họa trước...")
    print("─" * 60)
    run_demo(checker)

    # Đánh giá mô hình
    evaluate_model(checker)

    # Chuyển sang chế độ tương tác
    interactive_mode(checker)


if __name__ == "__main__":
    main()
