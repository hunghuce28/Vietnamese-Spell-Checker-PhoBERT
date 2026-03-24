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

## 🏗️ Kiến trúc Hệ Thống

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
│   Chấm Điểm và Xếp Hạng qua PhoBERT              │
│   - Đặt MASK vào câu: "... đi [MASK] ở chường"   │
│   - Yêu cầu PhoBERT (20GB Text) dự đoán xác suất │
│   - Áp dụng Edit Distance Penalty (Phạt điểm)    │
│   - Cập nhật Context từ Trái sang Phải           │
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

### 2. Phạt Khoảng Cách (Edit Distance Penalty)
Trong Non-word Error, điểm của PhoBERT sẽ bị trừ đi một lượng tương ứng với `Edit_Distance * 1.5`. 
VD: Để sửa chữ `toi` thành `tôi` (khoảng cách = 1) sẽ được ưu tiên cao hơn và dễ thắng chữ `thi` (khoảng cách = 2) dù PhoBERT có cho điểm Logits chữ `thi` cao hơn.

### 3. Sửa Lỗi Ngữ Cảnh Cuốn Chiếu (Left-to-Right Context Updating)
Nâng cấp cốt lõi so với N-Gram: Khi một từ sai ở vị trí (i) được sửa thành từ đúng, nó sẽ **Ngay lập tức** được lấp vào danh sách Token để làm "tọa độ ngữ cảnh" soi tiếp chữ sai ở vị trí (i+1). Tính năng này trị hoàn toàn lỗi "chập toàn câu" - khi liên tiếp nhiều từ bị viết sai cạnh nhau!

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
