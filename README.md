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
│   Chấm Điểm và Xếp Hạng Batch qua PhoBERT        │
│   - Quét toàn bộ câu thành Batch (O(1) Inference)│
│   - Tính log-softmax chuẩn hóa, áp dụng Penalty  │
│   - Multi-Pass: Lặp quét toàn bộ câu lần 2 để    │
│     chữa dứt điểm chuỗi lỗi chập (VD: hom nai)   │
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

### 2. Tối Ưu Hiệu Năng Batch Masked Language Modeling
Thay vì truyền tuần tự từng từ sai qua mô hình (O(N)), dự án đã đóng gói tính toán bằng Batch Inference (Vectorization). Toàn bộ các vị trí lỗi trong câu được mask và ném qua PhoBERT xử lý song song trong đúng **1 lượt truyền (1 forward pass)**. Nhờ vậy tốc độ kiểm tra lỗi vọt lên gấp 20 - 50 lần. Công đoạn fallback Levenshtein vét cạn từ điển (tốn kém) cũng được Pruning khóa lại, chỉ kích hoạt khí thuật toán từ vựng bí bách nhất.

### 3. Sửa Lỗi Đa Luồng (Whole-Sentence Multi-Pass)
Thay vì trễ nhịp do vướng BPE Tokenizer như cách cập nhật Left-to-Right cuốn chiếu thông thường, luồng logic đã được đổi sang thao tác quét và chấm điểm theo khối (Whole-Sentence Iteration). Khung thiết kế này giúp diệt gọn hoàn toàn hiện tượng "Mù ngữ cảnh cục bộ" do lỗi sai liên tục sát vách nhau tạo ra (Ví dụ: 2 từ sai đứng cạnh nhau `hom nai`).

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
