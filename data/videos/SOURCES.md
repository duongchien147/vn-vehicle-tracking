# Nguồn video

Tất cả clip là stock footage miễn phí, dùng theo license của nền tảng:
- Pexels License: https://www.pexels.com/license/ (dùng miễn phí, không bắt buộc ghi nguồn)
- Pixabay Content License: https://pixabay.com/service/license-summary/ (dùng miễn phí, không bắt buộc ghi nguồn)

Báo cáo vẫn nên ghi nguồn. Bản gốc 1080p/4K nằm trong `data/raw/`; bản làm việc 1280px trong `data/videos/`.

Cột "Trôi camera": dịch chuyển lớn nhất của khung hình so với frame đầu (ước lượng bằng optical flow + RANSAC, ở độ rộng 1280px). Camera cố định chuẩn ≈ 4px. Clip trôi nhiều không nên dùng để tính sai số đếm (vạch đếm cố định trên màn hình sẽ trượt so với mặt đường).

| File làm việc | Bối cảnh | Dài | Trôi camera | Dùng cho | Nguồn | Tác giả |
|---|---|---|---|---|---|---|
| nga_tu_ngang_tam_mat.mp4 | Ngã tư, ngang tầm mắt, ban ngày | 9.7s | ~15px (chân máy) | **Đánh giá đếm** | https://pixabay.com/videos/village-roas-street-traffic-181829/ | Pixabay |
| vong_xoay_cao.mp4 | Vòng xoay, nhìn từ trên cao, ban ngày | 11.5s | ~30px | **Đánh giá đếm**, demo | https://www.pexels.com/video/bustling-urban-traffic-at-busy-intersection-39119259/ | Phát (Pexels) |
| nga_tu_dem.mp4 | Ngã tư ban đêm, nhìn từ trên cao | 9.3s | ~86px | Đếm (thận trọng), cảnh đêm | https://www.pexels.com/video/bustling-night-traffic-in-busy-asian-city-36135626/ | J-Steve (Pexels) |
| hcm_duong_pho_cao.mp4 | Đường phố TP.HCM, nhìn từ trên cao, ban ngày | 15.0s | ~117px | Demo, tracking | https://www.pexels.com/video/traffic-on-street-13275193/ | Pew (Pexels) |
| nga_tu_drone_tren_xuong.mp4 | Ngã tư, drone nhìn thẳng xuống | 10.6s | ~98px | Demo, xe nhỏ | https://pixabay.com/videos/roads-cars-traffic-intersection-220451/ | Pixabay |
| ket_xe_cao.mp4 | Kẹt xe giờ cao điểm, nhìn từ trên cao | 7.9s | ~277px | Tracking (che khuất dày) | https://www.pexels.com/video/dense-urban-traffic-jam-with-motorbikes-and-cars-39221451/ | Phát (Pexels) |
| nga_tu_dem_mua.mp4 | Ngã tư TP.HCM ban đêm, mưa, ngang tầm mắt | 15.4s | lớn (cầm tay) | Phân tích lỗi (đêm + mưa) | https://www.pexels.com/video/busy-night-intersection-in-ho-chi-minh-city-33486917/ | R (Pexels) |
| quoc_lo_drone.mp4 | Quốc lộ 1, drone bay | 20.0s | lớn (drone bay) | Phân tích lỗi (xe tải/bus, xe nhỏ) | https://www.pexels.com/video/aerial-view-of-traffic-on-busy-vietnamese-highway-32971521/ | Duy (Pexels) |

## Video YouTube (license YouTube chuẩn — chỉ dùng cho mục đích học tập, ghi nguồn, không đăng lại)

Bản gốc 1080p do người làm đồ án tự tải (ytsave), lưu ở `data/raw/youtube_<id>*.mp4`. Mọi clip dưới đây dài 60s,
1920x1080, cắt từ đoạn camera đứng yên do `scripts/check_videos.py` tìm ra (ổn định 98%, 0 lần cắt cảnh).

| File làm việc | Bối cảnh | Cắt từ | Dùng cho | Nguồn |
|---|---|---|---|---|
| nga_tu_hanoi_a.mp4 | Ngã tư Hà Nội, nhìn chéo từ tầng cao, ban ngày, xe dày 4 hướng | 00:35–01:35, 25fps | **Test** (không bao giờ train) | https://www.youtube.com/watch?v=Lzds95Kfh54 — Raw Street Capture 101 |
| nga_tu_hanoi_b.mp4 | Như trên | 01:45–02:45, 25fps | **Chỉ minh hoạ, KHÔNG dùng đánh giá**: trùng đoạn lấy 10 frame val (1:45–2:39) | như trên |
| nga_tu_hanoi_c.mp4 | Như trên | 05:45–06:45, 25fps | **Chỉ minh hoạ, KHÔNG dùng đánh giá**: nằm trong đoạn lấy frame train (5:33–7:28) — mô hình đã học cảnh này | như trên |
| pvd_goc_a.mp4 | Đại lộ Phạm Văn Đồng (TP.HCM), buổi chiều, góc A (nhìn dọc đại lộ, đông xe) | 2:05:00–2:06:00 | **Chỉ test** | https://www.youtube.com/watch?v=VswAfC4fER0 — Vietnam Naturally |
| pvd_goc_b.mp4 | Như trên, góc B (chiều ngược lại, lá cây che mép dưới, lóa nắng) | 2:25:00–2:26:00 | **Chỉ test** | như trên |
| pvd_goc_c.mp4 | Như trên, góc C (nhìn về giao lộ, nhiều xe máy) | 1:50:00–1:51:00 | **Chỉ test** | như trên |
| ham_chui_dem_1.mp4 | Lối vào hầm chui TP.HCM, ban đêm, một góc cố định suốt 3 giờ | 0:10:00–0:11:00 | **Chỉ test** (cảnh đêm) | https://www.youtube.com/watch?v=-Ta6lRLNnnE — Midnight Soundscape |
| ham_chui_dem_2.mp4 | Như trên | 1:20:00–1:21:00 | **Chỉ test** | như trên |
| ham_chui_dem_3.mp4 | Như trên | 2:40:00–2:41:00 | **Chỉ test** | như trên |

Ghi chú về video gốc:
- Hà Nội (9:14): có zoom quanh 02:50, 04:30, 05:30, 07:40, 08:20, 08:50; đoạn ổn định ≥60s: 00:28–02:56, 02:56–04:32, 05:31–07:30.
  Bản 360p cũ (tải bằng yt-dlp) được chuyển sang `data/raw/old_360p/`, không dùng nữa.
- Phạm Văn Đồng (2:32:57): 3 góc máy xen kẽ, đổi góc ở khoảng 0:19:55, 0:24:37, 0:35:48, 0:56:10, 1:07:28–1:08:08,
  1:39:39, 1:47:17, 1:54:40–1:55:00, 1:56:25, 2:21:02. Góc A: 0:00–0:24, 0:56–1:07, 1:56–2:21; góc B: 0:24–0:56,
  1:08–1:47, 2:21–2:33; góc C: 1:47–1:56.
- Hầm chui (3:10:00): ổn định 100%. Đã kiểm tra KHÔNG phải video lặp (frame cách nhau 78/60/45 phút khác nhau
  ~2.7–3.8% điểm ảnh, bằng mức của hai thời điểm bất kỳ).
- Video phố cổ Hà Nội https://www.youtube.com/watch?v=-g79sgafH1o (Harry Mateman) **không dùng**: lia/zoom liên tục,
  hơn 10 lần cắt cảnh.

## Clip dự phòng

`data/raw/` còn giữ các clip dự phòng chưa dùng: pixabay_32499 (37s, cận cảnh, cầm tay), pixabay_220452/220453/220480 (trôi nhiều hoặc video dọc), vehicles.mp4 (cao tốc nước ngoài, nguồn của sample_8s.mp4).
