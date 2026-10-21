# Kết quả rà soát xe qua vạch — 3 clip hầm chui đêm

Đếm trực quan từ video gốc theo tài liệu `huong_dan_dem_tay.md`. Đây là bộ kết quả do AI rà ảnh, có nhật ký và ảnh đối chiếu; chưa phải nhãn chuẩn đã được người kiểm duyệt xác nhận. Các file đếm cũ trong dự án không được đọc hoặc ghi đè.

| Clip | Xe máy | Ô tô | Buýt | Tải | Tổng đã ghi | Chờ xác nhận |
|---|---:|---:|---:|---:|---:|---:|---:|
| ham_chui_dem_1 | 59 | 9 | 0 | 1 | **69** | 0 |
| ham_chui_dem_2 | 72 | 6 | 0 | 1 | **79** | 0 |
| ham_chui_dem_3 | 75 | 17 | 1 | 2 | **95** | **1 xe máy** |

Tất cả các lượt đã ghi đi lên trên ảnh, tương ứng `in` với vạch mặc định. `out = 0`. Nhánh đường trái/phải trong nhật ký chỉ là vị trí trên ảnh, không phải chiều vào/ra.

## Cách tính xe ở đầu và cuối clip

- Vạch nằm ở `y = 0.6 × 1080 = 648 px`, chạy ngang từ trái sang phải.
- Chỉ ghi một lượt khi **điểm giữa đáy xe** đi từ một phía sang phía kia của vạch. Xe máy chở nhiều người vẫn là một xe.
- Xe có đầu/thân nằm trên vạch ở đầu clip vẫn có thể được tính: xem đáy xe đã qua chưa. Nếu đáy còn ở dưới vạch và sau đó đi lên qua vạch trong clip thì tính.
- Nếu đáy xe đã nằm phía trên vạch ngay khi clip bắt đầu, không cộng chỉ vì xe đang xuất hiện trong hình. Xe chưa qua khi clip kết thúc cũng không cộng.
- Xe bị lan can che vẫn tính khi đủ bằng chứng trước/sau rằng xe đã qua trong clip; thời điểm được đánh dấu là ước lượng.
- Hai xe bán tải ở clip 3 được xếp `truck` theo tiêu chí có thùng hàng riêng của tài liệu. Nếu bộ nhãn của bạn dùng quy ước khác cho bán tải, cần đổi lớp nhất quán, tổng xe không đổi.

## Những trường hợp cần biết trước khi dùng số liệu

**Clip 1 nhảy cảnh tại frame 1448, khoảng 48.267 giây.** Frame 1447 và 1448 không nối tiếp chuyển động bình thường. Xe cứu hộ bên trái, ô tô trắng và xe máy áo đỏ sẫm bên phải còn chưa qua vạch ở cảnh trước rồi biến mất khi cắt. Không cộng ba xe đó bằng cách suy diễn chuyển động qua đoạn bị mất. Ô tô đang được chở trên xe cứu hộ không tính thành một lượt lưu thông riêng. Hai xe máy ở cảnh sau được theo dõi lại; xe bên phải qua khoảng 48.4 giây đã được tính. Xem [ảnh điểm cắt](evidence/clip1_cut_48s.jpg). Vì có điểm cắt, đây là số lượt quan sát được trong file, không đại diện một phút giao thông liên tục.

**Clip 1 và clip 2 trùng nội dung:** frame 412–1799 của clip 2 tương ứng frame 0–1387 của clip 1, lệch **412 frame = 13.733333 giây**. Đã đối chiếu hình ảnh và chữ ký ảnh của toàn bộ 1.388 cặp frame; sai khác nhỏ phù hợp việc mã hóa video lại. Clip 2 gồm 27 lượt ở đoạn đầu và 52 lượt thuộc đoạn trùng. Các lượt trùng được chuyển mốc chính xác +412 frame từ nhật ký clip 1, không phải một lần đếm độc lập. **Không cộng ba tổng thành số xe duy nhất.**

**Xe máy sát lan can ở đầu clip 1:** đối chiếu đoạn tương ứng trong clip 2 cho thấy xe đã qua hoặc vừa qua trước điểm bắt đầu clip 1. Không ghi lại vào clip 1; ghi ở clip 2 khoảng 13.63 giây. Bánh sau bị che nên thời điểm là suy đoán; đây là một quyết định ở ranh giới clip cần ưu tiên kiểm tra nếu làm nhãn chuẩn. Xem [ảnh đối chiếu đầu clip](evidence/initial_boundary.jpg).

**Một lượt chưa chốt ở cuối clip 3:** xe máy chở hai người, áo tối, đi sát lan can nhánh trái vào khoảng 59.9 giây. Điểm đáy bị che và sát ranh giới kết thúc nên chưa đủ chắc đã qua trước frame 1799. Lượt `C3_096` nằm trong `pending_review.json`, **không nằm trong `manual_counts.csv`**. Nếu xác nhận có qua thì clip 3 thành **96 xe**, trong đó 76 xe máy. Xem [ảnh cuối clip 3](evidence/clip3_last_frames.jpg). Xe máy đeo ba lô ở nhánh phải cuối clip 3 cũng chưa qua vạch, được loại; xem [ảnh nhánh phải](evidence/clip3_right_not_crossed.jpg).

## Mức độ kiểm tra và giới hạn thời gian

Đã rà chuỗi ảnh gốc cách nhau 0.5 giây ở toàn bộ clip 1, clip 3 và phần đầu không trùng của clip 2; sau đó rà từng lượt bằng ảnh trước/sau mốc khoảng 0.25 giây. Những điểm đầu/cuối và đoạn cắt được mở dày hơn, có kiểm tra tới các frame liền kề. Phần trùng clip 2 được đối chiếu với clip 1 như mô tả trên. Không dùng số đếm hoặc video có khung nhận diện của mô hình để suy ra kết quả.

`frame` và `time_s` trong CSV là **mốc ước lượng được quy về frame video**, không phải khẳng định đã tìm chính xác frame đầu tiên qua vạch. Nhiều lượt có thể lệch vài phần mười giây; trường hợp che khuất hoặc xe đi chậm có thể lệch hơn. Ảnh trước/sau là phương tiện đối chiếu, không phải khoảng sai số đã được kiểm định. Khi đánh giá sự kiện với ngưỡng thời gian rất hẹp, cần tinh chỉnh mốc trên video trước.

## Các file bàn giao

Mỗi thư mục `ham_chui_dem_N` có:

- `manual_counts.csv`: bốn cột `frame,time_s,class_name,direction`, cùng cấu trúc đầu ra công cụ đếm của dự án.
- `count_config.json`: nguồn video, FPS, vạch và điểm neo; có ghi rõ phương pháp AI và `human_verified: false`.
- `nhat_ky.md`: nhật ký từng lượt để đọc nhanh.
- `audit.json`: mô tả xe, nhánh đường, cờ che khuất, ảnh đối chiếu và nguồn chuyển mốc của đoạn trùng.

`evidence/review_cN_XX.jpg` là các tờ đối chiếu, mỗi dòng một lượt, ba ảnh quanh mốc đã ghi. Với clip 2, ảnh mang tên `review_c2` chỉ bao gồm 27 lượt đầu; các lượt còn lại tham chiếu ảnh clip 1 trong `audit.json`. Ảnh clip 3 vẫn giữ ứng viên số 96 để bạn kiểm tra, dù CSV chính đã loại lượt này.

`summary.json` chứa tổng theo lớp; `pending_review.json` chứa lượt chưa chốt. Đã kiểm tra lại số dòng CSV, thứ tự thời gian, miền frame 0–1799, nhãn lớp/chiều và quan hệ `time_s = frame / 30`. Chưa chạy đối chiếu với pipeline hoặc nhập kết quả vào thư mục nhãn chuẩn của dự án.
