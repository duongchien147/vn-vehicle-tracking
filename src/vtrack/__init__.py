"""Phát hiện, phân loại và theo dõi phương tiện giao thông Việt Nam từ video."""

CLASS_NAMES = ["motorbike", "car", "bus", "truck"]

# Vùng bỏ qua (theo quy ước DontCare của KITTI / iscrowd của COCO / ignored region của VisDrone): xe quá mơ hồ để
# gán nhãn chắc chắn (bị cây che gần hết...). Chỉ có trong nhãn gán tay; khi chấm điểm, phát hiện nằm trong vùng này
# không bị tính sai và bỏ sót nó cũng không bị trừ; khi train thì bỏ các box này. Id đứng sau 4 lớp chính.
IGNORE_LABEL = "ignore"
IGNORE_CLASS_ID = len(CLASS_NAMES)
