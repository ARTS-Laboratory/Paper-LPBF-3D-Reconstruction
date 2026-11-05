import numpy as np
import cv2

# Original BIN
with open("raw/optical_90983.bin", "rb") as f:
    height, width = np.fromfile(f, dtype="<u4", count=2)
    original = np.fromfile(f, dtype="<u2").reshape(height, width)

# Converted PNG
png = cv2.imread(
    "processed/optical images/optical_90983.png",
    cv2.IMREAD_UNCHANGED
)

print("PNG dtype:", png.dtype)
print("PNG shape:", png.shape)
print("Exact pixel match:", np.array_equal(original, png >> 4))
print("Maximum difference:", np.max(
    np.abs(original.astype(np.int32) - (png >> 4).astype(np.int32))
))