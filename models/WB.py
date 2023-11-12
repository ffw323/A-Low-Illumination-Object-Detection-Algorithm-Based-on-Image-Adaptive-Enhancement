import cv2
import numpy as np

# 读取图像
image = cv2.imread('img.png')

# 色调映射处理
def color_mapping(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return result

color_mapped = color_mapping(image)

# 伽马校正处理
def gamma_correction(image, gamma):
    result = np.power(image / 255.0, gamma)
    result = (result * 255).astype(np.uint8)
    return result

gamma = 1.5
gamma_corrected = gamma_correction(color_mapped, gamma)

# 对比度增强处理
def contrast_enhancement(image, alpha, beta):
    result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return result

alpha = 1.5
beta = 0
contrast_enhanced = contrast_enhancement(gamma_corrected, alpha, beta)

# 锐化处理
def sharpen(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    result = cv2.filter2D(image, -1, kernel)
    return result

sharpened = sharpen(contrast_enhanced)

# 保存输出图像
cv2.imwrite('color_mapped.jpg', color_mapped)
cv2.imwrite('gamma_corrected.jpg', gamma_corrected)
cv2.imwrite('contrast_enhanced.jpg', contrast_enhanced)
cv2.imwrite('sharpened.jpg', sharpened)

# 显示输出图像
cv2.imshow('Color Mapped', color_mapped)
cv2.imshow('Gamma Corrected', gamma_corrected)
cv2.imshow('Contrast Enhanced', contrast_enhanced)
cv2.imshow('Sharpened', sharpened)
cv2.waitKey(0)
cv2.destroyAllWindows()
