import cv2
from basicsr import calculate_psnr
import numpy as np
from basicsr.utils.registry import METRIC_REGISTRY
# 读取原始图像和重建图像
img = cv2.imread('C://Users//leeseo//Desktop//MAN-main//test//HR//3606.png')
img2 = cv2.imread('C://Users//leeseo//Desktop//shiyan data//A7.12 shiyan//test//LDMN//3606.png')  #'C://Users//leeseo//Desktop//MAN-main//test//Paied//SRRES//3694.png'
#img2 = cv2.imread('C://Users//leeseo//Desktop//MAN-main//test//car_x4//3603.png')1srresnetsand_x2
PSNR = calculate_psnr(img, img2, crop_border=2)
# 计算原始图像和重建图像之间的 MSE

print("PSNR",PSNR)

#import cv2

# 读取原始图像和重建图像
#original_img = cv2.imread('C://Users//leeseo//Desktop//MAN-main//test//HR//3620.png')
#reconstructed_img = cv2.imread('C://Users//leeseo//Desktop//MAN-main//test//Paied//LDMN//3620.png')

# 计算原始图像和重建图像之间的 MSE
#mse = np.mean((original_img - reconstructed_img) ** 2)

# 如果图像像素的范围是 [0, 255]，则 PSNR 的峰值为 255
#psnr = 10 * np.log10((255 ** 2) / mse)

#print(f"PSNR 值为: {psnr} dB")
