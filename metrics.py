from skimage.metrics import structural_similarity as ssim
from lpips import lpips
import cv2

# Load your two images (replace 'image1_path' and 'image2_path' with the paths to your images)
image1 = cv2.imread('/data/home1/saichandra/Vardhan/projectAIP/pytorch-nerf/results/tiny_nerf/tiny_param_4/bike/test_img.png')  # Load your first image
image2 = cv2.imread('/data/home1/saichandra/Vardhan/projectAIP/pytorch-nerf/results/tiny_nerf/tiny_param_4/bike/iter_20000.png')  # Load your second image()

# Calculate SSIM
ssim_value, _ = ssim(image1, image2, full=True)

# Calculate LPIPS
loss_fn = lpips.LPIPS(net='alex')  # Choose the network architecture (e.g., 'alex' or 'vgg')
lpips_value = loss_fn(image1, image2)

print("SSIM:", ssim_value)
print("LPIPS:", lpips_value)
