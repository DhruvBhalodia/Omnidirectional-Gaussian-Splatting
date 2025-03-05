<p align="center">
  <h1 align="center">Omnidirectional Dynamic Gaussian Splatting (ODGS) - SSIM-Based Adaptive Densification</h1>
</p>


---
This project is an improvement of ["ODGS: 3D Scene Reconstruction from Omnidirectional Images with 3D Gaussian Splatting."](https://arxiv.org/abs/2410.20686)

---

## **📌 Project Overview**
This project improves **Omnidirectional Dynamic Gaussian Splatting (ODGS)** by introducing **SSIM-based Adaptive Densification**. Instead of using gradients to decide where to add new Gaussians, we now use **SSIM (Structural Similarity Index)** error maps to focus densification in **blurry or missing detail areas**, resulting in **higher-quality 3D reconstruction**.

---

## **What Has Been Done?**
### **Original Method (Before Improvement)**
- Used **gradient-based densification** to add Gaussians where gradients were high.
- This method was **not directly related to image quality** and sometimes added unnecessary Gaussians.
  
### **Improvements (After Modification)**
- **SSIM-based Adaptive Densification** replaces gradient-based densification.
- Gaussians are now added where **SSIM error is high**, ensuring sharper edges and better details.
- **Faster convergence** and **higher quality** 3D reconstructions.

---

## Installation
~~~bash
git clone https://github.com/DhruvBhalodia/Omnidirectional-Gaussian-Splatting.git

# Set Environment
conda env create --file environment.yaml
conda activate ODGS
pip install submodules/simple-knn
pip install submodules/odgs-gaussian-rasterization
~~~

## Dataset
The dataset used in this project can be found at the following link:  
[ODGS Dataset](https://drive.google.com/drive/folders/1xLdy0Zh6K1vAN_WpTWg4RTTUPxxv8RFp?usp=sharing)


## Training (Optimization)
ODGS requires optimization for each scene. Run the script below to start optimization:
~~~python
python train.py -s <source(dataset)_path> -m <output_path> --eval
~~~

### Rendering a Scene
To render images from trained Gaussians:
```bash
python render.py --model_path <output_model_path> --iteration <iteration_number>
```

### Visualizing 3D Scene
To export Gaussians as a `.ply` file for 3D visualization:
```bash
python export_ply.py --model_path <output_model_path> --iteration <iteration_number>
```

---

## How the New Densification Works
### **Step 1: Compute SSIM error map**
```python
ssim_map = ssim(image, gt_image, return_map=True)
ssim_error = 1.0 - ssim_map  # Convert SSIM to an error map
```

### **Step 2: Map 2D SSIM errors to 3D Gaussians**
```python
ssim_error_per_gaussian = map_ssim_to_gaussians(ssim_error, gaussians.get_xyz, viewpoint_cam)
```

### **Step 3: Use SSIM error instead of gradients to decide where to add Gaussians**
```python
threshold = torch.mean(ssim_error_per_gaussian)
selected_pts_mask = ssim_error_per_gaussian > threshold  # Only add Gaussians in high-error areas
```

### **Step 4: Modify `densify_and_prune()` to use SSIM errors**
```python
self.densify_and_prune(ssim_error_per_gaussian)
```
---

## **📌 Credits & Original Work**
This project is based on the original work by **Inria GRAPHDECO Research Group**.
Official Repository: [Original GitHub Repository](https://github.com/esw0116/ODGS) 

The improvements in this project introduce **SSIM-based Adaptive Densification**, replacing the original gradient-based densification method.

---

## **📌 Contributors**
- **Dhruv Bhalodia**
- **Supervisor: Dr. Deepika Gupta**  
