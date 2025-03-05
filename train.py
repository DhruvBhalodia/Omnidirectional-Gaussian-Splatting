#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import random
import torch.nn.functional as F
from utils.loss_utils import l1_loss, ssim, est_wsmap
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
import time
from tqdm import tqdm
from utils.image_utils import psnr
from lpipsPyTorch import lpips
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import save_image
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import warnings
warnings.filterwarnings('ignore')


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", ncols=120)
    first_iter += 1
    time_record = 0
    check_10m, check_100m = 1, 1
    for iteration in range(first_iter, opt.iterations + 1):        
        pvt = time.time()
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        psi = render_pkg["psi"]
        lat, lon = render_pkg["lat"], render_pkg["lon"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        ssim_map = ssim(image, gt_image, return_map=True)  
        ssim_error = 1.0 - ssim_map  # Convert SSIM into an error map

        # Compute final loss
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * torch.mean(ssim_error)
        loss.backward()

        # Map SSIM error to Gaussians
        ssim_error_per_gaussian = map_ssim_to_gaussians(ssim_error, gaussians.get_xyz, viewpoint_cam)

        iter_end.record()
        cvt = time.time()
        time_record = time_record + (cvt-pvt)
        
        # Temporally add testing_iterations if time > 10min, time > 100min
        if time_record > 600 and check_10m == 1:
            testing_iterations += [iteration]
            saving_iterations += [iteration]
            check_10m = 0
        if time_record > 6000 and check_100m == 1:
            testing_iterations += [iteration]
            saving_iterations += [iteration]
            check_100m = 0
        
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "#pts": gaussians._xyz.shape[0]})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
    
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            # Densification
            pvt = time.time()

            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(ssim_error_per_gaussian, 0.005, scene.cameras_extent + 100, size_threshold, lat)
            
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            
            # Optimizer step
            if iteration < opt.iterations and (iteration+1) % 1 == 0:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            cvt = time.time()
            time_record = time_record + (cvt-pvt)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        if check_100m == 0:
            break


def project_gaussians_to_2D(gaussians_xyz, viewpoint_camera):
    # Get the view-projection matrix from the camera
    view_proj_matrix = viewpoint_camera.view_projection_matrix  # Shape: [4, 4]
    
    # Convert 3D points to homogeneous coordinates (N, 4)
    ones = torch.ones((gaussians_xyz.shape[0], 1), device=gaussians_xyz.device)
    gaussians_homogeneous = torch.cat([gaussians_xyz, ones], dim=-1)  # Shape: [N, 4]
    
    # Apply the view-projection transformation
    projected = torch.matmul(gaussians_homogeneous, view_proj_matrix.t())  # Shape: [N, 4]
    
    # Normalize by the w component to get 2D coordinates
    projected_2D = projected[:, :2] / projected[:, 3:4]  # Shape: [N, 2]
    
    return projected_2D

def map_ssim_to_gaussians(ssim_error, gaussians_xyz, viewpoint_camera):
    # Project 3D Gaussian positions to 2D normalized coordinates (range: [-1, 1])
    projected_2D = project_gaussians_to_2D(gaussians_xyz, viewpoint_camera)  # Shape: [N, 2]
    
    # Get the height and width of the SSIM error map
    H, W = ssim_error.shape[-2:]
    
    # Convert normalized coordinates to pixel coordinates
    projected_2D_pixel = projected_2D.clone()
    projected_2D_pixel[:, 0] = (projected_2D[:, 0] + 1) * (W - 1) / 2.0
    projected_2D_pixel[:, 1] = (projected_2D[:, 1] + 1) * (H - 1) / 2.0
    
    # Normalize pixel coordinates back to [-1, 1] (required for grid_sample)
    norm_x = (projected_2D_pixel[:, 0] / (W - 1)) * 2 - 1
    norm_y = (projected_2D_pixel[:, 1] / (H - 1)) * 2 - 1
    grid = torch.stack((norm_x, norm_y), dim=-1)  # Shape: [N, 2]
    
    # Reshape grid to [1, N, 1, 2] for grid_sample
    grid = grid.unsqueeze(0).unsqueeze(2)
    
    # Prepare the SSIM error map with shape [1, 1, H, W]
    ssim_error = ssim_error.unsqueeze(0).unsqueeze(0)
    
    # Sample the SSIM error map at the Gaussian positions using bilinear interpolation
    ssim_error_per_gaussian = F.grid_sample(ssim_error, grid, mode='bilinear', align_corners=True)
    
    # Remove extra dimensions to obtain a tensor of shape [N]
    ssim_error_per_gaussian = ssim_error_per_gaussian.squeeze()
    
    return ssim_error_per_gaussian

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},)
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test, wspsnr_test, ssim_test, wsssim_test, lpips_test = 0.0, 0.0, 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    ws_map = est_wsmap(image)
                    
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    test_psnr, test_wspsnr = psnr(image, gt_image, ws_map)
                    test_ssim, test_wsssim = ssim(image, gt_image, ws_map=ws_map)
                    psnr_test += test_psnr.mean().double()
                    wspsnr_test += test_wspsnr.mean().double()
                    ssim_test += test_ssim.mean().double()
                    wsssim_test += test_wsssim.mean().double()
                    lpips_test += lpips(image, gt_image).mean().double()

                    render_image_path = os.path.join(scene.model_path, "render")
                    if not os.path.exists(render_image_path):
                        os.mkdir(render_image_path)
                    if config['name'] == 'train' or idx % 5 == 0:
                        save_image(image, os.path.join(render_image_path, "iteration_{}_{}_{}.png".format(config['name'], iteration, viewpoint.image_name)))
                        save_image(gt_image, os.path.join(render_image_path, "gt_{}_{}.png".format(config['name'], viewpoint.image_name)))

                psnr_test /= len(config['cameras'])
                wspsnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                wsssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print(f"\n[ITER {iteration}] Eval {config['name']}: \tL1 {l1_test:.5f} PSNR {psnr_test:.3f} SSIM {ssim_test:.4f} LPIPS {lpips_test:.4f} #Points {scene.gaussians._xyz.shape[0]:d}")
                with open(os.path.join(scene.model_path, f"eval_{config['name']}.txt"), "a") as f:
                    f.write(f"{iteration}, {psnr_test:.4f}, {ssim_test:.4f}, {lpips_test:.4f}\n")
                scene.model_path
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000, 7000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    pvt = time.time()
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    print(f"Learning Gaussian {time.time()-pvt}")

    # All done
    print("\nTraining complete.")
