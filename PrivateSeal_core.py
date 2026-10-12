import torch
import cv2
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 
os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache" 

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.profiler
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
from diffusers import StableDiffusionXLImg2ImgPipeline
import kornia.augmentation as K
import bchlib
import time
# --- 每次改算法只改这里 ---
EXP_NAME = "final_version" 
# -----------------------

timestamp = time.strftime("%Y%m%d_%H%M")
save_root = f"experiments/{timestamp}_{EXP_NAME}"
os.makedirs(save_root, exist_ok=True)

log_save_path = os.path.join(save_root, "detailed_log.csv")
plot_save_path = os.path.join(save_root, "result_histogram.png")
bad_cases_dir = os.path.join(save_root, "bad_cases")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_id = 'runwayml/stable-diffusion-v1-5'


aug_pipeline = K.AugmentationSequential(
    # K.RandomGaussianBlur(kernel_size=(5,5), sigma=(0.1,2.0), p=0.5),
    K.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), p=1),
    data_keys=["input"],
    same_on_batch=False
)


k_enlarge = 100
def cos_correlation(vec1, vec2):
    vec1 = vec1.flatten().float()
    vec2 = vec2.flatten().float()
    return torch.dot(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2) + 1e-6) * k_enlarge


import lpips
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

class Evaluator:
    def __init__(self, device='cuda'):
        self.device = device  
        self.lpips = lpips.LPIPS(net='alex').to(device).eval()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)

    
    def evaluate(self, img_original, img_watermarked):
        with torch.no_grad():
            img_orig = img_original.to(self.device)
            img_wm = img_watermarked.to(self.device)

            img_orig_norm = img_orig * 2 - 1
            img_wm_norm = img_wm * 2 - 1  
            
            lpips_score = self.lpips(img_orig_norm, img_wm_norm).item()

            psnr_score = self.psnr(img_orig, img_wm).item() 
            ssim_score = self.ssim(img_orig, img_wm).item()
        
        return {
            "PSNR": round(psnr_score, 2),
            "SSIM": round(ssim_score, 4), 
            "LPIPS": round(lpips_score, 4)
        } 


class DiffJPEG(nn.Module):
    def __init__(self, height=512, width=512, differentiable=True, quality=50):
        super(DiffJPEG, self).__init__()
        self.differentiable = differentiable
        self.quality = quality
        self.height = height
        self.width = width

    def forward(self, x):
        
        if self.differentiable:
           
            factor = self.quality / 100.0 
            if factor > 0.8: scale = 1.0
            else: scale = max(0.1, factor) 
            
            h_low = int(self.height * scale)
            w_low = int(self.width * scale)
            

            x_down = F.interpolate(x, size=(h_low, w_low), mode='bilinear', align_corners=False)
            x_up = F.interpolate(x_down, size=(self.height, self.width), mode='bilinear', align_corners=False)
            
            noise = (torch.rand_like(x) - 0.5) * (1.0 - factor) * 0.5
            
            b, c, h, w = x.shape
            block_n = (torch.rand((b, c, h//8, w//8), device=x.device, dtype=x.dtype) - 0.5)
            block_noise = F.interpolate(block_n, size=(h, w), mode='nearest') * (1.0 - factor) * 0.5
            
            return x_up + noise + block_noise
        else:
            return x

class StableDiffusionAttacker:
    def __init__(self):
        self.device = device
        print(f"Model loading:{model_id}.")
        
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype = torch.bfloat16,
            safety_checker = None,          
            requires_safety_checker = False
        ).to(device)
        
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

        self.vae.eval()
        self.unet.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.unet.parameters():
            param.requires_grad = False

        print("Model loaded successfully.")

        self.jpeg_simulator = DiffJPEG(height=512, width=512, quality=50).to(device)
        print("JPEG Simulator loaded.")
        
        self.evaluator = Evaluator(device=device)
        self.JPEG_Trainer = DiffJPEG()
        
        self.loss_lpips_func = lpips.LPIPS(net='vgg').to(device).eval() 

        for param in self.loss_lpips_func.parameters():
            param.requires_grad = False  

        
        # print("Loading SD 2.1 for tampering...")
        # self.sd21_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        #     "Manojb/stable-diffusion-2-1-base", 
        #     torch_dtype=torch.bfloat16,
        #     safety_checker=None,
        #     requires_safety_checker=False
        # ).to(device)

        # self.sd21_vae = self.sd21_pipe.vae
        # self.sd21_unet = self.sd21_pipe.unet
        # self.sd21_scheduler = DDIMScheduler.from_config(self.sd21_pipe.scheduler.config)

        # self.sd21_vae.eval()
        # self.sd21_unet.eval()
        # for param in self.sd21_vae.parameters(): param.requires_grad = False
        # for param in self.sd21_unet.parameters(): param.requires_grad = False

        # print("SD 2.1 loaded successfully.")
        
        #self.unet.enable_gradient_checkpointing()       
        #self.sdxl_unet.enable_gradient_checkpointing()
        
        # self.m_bits = m_bits
        # self.D_latent = 4*64*64
        
        # if not os.path.exists(key_path):
        #     raise FileNotFoundError(f"Private key not found at {key_path}, Please generate it first.")

        # self.P = torch.load(key_path, map_location=self.device)
        # self.P.requires_grad = False

        # assert self.P.shape == (self.m_bits, self.D_latent), "Key dimensions mismatch!"
    
    def set_key(self, key_path, m_bits):
        self.m_bits = m_bits
        self.D_latent = 4 * 64 * 64
        
        if not os.path.exists(key_path):
            raise FileNotFoundError(f"🚨 Private key not found at {key_path}, Please generate it first.")

        self.P = torch.load(key_path, map_location=self.device)
        self.P.requires_grad = False
        
        assert self.P.shape == (self.m_bits, self.D_latent), "Key dimensions mismatch!"
        print(f"Successfully loaded {m_bits}-bit key from {key_path}")
       
    def preprocess_image(self, image_input):
        if isinstance(image_input, list):
            tensor_list = []
            for img_path in image_input:
                image = Image.open(img_path).convert('RGB').resize((512, 512))
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
                tensor_list.append(transform(image))
            return torch.stack(tensor_list).to(device).to(torch.bfloat16)
        
        elif isinstance(image_input, str):
            image = Image.open(image_input)
        elif isinstance(image_input, np.ndarray):
            image = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise ValueError(f"Unsupported input type:{type(image_input)}. Please input a file path, PIL Image, or OpenCV array.")
        
        image = image.convert('RGB').resize((512, 512))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        return transform(image).unsqueeze(0).to(device).to(torch.bfloat16)

    def image_to_latent(self, image_tensor):
        with torch.no_grad():
            latents = self.vae.encode(image_tensor).latent_dist.mode()
            latents = latents * self.vae.config.scaling_factor
        return latents

    def latent_to_image(self, latents):
        latents = latents.to(dtype=self.vae.dtype)
        latents = latents / self.vae.config.scaling_factor
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0,1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return (image * 255).round().astype('uint8')[0]

    def compute_sdxl_loss(self, latents_clean_sdxl, z_wm_15_bf, current_t):

        B = z_wm_15_bf.shape[0]

        image_wm_rec = self.vae.decode(z_wm_15_bf / self.vae.config.scaling_factor).sample

        image_wm_1024 = F.interpolate(image_wm_rec, size=(1024, 1024), mode='bilinear', align_corners=False)

        latents_wm_sdxl = self.sdxl_vae.encode(image_wm_1024).latent_dist.mode()
        latents_wm_sdxl = latents_wm_sdxl * self.sdxl_vae.config.scaling_factor

        noise = torch.randn_like(latents_wm_sdxl)

        z_wm_t_sdxl = self.sdxl_scheduler.add_noise(latents_wm_sdxl, noise, current_t)

        added_cond_kwargs = {
            "text_embeds": self.sdxl_empty_pooled_embeds.expand(B, -1),
            "time_ids": self.sdxl_add_time_ids.expand(B, -1)
        }
        
        with torch.no_grad():   
            z0_t_sdxl = self.sdxl_scheduler.add_noise(latents_clean_sdxl, noise, current_t)
            target_noise_pred = self.sdxl_unet(
                                    z0_t_sdxl,
                                    current_t,
                                    encoder_hidden_states=self.sdxl_empty_text_embeds.expand(B, -1, -1),
                                    added_cond_kwargs=added_cond_kwargs
                                ).sample
        
        noise_pred = self.sdxl_unet(
            z_wm_t_sdxl,
            current_t,
            encoder_hidden_states=self.sdxl_empty_text_embeds.expand(B, -1, -1),
            added_cond_kwargs=added_cond_kwargs
        ).sample

        loss_sdxl = F.mse_loss(noise_pred, target_noise_pred)
        
        return loss_sdxl

    def compute_loss(self, z0, z0_bf, delta, delta_bf, current_t, img_wm_recon, fixed_budget, z0_vecs_precalc,
                     img_clean_norm_precalc, jpeg_norms_precalc, z0_norms_precalc, jpeg_vecs_precalc, empty_prompt_precalc,
                     semantic_target = None, lambda_stability = 1, 
                     lambda_semantic = 1, lambda_ortho = 1, lambda_lpips = 1, lambda_ensemble = 1, lambda_jpeg = 1):

        noise = torch.randn_like(z0_bf)
        B = z0.shape[0]

        z_t = self.scheduler.add_noise(z0_bf, noise, current_t)
        
        boost_factor = 2.0
        z_input_bf = z_t + boost_factor * delta_bf
        
        with torch.no_grad():
            target_noise_pred = self.unet(z_t, current_t, encoder_hidden_states=empty_prompt_precalc).sample
            
        adv_noise_pred = self.unet(z_input_bf, current_t,
                                    encoder_hidden_states = empty_prompt_precalc).sample
        loss_stability = F.mse_loss(adv_noise_pred, target_noise_pred)

        img_wm_norm = img_wm_recon.float().clamp(-1.0, 1.0)

        loss_perceptual = self.loss_lpips_func(img_clean_norm_precalc, img_wm_norm).mean() * k_enlarge

        # delta_vecs = delta.view(B, -1)
        # dot_jpeg = (delta_vecs * jpeg_vecs_precalc).sum(dim=1)
        # cos_jpeg = dot_jpeg / (fixed_budget * jpeg_norms_precalc)
        # loss_jpeg = torch.abs(cos_jpeg * k_enlarge).mean()
        

        # k_exponent = 5.0
        # correlation = (delta_vecs * z0_vecs_precalc).sum(dim=1) / (fixed_budget * z0_norms_precalc)
        # loss_ortho = (torch.exp(torch.abs(correlation) * k_exponent) - 1.0).mean()
        # # loss_ortho = (correlation / (delta.norm() * z0.norm() + 1e-6)) ** 2
   
        if semantic_target is not None:
            loss_semantic = (delta - semantic_target).norm()
            loss = lambda_stability * loss_stability + lambda_semantic * loss_semantic #+ lambda_ortho * loss_ortho
        else:
            loss = lambda_stability * loss_stability #+ lambda_ortho * loss_ortho 
                
        loss = loss + lambda_lpips * loss_perceptual #+ lambda_jpeg * loss_jpeg
        
        return loss, adv_noise_pred, target_noise_pred

    def find_zero_space_watermark(self, image_input, batch_size, target_messages, 
                                  semantic_target = None, fixed_budget = 25,
                                  t_min = 100, t_max = 600, learning_rate = 0.1, attack_steps = 500,
                                  lambda_stability = 10, lambda_semantic = 1, 
                                  lambda_ortho = 0,
                                  lambda_reconstruction = 1, lambda_augmentation = 0, lambda_lpips = 1, 
                                  lambda_ensemble = 1, lambda_jpeg = 1, lambda_message = 1):
        image_tensor = self.preprocess_image(image_input)
        z0 = self.image_to_latent(image_tensor).float().detach()
        B = z0.shape[0]        

        z0_bf = z0.to(torch.bfloat16).detach()        
         
        # Reparameterization
        delta_init = torch.randn_like(z0, dtype=torch.float32)

        k_control = 0.8
        fixed_budget = fixed_budget * k_control

        norms = delta_init.view(B, -1).norm(dim=1).view(B, 1, 1, 1) + 1e-6
        delta_raw = (delta_init / norms * fixed_budget).detach().requires_grad_(True)
        
        with torch.no_grad():

            z0_input = z0.to(dtype=torch.bfloat16)
            img_clean_recon = self.vae.decode(z0_input / self.vae.config.scaling_factor).sample
            
            z0_recon = self.vae.encode(img_clean_recon).latent_dist.mode()
            z0_recon = z0_recon * self.vae.config.scaling_factor
            z0_recon_float_precalc = z0_recon.float().detach()
            
            img_clean_norm_precalc = img_clean_recon.float().clamp(-1.0, 1.0).detach()
            
            img_clean_jpeg = self.JPEG_Trainer.forward(img_clean_recon).to(dtype=torch.bfloat16)
            z_clean_jpeg_precalc = self.vae.encode(img_clean_jpeg).latent_dist.mode().detach()
            
            empty_prompt_precalc = self.pipe.encode_prompt(
                prompt="", device=device, num_images_per_prompt=B, do_classifier_free_guidance=True
            )[0].detach()
            
            # image_clean_rec = self.vae.decode(z0_bf / self.vae.config.scaling_factor).sample
            # image_clean_1024 = F.interpolate(image_clean_rec, size=(1024, 1024), mode='bilinear', align_corners=False)
            # latents_clean_sdxl = self.sd21_vae.encode(image_clean_1024).latent_dist.mode()
            # latents_clean_sdxl_precalc = latents_clean_sdxl * self.sd21_vae.config.scaling_factor

            z0_vecs_precalc = z0_recon_float_precalc.view(B, -1).detach()
            z0_norms_precalc = z0_vecs_precalc.norm(dim=1) + 1e-6
            
            jpeg_vecs_precalc = z_clean_jpeg_precalc.view(B, -1).detach()
            jpeg_norms_precalc = jpeg_vecs_precalc.norm(dim=1) + 1e-6

            target_norms = torch.full((B, 1), fixed_budget, device=device, dtype=torch.float32)
            
            
        is_augmentation = False
        if lambda_augmentation != 0:
            is_augmentation = True

        if is_augmentation == False:
            with torch.no_grad():
                z0_input = z0.to(dtype=torch.bfloat16)
                img_clean_recon = self.vae.decode(z0_input / self.vae.config.scaling_factor).sample
                z0_recon = self.vae.encode(img_clean_recon).latent_dist.mode()
                z0_recon = z0_recon * self.vae.config.scaling_factor

        else:
            with torch.no_grad():
                z0_input = z0.to(dtype=torch.bfloat16)
                img_clean_recon = self.vae.decode(z0_input / self.vae.config.scaling_factor).sample
                z0_recon = self.vae.encode(img_clean_recon).latent_dist.mode()
                z0_recon = z0_recon * self.vae.config.scaling_factor

                
                img_clean_recon_f32 = img_clean_recon.float()
                img_clean_augmented_f32 = aug_pipeline(img_clean_recon_f32)
                
                img_clean_augmented = img_clean_augmented_f32.to(dtype=torch.bfloat16)
                z0_augmented = self.vae.encode(img_clean_augmented).latent_dist.mode()
                z0_augmented = z0_augmented * self.vae.config.scaling_factor


        optimizer = optim.Adam([delta_raw], lr = learning_rate, fused=True)

        loss_history = []

        best_delta = None
        
        # min_stability_loss = float('inf')
        # best_contrast_metric = float('inf')
        # best_step_num = -1
        
        delta_history = torch.empty((attack_steps, B, *z0.shape[1:]), device=device, dtype=torch.bfloat16)
        per_sample_loss_history = torch.empty((attack_steps, B), device=device, dtype=torch.float32)
        metric_history = torch.empty((attack_steps, B), device=device, dtype=torch.float32)
        
        # activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        # with torch.profiler.profile(
        #     activities=activities,
        #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
        #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_dir'),
        #     record_shapes=True,
        #     profile_memory=True,
        #     with_stack=True
        # ) as prof:
            
        loss_stab_accum_tensor = torch.tensor(0.0, device=device)
        for step in range(attack_steps):
            optimizer.zero_grad()

            norms = delta_raw.view(B, -1).norm(dim=1).view(B, 1, 1, 1) + 1e-6
            delta = delta_raw / norms * fixed_budget
            
            z_wm = z0 + delta
            delta_bf = delta.to(torch.bfloat16)
            z_wm_bf = z0_bf + delta_bf
            
            img_wm_recon = self.vae.decode(z_wm_bf / self.vae.config.scaling_factor).sample
            z_reconstruction = self.vae.encode(img_wm_recon).latent_dist.mode()
            z_reconstruction = z_reconstruction * self.vae.config.scaling_factor
            
            if is_augmentation == True:
                img_augmented = aug_pipeline(img_wm_recon.float())
                z_augmented = self.vae.encode(img_augmented.to(torch.bfloat16)).latent_dist.mode()
                z_augmented = z_augmented * self.vae.config.scaling_factor
            

            watermark_rec = z_reconstruction.float() - z0_recon_float_precalc
            

            loss_cos_rec = (1 - F.cosine_similarity(watermark_rec.view(B, -1), delta.view(B, -1), dim=1)).mean()

            rec_norms = watermark_rec.view(B, -1).norm(dim=1, keepdim=True)
            loss_mag_rec = F.relu(target_norms - rec_norms).mean()
            
            loss_reconstruction = (loss_cos_rec + 0.5 * loss_mag_rec) 
            loss_structure = lambda_reconstruction * loss_reconstruction
            
            if is_augmentation:
                watermark_augmented = z_augmented.float() - z0_augmented.float()
                loss_cos_aug = (1 - F.cosine_similarity(watermark_augmented.view(B, -1), delta.view(B, -1), dim=1)).mean()
                aug_norms = watermark_augmented.view(B, -1).norm(dim=1, keepdim=True)
                loss_mag_aug = F.relu(target_norms - aug_norms).mean()
                loss_augmentation = (loss_cos_aug + 0.5 * loss_mag_aug)
                loss_structure = loss_structure + lambda_augmentation * loss_augmentation

            z0_recon_vec = z0_recon_float_precalc.view(B, -1).to(device)

            t_sample = torch.randint(low = t_min, high = t_max, size = (B,), device = device).long()
            
            delta_vecs = delta.view(B, -1)
            dot_products = (delta_vecs * z0_vecs_precalc).sum(dim=1) 

            cos_sim_recon = dot_products / (fixed_budget * z0_norms_precalc)
            
            
            loss_ortho_recon = torch.abs(cos_sim_recon).mean() 
            
            # if lambda_ensemble > 0:
            #     loss_sdxl = self.compute_sdxl_loss(latents_clean_sdxl = latents_clean_sdxl_precalc,
            #                                         z_wm_15_bf = z_wm_bf, current_t = t_sample)
            
            z_wm_vecs = z_wm.view(B, -1)  
            

            projections = torch.matmul(z_wm_vecs, self.P.T)  
            

            target_amplitude = fixed_budget / np.sqrt(self.m_bits)
            target_projections = target_messages * target_amplitude
            

            loss_message = F.mse_loss(projections, target_projections)
            
            loss_main, adv_noise_pred, target_noise_pred = self.compute_loss(
                        z0 = z0, delta = delta, current_t=t_sample, 
                        z0_bf = z0_bf, delta_bf = delta_bf, img_wm_recon = img_wm_recon,
                        img_clean_norm_precalc=img_clean_norm_precalc,
                        # z_clean_jpeg_precalc=z_clean_jpeg_precalc,
                        # z0_recon_float_precalc=z0_recon_float_precalc,
                        empty_prompt_precalc=empty_prompt_precalc,
                        lambda_stability=lambda_stability, 
                        lambda_ortho=lambda_ortho, 
                        semantic_target=None, lambda_lpips = lambda_lpips, 
                        lambda_ensemble = lambda_ensemble,
                        lambda_jpeg=lambda_jpeg,
                        fixed_budget=fixed_budget,
                        z0_vecs_precalc=z0_vecs_precalc,
                        z0_norms_precalc=z0_norms_precalc,
                        jpeg_vecs_precalc=jpeg_vecs_precalc,
                        jpeg_norms_precalc=jpeg_norms_precalc
                    ) 
            
            total_loss = loss_structure + loss_main + lambda_message * loss_message#+ loss_ortho_recon #+ lambda_ensemble * loss_sdxl
            total_loss.backward()
            loss_stab_accum_tensor += total_loss.detach()

            optimizer.step()
            
            delta_history[step] = delta.detach().to(torch.bfloat16)
            
            with torch.no_grad():
                loss_vec = F.mse_loss(adv_noise_pred, target_noise_pred, reduction='none').view(B, -1).mean(dim=1)
                per_sample_loss_history[step] = loss_vec
                # metric_history[step] = torch.abs(cos_sim_recon * k_enlarge).detach()

                metric_history[step] = loss_main
                
            # if step % 20 ==0:
            #    print(f"step{step}: loss_structure:{loss_structure:.4f}, loss_main:{loss_main:.4f}, loss_message:{loss_message:.4f}")       
        
        valid_mask = metric_history < 2.0 

        best_delta = torch.empty_like(delta)
        
        for b in range(B):
            b_mask = valid_mask[:, b]
            if b_mask.any():

                valid_b_losses = torch.where(b_mask, per_sample_loss_history[:, b], torch.tensor(float('inf'), device=device))
                best_idx = torch.argmin(valid_b_losses).item()
                best_delta[b] = delta_history[best_idx, b].to(torch.float32)
            else:
                best_delta[b] = delta.detach()[b] 
        
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"当前 Batch Size ({batch_size}) 的峰值显存占用: {peak_mem:.2f} GB")
        
        return best_delta, z0 #, best_step_num, min_stability_loss, best_contrast_metric

    def verify_watermark(self, z0, delta, target_messages, strength=0.3,watermark_scale = 1.0, tamper_pipe = None):
        z_watermarked = z0 + delta * watermark_scale

        arr_wm_uint8 = self.latent_to_image(z_watermarked)
        arr_clean_uint8 = self.latent_to_image(z0)
        
        arr_wm_fp32 = arr_wm_uint8.astype(np.float32)
        arr_clean_fp32 = arr_clean_uint8.astype(np.float32)
        
        img_wm_pil = Image.fromarray(arr_wm_uint8)
        img_clean_pil = Image.fromarray(arr_clean_uint8)

        
        detector = delta.flatten() / delta.norm()
        
        
        pipe_to_use = tamper_pipe if tamper_pipe is not None else self.pipe
        print(f"Tamper using model: {pipe_to_use.config._name_or_path}")
        
        img_tampered = pipe_to_use(
            #prompt = "masterpiece, best quality, high resolution, 8k",
            prompt = "add a dog",
            image = img_wm_pil,
            strength = strength,
            guidance_scale=7.5
        ).images[0]
        img_contrast = pipe_to_use(
            #prompt = "masterpiece, best quality, high resolution, 8k",
            prompt = "add a dog",
            image = img_clean_pil,
            strength = strength,
            guidance_scale=7.5
        ).images[0]

        z_tampered = self.image_to_latent(self.preprocess_image(img_tampered))
        z_contrast = self.image_to_latent(self.preprocess_image(img_contrast))
        
        
        random_noise = torch.randn_like(z0)


        tensor_clean = torch.from_numpy(arr_clean_uint8).float() / 255.0
        tensor_wm = torch.from_numpy(arr_wm_uint8).float() / 255.0
        
        tensor_clean = tensor_clean.permute(2, 0, 1).unsqueeze(0) 
        tensor_wm = tensor_wm.permute(2, 0, 1).unsqueeze(0)


        metrics = self.evaluator.evaluate(tensor_clean, tensor_wm)
        psnr = metrics['PSNR']
        ssim = metrics['SSIM']
        lpips = metrics['LPIPS']
        print(f"Quality Metrics -> PSNR: {psnr} | SSIM: {ssim} | LPIPS: {lpips}")
        

        z_recon = self.image_to_latent(self.preprocess_image(img_wm_pil))
        score_vae_only = cos_correlation(detector, (z_recon).flatten()).item()
        print(f"Score after VAE round-trip (No Diffusion): {score_vae_only}")


        def decode_and_eval(latent_tensor):
            vec = latent_tensor.view(1, -1).float() 
            extracted = torch.matmul(vec, self.P.T) 
            pred_bits = torch.sign(extracted)       

            acc = (pred_bits == target_messages).float().mean().item()
            return acc
            
        acc_watermarked = decode_and_eval(z_watermarked)
        acc_tampered = decode_and_eval(z_tampered)
        acc_contrast = decode_and_eval(z_contrast)
        


        print("Result (Bit Accuracy):")
        print(f"Accuracy of watermarked image: {acc_watermarked * 100:.1f}%")
        print(f"Accuracy of tampered image: {acc_tampered * 100:.1f}%")
        print(f"Accuracy of contrast image: {acc_contrast * 100:.1f}%")
        
        jpeg_scores = {}
        jpeg_clean_scores = {}
        jpeg_qualities = [90, 70, 50, 30] 
        
        img_cv2 = cv2.cvtColor(np.array(img_wm_pil), cv2.COLOR_RGB2BGR)
        for q in jpeg_qualities:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
            _, enc_data = cv2.imencode('.jpg', img_cv2, encode_param)
            img_jpeg_distorted = cv2.imdecode(enc_data, 1) 
            z_jpeg = self.image_to_latent(self.preprocess_image(img_jpeg_distorted))
            
            acc_jpeg = decode_and_eval(z_jpeg)
            jpeg_scores[f"jpeg_{q}"] = acc_jpeg
            print(f"Quality {q}: Accuracy = {acc_jpeg * 100:.1f}%")
            
        img_clean_cv2 = cv2.cvtColor(np.array(img_clean_pil), cv2.COLOR_RGB2BGR)
        for q in jpeg_qualities:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
            _, enc_clean_data = cv2.imencode('.jpg', img_clean_cv2, encode_param)
            img_jpeg_clean_distorted = cv2.imdecode(enc_clean_data, 1) 
            z_clean_jpeg = self.image_to_latent(self.preprocess_image(img_jpeg_clean_distorted))

            acc_clean_jpeg = decode_and_eval(z_clean_jpeg)
            jpeg_clean_scores[f"jpeg_clean_{q}"] = acc_clean_jpeg
            print(f"Quality {q}: Accuracy_contrast = {acc_clean_jpeg * 100:.1f}%")
        
        return acc_watermarked, acc_tampered, acc_contrast, score_vae_only, jpeg_scores, jpeg_clean_scores, psnr, ssim, lpips
    
    
import glob
import pandas as pd
import shutil

def main():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    current_script = __file__ 

    print("Initializing Attacker and loading models into VRAM (Only ONCE)...")
    attacker = StableDiffusionAttacker()
    
    for strength in [0.3]:
        for message in [30]:
            print(f"\n==================================================")
            print(f"开始实验组: Strength = {strength}, m_bits = {message}")
            print(f"==================================================")

            current_save_root = os.path.join(save_root, f"strength_{strength}_mbits_{message}")
            os.makedirs(current_save_root, exist_ok=True)
            
            shutil.copy(current_script, os.path.join(current_save_root, "source_code_backup.py"))

            log_save_path = os.path.join(current_save_root, "detailed_log.csv")
            bad_cases_dir = os.path.join(current_save_root, "bad_cases")
            assets_dir = os.path.join(current_save_root, "saved_assets")
            os.makedirs(bad_cases_dir, exist_ok=True)
            os.makedirs(assets_dir, exist_ok=True)

            KeyPATH = "/root/autodl-tmp/watermark_project/Multi_bit"
            KEY_PATH = os.path.join(KeyPATH, f"global_watermark_key_{message}.pt")
            
            attacker.set_key(key_path=KEY_PATH, m_bits=message)
            
            N_loop = 1000
            dataset_path = "/root/autodl-fs/VINE/W-Bench/SVD_1K/image/*.png"
            all_images = glob.glob(dataset_path)
            all_images.sort()
            images = all_images[:N_loop]
            
            print(f"Successfully load {len(images)} images for testing.")

            all_results = []
            
            start_idx = 0
            print(f"DEBUG: 正在扫描路径: {os.path.join(assets_dir, '*_delta.pt')}")
            print(f"DEBUG: assets_dir 目录是否存在: {os.path.isdir(assets_dir)}")
            existing_files = glob.glob(os.path.join(assets_dir, "*_delta.pt"))
            if existing_files:
                indices = [int(os.path.basename(f).split('_')[0]) for f in existing_files]
                start_idx = max(indices) + 1
                print(f"检测到已生成的张数: {len(indices)}, 最大索引: {max(indices)}")
                print(f"将从索引 {start_idx} 处开始续跑")

            else:
                print("未检测到已有进度，从头开始。")

            
            batch_size = 4
            from tqdm import tqdm
            for i in tqdm(range(start_idx, N_loop, batch_size), desc = "Running experiment"):
                batch_paths = images[i : i + batch_size]
                current_B = len(batch_paths) 
                
                target_messages = torch.randint(0, 2, (current_B, attacker.m_bits), device=device).float() * 2 - 1

                best_delta_batch, z0_batch = attacker.find_zero_space_watermark(target_messages=target_messages, image_input=batch_paths, batch_size=batch_size)
                
                for b in range(current_B):
                    img_path = batch_paths[b]
                    global_idx = i + b
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    single_z0 = z0_batch[b:b+1] 
                    single_delta = best_delta_batch[b:b+1]
                    
                    ori_save_path = os.path.join(assets_dir, f"{global_idx:04d}_{base_name}_orig.png")
                    img_orig_uint8 = attacker.latent_to_image(z0_batch[b:b+1])
                    cv2.imwrite(ori_save_path, cv2.cvtColor(img_orig_uint8, cv2.COLOR_RGB2BGR))

                    single_msg = target_messages[b].detach().cpu().numpy()
                    msg_save_path = os.path.join(assets_dir, f"{global_idx:04d}_{base_name}_msg.npy")
                    np.save(msg_save_path, single_msg)
                    with open(os.path.join(assets_dir, f"{global_idx:04d}_{base_name}_msg.txt"), "w") as f:
                        f.write(str(single_msg.tolist()))

                    delta_save_path = os.path.join(assets_dir, f"{global_idx:04d}_{base_name}_delta.pt")
                    torch.save(single_delta.detach().cpu(), delta_save_path)
                    
                    z_watermarked = z0_batch[b:b+1] + single_delta
                    img_wm_uint8 = attacker.latent_to_image(z_watermarked)
                    img_save_path = os.path.join(assets_dir, f"{global_idx:04d}_{base_name}_wm.png")
                    cv2.imwrite(img_save_path, cv2.cvtColor(img_wm_uint8, cv2.COLOR_RGB2BGR))
                    
                    
                    single_target_msg = target_messages[b:b+1]
                    
                    print("\n SD1.5 tamper")
                    acc_watermarked, acc_tampered_sd15, acc_contrast_sd15,score_vae_only, jpeg_scores, jpeg_clean_scores, psnr, ssim, lpips = attacker.verify_watermark(single_z0, single_delta, strength=strength, target_messages=single_target_msg)
                    
                    current_data = {
                        "img_idx": global_idx,
                        "img_name": os.path.basename(img_path),
                        "acc_watermarked": acc_watermarked,
                        "acc_tampered_sd15": acc_tampered_sd15,
                        "acc_contrast_sd15": acc_contrast_sd15,
                        "vae_only_score": score_vae_only,
                        "psnr": psnr,
                        "ssim": ssim,
                        "lpips": lpips,
                        "jpeg_90": jpeg_scores.get("jpeg_90", 0),
                        "jpeg_70": jpeg_scores.get("jpeg_70", 0),
                        "jpeg_50": jpeg_scores.get("jpeg_50", 0),
                        "jpeg_30": jpeg_scores.get("jpeg_30", 0),
                        "jpeg_clean_90": jpeg_clean_scores.get("jpeg_clean_90", 0),
                        "jpeg_clean_70": jpeg_clean_scores.get("jpeg_clean_70", 0),
                        "jpeg_clean_50": jpeg_clean_scores.get("jpeg_clean_50", 0),
                        "jpeg_clean_30": jpeg_clean_scores.get("jpeg_clean_30", 0)
                    }

                    all_results.append(current_data)

                    if (i + 1) % 10 == 0:
                        temp_df = pd.DataFrame(all_results)
                        temp_df.to_csv(log_save_path, index=False)
                        print(f"--- 已存档至第 {i+1} 张图 ---")
                
            df = pd.DataFrame(all_results)
            df.to_csv(log_save_path, index = False)
            

            plt.figure(figsize=(10, 6))
            plt.hist(df['acc_contrast_sd15'], bins=40, alpha=0.5, color='gray', label='Clean/Original')
            plt.hist(df['acc_tampered_sd15'], bins=40, alpha=0.5, color='orange', label='Tampered (SD1.5)')
            plt.hist(df['acc_watermarked'], bins=40, alpha=0.5, color='blue', label='Watermarked (Ideal)')
            plt.xlabel("Bit Accuracy")
            plt.ylabel("Frequency")
            plt.title(f"Distribution: {EXP_NAME} (SD1.5, Strength={strength}, bits={message})")
            plt.legend()
            plt.grid(True, alpha=0.3)
            save_path_sd15 = os.path.join(current_save_root, "result_histogram_SD15.png")
            plt.savefig(save_path_sd15)
            plt.close() 

            print(f"当前组别实验结束，结果存入{current_save_root}\n")

    print("所有实验组遍历执行完毕")



if __name__ == "__main__":
    main()