# Motus Policy for RoboTwin

import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
import sys
import os
import logging
from typing import List, Dict, Any, Optional
from collections import deque
import yaml
from PIL import Image
from transformers import AutoProcessor
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add model paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "models"))

from models.motus import Motus, MotusConfig

# Add bak path for T5EncoderModel
BAK_ROOT = str((Path(__file__).parent / "bak").resolve())
if BAK_ROOT not in sys.path:
    sys.path.insert(0, BAK_ROOT)

from wan.modules.t5 import T5EncoderModel

from utils.image_utils import resize_with_padding

logger = logging.getLogger(__name__)

class MotusPolicy:
    """
    Motus Policy wrapper for RoboTwin evaluation.
    Implements the joint video-action diffusion model for robotic control.
    """
    
    def __init__(self, checkpoint_path: str, config_path: str, wan_path: str, device: str = "cuda", log_dir: Optional[str] = None, task_name: Optional[str] = None):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.wan_path = wan_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config_dict = yaml.safe_load(f)
        
        # Initialize model
        self.model = self._load_model()

        # Initialize T5 encoder for language embeddings (WAN text encoder)
        self.t5_encoder = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=device,
            checkpoint_path=os.path.join(self.wan_path, 'Wan2.2-TI2V-5B', 'models_t5_umt5-xxl-enc-bf16.pth'),
            tokenizer_path=os.path.join(self.wan_path, 'Wan2.2-TI2V-5B', 'google/umt5-xxl'),
        )

        # Initialize VLM processor (for building Qwen2.5-VL inputs like in training)
        vlm_ckpt = self.config_dict['model']['vlm']['checkpoint_path']
        self.vlm_processor = AutoProcessor.from_pretrained(vlm_ckpt, trust_remote_code=True)
        
        # Initialize observation cache
        self.obs_cache = deque(maxlen=1)  # Only need current frame for LAWM
        self.action_cache = deque()
        
        # Model state
        self.current_state = None
        self.current_state_norm = None
        self.is_first_step = True
        
        # Action interpolation
        # self.arm_steps_length = self.config_dict['common']['arm_steps_length']
        self.prev_action = None

        # Load normalization stats for de/normalization
        self._load_normalization_stats()
        
        # Initialize image saving
        self.save_images = True
        # Resolve base log directory: prefer explicit arg, then env, fallback to project local 'logs'
        base_log_dir = log_dir or os.environ.get('LOG_DIR') or str(Path(__file__).resolve().parent.parent / "logs")
        task_dir_name = task_name or os.environ.get('TASK_NAME') or "default_task"
        self.save_dir = Path(base_log_dir) / "images" / task_dir_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.episode_count = 0
        self.step_count = 0

        logger.info("Motus Policy initialized successfully")

    def set_instruction(self, instruction: str):
        """Set the current instruction for the policy."""
        self.current_instruction = instruction
        logger.info(f"Instruction set: {instruction}")

    def _load_model(self) -> Motus:
        """Load the latest Motus model."""
        logger.info(f"Loading Motus model from {self.checkpoint_path}")

        # Create model config from yaml
        config = self._create_model_config()
        
        # Initialize model
        model = Motus(config)
        model = model.to(self.device)
        
        # Load checkpoint using model's built-in method
        try:
            logger.info(f"Loading checkpoint from {self.checkpoint_path}")
            model.load_checkpoint(self.checkpoint_path, strict=False)
            logger.info("Model checkpoint loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            logger.warning("Using model with random weights")
        
        model.eval()
        return model
    
    def _create_model_config(self) -> MotusConfig:
        """Create latest model configuration from yaml config."""
        common = self.config_dict['common']
        model_cfg = self.config_dict['model']

        # Derive Action Expert dims
        hidden_size = model_cfg['action_expert']['hidden_size']
        ffn_multiplier = model_cfg['action_expert']['ffn_dim_multiplier']

        # Build config dataclass
        config = MotusConfig(
            # Video model settings
            wan_checkpoint_path=model_cfg['wan']['checkpoint_path'],
            vae_path=model_cfg['wan']['vae_path'],
            wan_config_path=model_cfg['wan']['checkpoint_path'],  # Use same path as checkpoint
            video_precision=model_cfg['wan']['precision'],
            
            # VLM settings
            vlm_checkpoint_path=model_cfg['vlm']['checkpoint_path'],
            
            # Understanding Expert settings
            und_expert_hidden_size=model_cfg.get('und_expert', {}).get('hidden_size', 512),
            und_expert_ffn_dim_multiplier=model_cfg.get('und_expert', {}).get('ffn_dim_multiplier', 4),
            und_expert_norm_eps=model_cfg.get('und_expert', {}).get('norm_eps', 1e-5),
            und_layers_to_extract=None,  # Use default (all layers)
            
            # VLM adapter settings for understanding expert
            vlm_adapter_input_dim=model_cfg.get('und_expert', {}).get('vlm', {}).get('input_dim', 2048),
            vlm_adapter_projector_type=model_cfg.get('und_expert', {}).get('vlm', {}).get('projector_type', "mlp3x_silu"),
            
            # Action expert settings
            num_layers=30,
            action_state_dim=common['state_dim'],
            action_dim=common['action_dim'],
            action_expert_dim=hidden_size,
            action_expert_ffn_dim_multiplier=ffn_multiplier,
            action_expert_norm_eps=model_cfg['action_expert'].get('norm_eps', 1e-6),
            
            # Sampling settings
            global_downsample_rate=common['global_downsample_rate'],
            video_action_freq_ratio=common['video_action_freq_ratio'],
            num_video_frames=common['num_video_frames'],
            
            # Loss weights
            video_loss_weight=model_cfg['loss_weights']['video_loss_weight'],
            action_loss_weight=model_cfg['loss_weights']['action_loss_weight'],
            
            batch_size=1,

            # Video dimensions
            video_height=common['video_height'],
            video_width=common['video_width'],
            
            # Do not load WAN/VLM pretrained when using a trained checkpoint
            load_pretrained_backbones=False,
        )

        return config
    
    def update_obs(self, observation: Dict[str, Any]):
        """Update observation cache with new observation."""
        # Extract visual observations from three cameras
        if 'observation' in observation:
            obs_data = observation['observation']
            if 'head_camera' in obs_data and 'left_camera' in obs_data and 'right_camera' in obs_data:
                # Get images from all three cameras
                head_img = obs_data['head_camera']['rgb']  # Keep size (240x320)
                left_img = obs_data['left_camera']['rgb']  # Will resize to (120x160)
                right_img = obs_data['right_camera']['rgb']  # Will resize to (120x160)
                
                # Resize left and right images
                left_img_resized = cv2.resize(left_img, (160, 120))  # (width, height) -> 120x160
                right_img_resized = cv2.resize(right_img, (160, 120))  # (width, height) -> 120x160
                
                # Concatenate left and right horizontally first
                bottom_row = np.concatenate([left_img_resized, right_img_resized], axis=1)  # (120, 320, 3)
                
                # Now concatenate head (top) with bottom row (vertically)
                # head_img: (240, 320, 3)
                # bottom_row: (120, 320, 3) 
                # Final: (360, 320, 3) -> height=360, width=320
                image = np.concatenate([head_img, bottom_row], axis=0)
            else:
                raise ValueError("Missing camera data in observation dict")
        elif 'head_camera' in observation:
            image = observation['head_camera']
        elif 'image' in observation:
            image = observation['image']
        else:
            raise ValueError("No visual observation found in observation dict")

        # Resize to model expected size using resize_with_padding to avoid distortion
        target_size = (self.config_dict['common']['video_height'],
                      self.config_dict['common']['video_width'])

        # Convert to tensor first (without normalization)
        if isinstance(image, np.ndarray):
            # Convert from HWC to CHW first
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        else:
            image_tensor = image

        if image_tensor.shape[-2:] != target_size:
            # Convert tensor to numpy for resize_with_padding function
            # image_tensor shape: [1, C, H, W]
            image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H, W, C]

            # Use resize_with_padding to maintain aspect ratio and add padding
            resized_np = resize_with_padding(image_np, target_size)

            # Convert back to tensor [1, C, H, W] and normalize to [0, 1]
            # resize_with_padding returns uint8, so we need to normalize
            if resized_np.dtype == np.uint8:
                resized_np = resized_np.astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(resized_np).permute(2, 0, 1).unsqueeze(0)
        
        # Update cache
        self.obs_cache.append(image_tensor.to(self.device))

        # Extract robot state
        state = observation['joint_action']['vector']

        if isinstance(state, np.ndarray):
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        else:
            state_tensor = state.float().unsqueeze(0) if state.dim() == 1 else state.float()

        # Save raw and normalized state (model expects normalized)
        self.current_state = state_tensor.to(self.device)
        self.current_state_norm = self._normalize_actions(self.current_state).to(self.device)
    
    def get_action(self, instruction: str = None) -> List[np.ndarray]:
        """
        Get action predictions from the model.
        
        Returns:
            List of action arrays for execution
        """
        if len(self.obs_cache) == 0:
            raise ValueError("No observations in cache. Call update_obs first.")
        
        if self.current_state is None:
            raise ValueError("No robot state available. Call update_obs first.")
        
        # Get current frame
        current_frame = self.obs_cache[-1]  # [1, C, H, W]

        # Get instruction and encode with T5 (WAN text embeddings)
        scene_prefix = ("The whole scene is in a realistic, industrial art style with three views: "
                        "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
                        "The aloha robot is currently performing the following task: ")
        instruction = f"{scene_prefix}{self.current_instruction}"
        t5_out = self.t5_encoder([instruction], self.device)
        if isinstance(t5_out, torch.Tensor):
            # Expect [1, seq_len, dim] or [seq_len, dim]
            t5_list = [t5_out.squeeze(0)] if t5_out.dim() == 3 else [t5_out]
        elif isinstance(t5_out, list):
            t5_list = t5_out
        else:
            raise ValueError("Unexpected T5 encoder output format")

        # Build VLM inputs from first frame and instruction
        first_frame_pil = self._tensor_to_pil_image(current_frame.squeeze(0).cpu())
        vlm_inputs = self._preprocess_vlm_messages(instruction, first_frame_pil)

        # Run inference - Use configured inference steps
        num_inference_steps = self.config_dict['model']['inference']['num_inference_timesteps']
        with torch.no_grad():
            predicted_frames, predicted_actions = self.model.inference_step(
                first_frame=current_frame,
                # state=self.current_state_norm,
                state=self.current_state,
                num_inference_steps=num_inference_steps,
                language_embeddings=t5_list,
                vlm_inputs=[vlm_inputs],
            )

        # Save frame grid (condition frame + predicted frames)
        if predicted_frames is not None:
            # predicted_frames shape: [B, C, T, H, W] or [B, T, C, H, W]
            # Ensure it's [B, T, C, H, W] format
            if predicted_frames.dim() == 5:
                if predicted_frames.shape[1] == 3:  # [B, C, T, H, W]
                    predicted_frames_viz = predicted_frames.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
                else:  # [B, T, C, H, W]
                    predicted_frames_viz = predicted_frames
                
                # Take first batch and save grid
                condition_frame_viz = current_frame.squeeze(0)  # [C, H, W]
                predicted_frames_viz = predicted_frames_viz.squeeze(0)  # [T, C, H, W]
                
                self._save_frame_grid(condition_frame_viz, predicted_frames_viz)
                self.step_count += 1

        # Extract actions [T, D] in normalized space and denormalize back
        # actions_norm = predicted_actions.squeeze(0)  # [action_chunk_size, action_dim]
        # actions_real = self._denormalize_actions(actions_norm).cpu().numpy()
        actions_real = predicted_actions.squeeze(0).cpu().numpy()

        '''
        # Apply interpolation to each predicted action
        interpolated_actions = []
        for i, action in enumerate(actions_real):
            if i == 0:
                # For the first action, interpolate from previous action if available
                interpolated = self.interpolate_action(self.prev_action, action)
            else:
                # For subsequent actions, interpolate from previous predicted action
                interpolated = self.interpolate_action(actions_real[i-1], action)
            
            interpolated_actions.extend(interpolated)
        '''
        
        # Update previous action for next iteration
        self.prev_action = actions_real[-1].copy()

        # Cache actions for potential reuse
        # self.action_cache.extend(interpolated_actions)
        self.action_cache.extend(actions_real)

        # return interpolated_actions
        return actions_real

    def _tensor_to_pil_image(self, tensor_chw: torch.Tensor) -> Image.Image:
        """Convert [C, H, W] float tensor in [0,1] to PIL RGB image."""
        if tensor_chw.dtype != torch.float32:
            tensor_chw = tensor_chw.float()
        tensor_chw = tensor_chw.clamp(0, 1)
        np_img = (tensor_chw.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(np_img, mode='RGB')

    def _preprocess_vlm_messages(self, instruction: str, image: Image.Image) -> Dict[str, torch.Tensor]:
        """Build VLM inputs like training's preprocess_vlm_messages for Qwen2.5-VL."""
        # Build chat messages
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': instruction},
                    {'type': 'image', 'image': image},
                ]
            }
        ]
        # Apply chat template to get prompt text
        text = self.vlm_processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        # Tokenize and process image
        encoded = self.vlm_processor(text=[text], images=[image], return_tensors='pt')
        # Move tensors to device
        vlm_inputs = {
            'input_ids': encoded['input_ids'].to(self.device),
            'attention_mask': encoded['attention_mask'].to(self.device), 
            'pixel_values': encoded['pixel_values'].to(self.device),
            'image_grid_thw': encoded.get('image_grid_thw', None)
        }
        if vlm_inputs['image_grid_thw'] is not None:
            vlm_inputs['image_grid_thw'] = vlm_inputs['image_grid_thw'].to(self.device)
        return vlm_inputs

    def _load_normalization_stats(self):
        """Load action normalization stats from utils/stat.json."""
        try:
            stat_path = Path(__file__).parent / 'utils' / 'stat.json'
            with open(stat_path, 'r') as f:
                stat_data = yaml.safe_load(f) if stat_path.suffix in ['.yml', '.yaml'] else None
        except Exception:
            stat_data = None
        # Fallback to json module
        if stat_data is None:
            import json as _json
            with open(Path(__file__).parent / 'utils' / 'stat.json', 'r') as f:
                stat_data = _json.load(f)

        stats = stat_data.get('robotwin2')
        if stats is None:
            raise ValueError('Normalization stats for robotwin2 not found in stat.json')
        self.action_min = torch.tensor(stats['min'], dtype=torch.float32, device=self.device)
        self.action_max = torch.tensor(stats['max'], dtype=torch.float32, device=self.device)
        self.action_range = self.action_max - self.action_min

    def _normalize_actions(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize to [0,1]: (x - min) / (max - min). Supports [B,D] or [T,D] or [B,T,D]."""
        shape = x.shape
        x_flat = x.reshape(-1, shape[-1])
        norm = (x_flat - self.action_min.unsqueeze(0)) / self.action_range.unsqueeze(0)
        return norm.reshape(shape)

    def _denormalize_actions(self, y: torch.Tensor) -> torch.Tensor:
        """Denormalize from [0,1]: y * (max - min) + min. Supports [T,D] or [B,T,D]."""
        shape = y.shape
        y_flat = y.reshape(-1, shape[-1])
        denorm = y_flat * self.action_range.unsqueeze(0) + self.action_min.unsqueeze(0)
        return denorm.reshape(shape)
    
    def _create_frame_grid(self, condition_frame: torch.Tensor, predicted_frames: torch.Tensor) -> Image.Image:
        """
        Create a horizontal grid with condition frame + predicted frames.
        
        Args:
            condition_frame: [C, H, W] condition frame tensor
            predicted_frames: [T, C, H, W] predicted frames tensor
            
        Returns:
            PIL Image of the horizontal grid (5 frames total)
        """
        # Convert tensors to numpy arrays [H, W, C]
        def tensor_to_numpy(tensor):
            if tensor.dim() == 3:  # [C, H, W]
                tensor = tensor.permute(1, 2, 0)
            tensor = tensor.detach().cpu().float()
            tensor = torch.clamp(tensor, 0, 1)
            return (tensor.numpy() * 255).astype(np.uint8)
        
        # Convert condition frame
        condition_np = tensor_to_numpy(condition_frame)
        
        # Convert predicted frames (take first 4 frames)
        predicted_np = []
        # num_pred_frames = min(4, predicted_frames.shape[0])
        num_pred_frames = predicted_frames.shape[0]
        for i in range(num_pred_frames):
            frame_np = tensor_to_numpy(predicted_frames[i])
            predicted_np.append(frame_np)
        
        # Pad with last frame if we have less than 4 predicted frames
        while len(predicted_np) < 4:
            predicted_np.append(predicted_np[-1] if predicted_np else condition_np)
        
        # Create horizontal concatenation: condition + 4 predicted frames
        all_frames = [condition_np] + predicted_np[:4]
        
        # Concatenate horizontally
        grid_image = np.concatenate(all_frames, axis=1)  # [H, W*5, C]
        
        return Image.fromarray(grid_image)
    
    def _save_frame_grid(self, condition_frame: torch.Tensor, predicted_frames: torch.Tensor):
        """Save the frame grid to disk."""
        if not self.save_images:
            return
        
        try:
            # Create grid image
            grid_image = self._create_frame_grid(condition_frame, predicted_frames)
            
            # Create filename
            filename = f"episode_{self.episode_count:04d}_step_{self.step_count:04d}.png"
            save_path = self.save_dir / filename
            
            # Save image
            grid_image.save(save_path)
            logger.info(f"Saved frame grid to {save_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save frame grid: {e}")
    
    def interpolate_action(self, prev_action: np.ndarray, cur_action: np.ndarray) -> np.ndarray:
        """
        Action interpolation to reduce jitter.
        Based on the provided reference implementation.
        
        Args:
            prev_action: Previous action array
            cur_action: Current action array
            
        Returns:
            Interpolated actions as array of shape [num_steps, action_dim]
        """
        if prev_action is None:
            return cur_action[np.newaxis, :]
            
        # Calculate step size based on maximum difference
        steps = np.full_like(cur_action, self.arm_steps_length)
        diff = np.abs(cur_action - prev_action)
        step = np.ceil(diff / steps).astype(int)
        step = np.max(step)
        
        if step <= 1:
            return cur_action[np.newaxis, :]
        
        # Linear interpolation
        new_actions = np.linspace(prev_action, cur_action, step + 1)
        return new_actions[1:]  # Skip the first point (previous action)


def encode_obs(observation):
    """Post-Process Observation"""
    # No additional processing needed for LAWM
    return observation


def get_model(usr_args):
    """
    Initialize Motus model from deploy_policy.yml and eval.sh overrides.
    
    Args:
        usr_args: Arguments from eval.sh and deploy_policy.yml
        
    Returns:
        MotusPolicy: Initialized Motus policy model
    """
    # Extract configuration from usr_args
    checkpoint_path = usr_args.get('ckpt_setting') 
    
    # Get model config path
    policy_dir = Path(__file__).parent
    config_path = usr_args.get('model_config')
    config_path = policy_dir / "utils"/ "robotwin.yml"
    wan_path = usr_args.get('wan_path')
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize policy
    policy = MotusPolicy(
        checkpoint_path=checkpoint_path,
        wan_path=wan_path,
        config_path=str(config_path),
        device=device,
        log_dir=usr_args.get('log_dir'),
        task_name=usr_args.get('task_name')
    )
    
    return policy


def eval(TASK_ENV, model, observation):
    """
    Evaluation function for LAWM policy.
    
    Args:
        TASK_ENV: Task environment instance
        model: LAWMPolicy instance from get_model()
        observation: Current observation from environment
    """
    # Post-process observation
    obs = encode_obs(observation)
    
    # Get instruction from environment and set it
    instruction = TASK_ENV.get_instruction()
    model.set_instruction(instruction)

    # Always update observation at the start of each eval call
    model.update_obs(obs)

    # Get action predictions (and cache predicted frames for visualization)
    actions = model.get_action()
    
    # Execute each action step
    for action in actions:
        # Execute action (joint control mode)
        TASK_ENV.take_action(action, action_type='qpos')
        
        # Get new observation and update model
        # observation = TASK_ENV.get_obs()
        # obs = encode_obs(observation)
        # model.update_obs(obs)


def reset_model(model):  
    """
    Reset model cache at the beginning of each evaluation episode.
    
    Args:
        model: LAWMPolicy instance
    """
    # Clear observation and action caches
    model.obs_cache.clear()
    model.action_cache.clear()
    
    # Reset state
    model.current_state = None
    model.is_first_step = True
    
    # Reset interpolation state
    model.prev_action = None
    
    # Update episode counter and reset step counter
    model.episode_count += 1
    model.step_count = 0
    
    logger.info(f"Model reset completed for episode {model.episode_count}")