
import os
import sys
import json
import random
import re
import logging
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datetime import datetime

import torch
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from safetensors import safe_open

# ===== 配置日志 =====
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

IS_HPC = Path("/scratch").exists()

if IS_HPC:
    NETID = "xl5199"  # your NetID
    DATA_DIR = Path(f"/scratch/{NETID}/r2r_dataset")
    SHAREGPT_PATH = DATA_DIR / "sharegpt_data.json"

    from datetime import datetime
    RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = Path(f"/scratch/{NETID}/qwen3vl_runs/{RUN_ID}")
else:
    DATA_DIR = Path("/Users/roger/Downloads/dataset_train")
    SHAREGPT_PATH = DATA_DIR / "sharegpt_data.json"
    OUTPUT_DIR = DATA_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "model_name": "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
    "max_seq_length": 4096,
    "lora_rank": 16,
    "lora_alpha": 16,
    
    "num_images": 8,
    "image_size": (512, 512),
    
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 3,     
    "max_steps": -1,            
    "learning_rate": 2e-4,
    "warmup_ratio": 0.05,       
    "save_steps": 500,
    "logging_steps": 50,
    
    "eval_samples": 1000,     
}

# =====  Action Space =====
VALID_ANGLES = [15, 30, 45, 60, 90]      # θ ∈ {15, 30, 45, 60, 90}
VALID_DISTANCES = [25, 50, 75, 100]      # d ∈ {25, 50, 75, 100}

def load_and_resize_image(img_path, size=(512, 512)):
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(size, Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        logger.warning(f"Failed to load {img_path}: {e}")
        return None

def sample_image_indices_for_training(total, n):
    """
    采样图像索引 - 保证最后一张是当前观察
    
    策略: 
    - 最后一张 (索引 total-1) 一定被选中作为"当前观察"
    - 前 N-1 张从 [0, total-2] 范围内均匀采样作为"历史观察"
    
    示例 (total=60, n=8):
    - 历史索引: [0, 8, 17, 25, 34, 42, 51] (7张，均匀分布在0-58)
    - 当前索引: [59] (1张，最后一帧)
    - 返回: [0, 8, 17, 25, 34, 42, 51, 59]
    """
    if total <= n:
        return list(range(total))
    
    if n == 1:
        return [total - 1]
    
    n_history = n - 1
    history_range = total - 1
    
    if n_history == 1:
        history_indices = [0]
    else:
        step = (history_range - 1) / (n_history - 1)
        history_indices = [int(i * step) for i in range(n_history)]
    
    current_index = total - 1
    return history_indices + [current_index]

def build_navigation_prompt(instruction, n_images):
    """
    构建 Navigation Prompt
    
    图像排列: [历史1, 历史2, ..., 历史N-1, 当前]
    - Images 1 到 N-1: 历史观察 (均匀采样)
    - Image N: 当前观察 (最新一帧)
    """
    if n_images == 1:
        return f"""imagine you are a robot programmed for navigation tasks.
You have been given current observation:
[Image 1]
Your assigned task is: {instruction}
Analyze this image to decide your next move,
which could involve turning left or right by a specific degree,
moving forward a certain distance, or stop if the task is completed."""
    else:
        return f"""imagine you are a robot programmed for navigation tasks.
You have been given a video of historical observations:
[Images 1-{n_images-1}]
and current observation:
[Image {n_images}]
Your assigned task is: {instruction}
Analyze this series of images to decide your next move,
which could involve turning left or right by a specific degree,
moving forward a certain distance, or stop if the task is completed."""

# ===== dataset =====

class R2RVisionDataset:
    """
    Lazy loading dataset - 支持多图像输入
    
    图像采样策略:
    - 前 N-1 张: 历史观察 (从轨迹开始到倒数第二帧均匀采样)
    - 第 N 张: 当前观察 (最后一帧，用于决策)
    """
    def __init__(self, data_items, data_dir, num_images=8, image_size=(512, 512)):
        self.data_items = data_items
        self.data_dir = Path(data_dir)
        self.num_images = num_images
        self.image_size = image_size
        
    def __len__(self):
        return len(self.data_items)
    
    def __getitem__(self, idx):
        item = self.data_items[idx]
        msgs = item["messages"]
        img_paths = item["images"]
        
        selected_indices = sample_image_indices_for_training(len(img_paths), self.num_images)
        selected_paths = [img_paths[i] for i in selected_indices]
        
        images = []
        for img_path in selected_paths:
            full_path = self.data_dir / img_path
            img = load_and_resize_image(full_path, self.image_size)
            if img is None:
                img = Image.new("RGB", self.image_size, (0, 0, 0))
            images.append(img)
        
        user_text = msgs[0]["content"]
        instruction = re.sub(r'<image>', '', user_text).strip()
        assistant_answer = msgs[1]["content"]
        
        # build Navigation Prompt
        nav_prompt = build_navigation_prompt(instruction, len(images))
        
        #user content: [image1, image2, ..., imageN, text]
        user_content = []
        for img in images:
            user_content.append({"type": "image", "image": img})
        user_content.append({"type": "text", "text": nav_prompt})
        
        # return Unsloth Vision SFT format
        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": assistant_answer}]},
            ]
        }

# ===== evaluation functions =====

def snap_to_discrete(value, valid_values):
    """将预测值映射到最近的离散值"""
    if not valid_values:
        return value
    return min(valid_values, key=lambda x: abs(x - value))

def parse_action(text, snap_to_valid=False):
    """解析动作文本，返回结构化的动作信息"""
    text = text.lower().strip()
    text = re.sub(r'^the next action is\s*', '', text, flags=re.IGNORECASE)
    text = text.strip().rstrip('.')
    
    # pattern 1: turn left/right X degrees
    turn_match = re.search(r'turn\s+(left|right)\s+(\d+)\s*(?:degrees?|deg)?', text)
    if turn_match:
        angle = int(turn_match.group(2))
        if snap_to_valid:
            angle = snap_to_discrete(angle, VALID_ANGLES)
        return {
            "type": "turn",
            "direction": turn_match.group(1),
            "angle": angle,
            "raw": text
        }
    
    # pattern 2: move forward X cm
    move_match = re.search(r'(?:move\s+)?forward\s+(\d+)\s*(?:cm|centimeters?)?', text)
    if move_match:
        distance = int(move_match.group(1))
        if snap_to_valid:
            distance = snap_to_discrete(distance, VALID_DISTANCES)
        return {
            "type": "move",
            "direction": "forward",
            "distance": distance,
            "raw": text
        }
    
    # pattern 3: stop
    if 'stop' in text:
        return {"type": "stop", "raw": text}
    
    # unable to parse
    return {"type": "unknown", "raw": text}

def is_valid_action(action):
    """检查动作是否在合法的离散 action space 中"""
    if action["type"] == "turn":
        return action.get("angle") in VALID_ANGLES
    elif action["type"] == "move":
        return action.get("distance") in VALID_DISTANCES
    elif action["type"] == "stop":
        return True
    return False

def compare_actions_discrete(pred_action, gt_action):
    """离散 Action Space 的严格比较"""
    result = {
        "exact_match": False,
        "type_match": False,
        "direction_match": False,
        "pred_is_valid": is_valid_action(pred_action),
        "gt_is_valid": is_valid_action(gt_action),
    }
    
    if pred_action["type"] == gt_action["type"]:
        result["type_match"] = True
        
        if pred_action["type"] == "stop":
            result["exact_match"] = True
            result["direction_match"] = True
            return result
        
        if pred_action["type"] == "turn":
            if pred_action.get("direction") == gt_action.get("direction"):
                result["direction_match"] = True
                if pred_action.get("angle") == gt_action.get("angle"):
                    result["exact_match"] = True
        
        if pred_action["type"] == "move":
            result["direction_match"] = True
            if pred_action.get("distance") == gt_action.get("distance"):
                result["exact_match"] = True
    
    return result

def evaluate_model(model, tokenizer, test_data, data_dir, num_images, image_size, limit=None):
    """
    在 test set 上评估模型 (离散 Action Space)
    
    使用与训练完全相同的:
    1. 图像采样策略 (sample_image_indices_for_training)
    2. Navigation Prompt (build_navigation_prompt)
    """
    FastVisionModel.for_inference(model)
    
    results = {
        "total": 0,
        "exact_match": 0,
        "type_match": 0,
        "direction_match": 0,
        "valid_predictions": 0,
        "unknown_predictions": 0,
        "predictions": [],
        "gt_type_distribution": {"turn": 0, "move": 0, "stop": 0, "unknown": 0},
        "pred_type_distribution": {"turn": 0, "move": 0, "stop": 0, "unknown": 0},
        "exact_match_snapped": 0,
        "config": {
            "num_images": num_images,
            "image_size": image_size,
        },
        # Turn direction detailed statistics
        "turn_stats": {
            "total_turn_gt": 0,           # GT is turn total count
            "type_match_turn": 0,         # prediction is also turn
            "direction_correct": 0,       # direction correct (left/right)
            "direction_wrong": 0,         # direction wrong
            "angle_correct": 0,           # direction and angle correct
            # confusion matrix
            "gt_left_pred_left": 0,
            "gt_left_pred_right": 0,
            "gt_right_pred_left": 0,
            "gt_right_pred_right": 0,
        },
        # Move distance statistics
        "move_stats": {
            "total_move_gt": 0,
            "type_match_move": 0,
            "distance_correct": 0,
        },
    }
    
    logger.info(f"Evaluation 配置: {num_images} 张图像 (前{num_images-1}历史 + 1当前), 尺寸 {image_size}")
    logger.info(f"合法角度: {VALID_ANGLES}")
    logger.info(f"合法距离: {VALID_DISTANCES}")
    logger.info(f"生成设置: temperature=None, do_sample=False (deterministic)")
    
    for item in tqdm(test_data[:limit] if limit else test_data, desc="Evaluating"):
        msgs = item["messages"]
        img_paths = item["images"]
        
        # ===== use the same sampling strategy as training! =====
        selected_indices = sample_image_indices_for_training(len(img_paths), num_images)
        selected_paths = [img_paths[i] for i in selected_indices]
        
        images = []
        for img_path in selected_paths:
            full_path = data_dir / img_path
            if not full_path.exists():
                continue
            img = load_and_resize_image(full_path, image_size)
            if img is not None:
                images.append(img)
        
        if len(images) == 0:
            continue
        
        # Get raw instruction and GT
        user_text_raw = msgs[0]["content"]
        gt_answer = msgs[1]["content"]
        instruction = re.sub(r'<image>', '', user_text_raw).strip()
        
        # ===== use the same Navigation Prompt as training! =====
        nav_prompt = build_navigation_prompt(instruction, len(images))
        
        # Construct multi-image messages
        msg_content = [{"type": "image"} for _ in images] + [{"type": "text", "text": nav_prompt}]
        messages = [{"role": "user", "content": msg_content}]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        
        # Inference
        try:
            inputs = tokenizer(
                images,
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to("cuda")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    temperature=None,
                    do_sample=False,
                    use_cache=True,
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if input_text in generated:
                pred_text = generated[len(input_text):].strip()
            else:
                pred_text = generated.strip()
            
        except Exception as e:
            logger.warning(f"Error: {e}")
            continue
        
        gt_action = parse_action(gt_answer, snap_to_valid=False)
        pred_action_raw = parse_action(pred_text, snap_to_valid=False)
        pred_action_snapped = parse_action(pred_text, snap_to_valid=True)
        
        # Compare
        comparison = compare_actions_discrete(pred_action_raw, gt_action)
        comparison_snapped = compare_actions_discrete(pred_action_snapped, gt_action)
        
        results["total"] += 1
        results["gt_type_distribution"][gt_action["type"]] += 1
        results["pred_type_distribution"][pred_action_raw["type"]] += 1
        
        if pred_action_raw["type"] == "unknown":
            results["unknown_predictions"] += 1
        
        if comparison["exact_match"]:
            results["exact_match"] += 1
        if comparison["type_match"]:
            results["type_match"] += 1
        if comparison["direction_match"]:
            results["direction_match"] += 1
        if comparison["pred_is_valid"]:
            results["valid_predictions"] += 1
        if comparison_snapped["exact_match"]:
            results["exact_match_snapped"] += 1
        
        # ===== Turn analysis =====
        if gt_action["type"] == "turn":
            results["turn_stats"]["total_turn_gt"] += 1
            gt_dir = gt_action.get("direction")
            
            if pred_action_raw["type"] == "turn":
                results["turn_stats"]["type_match_turn"] += 1
                pred_dir = pred_action_raw.get("direction")
                
                # confusion matrix
                if gt_dir == "left" and pred_dir == "left":
                    results["turn_stats"]["gt_left_pred_left"] += 1
                elif gt_dir == "left" and pred_dir == "right":
                    results["turn_stats"]["gt_left_pred_right"] += 1
                elif gt_dir == "right" and pred_dir == "left":
                    results["turn_stats"]["gt_right_pred_left"] += 1
                elif gt_dir == "right" and pred_dir == "right":
                    results["turn_stats"]["gt_right_pred_right"] += 1
                
                # direction correctness
                if gt_dir == pred_dir:
                    results["turn_stats"]["direction_correct"] += 1
                    if gt_action.get("angle") == pred_action_raw.get("angle"):
                        results["turn_stats"]["angle_correct"] += 1
                else:
                    results["turn_stats"]["direction_wrong"] += 1
        
        # ===== Move distance statistics =====
        if gt_action["type"] == "move":
            results["move_stats"]["total_move_gt"] += 1
            if pred_action_raw["type"] == "move":
                results["move_stats"]["type_match_move"] += 1
                if gt_action.get("distance") == pred_action_raw.get("distance"):
                    results["move_stats"]["distance_correct"] += 1
        
        results["predictions"].append({
            "ground_truth": gt_answer,
            "prediction": pred_text,
            "gt_action": gt_action,
            "pred_action_raw": pred_action_raw,
            "pred_action_snapped": pred_action_snapped,
            "comparison": comparison,
            "comparison_snapped": comparison_snapped,
            "num_images_used": len(images),
        })
    
    # calculate overall metrics
    total = results["total"]
    if total > 0:
        results["exact_match_accuracy"] = results["exact_match"] / total
        results["type_accuracy"] = results["type_match"] / total
        results["direction_accuracy"] = results["direction_match"] / total
        results["valid_prediction_rate"] = results["valid_predictions"] / total
        results["unknown_rate"] = results["unknown_predictions"] / total
        results["exact_match_accuracy_snapped"] = results["exact_match_snapped"] / total
    else:
        results["exact_match_accuracy"] = 0
        results["type_accuracy"] = 0
        results["direction_accuracy"] = 0
        results["valid_prediction_rate"] = 0
        results["unknown_rate"] = 0
        results["exact_match_accuracy_snapped"] = 0
    
    # ===== Calculate Turn detailed accuracy =====
    ts = results["turn_stats"]
    if ts["total_turn_gt"] > 0:
        ts["turn_type_accuracy"] = ts["type_match_turn"] / ts["total_turn_gt"]
    else:
        ts["turn_type_accuracy"] = 0
    
    if ts["type_match_turn"] > 0:
        ts["turn_direction_accuracy"] = ts["direction_correct"] / ts["type_match_turn"]
        ts["turn_angle_accuracy"] = ts["angle_correct"] / ts["type_match_turn"]
        ts["turn_confusion_rate"] = ts["direction_wrong"] / ts["type_match_turn"]
    else:
        ts["turn_direction_accuracy"] = 0
        ts["turn_angle_accuracy"] = 0
        ts["turn_confusion_rate"] = 0
    
    # Left turn / Right turn accuracy
    gt_left_total = ts["gt_left_pred_left"] + ts["gt_left_pred_right"]
    gt_right_total = ts["gt_right_pred_left"] + ts["gt_right_pred_right"]
    ts["turn_left_accuracy"] = ts["gt_left_pred_left"] / gt_left_total if gt_left_total > 0 else 0
    ts["turn_right_accuracy"] = ts["gt_right_pred_right"] / gt_right_total if gt_right_total > 0 else 0
    
    # ===== Calculate Move detailed accuracy =====
    ms = results["move_stats"]
    if ms["total_move_gt"] > 0:
        ms["move_type_accuracy"] = ms["type_match_move"] / ms["total_move_gt"]
    else:
        ms["move_type_accuracy"] = 0
    
    if ms["type_match_move"] > 0:
        ms["move_distance_accuracy"] = ms["distance_correct"] / ms["type_match_move"]
    else:
        ms["move_distance_accuracy"] = 0
    
    return results

def verify_lora_weights(lora_path):
    """验证 LoRA 权重是否真的被训练了"""
    try:
        with safe_open(str(lora_path / "adapter_model.safetensors"), framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                n_zeros = (tensor == 0).sum() / tensor.numel()
                assert(n_zeros.item() != tensor.numel())
        logger.info("✓ LoRA weights verified (non-zero)")
        return True
    except Exception as e:
        logger.error(f"LoRA verification failed: {e}")
        return False

# ===== Main Function =====

def main():
    logger.info("="*70)
    logger.info("Qwen3-VL R2R SFT Training (HPC Mode) - Full Version")
    logger.info(f"Start time: {datetime.now()}")
    logger.info(f"DATA_DIR: {DATA_DIR}")
    logger.info(f"OUTPUT_DIR: {OUTPUT_DIR}")
    logger.info("="*70)
    
    # Print configuration
    logger.info("\n【训练配置】")
    for k, v in CONFIG.items():
        logger.info(f"  {k}: {v}")
    
    # check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"\nGPU: {gpu_name}, Memory: {gpu_mem:.1f} GB")
    else:
        logger.error("No GPU available! Exiting...")
        sys.exit(1)
    
    # ===== 1. Load data =====
    logger.info("\n" + "="*60)
    logger.info("【Step 1】Loading data...")
    logger.info("="*60)
    
    with open(SHAREGPT_PATH, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    logger.info(f"Total samples: {len(all_data)}")
    
    def validate_item(item):
        msgs = item.get("messages", [])
        imgs = item.get("images", [])
        if len(msgs) < 2 or len(imgs) < 1:
            return False
        if msgs[0].get("role") != "user" or msgs[1].get("role") != "assistant":
            return False
        return True
    
    valid_data = [item for item in all_data if validate_item(item)]
    logger.info(f"Valid samples: {len(valid_data)}")
    
    # Train/Test Split (80/20)
    random.seed(42)
    random.shuffle(valid_data)
    split_idx = int(len(valid_data) * 0.8)
    train_data = valid_data[:split_idx]
    test_data = valid_data[split_idx:]
    logger.info(f"Train: {len(train_data)}, Test: {len(test_data)}")
    
    # ===== 2. Load model =====
    logger.info("\n" + "="*60)
    logger.info("【Step 2】Loading model (Unsloth FastVisionModel)...")
    logger.info("="*60)
    
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=CONFIG["model_name"],
        max_seq_length=CONFIG["max_seq_length"],
        load_in_4bit=True,
    )
    
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=CONFIG["lora_rank"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
        use_gradient_checkpointing="unsloth",  # Unsloth optimization
    )
    logger.info(f"✓ Model loaded with LoRA rank={CONFIG['lora_rank']}")
    
    # ===== 3. Prepare dataset =====
    logger.info("\n" + "="*60)
    logger.info("【Step 3】Preparing dataset...")
    logger.info("="*60)
    
    num_images = CONFIG["num_images"]
    image_size = CONFIG["image_size"]
    
    converted_train_dataset = R2RVisionDataset(
        train_data, DATA_DIR, 
        num_images=num_images,
        image_size=image_size
    )
    logger.info(f"Training dataset size: {len(converted_train_dataset)}")
    logger.info(f"每个样本使用 {num_images} 张图像, 尺寸 {image_size}")
    logger.info(f"采样策略: 前 {num_images-1} 张历史观察 (均匀采样) + 1 张当前观察 (最后一帧)")
    
    # ===== 4. train =====
    logger.info("\n" + "="*60)
    logger.info("【Step 4】Starting training...")
    logger.info("="*60)
    
    FastVisionModel.for_training(model)  # Switch to training mode
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=converted_train_dataset,
        args=SFTConfig(
            per_device_train_batch_size=CONFIG["batch_size"],
            gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
            num_train_epochs=CONFIG["num_train_epochs"],
            max_steps=CONFIG["max_steps"],
            learning_rate=CONFIG["learning_rate"],
            warmup_ratio=CONFIG["warmup_ratio"],
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=CONFIG["logging_steps"],
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=str(OUTPUT_DIR / "sft_checkpoints"),
            save_strategy="steps",
            save_steps=CONFIG["save_steps"],
            report_to="none",
            # For vision models
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=4,
            max_seq_length=CONFIG["max_seq_length"],
        ),
    )
    
    # Display GPU memory
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    logger.info(f"{start_gpu_memory} GB of memory reserved before training.")
    
    # Start training
    trainer_stats = trainer.train()
    
    # Display GPU memory usage after training
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    
    train_time = trainer_stats.metrics['train_runtime']
    logger.info(f"\n【训练完成】")
    logger.info(f"  训练时间: {train_time:.0f} 秒 ({train_time/60:.2f} 分钟)")
    logger.info(f"  Peak reserved memory = {used_memory} GB")
    logger.info(f"  Peak reserved memory for training = {used_memory_for_lora} GB")
    logger.info(f"  Peak reserved memory % of max memory = {used_percentage}%")
    
    # ===== 5. Save model =====
    logger.info("\n" + "="*60)
    logger.info("【Step 5】Saving model...")
    logger.info("="*60)
    
    lora_path = OUTPUT_DIR / "r2r_sft_lora"
    model.save_pretrained(str(lora_path))
    tokenizer.save_pretrained(str(lora_path))
    logger.info(f"Model saved to {lora_path}")
    
    # Verify LoRA weights
    verify_lora_weights(lora_path)
    
    # ===== 6. Evaluation =====
    logger.info("\n" + "="*60)
    logger.info("【Step 6】Evaluating on test set...")
    logger.info("="*60)
    
    eval_results = evaluate_model(
        model, tokenizer, test_data, DATA_DIR,
        num_images=num_images,
        image_size=image_size,
        limit=CONFIG["eval_samples"]
    )
    
    # Print evaluation results
    logger.info(f"\n{'='*60}")
    logger.info(f"评估结果 (共 {eval_results['total']} 样本) - 离散 Action Space")
    logger.info(f"{'='*60}")
    
    logger.info(f"\n【准确率指标】")
    logger.info(f"  Exact Match Accuracy:     {eval_results['exact_match_accuracy']:.2%}")
    logger.info(f"  Action Type Accuracy:     {eval_results['type_accuracy']:.2%}")
    logger.info(f"  Direction Accuracy:       {eval_results['direction_accuracy']:.2%}")
    
    logger.info(f"\n【鲁棒性指标 (Robustness)】")
    logger.info(f"  Valid Prediction Rate:    {eval_results['valid_prediction_rate']:.2%}")
    logger.info(f"  Unknown Rate (解析失败):  {eval_results['unknown_rate']:.2%}")
    
    logger.info(f"\n【Snap 后的准确率 (仅供参考)】")
    logger.info(f"  Exact Match (snapped):    {eval_results['exact_match_accuracy_snapped']:.2%}")
    
    logger.info(f"\n【GT 动作类型分布】")
    for action_type, count in eval_results['gt_type_distribution'].items():
        if count > 0:
            logger.info(f"  {action_type}: {count} ({count/eval_results['total']:.1%})")
    
    logger.info(f"\n【预测动作类型分布】")
    for action_type, count in eval_results['pred_type_distribution'].items():
        if count > 0:
            logger.info(f"  {action_type}: {count} ({count/eval_results['total']:.1%})")
    
    # ===== Turn analysis =====
    ts = eval_results["turn_stats"]
    logger.info(f"\n{'='*60}")
    logger.info(f"【Turn 方向细分统计】")
    logger.info(f"{'='*60}")
    logger.info(f"  GT 是 turn 的样本数: {ts['total_turn_gt']}")
    logger.info(f"  预测也是 turn:       {ts['type_match_turn']} ({ts['turn_type_accuracy']:.1%})")
    logger.info(f"  方向正确 (left/right): {ts['direction_correct']} ({ts['turn_direction_accuracy']:.1%} of type-matched)")
    logger.info(f"  方向错误:              {ts['direction_wrong']} ({ts['turn_confusion_rate']:.1%} of type-matched)")
    logger.info(f"  角度也正确:            {ts['angle_correct']} ({ts['turn_angle_accuracy']:.1%} of type-matched)")
    
    logger.info(f"\n  【左转/右转分别准确率】")
    logger.info(f"    Turn Left Accuracy:  {ts['turn_left_accuracy']:.2%}  (GT=left, Pred=left)")
    logger.info(f"    Turn Right Accuracy: {ts['turn_right_accuracy']:.2%}  (GT=right, Pred=right)")
    
    logger.info(f"\n  【Turn 方向混淆矩阵】")
    logger.info(f"                      预测 Left    预测 Right")
    logger.info(f"    GT Left           {ts['gt_left_pred_left']:>5}        {ts['gt_left_pred_right']:>5}")
    logger.info(f"    GT Right          {ts['gt_right_pred_left']:>5}        {ts['gt_right_pred_right']:>5}")
    
    # ===== Move distance analysis =====
    ms = eval_results["move_stats"]
    logger.info(f"\n【Move 距离统计】")
    logger.info(f"  GT 是 move 的样本数: {ms['total_move_gt']}")
    logger.info(f"  预测也是 move:       {ms['type_match_move']} ({ms['move_type_accuracy']:.1%})")
    logger.info(f"  距离也正确:          {ms['distance_correct']} ({ms['move_distance_accuracy']:.1%} of type-matched)")
    
    # ===== 7. Save evaluation results =====
    logger.info("\n" + "="*60)
    logger.info("【Step 7】Saving evaluation results...")
    logger.info("="*60)
    
    # Save summary results
    eval_output = {
        "model": "Qwen3-VL-8B-Instruct + SFT LoRA (Unsloth)",
        "task": "R2R Action Prediction",
        "config": CONFIG,
        "train_samples": len(converted_train_dataset),
        "test_samples": eval_results["total"],
        "training_time_seconds": train_time,
        "training_time_minutes": train_time / 60,
        "generation_config": {
            "temperature": None,
            "do_sample": False,
            "max_new_tokens": 64,
            "note": "Deterministic generation for reproducible evaluation"
        },
        "action_space": {
            "angles": VALID_ANGLES,
            "distances": VALID_DISTANCES,
            "action_types": ["turn", "move", "stop"],
        },
        "metrics_strict": {
            "exact_match_accuracy": eval_results["exact_match_accuracy"],
            "type_accuracy": eval_results["type_accuracy"],
            "direction_accuracy": eval_results["direction_accuracy"],
        },
        "robustness_metrics": {
            "valid_prediction_rate": eval_results["valid_prediction_rate"],
            "unknown_rate": eval_results["unknown_rate"],
            "note": "unknown_rate = 无法解析为合法动作的预测比例"
        },
        "metrics_snapped": {
            "exact_match_accuracy": eval_results["exact_match_accuracy_snapped"],
            "note": "预测值 snap 到最近离散值后的准确率 (仅供参考)",
        },
        "counts": {
            "exact_match": eval_results["exact_match"],
            "type_match": eval_results["type_match"],
            "direction_match": eval_results["direction_match"],
            "valid_predictions": eval_results["valid_predictions"],
            "unknown_predictions": eval_results["unknown_predictions"],
        },
        "gt_type_distribution": eval_results["gt_type_distribution"],
        "pred_type_distribution": eval_results["pred_type_distribution"],
        # Turn
        "turn_stats": eval_results["turn_stats"],
        # Move 
        "move_stats": eval_results["move_stats"],
    }
    
    eval_json_path = OUTPUT_DIR / "evaluation_results.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_output, f, ensure_ascii=False, indent=2)
    logger.info(f"✓ 评估摘要已保存到: {eval_json_path}")
    
    # 保存完整预测结果
    all_predictions_path = OUTPUT_DIR / "all_predictions.json"
    with open(all_predictions_path, "w", encoding="utf-8") as f:
        json.dump(eval_results["predictions"], f, ensure_ascii=False, indent=2)
    logger.info(f"✓ 完整预测结果已保存到: {all_predictions_path}")
    logger.info(f"  共 {len(eval_results['predictions'])} 条记录")
    
    # 保存错误样本
    error_predictions = [
        p for p in eval_results["predictions"] 
        if not p["comparison"]["exact_match"]
    ]
    error_predictions_path = OUTPUT_DIR / "error_predictions.json"
    with open(error_predictions_path, "w", encoding="utf-8") as f:
        json.dump(error_predictions, f, ensure_ascii=False, indent=2)
    logger.info(f"✓ 错误预测已保存到: {error_predictions_path}")
    logger.info(f"  共 {len(error_predictions)} 条错误记录")
    
    # ===== Done =====
    logger.info("\n" + "="*70)
    logger.info("All done!")
    logger.info(f"Finished at {datetime.now()}")
    logger.info("="*70)
    logger.info(f"\n输出文件:")
    logger.info(f"  - {lora_path}/ (LoRA 权重)")
    logger.info(f"  - {eval_json_path} (评估结果)")
    logger.info(f"  - {all_predictions_path} (完整预测)")
    logger.info(f"  - {error_predictions_path} (错误样本)")

if __name__ == "__main__":
    main()
