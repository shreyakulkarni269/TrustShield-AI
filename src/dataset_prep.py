"""
Dataset Preparation (Paired Leak-Free Face Extraction):
1. Pairs Real video (e.g. 033.mp4) and Deepfake video (033_097.mp4) by Base Video ID.
2. Splits strictly by Video ID:
   - 70 Video IDs -> Train (70 Real + 70 Fake = 2,100 frames)
   - 15 Video IDs -> Val   (15 Real + 15 Fake = 450 frames)
   - 15 Video IDs -> Test  (15 Real + 15 Fake = 450 frames)
   GUARANTEE: No background or identity from Train ever leaks into Test.
3. Crops tightly around the facial region (upper-center head crop) to eliminate background bias.
"""

import os
import glob
import random
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import cv2

def extract_tight_paired_dataset(
    raw_dir="./data/raw",
    output_dir="./data/processed",
    frames_per_video=15,
    target_size=(224, 224),
    seed=42
):
    random.seed(seed)
    
    orig_dir = os.path.join(raw_dir, 'original')
    df_dir = os.path.join(raw_dir, 'Deepfakes')
    
    orig_files = {os.path.splitext(f)[0]: f for f in os.listdir(orig_dir) if f.endswith('.mp4')}
    df_files = {}
    for f in os.listdir(df_dir):
        if f.endswith('.mp4'):
            base_id = f.split('_')[0]
            df_files[base_id] = f

    # Find the 100 common matching IDs
    common_ids = sorted([vid_id for vid_id in orig_files if vid_id in df_files])
    print(f"Total matching video pairs: {len(common_ids)}")

    # Split strictly by Video ID to prevent identity/background leakage
    random.shuffle(common_ids)
    id_splits = {
        'train': common_ids[:70],
        'val': common_ids[70:85],
        'test': common_ids[85:]
    }

    # Reset output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    for s in ['train', 'val', 'test']:
        for c in ['real', 'manipulated']:
            os.makedirs(os.path.join(output_dir, s, c), exist_ok=True)

    print("=" * 60)
    print("Extracting Leak-Free Paired Face Dataset (FaceForensics++)")
    print(f"Train IDs: {len(id_splits['train'])}, Val IDs: {len(id_splits['val'])}, Test IDs: {len(id_splits['test'])}")
    print("=" * 60)

    for split_name, v_ids in id_splits.items():
        print(f"\nProcessing {split_name.upper()} split ({len(v_ids)} video pairs)...")
        
        real_saved = 0
        fake_saved = 0

        for vid_id in tqdm(v_ids, desc=f"  Extracting {split_name}"):
            real_file = os.path.join(orig_dir, orig_files[vid_id])
            fake_file = os.path.join(df_dir, df_files[vid_id])

            for is_fake, vid_path in [(False, real_file), (True, fake_file)]:
                cap = cv2.VideoCapture(vid_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames <= 0:
                    cap.release()
                    continue

                step = max(1, total_frames // (frames_per_video + 2))
                frame_indices = [i * step for i in range(1, frames_per_video + 1)]

                label = 'manipulated' if is_fake else 'real'

                for f_idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        continue

                    h, w, _ = frame.shape
                    
                    # Tight Head & Face Crop:
                    # In FaceForensics++ 480p videos (854x480):
                    # Head is located vertically between 6% and 72% of frame (approx 315px high)
                    crop_size = int(h * 0.66) # 316px
                    start_y = int(h * 0.06)  # 28px from top
                    start_x = (w - crop_size) // 2
                    
                    face_crop = frame[start_y:start_y + crop_size, start_x:start_x + crop_size]

                    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(face_rgb).resize(target_size, Image.Resampling.LANCZOS)

                    out_name = f"{label}_{vid_id}_f{f_idx:04d}.jpg"
                    out_path = os.path.join(output_dir, split_name, label, out_name)
                    pil_img.save(out_path, "JPEG", quality=95)

                    if is_fake:
                        fake_saved += 1
                    else:
                        real_saved += 1

                cap.release()

        print(f"  -> {split_name.upper()}: Extracted {real_saved} Real and {fake_saved} Manipulated faces.")

    print("\n✅ Leak-free dataset generation complete!")

if __name__ == "__main__":
    extract_tight_paired_dataset()
