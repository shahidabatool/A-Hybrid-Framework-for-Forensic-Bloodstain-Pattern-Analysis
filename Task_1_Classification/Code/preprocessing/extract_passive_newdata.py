import os
import shutil

ROOT = "/Users/shahidabatool/Desktop/MRP/Raw_dataset /Passive_drip_newdata"
DST_DIR = "/Users/shahidabatool/Desktop/MRP/Raw_dataset /Passive_drip_extracted"

def main():
    os.makedirs(DST_DIR, exist_ok=True)
    
    # Delete old files if any
    for f in os.listdir(DST_DIR):
        fpath = os.path.join(DST_DIR, f)
        if os.path.isfile(fpath):
            os.unlink(fpath)
            
    heights = ["paper25cm", "paper50cm", "paper75cm", "paper100cm"]
    copied_count = 0
    missing_count = 0
    
    print("=== Extraction Plan & Audit ===")
    for h in heights:
        h_path = os.path.join(ROOT, h)
        if not os.path.exists(h_path):
            print(f"[!] Height folder not found: {h_path}")
            continue
            
        subdirs = sorted([d for d in os.listdir(h_path) if os.path.isdir(os.path.join(h_path, d))])
        print(f"Processing {h}: {len(subdirs)} subfolders")
        
        h_copied = 0
        for d in subdirs:
            sub_path = os.path.join(h_path, d)
            
            # Find the best candidate file
            # Candidates in order of preference:
            # 1. d.jpg (e.g., 9-10.jpg)
            # 2. dedit.jpg or d_crop.jpg
            # 3. Any file containing d and ending in .jpg or .png
            candidates = [
                f"{d}.jpg",
                f"{d}.png",
                f"{d}edit.jpg",
                f"{d}-crop.jpg",
                f"{d}_cropped.jpg"
            ]
            
            selected_file = None
            for cand in candidates:
                cand_path = os.path.join(sub_path, cand)
                if os.path.exists(cand_path):
                    selected_file = cand
                    break
                    
            if not selected_file:
                # Fallback: search for any .jpg or .png containing the folder name
                all_files = os.listdir(sub_path)
                for f in all_files:
                    if d in f and f.lower().endswith(('.jpg', '.png', '.jpeg')):
                        selected_file = f
                        break
                        
            if selected_file:
                src_path = os.path.join(sub_path, selected_file)
                # Rename the file to prevent overwrite collisions (since folders in different heights share names)
                # E.g., paper50cm_9-10.jpg
                ext = os.path.splitext(selected_file)[1]
                new_filename = f"{h}_{d}{ext}"
                dst_path = os.path.join(DST_DIR, new_filename)
                
                shutil.copy(src_path, dst_path)
                h_copied += 1
                copied_count += 1
            else:
                print(f"  [!] No matching image found in {h}/{d} (files: {os.listdir(sub_path)})")
                missing_count += 1
                
        print(f"  -> Extracted {h_copied}/{len(subdirs)} images from {h}")
        
    print("\n=== Extraction Summary ===")
    print(f"Total images extracted: {copied_count}")
    print(f"Total subfolders missing images: {missing_count}")
    print(f"Extracted images saved in: {DST_DIR}")

if __name__ == "__main__":
    main()
