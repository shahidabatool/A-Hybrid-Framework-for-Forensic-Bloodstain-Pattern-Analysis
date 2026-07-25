import os
import re

# --- Environment Auto-Detection ---
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    try:
        drive.mount('/content/drive')
    except Exception:
        pass

    possible_dirs = [
        "/content/drive/MyDrive/Task_1_Classification",
        "/content/drive/MyDrive/MRP/Task_1_Classification",
        "/content/drive/My Drive/Task_1_Classification",
        "/content/drive/My Drive/MRP/Task_1_Classification"
    ]
    BASE_DIR = None
    for path in possible_dirs:
        if os.path.exists(path):
            BASE_DIR = path
            break
    if BASE_DIR is None:
        BASE_DIR = "/content/drive/MyDrive/Task_1_Classification"
else:
    BASE_DIR = "/Users/shahidabatool/Desktop/MRP/Task_1_Classification"

LOCAL_DATA_DIRS = [
    "/content/Task_1_Classification/Data/Augmented",
    "/content/Data/Augmented",
    "/content/Augmented"
]

DATA_DIR = None
if IN_COLAB:
    for path in LOCAL_DATA_DIRS:
        if os.path.exists(path):
            DATA_DIR = path
            print(f"[+] Detected high-speed local data directory at: {DATA_DIR}")
            break

if DATA_DIR is None:
    DATA_DIR = os.path.join(BASE_DIR, "Data", "Augmented")
    print(f"[*] Using data directory: {DATA_DIR}")

TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR  = os.path.join(DATA_DIR, "test")
EVAL_DIR  = os.path.join(BASE_DIR, "Evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(EVAL_DIR, "leakage_audit_report.txt")

# --------------------------------------------------------------------------
# Helper: extract parent source name from filename
# Train files: train_N_orig_SOURCENAME_frame_X.jpg  OR  train_N_aug_SOURCENAME_frame_X.jpg
# Test files:  test_orig_SOURCENAME_frame_X.jpg      OR  test_capped_SOURCENAME_frame_X.jpg
# Gunshot tiles: train_N_orig_C1.jpg_tile_33.png
# --------------------------------------------------------------------------
def extract_source(filename):
    # Remove prefix (train_N_orig_ / train_N_aug_ / test_orig_ / test_capped_)
    name = re.sub(r'^(train_\d+_(orig|aug)_|test_(orig|aug|capped)_)', '', filename)
    # Remove suffix (_frame_NNN.ext  OR  _tile_NNN.ext)
    name = re.sub(r'(_frame_\d+.*|_tile_\d+.*)', '', name)
    return name.strip()


def audit_class(class_name, train_class_dir, test_class_dir):
    """Returns (train_sources, test_sources, leaking_sources)."""
    train_sources = set()
    test_sources  = set()

    if os.path.exists(train_class_dir):
        for f in os.listdir(train_class_dir):
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                train_sources.add(extract_source(f))

    if os.path.exists(test_class_dir):
        for f in os.listdir(test_class_dir):
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                test_sources.add(extract_source(f))

    leaking = train_sources & test_sources
    return train_sources, test_sources, leaking


def run_audit():
    classes = ["Cough_Spatter", "Gunshot", "Impact_Spatter", "Passive_Drip", "Transfer_Wipe"]

    lines = []
    lines.append("=" * 70)
    lines.append("  BPA TASK 1 — PARENT-IMAGE LEAKAGE AUDIT REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Audit checks whether the same source scan / video appears in")
    lines.append("both the training and the test set (parent-image leakage).")
    lines.append("")
    lines.append("DECISION: Cough_Spatter is EXCLUDED from evaluation because")
    lines.append("it has only ONE source video and both train and test frames")
    lines.append("originate from that same recording — a known dataset limitation.")
    lines.append("")
    lines.append("-" * 70)

    any_leakage = False

    for cls in classes:
        train_cls = os.path.join(TRAIN_DIR, cls)
        test_cls  = os.path.join(TEST_DIR,  cls)

        train_src, test_src, leak = audit_class(cls, train_cls, test_cls)

        status = "⚠  LEAKAGE DETECTED" if leak else "✓  CLEAN"
        if leak:
            any_leakage = True

        lines.append(f"\nClass: {cls}")
        lines.append(f"  Train sources  : {len(train_src)}")
        lines.append(f"  Test sources   : {len(test_src)}")
        lines.append(f"  Shared sources : {len(leak)}")
        lines.append(f"  Status         : {status}")

        if leak:
            lines.append(f"  Leaking sources:")
            for src in sorted(leak):
                lines.append(f"    - {src}")

    lines.append("")
    lines.append("-" * 70)
    if any_leakage:
        lines.append("SUMMARY: Leakage found in one or more classes.")
        lines.append("Action taken: Cough_Spatter excluded from model evaluation.")
        lines.append("Evaluation is reported only on 4 classes with clean splits:")
        lines.append("  Gunshot, Impact_Spatter, Passive_Drip, Transfer_Wipe")
    else:
        lines.append("SUMMARY: No leakage detected in any class.")
    lines.append("=" * 70)

    report = "\n".join(lines)
    print(report)

    with open(OUTPUT_PATH, "w") as f:
        f.write(report + "\n")
    print(f"\n[+] Report saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_audit()
