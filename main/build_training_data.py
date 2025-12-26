# build_training_data.py
import pandas as pd
import os


INPUT_PATH = "resume_data.csv"      # your Kaggle file
OUTPUT_PATH = "training_data.csv"   # will be created


def build_text(row, cols):
    """Safely join multiple text columns into one block."""
    parts = []
    for c in cols:
        if c in row:
            val = row[c]
            if isinstance(val, str):
                t = val.strip()
                if t:
                    parts.append(t)
    return "\n".join(parts)


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Could not find {INPUT_PATH} in current folder")

    df = pd.read_csv(INPUT_PATH)

    # Columns from your Kaggle file:
    # - Resume side
    # - Job description side
    # - Label: matched_score
    jd_cols = [
        "\ufeffjob_position_name",   # note: BOM char at start
        "educationaL_requirements",
        "experiencere_requirement",
        "age_requirement",
        "responsibilities.1",
        "skills_required",
    ]

    resume_cols = [
        "career_objective",
        "skills",
        "degree_names",
        "major_field_of_studies",
        "professional_company_names",
        "positions",
        "responsibilities",
    ]

    if "matched_score" not in df.columns:
        raise KeyError("Column 'matched_score' not found in dataset")

    # Drop rows without score
    df = df.dropna(subset=["matched_score"])

    # High-confidence positives and negatives
    pos = df[df["matched_score"] >= 0.75]
    neg = df[df["matched_score"] <= 0.50]

    if pos.empty or neg.empty:
        raise ValueError("Not enough positive/negative samples after thresholding")

    # Balance classes
    n = min(len(pos), len(neg))
    pos = pos.sample(n, random_state=42)
    neg = neg.sample(n, random_state=42)

    balanced = pd.concat([pos, neg], axis=0).sample(frac=1.0, random_state=42)

    jd_texts = []
    resume_texts = []
    labels = []

    for _, row in balanced.iterrows():
        jd_text = build_text(row, jd_cols)
        res_text = build_text(row, resume_cols)

        # Fallback in case text is accidentally empty
        if not jd_text.strip():
            jd_text = str(row.get("\ufeffjob_position_name", ""))

        if not res_text.strip():
            res_text = str(row.get("career_objective", ""))

        jd_texts.append(jd_text)
        resume_texts.append(res_text)

        label = 1 if row["matched_score"] >= 0.75 else 0
        labels.append(label)

    out = pd.DataFrame({
        "jd_text": jd_texts,
        "resume_text": resume_texts,
        "label": labels,
    })

    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Saved {len(out)} samples to {OUTPUT_PATH}")
    print(out["label"].value_counts())


if __name__ == "__main__":
    main()
