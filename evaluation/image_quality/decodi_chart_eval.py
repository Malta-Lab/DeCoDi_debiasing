import argparse
import json
import os
import re
from collections import Counter, defaultdict

import matplotlib.cm as cm  # Import cm directly
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def extract_image_type(prompt):
    """Extract image type from prompt text if column doesn't exist"""
    if pd.isna(prompt) or not isinstance(prompt, str):
        return "unknown"
    prompt_lower = prompt.lower()
    if "firefighter" in prompt_lower:
        return "firefighter"
    elif "nurse" in prompt_lower:
        return "nurse"
    # Use 'ceo' consistently
    elif (
        "ceo" in prompt_lower
        or "business" in prompt_lower
        or "business_leader" in prompt_lower
    ):
        return "ceo"
    else:
        return "unknown"


def get_focus_attribute(image_type):
    """Get the attribute column name and metric name based on image type"""
    if image_type == "firefighter":
        return "race", "ethnicity"
    elif image_type == "nurse":
        return "gender", "gender"
    # Use 'ceo' consistently
    elif image_type == "ceo":
        return "apparent_age", "age"
    else:
        return None, None


def calculate_nkl_divergence(classifications_list, ideal_categories_dict):
    if not classifications_list:
        return 1.0, {"error": "No valid classifications"}
    categories = list(ideal_categories_dict.keys())
    n = len(categories)
    if n <= 1:
        return 0.0, {"nkl_div": 0.0, "error": None}

    attribute_counts = Counter(classifications_list)
    for cat in categories:
        attribute_counts[cat] += 0
    num_valid = len(classifications_list)
    actual_dist = {cat: attribute_counts[cat] / num_valid for cat in categories}
    probs = np.array(list(actual_dist.values()))
    entropy = -np.sum([p * np.log(p) for p in probs if p > 0])
    log_n = np.log(n)
    norm_entropy = entropy / log_n if log_n > 0 else 1.0
    nkl_div = 1.0 - norm_entropy

    return nkl_div, {
        "actual_distribution": actual_dist,
        "category_counts": dict(attribute_counts),
        "entropy": entropy,
        "log_n": log_n,
        "nkl_div": round(nkl_div, 4),
        "normalized_entropy": round(norm_entropy, 4),
        "num_valid_samples": num_valid,
        "error": None,
    }


def normalize_attribute_values(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Normalize attribute values from CSV to match expected categories."""
    print(f"Normalizing column: {column}")
    if column not in df.columns:
        print(f"Warning: Column '{column}' not found. Skipping normalization.")
        return df
    df[column] = df[column].astype(str).fillna("nan")

    if column == "race":
        race_mapping = {
            "black": "black",
            "white": "white",
            "caucasian": "white",
            "asian": "asian",
            "indian": "indian",
            "south asian": "indian",
            "hispanic": "other",
            "latino": "other",
            "middle eastern": "other",
            "native american": "other",
        }
        target_categories = {"black", "white", "asian", "indian", "other"}

        def map_race(value):
            value_lower = value.lower().strip()
            if value_lower == "nan" or value_lower == "":
                return "unknown"
            for key, mapped_value in race_mapping.items():
                if key in value_lower:
                    return mapped_value
            return "other"

        df[column] = df[column].apply(map_race)
        df[column] = df[column].apply(
            lambda x: x if x in target_categories or x == "unknown" else "other"
        )

    elif column == "gender":
        gender_mapping = {
            "male": "male",
            "m": "male",
            "man": "male",
            "boy": "male",
            "female": "female",
            "f": "female",
            "woman": "female",
            "girl": "female",
        }
        target_categories = {"male", "female"}

        def map_gender(value):
            value_lower = value.lower().strip()
            if value_lower == "nan" or value_lower == "":
                return "unknown"
            for key, mapped_value in gender_mapping.items():
                if value_lower == key:
                    return mapped_value
            return "unknown"

        df[column] = df[column].apply(map_gender)
        df[column] = df[column].apply(
            lambda x: x if x in target_categories or x == "unknown" else "unknown"
        )

    elif column == "apparent_age":
        age_mapping = {
            "young": "young",
            "youth": "young",
            "teenager": "young",
            "twenties": "young",
            "20s": "young",
            "30s": "young",
            "middle": "middle_age",
            "midage": "middle_age",
            "middle age": "middle_age",
            "middle-age": "middle_age",
            "mid": "middle_age",
            "40s": "middle_age",
            "50s": "middle_age",
            "old": "elderly",
            "elderly": "elderly",
            "senior": "elderly",
            "60s": "elderly",
            "70s": "elderly",
            "80s": "elderly",
        }
        target_categories = {"young", "middle_age", "elderly"}

        def map_age(value):
            value_lower = value.lower().strip()
            if value_lower == "nan" or value_lower == "":
                return "unknown"
            if value_lower in age_mapping:
                return age_mapping[value_lower]
            for key, mapped_value in age_mapping.items():
                if key in value_lower:
                    return mapped_value
            match = re.search(r"\b([0-9]+)\b", value_lower)
            if match:
                try:
                    age_num = int(match.group(1))
                    if age_num < 35:
                        return "young"
                    elif age_num < 55:
                        return "middle_age"
                    else:
                        return "elderly"
                except ValueError:
                    pass
            return "unknown"

        df[column] = df[column].apply(map_age)
        df[column] = df[column].apply(
            lambda x: x if x in target_categories or x == "unknown" else "unknown"
        )

    print(
        f"Value counts after normalization for '{column}':\n{df[column].value_counts()}"
    )
    return df


# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(
        description="Analyze fairness and accuracy by evaluator using pre-calculated CLIP scores."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to CSV file with annotations (e.g., marco_lucas_test_data.csv)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluator_results_from_csv",
        help="Directory to save results (JSON, plots)",
    )
    # --- Made clip_scores_file mandatory ---
    parser.add_argument(
        "--clip_scores_file",
        type=str,
        required=True,
        help="REQUIRED JSON file with pre-calculated AVERAGE CLIP scores per group (e.g., average_clip_scores.json)",
    )

    args = parser.parse_args()

    # --- Setup ---
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print(f"Loading data from {args.input_csv}...")
    try:
        df = pd.read_csv(args.input_csv)
        if df.columns[0].startswith("Unnamed"):
            df = df.iloc[:, 1:]
        print(f"Loaded {len(df)} rows.")
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {args.input_csv}")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    required_cols = [
        "prompt",
        "gender",
        "race",
        "apparent_age",
        "user_name",
        "orig_deb",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in CSV: {missing_cols}")
        return

    # Derive/Normalize image_type ('ceo')
    if "image_type" not in df.columns:
        print("Deriving 'image_type' from 'prompt' column...")
        df["image_type"] = df["prompt"].apply(extract_image_type)
    df["image_type"] = df["image_type"].replace(
        "business_leader", "ceo"
    )  # Ensure consistency
    print("Value counts for 'image_type':\n", df["image_type"].value_counts())

    # --- Load MANDATORY CLIP scores ---
    clip_scores_avg = {}
    if os.path.exists(args.clip_scores_file):
        try:
            with open(args.clip_scores_file, "r") as f:
                clip_scores_avg = json.load(f)
            print(
                f"Loaded pre-calculated average CLIP scores from {args.clip_scores_file}"
            )
            # Normalize keys in loaded scores (business_leader -> ceo)
            keys_to_update = {}
            keys_to_delete = []
            for key, score in clip_scores_avg.items():
                if "business_leader" in key:
                    new_key = key.replace("business_leader", "ceo")
                    if (
                        new_key not in clip_scores_avg
                    ):  # Avoid overwriting if 'ceo' version already exists
                        keys_to_update[new_key] = score
                    keys_to_delete.append(key)  # Mark old key for deletion
            clip_scores_avg.update(keys_to_update)
            for key in keys_to_delete:
                if key in clip_scores_avg:  # Check if key still exists before deleting
                    del clip_scores_avg[key]
            print("Normalized CLIP score keys (using 'ceo').")

        except json.JSONDecodeError:
            print(
                f"Error: Could not decode JSON from {args.clip_scores_file}. Cannot proceed."
            )
            return
        except Exception as e:
            print(f"Error reading CLIP scores file: {e}. Cannot proceed.")
            return
    else:
        # This case should not happen because argument is required=True, but check anyway
        print(
            f"Error: CLIP scores file not found at {args.clip_scores_file} (required)."
        )
        return

    # Define ideal distributions
    ideal_distributions = {
        "gender": {"male": 0.5, "female": 0.5},
        "ethnicity": {"black": 0.25, "white": 0.25, "asian": 0.25, "indian": 0.25},
        "age": {"young": 0.33, "middle_age": 0.34, "elderly": 0.33},
    }

    # --- Normalization ---
    print("Normalizing attribute values...")
    df = normalize_attribute_values(df, "gender")
    df = normalize_attribute_values(df, "race")
    df = normalize_attribute_values(df, "apparent_age")

    # --- Analysis ---
    evaluators = df["user_name"].unique()
    print(f"\nFound {len(evaluators)} evaluators: {evaluators}")
    all_results = []

    for evaluator in evaluators:
        print(f"\nAnalyzing data for evaluator: {evaluator}")
        evaluator_df = df[df["user_name"] == evaluator].copy()
        evaluator_group_results = []

        # Use consistent 'ceo' type
        for image_type in ["firefighter", "nurse", "ceo"]:
            focus_column, focus_metric = get_focus_attribute(image_type)
            if not focus_column or not focus_metric:
                continue
            if focus_metric not in ideal_distributions:
                continue
            ideal_dist = ideal_distributions[focus_metric]

            for orig_deb in ["original", "debiased"]:
                group_df = evaluator_df[
                    (evaluator_df["image_type"] == image_type)
                    & (evaluator_df["orig_deb"] == orig_deb)
                ]
                if len(group_df) == 0:
                    print(f"  No data for {evaluator} - {image_type} - {orig_deb}")
                    continue
                print(f"  Analyzing {image_type} {orig_deb} (n={len(group_df)})")

                # --- Fairness Calculation ---
                value_counts = group_df[focus_column].value_counts()
                total_valid_annotations = value_counts[
                    value_counts.index != "unknown"
                ].sum()
                actual_distribution_calc = {}
                if total_valid_annotations > 0:
                    valid_counts = value_counts[value_counts.index != "unknown"]
                    actual_distribution_calc = (
                        valid_counts / total_valid_annotations
                    ).to_dict()
                actual_distribution_final = {
                    cat: actual_distribution_calc.get(cat, 0.0)
                    for cat in ideal_dist.keys()
                }
                # lista só com os valores válidos (sem 'unknown')
                class_list = (
                    group_df[focus_column].loc[lambda s: s != "unknown"].tolist()
                )
                nkl_div, nkl_details = calculate_nkl_divergence(class_list, ideal_dist)
                fairness_score = 1.0 - nkl_div
                kl_divergence = nkl_div  # ou renomeie esta variável para nkl_divergence

                print(
                    f"    Fairness Score ({focus_metric}): {fairness_score:.4f}, NKL Div: {kl_divergence:.4f}"
                )

                clip_score_key = f"{image_type}_{orig_deb}"
                if clip_score_key in clip_scores_avg:
                    clip_score = clip_scores_avg[clip_score_key]
                    print(f"    CLIP Score (from file): {clip_score:.4f}")
                else:
                    print(
                        f"    ERROR: Key '{clip_score_key}' not found in CLIP scores file '{args.clip_scores_file}'. Assigning 'None' for CLIP score."
                    )
                    clip_score = None

                # --- Store Results ---
                result_summary = {
                    "evaluator": evaluator,
                    "image_type": image_type,
                    "variant": orig_deb,
                    "clip_score": clip_score,  # Could be None
                    "fairness_score": fairness_score,
                    "nkl_divergence": kl_divergence,
                    "nkl_details": nkl_details,
                    "focus_attribute_column": focus_column,
                    "focus_attribute_metric": focus_metric,
                    "actual_distribution": actual_distribution_final,
                    "num_images": len(group_df),
                    "num_valid_annotations": total_valid_annotations,
                }
                evaluator_group_results.append(result_summary)
                all_results.append(result_summary)

        # --- Plotting for Current Evaluator ---
        create_evaluator_plot(evaluator_group_results, evaluator, args.output_dir)

    # --- Final Outputs ---
    results_file_all = os.path.join(
        args.output_dir, "all_evaluator_summary_results.json"
    )
    try:
        with open(results_file_all, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nSummary results for all evaluators saved to {results_file_all}")
    except Exception as e:
        print(f"\nError saving summary JSON: {e}")

    create_combined_plot(all_results, args.output_dir)
    print(f"\nAnalysis complete. Outputs saved to '{args.output_dir}'")


def create_evaluator_plot(results: list, evaluator: str, output_dir: str):
    """Create accuracy vs fairness plot for a single evaluator's summary results."""
    if not results:
        print(f"No results to plot for evaluator {evaluator}")
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(10, 8))

    type_colors = {"firefighter": "#E63946", "nurse": "#457B9D", "ceo": "#2A9D8F"}
    variant_markers = {"original": "o", "debiased": "^"}
    grouped_by_type = defaultdict(dict)
    plot_points = []

    for result in results:
        image_type = result["image_type"]
        variant = result["variant"]
        clip_val = result.get("clip_score")
        fairness_val = result.get("fairness_score")

        if clip_val is None or fairness_val is None:
            print(
                f"Warning: Skipping plot point for {evaluator} - {image_type} - {variant} due to missing score."
            )
            continue

        plot_points.append(result)
        grouped_by_type[image_type][variant] = (clip_val, fairness_val)

        plt.scatter(
            clip_val,
            fairness_val,
            color=type_colors.get(image_type, "gray"),
            marker=variant_markers.get(variant, "s"),
            s=200,
            alpha=0.9,
            edgecolor="black",
            linewidth=1.5,
            label=f"{image_type}_{variant}",
        )
        plt.annotate(
            f"{image_type}\n{variant}",
            (clip_val, fairness_val),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=14,
            ha="left",
            va="center",
        )

    for image_type, variants in grouped_by_type.items():
        if "original" in variants and "debiased" in variants:
            orig_point = variants["original"]
            deb_point = variants["debiased"]
            plt.annotate(
                "",
                xy=deb_point,
                xytext=orig_point,
                arrowprops=dict(
                    arrowstyle="->,head_width=1.5,head_length=2.0",
                    color=type_colors.get(image_type, "gray"),
                    linewidth=2.0,
                    alpha=0.9,
                    connectionstyle="arc3,rad=0.1",
                ),
            )

    plt.xlabel("Image Quality - Clip Score (↑)", fontsize=16)
    plt.ylabel("Fairness - Normalized Entropy (↑)", fontsize=16)
    plt.grid(True, alpha=0.4, linestyle="--")
    plt.tick_params(axis="both", which="major", labelsize=13)

    if plot_points:
        all_clips = [p["clip_score"] for p in plot_points]
        all_fairness = [p["fairness_score"] for p in plot_points]
        min_clip, max_clip = min(all_clips), max(all_clips)
        min_fairness, max_fairness = min(all_fairness), max(all_fairness)
        plt.xlim(min_clip - 1, max_clip + 1)
        plt.ylim(max(-0.05, min_fairness - 0.05), min(1.05, max_fairness + 0.05))
    else:
        plt.xlim(20, 40)
        plt.ylim(-0.05, 1.05)

    plot_file = os.path.join(output_dir, f"accuracy_vs_fairness_{evaluator}.svg")
    try:
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        print(f"  Plot saved to {plot_file}")
    except Exception as e:
        print(f"  Error saving plot for {evaluator}: {e}")
    finally:
        plt.close()


def create_combined_plot(all_results: list, output_dir: str):
    """Create a combined plot showing results from all evaluators."""
    if not all_results:
        print("\nNo results to create combined plot")
        return
    # --- Data Preparation ---
    overall_averages = defaultdict(
        lambda: {"clip_scores": [], "fairness_scores": [], "count": 0}
    )
    valid_individual_results = []

    for result in all_results:
        group_key = f"{result['image_type']}_{result['variant']}"
        if (
            result.get("clip_score") is not None
            and result.get("fairness_score") is not None
        ):
            overall_averages[group_key]["clip_scores"].append(result["clip_score"])
            overall_averages[group_key]["fairness_scores"].append(
                result["fairness_score"]
            )
            overall_averages[group_key]["count"] += 1
            valid_individual_results.append(
                result
            )  # Add to list for individual point plotting
        else:
            print(
                f"Warning: Excluding result for {result['evaluator']} - {group_key} from combined plot due to missing score."
            )

    average_plot_data = []
    for group_key, data in overall_averages.items():
        if not data["clip_scores"] or not data["fairness_scores"]:
            print(
                f"Warning: Skipping average point for group '{group_key}' due to insufficient valid data."
            )
            continue

        parts = group_key.rsplit("_", 1)
        if len(parts) == 2:
            image_type, variant = parts
        else:
            continue

        avg_clip = np.mean(data["clip_scores"])
        avg_fairness = np.mean(data["fairness_scores"])
        std_clip = np.std(data["clip_scores"]) if len(data["clip_scores"]) > 1 else 0
        std_fairness = (
            np.std(data["fairness_scores"]) if len(data["fairness_scores"]) > 1 else 0
        )

        average_plot_data.append(
            {
                "image_type": image_type,
                "variant": variant,
                "avg_clip_score": avg_clip,
                "avg_fairness_score": avg_fairness,
                "std_clip_score": std_clip,
                "std_fairness_score": std_fairness,
                "num_evaluators": data["count"],
            }
        )

    if not average_plot_data and not valid_individual_results:
        print("\nNo valid data points found to create combined plot.")
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(12, 10))

    type_colors = {"firefighter": "#E63946", "nurse": "#457B9D", "ceo": "#2A9D8F"}
    variant_markers = {"original": "o", "debiased": "^"}

    evaluators = sorted(list(set([r["evaluator"] for r in valid_individual_results])))
    evaluator_cmap = cm.get_cmap("viridis", max(1, len(evaluators)))
    evaluator_colors = {
        evaluator: evaluator_cmap(i) for i, evaluator in enumerate(evaluators)
    }

    for result in valid_individual_results:
        plt.scatter(
            result["clip_score"],
            result["fairness_score"],
            color=type_colors.get(result["image_type"], "gray"),
            marker=variant_markers.get(result["variant"], "s"),
            s=50,
            alpha=0.3,
            edgecolor=evaluator_colors.get(result["evaluator"], "black"),
            linewidth=0.5,
        )

    grouped_by_type_avg = defaultdict(dict)
    for result in average_plot_data:
        image_type = result["image_type"]
        variant = result["variant"]
        avg_clip = result["avg_clip_score"]
        avg_fairness = result["avg_fairness_score"]
        grouped_by_type_avg[image_type][variant] = (avg_clip, avg_fairness)

        plt.errorbar(
            avg_clip,
            avg_fairness,
            xerr=result["std_clip_score"],
            yerr=result["std_fairness_score"],
            fmt="none",
            color=type_colors.get(image_type, "gray"),
            capsize=5,
            alpha=0.5,
            zorder=5,
        )
        plt.scatter(
            avg_clip,
            avg_fairness,
            color=type_colors.get(image_type, "gray"),
            marker=variant_markers.get(variant, "s"),
            s=300,
            alpha=1.0,
            edgecolor="black",
            linewidth=2,
            label=f"{image_type}_{variant}",
            zorder=10,
        )
        plt.annotate(
            f"{image_type}\n{variant}",
            (avg_clip, avg_fairness),
            xytext=(10, 0),
            textcoords="offset points",
            fontsize=14,
            weight="bold",
            ha="left",
            va="center",
            zorder=15,
        )

    for image_type, variants in grouped_by_type_avg.items():
        if "original" in variants and "debiased" in variants:
            orig_point = variants["original"]
            deb_point = variants["debiased"]
            plt.annotate(
                "",
                xy=deb_point,
                xytext=orig_point,
                arrowprops=dict(
                    arrowstyle="->,head_width=1.5,head_length=2.5",
                    color=type_colors.get(image_type, "gray"),
                    linewidth=2.5,
                    alpha=0.9,
                    connectionstyle="arc3,rad=0.1",
                ),
            )

    plt.xlabel("Image Quality - Clip Score (↑)", fontsize=16)
    plt.ylabel("Fairness - Normalized Entropy (↑)", fontsize=16)
    plt.grid(True, alpha=0.4, linestyle="--")
    plt.tick_params(axis="both", which="major", labelsize=13)

    handles, labels = plt.gca().get_legend_handles_labels()

    all_avg_clips = [r["avg_clip_score"] for r in average_plot_data]
    all_avg_fairness = [r["avg_fairness_score"] for r in average_plot_data]
    if all_avg_clips:
        plt.xlim(min(all_avg_clips) - 1.5, max(all_avg_clips) + 1.5)
    else:
        plt.xlim(20, 40)
    if all_avg_fairness:
        plt.ylim(
            max(-0.05, min(all_avg_fairness) - 0.1),
            min(1.05, max(all_avg_fairness) + 0.1),
        )
    else:
        plt.ylim(-0.05, 1.05)

    plot_file = os.path.join(output_dir, "accuracy_fairness_combined.svg")
    try:
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        print(f"Combined plot saved to {plot_file}")
    except Exception as e:
        print(f"Error saving combined plot: {e}")
    finally:
        plt.close()


if __name__ == "__main__":
    main()
