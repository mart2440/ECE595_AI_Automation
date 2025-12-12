import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("results/figures", exist_ok=True)

# ---------- LOAD DATA ----------
def load_scores(filepath):
    df = pd.read_csv(
        filepath,
        engine="python",
        on_bad_lines="skip"
    )

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    return df

# ---------- GRAPH 1: Average score by variant ----------
def plot_avg_by_variant(df, model_name):
    avg_scores = df.groupby("variant")["score"].mean()

    plt.figure()
    avg_scores.plot(kind="bar")
    plt.ylim(0, 2)
    plt.ylabel("Average Score (0–2)")
    plt.title(f"{model_name}: Average Score by Prompt Variant")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"results/figures/{model_name}_avg_by_variant.png")
    plt.close()

# ---------- GRAPH 2: Score distribution ----------
def plot_score_distribution(df, model_name):
    score_counts = df["Score"].value_counts().sort_index()

    plt.figure()
    score_counts.plot(kind="bar")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title(f"{model_name}: Score Distribution")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"results/figures/{model_name}_score_distribution.png")
    plt.close()

# ---------- MAIN ----------
if __name__ == "__main__":
    # CHANGE FILENAMES AS NEEDED
    gemma = load_scores("gemma_scores.csv")
    llama = load_scores("llama_scores.csv")

    plot_avg_by_variant(gemma, "Gemma")
    plot_score_distribution(gemma, "Gemma")

    plot_avg_by_variant(llama, "LLaMA")
    plot_score_distribution(llama, "LLaMA")

    print("Graphs generated successfully.")
