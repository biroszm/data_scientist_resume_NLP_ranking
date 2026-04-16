import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# ==========================================
# 1) LOAD FILES
# ==========================================
df = pd.read_csv("Resume.csv")

with open("job_description.txt", "r", encoding="utf-8") as f:
    jd_text_raw = f.read()

# Detect resume text column
if "Resume_str" in df.columns:
    resume_col = "Resume_str"
elif "Resume.str" in df.columns:
    resume_col = "Resume.str"
else:
    raise KeyError("Could not find 'Resume_str' or 'Resume.str' in Resume.csv")

# ==========================================
# 2) BASIC CLEANING
# ==========================================
GENERIC_WORDS = {
    "good", "great", "excellent", "strong", "motivated", "passionate",
    "hardworking", "responsible", "experience", "experienced",
    "work", "worked", "working", "ability", "abilities", "skill", "skills",
    "knowledge", "professional", "including", "various", "multiple",
    "team", "dynamic", "dedicated", "proven", "success", "successful",
    "detail", "oriented", "results", "driven", "innovative", "creative",
    "using", "used", "use", "project", "projects", "year", "years",
    "candidate", "candidates", "role", "job", "position"
}

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "for", "to", "of", "in",
    "on", "at", "by", "with", "from", "as", "is", "are", "was", "were", "be", "been",
    "being", "this", "that", "these", "those", "it", "its", "their", "his", "her", "our",
    "your", "my", "we", "they", "he", "she", "you", "i", "will", "would", "can", "could",
    "should", "must", "may", "might", "into", "about", "over", "after", "before", "during",
    "within", "through", "across", "such", "than", "also", "not", "no", "yes"
}

def normalize_text(text):
    text = str(text).lower()
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9+#.\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_basic(text):
    text = normalize_text(text)
    tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9+#.]{1,}\b", text)
    tokens = [
        t for t in tokens
        if t not in STOPWORDS
        and t not in GENERIC_WORDS
        and len(t) > 1
    ]
    return tokens

jd_text = normalize_text(jd_text_raw)
jd_tokens = set(tokenize_basic(jd_text))

# ==========================================
# 3) DATA SCIENTIST HARD SKILLS
# ==========================================
HARD_SKILLS = {
    "python": [r"\bpython\b"],
    "sql": [r"\bsql\b", r"\bmysql\b", r"\bpostgresql\b", r"\bpostgres\b", r"\bsql server\b"],
    "r": [r"\br\b", r"\br language\b", r"\brstudio\b"],
    "pandas": [r"\bpandas\b"],
    "numpy": [r"\bnumpy\b"],
    "scikit_learn": [r"\bscikit learn\b", r"\bsklearn\b"],
    "tensorflow": [r"\btensorflow\b"],
    "pytorch": [r"\bpytorch\b"],
    "machine_learning": [r"\bmachine learning\b", r"\bml\b"],
    "deep_learning": [r"\bdeep learning\b", r"\bneural network[s]?\b"],
    "statistics": [
        r"\bstatistics\b", r"\bstatistical\b", r"\bhypothesis testing\b",
        r"\bprobability\b", r"\bregression\b", r"\bclassification\b",
        r"\bstatistical modeling\b"
    ],
    "data_visualization": [
        r"\bdata visualization\b", r"\bvisualisation\b", r"\bvisualization\b",
        r"\bmatplotlib\b", r"\bseaborn\b", r"\bplotly\b"
    ],
    "tableau": [r"\btableau\b"],
    "power_bi": [r"\bpower bi\b", r"\bpowerbi\b"],
    "excel": [r"\bexcel\b", r"\bmicrosoft excel\b"],
    "spark": [r"\bspark\b", r"\bpyspark\b", r"\bapache spark\b"],
    "hadoop": [r"\bhadoop\b"],
    "aws": [r"\baws\b", r"\bamazon web services\b", r"\bsagemaker\b"],
    "azure": [r"\bazure\b", r"\bazure ml\b", r"\bazure machine learning\b"],
    "gcp": [r"\bgcp\b", r"\bgoogle cloud\b", r"\bbigquery\b"],
    "nlp": [r"\bnlp\b", r"\bnatural language processing\b"],
    "computer_vision": [r"\bcomputer vision\b", r"\bimage classification\b", r"\bobject detection\b"],
    "time_series": [r"\btime series\b", r"\bforecasting\b"],
    "a_b_testing": [r"\ba ?b testing\b", r"\bab testing\b", r"\bexperiment design\b"],
    "etl": [r"\betl\b", r"\bdata pipeline[s]?\b", r"\bdata engineering\b"],
    "git": [r"\bgit\b", r"\bgithub\b", r"\bgitlab\b"]
}

# ==========================================
# 4) DATA SCIENTIST CONCEPT MAP
# ==========================================
CONCEPTS = {
    "data_analysis": [
        r"\bdata analysis\b",
        r"\banaly[sz]ed data\b",
        r"\banaly[sz]ing data\b",
        r"\bexploratory data analysis\b",
        r"\beda\b",
        r"\bderived insights\b",
        r"\bgenerated insights\b",
        r"\bdata driven insights\b",
        r"\binsight generation\b",
        r"\binsights from data\b"
    ],
    "machine_learning_modeling": [
        r"\bbuilt machine learning model[s]?\b",
        r"\bdeveloped machine learning model[s]?\b",
        r"\btrained model[s]?\b",
        r"\bpredictive model(?:ing)?\b",
        r"\bclassification model[s]?\b",
        r"\bregression model[s]?\b",
        r"\bforecasting model[s]?\b",
        r"\bmodel development\b",
        r"\bmodel building\b",
        r"\bdeployed model[s]?\b"
    ],
    "feature_engineering": [
        r"\bfeature engineering\b",
        r"\bengineered feature[s]?\b",
        r"\bfeature selection\b",
        r"\bfeature extraction\b",
        r"\bconstructed feature[s]?\b",
        r"\bcreated feature[s]?\b"
    ],
    "data_cleaning_preprocessing": [
        r"\bdata cleaning\b",
        r"\bcleaned data\b",
        r"\bdata preprocessing\b",
        r"\bpre processed data\b",
        r"\bpreprocessing\b",
        r"\bwrangled data\b",
        r"\bdata wrangling\b",
        r"\btransformed data\b",
        r"\bhandled missing data\b"
    ],
    "visualization_reporting": [
        r"\bcreated dashboard[s]?\b",
        r"\bbuilt dashboard[s]?\b",
        r"\bdesigned dashboard[s]?\b",
        r"\bdata visualization\b",
        r"\bvisualized data\b",
        r"\breporting\b",
        r"\bdeveloped reports\b",
        r"\bbusiness intelligence\b",
        r"\bbi dashboard[s]?\b"
    ],
    "stakeholder_communication": [
        r"\bpresented findings\b",
        r"\bpresented insight[s]?\b",
        r"\bcommunicated results\b",
        r"\bexplained model[s]?\b",
        r"\btranslated technical findings\b",
        r"\btranslated data into business\b",
        r"\bworked with stakeholders\b",
        r"\bstakeholder management\b",
        r"\bcross functional collaboration\b",
        r"\bpartnered with product\b",
        r"\bpartnered with business teams\b",
        r"\bcommunicated with clients\b"
    ],
    "business_problem_solving": [
        r"\bsolved business problem[s]?\b",
        r"\bbusiness problem solving\b",
        r"\bdecision support\b",
        r"\bdata driven decision making\b",
        r"\boptimized process(?:es)?\b",
        r"\bimproved business performance\b",
        r"\bidentified business opportunity\b"
    ],
    "experimentation": [
        r"\ba ?b testing\b",
        r"\bab testing\b",
        r"\bdesigned experiments\b",
        r"\bexperiment design\b",
        r"\bcausal inference\b",
        r"\btest hypothesis\b",
        r"\bhypothesis testing\b"
    ],
    "deployment_mlops": [
        r"\bmodel deployment\b",
        r"\bdeployed machine learning model[s]?\b",
        r"\bproductionized model[s]?\b",
        r"\bmlops\b",
        r"\bmodel monitoring\b",
        r"\bapi deployment\b",
        r"\bdeployed to production\b"
    ],
    "team_leadership": [
        r"\bled a team\b",
        r"\bleading a team\b",
        r"\bmanaged a team\b",
        r"\bteam lead\b",
        r"\bteam leadership\b",
        r"\bsupervised analyst[s]?\b",
        r"\bmentored junior data scientist[s]?\b",
        r"\bmanaged direct reports\b",
        r"\bcoordinated cross functional teams\b"
    ],
    "research_and_innovation": [
        r"\bresearch\b",
        r"\bresearched model[s]?\b",
        r"\bprototyped solution[s]?\b",
        r"\btested new algorithm[s]?\b",
        r"\bbenchmark(?:ed|ing)? model[s]?\b",
        r"\bcompared algorithm[s]?\b"
    ],
    "nlp_work": [
        r"\bnatural language processing\b",
        r"\bnlp\b",
        r"\btext classification\b",
        r"\btopic modeling\b",
        r"\bsentiment analysis\b",
        r"\bentity recognition\b",
        r"\btransformer model[s]?\b"
    ],
    "time_series_work": [
        r"\btime series\b",
        r"\bforecasting\b",
        r"\bdemand forecasting\b",
        r"\bsales forecasting\b",
        r"\btrend analysis\b"
    ],
    "data_pipeline_building": [
        r"\bbuilt data pipeline[s]?\b",
        r"\bdeveloped data pipeline[s]?\b",
        r"\betl\b",
        r"\belt\b",
        r"\bautomated data ingestion\b",
        r"\bautomated data processing\b",
        r"\bdata workflow[s]?\b"
    ]
}

# ==========================================
# 5) HELPERS
# ==========================================
def text_matches_any_pattern(text, patterns):
    return any(re.search(p, text) for p in patterns)

def find_matches(text, pattern_dict):
    matched = []
    for label, patterns in pattern_dict.items():
        for pattern in patterns:
            if re.search(pattern, text):
                matched.append(label)
                break
    return matched

def to_rank_desc(series):
    return series.rank(method="min", ascending=False).astype(int)

def wrap_label(s, width=16):
    words = s.replace("_", " ").split()
    lines = []
    cur = []
    cur_len = 0
    for w in words:
        add_len = len(w) + (1 if cur else 0)
        if cur_len + add_len <= width:
            cur.append(w)
            cur_len += add_len
        else:
            lines.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines)

# ==========================================
# 6) ACTIVATE JD-RELEVANT ITEMS
# ==========================================
active_hard_skills = {
    skill: patterns
    for skill, patterns in HARD_SKILLS.items()
    if text_matches_any_pattern(jd_text, patterns)
}

active_concepts = {
    concept: patterns
    for concept, patterns in CONCEPTS.items()
    if text_matches_any_pattern(jd_text, patterns)
}

if len(active_hard_skills) == 0:
    active_hard_skills = HARD_SKILLS.copy()

if len(active_concepts) == 0:
    active_concepts = CONCEPTS.copy()

HARD_SKILL_WEIGHT = 3
CONCEPT_WEIGHT = 2

# ==========================================
# 7) SCORE RESUMES
# ==========================================
basic_scores = []
hard_skill_only_scores = []
complex_scores = []
score_gain_hard_minus_basic = []
score_gain_complex_minus_basic = []

basic_keywords_counter = Counter()
hard_skill_counter = Counter()
concept_counter = Counter()

matched_basic_col = []
matched_hard_skills_col = []
matched_concepts_col = []

hard_skill_presence_rows = []
concept_presence_rows = []

for resume_text in df[resume_col].fillna(""):
    resume_text_norm = normalize_text(resume_text)

    resume_tokens = set(tokenize_basic(resume_text_norm))
    matched_basic_keywords = sorted(jd_tokens.intersection(resume_tokens))
    basic_score = len(matched_basic_keywords)

    matched_hard_skills = find_matches(resume_text_norm, active_hard_skills)
    hard_skill_only_score = len(matched_hard_skills)

    matched_concepts = find_matches(resume_text_norm, active_concepts)
    complex_score = len(matched_hard_skills) * HARD_SKILL_WEIGHT + len(matched_concepts) * CONCEPT_WEIGHT

    matched_basic_col.append(matched_basic_keywords)
    matched_hard_skills_col.append(matched_hard_skills)
    matched_concepts_col.append(matched_concepts)

    basic_scores.append(basic_score)
    hard_skill_only_scores.append(hard_skill_only_score)
    complex_scores.append(complex_score)
    score_gain_hard_minus_basic.append(hard_skill_only_score - basic_score)
    score_gain_complex_minus_basic.append(complex_score - basic_score)

    for token in matched_basic_keywords:
        basic_keywords_counter[token] += 1
    for hs in matched_hard_skills:
        hard_skill_counter[hs] += 1
    for c in matched_concepts:
        concept_counter[c] += 1

    hard_skill_presence_rows.append({skill: int(skill in matched_hard_skills) for skill in active_hard_skills.keys()})
    concept_presence_rows.append({concept: int(concept in matched_concepts) for concept in active_concepts.keys()})

df["basic_keyword_score"] = basic_scores
df["hard_skill_only_score"] = hard_skill_only_scores
df["complex_match_score"] = complex_scores
df["score_gain_hard_minus_basic"] = score_gain_hard_minus_basic
df["score_gain_complex_minus_basic"] = score_gain_complex_minus_basic
df["matched_basic_keywords"] = matched_basic_col
df["matched_hard_skills"] = matched_hard_skills_col
df["matched_concepts"] = matched_concepts_col

df["basic_rank"] = to_rank_desc(df["basic_keyword_score"])
df["hard_skill_rank"] = to_rank_desc(df["hard_skill_only_score"])
df["complex_rank"] = to_rank_desc(df["complex_match_score"])

df["hard_skill_points"] = df["hard_skill_only_score"] * HARD_SKILL_WEIGHT
df["concept_points"] = df["matched_concepts"].apply(len) * CONCEPT_WEIGHT

hard_skill_presence_df = pd.DataFrame(hard_skill_presence_rows)
concept_presence_df = pd.DataFrame(concept_presence_rows)

if "Category" not in df.columns:
    df["Category"] = "Unknown"

# ==========================================
# 8) PRINT SUMMARY
# ==========================================
print("\nACTIVE HARD SKILLS FROM JD:\n")
print(sorted(active_hard_skills.keys()))

print("\nACTIVE CONCEPTS FROM JD:\n")
print(sorted(active_concepts.keys()))

summary_cols = [
    "ID", "Category", "basic_keyword_score", "hard_skill_only_score",
    "complex_match_score", "basic_rank", "hard_skill_rank", "complex_rank"
]

print("\nTOP 20 RESUMES BY COMPLEX MATCH SCORE:\n")
print(df.sort_values("complex_match_score", ascending=False)[summary_cols].head(20).to_string(index=False))

detail_cols = [
    "ID", "Category", "matched_basic_keywords", "matched_hard_skills",
    "matched_concepts", "complex_match_score"
]
print("\nTOP 10 RESUMES WITH MATCH DETAILS:\n")
print(df.sort_values("complex_match_score", ascending=False)[detail_cols].head(10).to_string(index=False))

# ==========================================
# 9) SAVE OUTPUT TABLES
# ==========================================
output_cols = [
    "ID", "Category",
    "basic_keyword_score",
    "hard_skill_only_score",
    "complex_match_score",
    "score_gain_hard_minus_basic",
    "score_gain_complex_minus_basic",
    "basic_rank",
    "hard_skill_rank",
    "complex_rank",
    "matched_basic_keywords",
    "matched_hard_skills",
    "matched_concepts"
]

df.sort_values("complex_match_score", ascending=False)[output_cols].to_csv(
    "resume_match_portfolio_results.csv", index=False
)

top10_table_cols = [
    "ID", "Category", "hard_skill_only_score", "complex_match_score",
    "matched_hard_skills", "matched_concepts"
]
df.sort_values("complex_match_score", ascending=False)[top10_table_cols].head(10).to_csv(
    "resume_match_top10_table.csv", index=False
)

print("\nSaved results to:")
print(" - resume_match_portfolio_results.csv")
print(" - resume_match_top10_table.csv")

# ==========================================
# 10) CHART 1: PIPELINE DIAGRAM
# ==========================================
fig, ax = plt.subplots(figsize=(14, 3))
ax.axis("off")

steps = [
    ("Job Description", 0.05, 0.5),
    ("Text Cleaning", 0.23, 0.5),
    ("Basic Matcher", 0.41, 0.72),
    ("Hard Skill Matcher", 0.41, 0.5),
    ("Concept Matcher", 0.41, 0.28),
    ("Scoring & Ranking", 0.66, 0.5),
    ("Visualizations", 0.88, 0.5),
]

for label, x, y in steps:
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black")
    )

arrowprops = dict(arrowstyle="->", lw=1.5)
ax.annotate("", xy=(0.17, 0.5), xytext=(0.11, 0.5), xycoords="axes fraction", arrowprops=arrowprops)
ax.annotate("", xy=(0.34, 0.5), xytext=(0.28, 0.5), xycoords="axes fraction", arrowprops=arrowprops)
ax.annotate("", xy=(0.56, 0.5), xytext=(0.47, 0.68), xycoords="axes fraction", arrowprops=arrowprops)
ax.annotate("", xy=(0.56, 0.5), xytext=(0.47, 0.50), xycoords="axes fraction", arrowprops=arrowprops)
ax.annotate("", xy=(0.56, 0.5), xytext=(0.47, 0.32), xycoords="axes fraction", arrowprops=arrowprops)
ax.annotate("", xy=(0.80, 0.5), xytext=(0.74, 0.5), xycoords="axes fraction", arrowprops=arrowprops)

plt.title("Resume Matching Pipeline")
plt.tight_layout()
plt.show()

# ==========================================
# 11) CHART 2: SCORE DISTRIBUTION COMPARISON
# ==========================================
plt.figure(figsize=(11, 6))
plt.hist(df["basic_keyword_score"], bins=20, alpha=0.5, label="Basic keyword checker", edgecolor="black")
plt.hist(df["hard_skill_only_score"], bins=20, alpha=0.5, label="Hard-skill-only checker", edgecolor="black")
plt.hist(df["complex_match_score"], bins=20, alpha=0.5, label="Complex DS-specific checker", edgecolor="black")
plt.xlabel("Score")
plt.ylabel("Number of resumes")
plt.title("Resume Score Comparison: Basic vs Hard-Skill-Only vs Complex")
plt.legend()
plt.tight_layout()
plt.show()

# ==========================================
# 12) CHART 3: GAIN HISTOGRAM
# ==========================================
plt.figure(figsize=(10, 6))
plt.hist(df["score_gain_hard_minus_basic"], bins=20, alpha=0.6, label="Hard-skill-only - Basic", edgecolor="black")
plt.hist(df["score_gain_complex_minus_basic"], bins=20, alpha=0.6, label="Complex - Basic", edgecolor="black")
plt.xlabel("Score gain")
plt.ylabel("Number of resumes")
plt.title("Score Gain Compared with Basic Keyword Checker")
plt.legend()
plt.tight_layout()
plt.show()

# ==========================================
# 13) CHART 4: TOP MATCHED HARD SKILLS
# ==========================================
top_n = 15
top_hard_skills = hard_skill_counter.most_common(top_n)

if top_hard_skills:
    hs_names = [wrap_label(x[0]) for x in top_hard_skills]
    hs_freqs = [x[1] for x in top_hard_skills]

    plt.figure(figsize=(12, 6))
    plt.bar(hs_names, hs_freqs, edgecolor="black")
    plt.xlabel("Hard skill")
    plt.ylabel("Number of resumes matched")
    plt.title("Top Matched Hard Skills")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# ==========================================
# 14) CHART 5: TOP MATCHED CONCEPTS
# ==========================================
top_concepts = concept_counter.most_common(top_n)

if top_concepts:
    concept_names = [wrap_label(x[0]) for x in top_concepts]
    concept_freqs = [x[1] for x in top_concepts]

    plt.figure(figsize=(12, 6))
    plt.bar(concept_names, concept_freqs, edgecolor="black")
    plt.xlabel("Concept")
    plt.ylabel("Number of resumes matched")
    plt.title("Top Matched Data Scientist Concepts")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# ==========================================
# 15) CHART 6: CATEGORY AVERAGE SCORE
# ==========================================
category_avg = (
    df.groupby("Category")[["hard_skill_only_score", "complex_match_score"]]
    .mean()
    .sort_values("complex_match_score", ascending=False)
)

if len(category_avg) > 0:
    category_avg.plot(kind="bar", figsize=(12, 6), edgecolor="black")
    plt.xlabel("Category")
    plt.ylabel("Average score")
    plt.title("Average Resume Match Score by Category")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# ==========================================
# 16) CHART 7: CATEGORY BOXPLOT
# ==========================================
top_categories = df["Category"].value_counts().head(10).index.tolist()
box_df = df[df["Category"].isin(top_categories)].copy()

if len(box_df) > 0:
    categories_sorted = (
        box_df.groupby("Category")["complex_match_score"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    data = [box_df.loc[box_df["Category"] == cat, "complex_match_score"] for cat in categories_sorted]

    plt.figure(figsize=(12, 6))
    plt.boxplot(data, labels=categories_sorted)
    plt.xlabel("Category")
    plt.ylabel("Complex match score")
    plt.title("Distribution of Complex Match Scores by Category")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# ==========================================
# 17) CHART 8: HARD SKILL HEATMAP BY CATEGORY
# ==========================================
if len(active_hard_skills) > 0:
    hard_skill_cat_df = pd.concat([df[["Category"]].reset_index(drop=True), hard_skill_presence_df.reset_index(drop=True)], axis=1)
    hard_skill_heat = hard_skill_cat_df.groupby("Category").mean()

    top_categories_h = df["Category"].value_counts().head(10).index
    top_skills_h = [skill for skill, _ in hard_skill_counter.most_common(12)]

    hard_skill_heat = hard_skill_heat.loc[hard_skill_heat.index.intersection(top_categories_h)]
    hard_skill_heat = hard_skill_heat[top_skills_h] if len(top_skills_h) > 0 else hard_skill_heat

    if hard_skill_heat.shape[0] > 0 and hard_skill_heat.shape[1] > 0:
        plt.figure(figsize=(12, 7))
        plt.imshow(hard_skill_heat.values, aspect="auto")
        plt.colorbar(label="Share of resumes matched")
        plt.xticks(range(len(hard_skill_heat.columns)), [wrap_label(c, 12) for c in hard_skill_heat.columns], rotation=45, ha="right")
        plt.yticks(range(len(hard_skill_heat.index)), hard_skill_heat.index)
        plt.xlabel("Hard skill")
        plt.ylabel("Category")
        plt.title("Hard Skill Match Heatmap by Category")
        plt.tight_layout()
        plt.show()

# ==========================================
# 18) CHART 9: CONCEPT HEATMAP BY CATEGORY
# ==========================================
if len(active_concepts) > 0:
    concept_cat_df = pd.concat([df[["Category"]].reset_index(drop=True), concept_presence_df.reset_index(drop=True)], axis=1)
    concept_heat = concept_cat_df.groupby("Category").mean()

    top_categories_c = df["Category"].value_counts().head(10).index
    top_concepts_hm = [concept for concept, _ in concept_counter.most_common(12)]

    concept_heat = concept_heat.loc[concept_heat.index.intersection(top_categories_c)]
    concept_heat = concept_heat[top_concepts_hm] if len(top_concepts_hm) > 0 else concept_heat

    if concept_heat.shape[0] > 0 and concept_heat.shape[1] > 0:
        plt.figure(figsize=(12, 7))
        plt.imshow(concept_heat.values, aspect="auto")
        plt.colorbar(label="Share of resumes matched")
        plt.xticks(range(len(concept_heat.columns)), [wrap_label(c, 12) for c in concept_heat.columns], rotation=45, ha="right")
        plt.yticks(range(len(concept_heat.index)), concept_heat.index)
        plt.xlabel("Concept")
        plt.ylabel("Category")
        plt.title("Concept Match Heatmap by Category")
        plt.tight_layout()
        plt.show()

# ==========================================
# 19) CHART 10: RANKING CHANGE CHART
# ==========================================
top_rank_n = 10
top_complex_df = df.sort_values("complex_rank").head(top_rank_n).copy()

if len(top_complex_df) > 0:
    plot_df = top_complex_df[["ID", "basic_rank", "hard_skill_rank", "complex_rank"]].copy()
    plot_df = plot_df.sort_values("complex_rank")

    x_positions = [0, 1, 2]
    x_labels = ["Basic", "Hard-skill-only", "Complex"]

    plt.figure(figsize=(13, 8))

    # Use a categorical color palette
    colors = plt.get_cmap("tab10").colors

    for idx, (_, row) in enumerate(plot_df.iterrows()):
        y_values = [row["basic_rank"], row["hard_skill_rank"], row["complex_rank"]]
        color = colors[idx % len(colors)]

        plt.plot(
            x_positions,
            y_values,
            marker="o",
            linewidth=2,
            color=color,
            label=f"ID {row['ID']}"
        )

    plt.xticks(x_positions, x_labels)
    plt.gca().invert_yaxis()
    plt.ylabel("Rank (1 = best)")
    plt.title("Ranking Change of Top Complex-Scored Candidates")

    # Put legend outside plot so it stays readable
    plt.legend(
        title="Candidate IDs",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        frameon=True
    )

    plt.tight_layout()
    plt.show()

# ==========================================
# 20) CHART 11: EXPLAINABILITY FOR TOP CANDIDATE
# ==========================================
best_row = df.sort_values("complex_match_score", ascending=False).iloc[0]

best_hard_skills = best_row["matched_hard_skills"]
best_concepts = best_row["matched_concepts"]

contrib_labels = []
contrib_values = []

for hs in best_hard_skills:
    contrib_labels.append(f"HS: {hs}")
    contrib_values.append(HARD_SKILL_WEIGHT)

for c in best_concepts:
    contrib_labels.append(f"Concept: {c}")
    contrib_values.append(CONCEPT_WEIGHT)

if len(contrib_labels) > 0:
    contrib_pairs = sorted(zip(contrib_labels, contrib_values), key=lambda x: x[1], reverse=True)
    contrib_labels = [wrap_label(x[0], 18) for x in contrib_pairs]
    contrib_values = [x[1] for x in contrib_pairs]

    plt.figure(figsize=(12, max(5, len(contrib_labels) * 0.4)))
    plt.barh(contrib_labels, contrib_values, edgecolor="black")
    plt.xlabel("Score contribution")
    plt.ylabel("Matched feature")
    plt.title(f"Explainability: Score Contributions for Top Resume ID {best_row['ID']}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# ==========================================
# 21) CHART 12: TOP BASIC KEYWORDS
# ==========================================
top_basic = basic_keywords_counter.most_common(15)

if top_basic:
    basic_words = [wrap_label(x[0], 12) for x in top_basic]
    basic_freqs = [x[1] for x in top_basic]

    plt.figure(figsize=(12, 6))
    plt.bar(basic_words, basic_freqs, edgecolor="black")
    plt.xlabel("Keyword")
    plt.ylabel("Number of resumes matched")
    plt.title("Top Basic Matching Keywords")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# ==========================================
# 22) OPTIONAL: PORTFOLIO INSIGHT TABLE
# ==========================================
portfolio_summary = pd.DataFrame({
    "metric": [
        "Number of resumes",
        "Active hard skills from JD",
        "Active concepts from JD",
        "Average basic score",
        "Average hard-skill-only score",
        "Average complex score",
        "Top candidate ID",
        "Top candidate category",
        "Top candidate complex score"
    ],
    "value": [
        len(df),
        len(active_hard_skills),
        len(active_concepts),
        round(df["basic_keyword_score"].mean(), 2),
        round(df["hard_skill_only_score"].mean(), 2),
        round(df["complex_match_score"].mean(), 2),
        best_row["ID"],
        best_row["Category"],
        best_row["complex_match_score"]
    ]
})

portfolio_summary.to_csv("portfolio_project_summary.csv", index=False)
print("\nSaved summary table to: portfolio_project_summary.csv")