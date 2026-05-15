import os
import pygmo as pg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# 0. SETTINGS
# ==========================================================

GENERATE_ALLOCATION_CHART = False

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURE_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


# ==========================================================
# 1. LOCAL ECONOMY OPTIMIZATION MODEL
# ==========================================================

class LocalEconomyOptimization:
    def fitness(self, x):
        total_raw = np.sum(x)

        if total_raw <= 1e-12:
            allocation = np.array([25, 25, 25, 25])
        else:
            allocation = np.array(x) / total_raw * 100

        inv_umkm, inv_environment, prog_social, resilience_fund = allocation

        # Objective 1: Minimize economic cost
        f_econ = (
            0.8 * inv_umkm +
            1.2 * inv_environment +
            0.9 * prog_social +
            1.5 * resilience_fund
        )

        # Objective 2: Minimize environmental burden
        f_env = max(
            0,
            80 +
            0.4 * inv_umkm -
            0.8 * inv_environment -
            0.2 * resilience_fund
        )

        # Objective 3: Maximize social impact
        f_soc = (
            0.5 * inv_umkm +
            1.1 * prog_social +
            0.3 * resilience_fund
        )

        # Objective 4: Maximize resilience score
        f_res = (
            0.2 * inv_umkm +
            0.4 * inv_environment +
            0.9 * resilience_fund
        )

        # PyGMO minimizes all objectives.
        # Maximized objectives are converted into minimization form.
        return [f_econ, f_env, -f_soc, -f_res]

    def get_bounds(self):
        return ([0, 0, 0, 0], [1, 1, 1, 1])

    def get_nobj(self):
        return 4

    def get_name(self):
        return "Local Economy Optimization with Budget Normalization"


# ==========================================================
# 2. NORMALIZATION AND EVALUATION FUNCTIONS
# ==========================================================

def normalize_allocation(x):
    total = np.sum(x)

    if total <= 1e-12:
        return np.array([25, 25, 25, 25])

    return np.array(x) / total * 100


def evaluate_allocation(allocation):
    inv_umkm, inv_environment, prog_social, resilience_fund = allocation

    economic_cost = (
        0.8 * inv_umkm +
        1.2 * inv_environment +
        0.9 * prog_social +
        1.5 * resilience_fund
    )

    environmental_burden = max(
        0,
        80 +
        0.4 * inv_umkm -
        0.8 * inv_environment -
        0.2 * resilience_fund
    )

    social_impact = (
        0.5 * inv_umkm +
        1.1 * prog_social +
        0.3 * resilience_fund
    )

    resilience_score = (
        0.2 * inv_umkm +
        0.4 * inv_environment +
        0.9 * resilience_fund
    )

    return [
        economic_cost,
        environmental_burden,
        social_impact,
        resilience_score
    ]


# ==========================================================
# 3. NSGA-II OPTIMIZATION PROCESS
# ==========================================================

POPULATION_SIZE = 100
GENERATIONS = 500
SEED = 42

prob = pg.problem(LocalEconomyOptimization())
algo = pg.algorithm(pg.nsga2(gen=GENERATIONS, seed=SEED))
pop = pg.population(prob, size=POPULATION_SIZE, seed=SEED)

pop = algo.evolve(pop)

fits = pop.get_f()
raw_points = pop.get_x()

allocation_points = np.array([
    normalize_allocation(x)
    for x in raw_points
])

ndf, _, _, _ = pg.fast_non_dominated_sorting(points=fits)
non_dominated_idx = ndf[0]

non_dominated_fits = fits[non_dominated_idx]
non_dominated_points = allocation_points[non_dominated_idx]


# ==========================================================
# 4. OPTIMIZATION CONFIGURATION
# ==========================================================

optimization_config_df = pd.DataFrame({
    "Component": [
        "Optimization algorithm",
        "Population size",
        "Number of generations",
        "Number of decision variables",
        "Number of objectives",
        "Budget normalization",
        "Total budget allocation",
        "Decision bounds",
        "Policy sectors",
        "Optimization output",
        "Number of non-dominated solutions"
    ],
    "Value": [
        "NSGA-II",
        POPULATION_SIZE,
        GENERATIONS,
        4,
        4,
        "Applied",
        "100%",
        "0 to 1",
        "UMKM, environment, social, resilience",
        "Non-dominated policy allocation scenarios",
        len(non_dominated_fits)
    ]
})


# ==========================================================
# 5. FULL NON-DOMINATED SOLUTIONS
# ==========================================================

full_non_dominated_df = pd.DataFrame({
    "Solution": np.arange(1, len(non_dominated_points) + 1),
    "UMKM (%)": non_dominated_points[:, 0],
    "Env (%)": non_dominated_points[:, 1],
    "Social (%)": non_dominated_points[:, 2],
    "Resil (%)": non_dominated_points[:, 3],
    "Economic Cost": non_dominated_fits[:, 0],
    "Environmental Burden": non_dominated_fits[:, 1],
    "Social Impact": -non_dominated_fits[:, 2],
    "Resilience Score": -non_dominated_fits[:, 3]
}).round(2)


# ==========================================================
# 6. PARETO OBJECTIVE SUMMARY
# ==========================================================

objective_columns = [
    "Economic Cost",
    "Environmental Burden",
    "Social Impact",
    "Resilience Score"
]

pareto_objective_summary_df = full_non_dominated_df[objective_columns].agg(
    ["min", "max", "mean", "std"]
).T.reset_index()

pareto_objective_summary_df.columns = [
    "Objective",
    "Minimum",
    "Maximum",
    "Mean",
    "Standard Deviation"
]

pareto_objective_summary_df["Optimization Direction"] = [
    "Lower is better",
    "Lower is better",
    "Higher is better",
    "Higher is better"
]

pareto_objective_summary_df = pareto_objective_summary_df[
    [
        "Objective",
        "Optimization Direction",
        "Minimum",
        "Maximum",
        "Mean",
        "Standard Deviation"
    ]
].round(2)


# ==========================================================
# 7. REPRESENTATIVE POLICY ALLOCATION SCENARIOS
# ==========================================================

scenario_names = [
    "1: Extreme Econ",
    "2: Extreme Env",
    "3: Extreme Social",
    "4: Extreme Resil",
    "5: Balanced"
]

allocations = np.array([
    [100.00, 0.00, 0.00, 0.00],
    [0.00, 100.00, 0.00, 0.00],
    [0.00, 0.00, 100.00, 0.00],
    [0.00, 0.00, 0.00, 100.00],
    [35.00, 20.00, 25.00, 20.00]
])

policy_orientation = [
    "Analytical Benchmark",
    "Analytical Benchmark",
    "Analytical Benchmark",
    "Analytical Benchmark",
    "Realistic Compromise"
]

representative_policy_allocation_df = pd.DataFrame({
    "Scenario": scenario_names,
    "UMKM (%)": allocations[:, 0],
    "Env (%)": allocations[:, 1],
    "Social (%)": allocations[:, 2],
    "Resil (%)": allocations[:, 3],
    "Policy Orientation": policy_orientation
}).round(2)


# ==========================================================
# 8. OBJECTIVE PERFORMANCE OF SELECTED SCENARIOS
# ==========================================================

objective_values = np.array([
    evaluate_allocation(allocation)
    for allocation in allocations
])

objective_performance_df = pd.DataFrame({
    "Scenario": scenario_names,
    "Economic Cost": objective_values[:, 0],
    "Environmental Burden": objective_values[:, 1],
    "Social Impact": objective_values[:, 2],
    "Resilience Score": objective_values[:, 3]
}).round(2)


# ==========================================================
# 9. SCENARIO INTERPRETATION SUMMARY
# ==========================================================

scenario_interpretation_df = pd.DataFrame({
    "Scenario": scenario_names,
    "Main Interpretation": [
        "Lowest economic cost, but highest environmental burden",
        "Lowest environmental burden, but no direct social impact under the current model structure",
        "Highest social impact, but no direct resilience contribution under the current model structure",
        "Highest resilience score, but highest economic cost",
        "Moderate compromise across all objective dimensions"
    ]
})


# ==========================================================
# 10. PAPER-TO-OUTPUT MAPPING
# ==========================================================

output_manifest_df = pd.DataFrame({
    "Paper Item": [
        "Table 1",
        "Table 2",
        "Table 3",
        "Table 4",
        "Table 5",
        "Appendix",
        "Figure 1",
        "Figure 2"
    ],
    "Paper Description": [
        "Optimization Configuration",
        "Pareto Objective Summary",
        "Representative Policy Allocation Scenarios",
        "Objective Performance of Selected Scenarios",
        "Scenario Interpretation Summary",
        "Full Non-Dominated Solutions",
        "Pareto Front",
        "Objective Performance Comparison"
    ],
    "Repository File": [
        "outputs/optimization_configuration.csv",
        "outputs/pareto_objective_summary.csv",
        "outputs/representative_policy_allocation_scenarios.csv",
        "outputs/objective_performance_selected_scenarios.csv",
        "outputs/scenario_interpretation_summary.csv",
        "outputs/full_non_dominated_solutions.csv",
        "figures/pareto_front.png",
        "figures/objective_performance_comparison.png"
    ]
})


# ==========================================================
# 11. SAVE OUTPUT FILES
# ==========================================================

optimization_config_path = os.path.join(
    OUTPUT_DIR,
    "optimization_configuration.csv"
)

pareto_summary_path = os.path.join(
    OUTPUT_DIR,
    "pareto_objective_summary.csv"
)

representative_scenarios_path = os.path.join(
    OUTPUT_DIR,
    "representative_policy_allocation_scenarios.csv"
)

objective_performance_path = os.path.join(
    OUTPUT_DIR,
    "objective_performance_selected_scenarios.csv"
)

scenario_interpretation_path = os.path.join(
    OUTPUT_DIR,
    "scenario_interpretation_summary.csv"
)

full_non_dominated_path = os.path.join(
    OUTPUT_DIR,
    "full_non_dominated_solutions.csv"
)

output_manifest_path = os.path.join(
    OUTPUT_DIR,
    "output_manifest.csv"
)

optimization_config_df.to_csv(optimization_config_path, index=False)
pareto_objective_summary_df.to_csv(pareto_summary_path, index=False)
representative_policy_allocation_df.to_csv(representative_scenarios_path, index=False)
objective_performance_df.to_csv(objective_performance_path, index=False)
scenario_interpretation_df.to_csv(scenario_interpretation_path, index=False)
full_non_dominated_df.to_csv(full_non_dominated_path, index=False)
output_manifest_df.to_csv(output_manifest_path, index=False)


# ==========================================================
# 12. PRINT OUTPUT IN TERMINAL
# ==========================================================

print("\n" + "=" * 70)
print("LOCAL ECONOMY POLICY OPTIMIZATION RESULTS")
print("=" * 70)
print(f"Total population              : {len(fits)}")
print(f"Non-dominated solutions       : {len(non_dominated_fits)}")

print("\nOptimization Configuration")
print(optimization_config_df.to_string(index=False))

print("\nPareto Objective Summary")
print(pareto_objective_summary_df.to_string(index=False))

print("\nRepresentative Policy Allocation Scenarios")
print(representative_policy_allocation_df.to_string(index=False))

print("\nObjective Performance of Selected Scenarios")
print(objective_performance_df.to_string(index=False))

print("\nScenario Interpretation Summary")
print(scenario_interpretation_df.to_string(index=False))


# ==========================================================
# 13. FIGURE: PARETO FRONT
# ==========================================================

plt.figure(figsize=(10, 6))

sc = plt.scatter(
    non_dominated_fits[:, 0],
    -non_dominated_fits[:, 3],
    c=-non_dominated_fits[:, 2],
    cmap="viridis",
    s=60,
    alpha=0.8
)

plt.colorbar(sc, label="Social Impact")
plt.xlabel("Economic Cost")
plt.ylabel("Resilience Score")
plt.title("Pareto Front: Economic Cost and Resilience Trade-off")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

pareto_front_path = os.path.join(FIGURE_DIR, "pareto_front.png")
plt.savefig(pareto_front_path, dpi=300, bbox_inches="tight")
plt.show()


# ==========================================================
# 14. OPTIONAL FIGURE: BUDGET ALLOCATION COMPARISON
# ==========================================================

budget_allocation_path = None

if GENERATE_ALLOCATION_CHART:
    umkm = allocations[:, 0]
    env = allocations[:, 1]
    soc = allocations[:, 2]
    res = allocations[:, 3]

    fig, ax = plt.subplots(figsize=(11, 6))

    width = 0.6

    ax.bar(scenario_names, umkm, width, label="UMKM Allocation", color="#1f77b4")
    ax.bar(scenario_names, env, width, bottom=umkm, label="Environmental Allocation", color="#ff7f0e")
    ax.bar(scenario_names, soc, width, bottom=umkm + env, label="Social Allocation", color="#2ca02c")
    ax.bar(scenario_names, res, width, bottom=umkm + env + soc, label="Resilience Allocation", color="#d62728")

    ax.set_ylabel("Percentage Allocation (%)", fontsize=12)
    ax.set_title("Budget Allocation Comparison by Scenario", fontsize=14, pad=15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4)
    ax.set_ylim(0, 110)

    plt.xticks(rotation=15)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    budget_allocation_path = os.path.join(FIGURE_DIR, "budget_allocation_comparison.png")
    plt.savefig(budget_allocation_path, dpi=300, bbox_inches="tight")
    plt.show()


# ==========================================================
# 15. FIGURE: OBJECTIVE PERFORMANCE COMPARISON
# ==========================================================

economic_cost = objective_values[:, 0]
environmental_burden = objective_values[:, 1]
social_impact = objective_values[:, 2]
resilience_score = objective_values[:, 3]

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(scenario_names))
bar_width = 0.2

rects1 = ax.bar(
    x - 1.5 * bar_width,
    economic_cost,
    bar_width,
    label="Economic Cost",
    color="#1f77b4"
)

rects2 = ax.bar(
    x - 0.5 * bar_width,
    environmental_burden,
    bar_width,
    label="Environmental Burden",
    color="#ff7f0e"
)

rects3 = ax.bar(
    x + 0.5 * bar_width,
    social_impact,
    bar_width,
    label="Social Impact",
    color="#2ca02c"
)

rects4 = ax.bar(
    x + 1.5 * bar_width,
    resilience_score,
    bar_width,
    label="Resilience Score",
    color="#d62728"
)

ax.set_ylabel("Objective Value", fontsize=12)
ax.set_title("Objective Performance Comparison", fontsize=14, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(scenario_names, rotation=15)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f"{height:.1f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9
        )


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()

objective_performance_figure_path = os.path.join(
    FIGURE_DIR,
    "objective_performance_comparison.png"
)

plt.savefig(objective_performance_figure_path, dpi=300, bbox_inches="tight")
plt.show()


# ==========================================================
# 16. INTERPRETATION
# ==========================================================

print("\n" + "=" * 70)
print("INTERPRETATION SUMMARY")
print("=" * 70)

print("""
The optimization model generated non-dominated policy allocation alternatives under a fixed-budget setting.
The Pareto front indicates a trade-off between economic cost and resilience score. Higher resilience generally
requires higher economic commitment.

The first four selected scenarios are extreme analytical benchmarks. They are used to show the consequences of
prioritizing one dominant policy objective: economic efficiency, environmental improvement, social impact, or
resilience strengthening.

The balanced scenario is manually constructed as a realistic compromise benchmark. It is not claimed as the
mathematically optimal solution. Instead, it is included to provide a practical comparison because local governments
are unlikely to allocate the entire budget to only one policy sector.

The Pareto objective summary describes the range of non-dominated objective values. The full non-dominated
solution set is exported as an appendix file so that all allocation alternatives can be inspected beyond the five
representative scenarios.
""")


# ==========================================================
# 17. OUTPUT FILE LIST
# ==========================================================

print("\nGenerated files:")
print(f"1. {optimization_config_path}")
print(f"2. {pareto_summary_path}")
print(f"3. {representative_scenarios_path}")
print(f"4. {objective_performance_path}")
print(f"5. {scenario_interpretation_path}")
print(f"6. {full_non_dominated_path}")
print(f"7. {output_manifest_path}")
print(f"8. {pareto_front_path}")
print(f"9. {objective_performance_figure_path}")

if budget_allocation_path is not None:
    print(f"10. {budget_allocation_path}")
