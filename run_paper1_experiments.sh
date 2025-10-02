#!/bin/bash
# Script to run all experiments for Paper 1

# --- Configuration ---
TASKS=("airfoil" "concrete")
SURROGATES=("Linear" "RandomForest" "XGBoost" "MLP" "GaussianProcess")
BUDGETS=(20 50)
FRACTIONS=(0.1 0.5)
NUM_RUNS=10 # Use 10 runs for statistical significance
SEED=42

# --- Run Loop ---
for task in "${TASKS[@]}"; do
  for budget in "${BUDGETS[@]}"; do
    echo "--- Running Task: $task | Budget: $budget ---"
    python -m benchmarks.run_inverse_design \
      --task "$task" \
      --surrogates "${SURROGATES[@]}" \
      --fractions "${FRACTIONS[@]}" \
      --budget "$budget" \
      --num-runs "$NUM_RUNS" \
      --random-state "$SEED"
  done
done

echo "--- All experiments complete. ---"
