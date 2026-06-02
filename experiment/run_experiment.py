#!/usr/bin/env python3
"""
Run the full discharge summary comprehension experiment from config file.
Supports multiple models (OpenAI + Anthropic) with per-model iteration counts.
"""

import json
import os
import sys
import itertools
from persona_discharge_query import PersonaDischargeQueryEngine


def load_config(config_file: str = "experiment_config.json") -> dict:
    """Load experiment configuration."""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file '{config_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file: {e}")
        sys.exit(1)


def calculate_total_queries(config: dict) -> int:
    """Calculate total number of queries per single iteration."""
    persona_keys = ['age', 'gender', 'education', 'ethnicity', 'doctor_visit', 'er_visit_frequency']
    persona_values = [config['persona_variations'][k] for k in persona_keys]
    num_personas = len(list(itertools.product(*persona_values)))

    if config.get('max_personas'):
        num_personas = min(num_personas, config['max_personas'])

    num_ds = len(config['discharge_summary_ids'])
    num_questions = len(config['question_ids'])

    return num_personas * num_ds * num_questions


def print_experiment_plan(config: dict, queries_per_iter: int) -> int:
    """Print experiment plan and return total query count across all models/iterations."""
    models = config.get('models', [])
    # Backward compatibility: single model field
    if not models and config.get('model'):
        models = [{"model": config['model'], "provider": "openai", "iterations": 1}]

    total_all = 0
    print("\n" + "=" * 80)
    print("EXPERIMENT PLAN")
    print("=" * 80)
    print(f"Queries per iteration: {queries_per_iter:,}")
    print(f"Temperature: {config.get('temperature', 1.0)}")
    print(f"Reasoning: {'enabled' if config.get('enable_reasoning', False) else 'disabled'}")
    print("-" * 80)
    print(f"{'Model':<35} {'Provider':<12} {'Iters':<8} {'Total Queries'}")
    print("-" * 80)

    for m in models:
        iters = m.get('iterations', 1)
        total = queries_per_iter * iters
        total_all += total
        print(f"{m['model']:<35} {m.get('provider', 'openai'):<12} {iters:<8} {total:,}")

    print("-" * 80)
    print(f"{'GRAND TOTAL':<56} {total_all:,}")
    print("=" * 80)

    return total_all


def main():
    """Run the experiment from config file."""
    config_file = sys.argv[1] if len(sys.argv) > 1 else "experiment_config.json"
    print(f"Loading configuration from: {config_file}")
    config = load_config(config_file)

    queries_per_iter = calculate_total_queries(config)
    total_queries = print_experiment_plan(config, queries_per_iter)

    # Ask for confirmation
    if total_queries > 100:
        response = input(f"\nTotal {total_queries:,} queries will be made. Continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Experiment cancelled.")
            return

    # Initialize engine
    try:
        engine = PersonaDischargeQueryEngine()
    except ValueError as e:
        print(f"\nError: {e}")
        print("\nSet your API keys:")
        print("  export OPENAI_API_KEY='your-key'")
        print("  export ANTHROPIC_API_KEY='your-key'")
        return

    # Parse models list (backward compatible)
    models = config.get('models', [])
    if not models and config.get('model'):
        models = [{"model": config['model'], "provider": "openai", "iterations": 1}]

    # Create output directory
    output_dir = config.get('output_dir', 'results')
    os.makedirs(output_dir, exist_ok=True)

    # Run experiment for each model and iteration
    for model_cfg in models:
        model_name = model_cfg['model']
        provider = model_cfg.get('provider', 'openai')
        iterations = model_cfg.get('iterations', 1)

        for iteration in range(1, iterations + 1):
            print(f"\n{'=' * 80}")
            print(f"MODEL: {model_name} | PROVIDER: {provider} | ITERATION: {iteration}/{iterations}")
            print(f"{'=' * 80}")

            results = engine.run_full_experiment(
                persona_variations=config['persona_variations'],
                discharge_summary_ids=config['discharge_summary_ids'],
                question_ids=config['question_ids'],
                model=model_name,
                temperature=config.get('temperature', 1.0),
                max_personas=config.get('max_personas'),
                enable_reasoning=config.get('enable_reasoning', False),
                provider=provider
            )

            # Save results per model per iteration
            safe_model_name = model_name.replace("/", "_")
            output_file = os.path.join(output_dir, f"{safe_model_name}_iter{iteration}.json")
            engine.save_results(results, output_file)

            print(f"Saved: {output_file}")

    print(f"\nAll experiments complete! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
