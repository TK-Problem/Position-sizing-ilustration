import numpy as np


def generate_binary_outcomes(probability, n_observations, seed=None):
    """Generate a vector of binary outcomes (1=success, 0=failure)."""
    rng = np.random.default_rng(seed)
    return rng.binomial(n=1, p=probability, size=n_observations)


def flat_unit_staking(outcomes, stake_pct, start_budget):
    """Convert binary outcomes to cumulative profit using flat unit staking."""
    unit = start_budget * stake_pct
    stakes = np.where(outcomes == 1, unit, -unit)
    return start_budget + np.concatenate(([0], np.cumsum(stakes)))

def percent_budget_staking(outcomes, stake_pct, start_budget):
    """Convert binary outcomes to budget trajectory using % of current budget."""
    outcomes = np.asarray(outcomes)
    budget = np.empty(len(outcomes) + 1)
    budget[0] = start_budget
    for i, outcome in enumerate(outcomes):
        if outcome == 1:
            budget[i + 1] = budget[i] * (1 + stake_pct)
        else:
            budget[i + 1] = budget[i] * (1 - stake_pct)
    return budget
