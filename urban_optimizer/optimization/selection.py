"""Heuristiques de sélection sous budget pour le NDP.

Deux briques :
1. ``knapsack_dp_select`` — DP 0/1 exact (à coûts discrétisés) sur le score
   additif. Remplace une sélection gloutonne par BCR décroissant ; identique
   en complexité observable mais optimal sur la fonction additive.
2. ``local_search_2swap`` — post-traitement first-improvement qui rejoue le
   FW joint après chaque swap (in, out) pour casser les redondances entre
   interventions parallèles (cf. cannibalisation des corridors voisins).

Les deux fonctions opèrent sur une liste de ``NewArcEvaluation`` produite par
``_marginal_evaluate`` — elles ignorent le détail métier des interventions.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from urban_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


# ── 1. Knapsack DP 0/1 ───────────────────────────────────────────────────────

def knapsack_dp_select(
    evals: list,
    budget_eur: float,
    *,
    cost_step_eur: float = 50_000.0,
    score_attr: str = "annual_benefit_eur",
) -> tuple[list, float]:
    """Sélection optimale 0/1 sous budget par DP, à coûts discrétisés.

    Hypothèses :
    - score additif (somme des ``score_attr`` des candidats sélectionnés) ;
    - pas d'interaction entre candidats (capturée plus tard par le 2-swap).

    Discrétisation : chaque coût est arrondi vers le haut au pas
    ``cost_step_eur``. Avec n=60 et B = 50 M€ / 50 k€ = 1000, le DP prend
    ~50 ms. Pour de très grands B, augmenter ``cost_step_eur``.

    Retourne : (liste sélectionnée, coût réel total).
    """
    items = [e for e in evals if getattr(e, "is_worth_it", True)
             and getattr(e, score_attr, 0.0) > 0]
    if not items or budget_eur <= 0:
        return [], 0.0

    n = len(items)
    B = max(1, int(budget_eur // cost_step_eur))
    weights = [max(1, int(np.ceil(e.cost_eur / cost_step_eur))) for e in items]
    values = [float(getattr(e, score_attr)) for e in items]

    # DP O(n·B). On garde un tableau de booléens pour reconstruire.
    dp = np.zeros(B + 1, dtype=np.float64)
    keep = np.zeros((n, B + 1), dtype=bool)
    for i in range(n):
        w_i, v_i = weights[i], values[i]
        # Itération en sens décroissant pour éviter de réutiliser l'item i
        for b in range(B, w_i - 1, -1):
            cand = dp[b - w_i] + v_i
            if cand > dp[b]:
                dp[b] = cand
                keep[i, b] = True

    chosen: list = []
    b = B
    for i in range(n - 1, -1, -1):
        if keep[i, b]:
            chosen.append(items[i])
            b -= weights[i]

    spent = sum(e.cost_eur for e in chosen)
    if spent > budget_eur:
        # La discrétisation peut faire dépasser légèrement → on retire les
        # items les moins rentables jusqu'à respecter le budget réel.
        chosen.sort(key=lambda e: getattr(e, "bcr", 0.0))
        while chosen and spent > budget_eur:
            dropped = chosen.pop(0)
            spent -= dropped.cost_eur
        chosen.sort(key=lambda e: -getattr(e, "bcr", 0.0))
    return chosen, spent


# ── 2. Local search 2-swap (first-improvement) ───────────────────────────────

def local_search_2swap(
    chosen: list,
    all_evals: list,
    spent: float,
    budget_eur: float,
    *,
    joint_eval_fn: Callable[[list, float], "object | None"],
    benefit_attr: str = "joint_annual_benefit_eur",
    max_iter: int = 3,
    top_k_out: int = 3,
    top_k_in: int = 5,
    max_fw_calls: int = 12,
) -> tuple[list, float, object | None, int]:
    """Améliore un plan en échangeant (in, out) tant que le bénéfice joint monte.

    Stratégie *first-improvement* (s'arrête au 1er swap profitable de chaque
    tour) pour borner le coût en FW joints. Les listes de candidats sortants
    (``top_k_out`` plus bas BCR) et entrants (``top_k_in`` plus haut BCR hors
    plan) sont restreintes pour rester sous ~k_out × k_in FW joints par tour.

    Args:
        chosen: plan initial (issu du knapsack).
        all_evals: tous les candidats évalués (incluant ``chosen``).
        spent: coût initial du plan.
        budget_eur: contrainte.
        joint_eval_fn: callable ``(plan, spent) -> JointPlanResult`` ; un FW
            complet est lancé à chaque appel — c'est le coût dominant.
        benefit_attr: attribut sur lequel comparer les plans (joint_benef).
        max_iter: nb maximal de tours de swap.
        top_k_out / top_k_in: bornes sur l'exploration par tour.
        max_fw_calls: garde-fou absolu sur le nb de FW joints.

    Retourne : (chosen_final, spent_final, best_joint, nb_fw_calls).
    """
    chosen_ids = {id(e) for e in chosen}
    base_joint = joint_eval_fn(chosen, spent)
    fw_calls = 1
    if base_joint is None:
        return chosen, spent, None, fw_calls

    best_benef = float(getattr(base_joint, benefit_attr, 0.0))
    best_joint = base_joint
    cur_chosen = list(chosen)
    cur_spent = spent

    for it in range(max_iter):
        if fw_calls >= max_fw_calls:
            logger.info(
                f"  [2-swap] budget FW épuisé ({fw_calls}/{max_fw_calls}), arrêt"
            )
            break

        # Candidats sortants : ceux avec le plus bas BCR (les moins rentables marginalement)
        outs = sorted(cur_chosen, key=lambda e: getattr(e, "bcr", 0.0))[:top_k_out]
        # Candidats entrants : meilleurs BCR hors plan, ne touchant pas le budget négatif
        chosen_ids = {id(e) for e in cur_chosen}
        ins_pool = [
            e for e in all_evals
            if id(e) not in chosen_ids
            and getattr(e, "is_worth_it", True)
            and getattr(e, "annual_benefit_eur", 0.0) > 0
        ]
        ins = sorted(ins_pool, key=lambda e: -getattr(e, "bcr", 0.0))[:top_k_in]
        if not outs or not ins:
            break

        improved = False
        for c_out in outs:
            if improved or fw_calls >= max_fw_calls:
                break
            for c_in in ins:
                new_spent = cur_spent - c_out.cost_eur + c_in.cost_eur
                if new_spent > budget_eur:
                    continue
                new_plan = [e for e in cur_chosen if e is not c_out] + [c_in]
                jr = joint_eval_fn(new_plan, new_spent)
                fw_calls += 1
                if jr is None:
                    continue
                new_benef = float(getattr(jr, benefit_attr, 0.0))
                if new_benef > best_benef * 1.005:  # gain > 0.5 % pour éviter le bruit
                    logger.info(
                        f"  [2-swap iter {it+1}] swap accepté : "
                        f"benef {best_benef:,.0f} → {new_benef:,.0f} € "
                        f"(+{(new_benef - best_benef) / max(abs(best_benef), 1.0) * 100:.1f}%)"
                    )
                    best_benef = new_benef
                    best_joint = jr
                    cur_chosen = new_plan
                    cur_spent = new_spent
                    improved = True
                    break
                if fw_calls >= max_fw_calls:
                    break
        if not improved:
            logger.info(
                f"  [2-swap iter {it+1}] aucun swap profitable trouvé, arrêt"
            )
            break

    logger.info(
        f"  [2-swap] terminé : {fw_calls} FW joints, "
        f"benef {float(getattr(base_joint, benefit_attr, 0.0)):,.0f} → "
        f"{best_benef:,.0f} €"
    )
    return cur_chosen, cur_spent, best_joint, fw_calls
