"""
Policy layer for converting ML predictions to strategy decisions.

This layer:
1. Takes raw model outputs (probabilities, predictions)
2. Applies business rules and constraints
3. Generates scenario branches
4. Produces final JSON output
"""
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class PitDecision:
    """Single lap pit decision output."""
    pit_recommend_now: int  # 0 or 1
    pit_probability_next_1_lap: float
    pit_probability_next_2_laps: float
    recommended_compound_next: str
    compound_probabilities: Dict[str, float]
    target_stint_length_laps: int
    expected_total_race_time_s: Optional[float]
    expected_gain_vs_stayout_s: float
    confidence_band_s: Tuple[float, float]
    model_confidence: float


@dataclass
class StintPlan:
    """Single stint in the race plan."""
    from_lap: int
    to_lap: int
    compound: str
    planned_pit_lap: int


@dataclass
class RacePlan:
    """Overall race strategy plan."""
    planned_number_of_stops: int
    stints: List[StintPlan]
    scenario_branches: Dict[str, Dict[str, Any]]


@dataclass
class StrategyOutput:
    """Complete strategy output for a single lap."""
    decision: PitDecision
    plan: RacePlan
    live_context: Dict[str, Any]


class PolicyEngine:
    """Converts model predictions to strategy decisions."""
    
    def __init__(
        self,
        pit_threshold: float = 0.5,
        sc_window_laps: int = 5,
        rain_threshold: float = 0.4,
        undercut_gap_threshold_s: float = 2.0,
        min_stint_length: int = 5,
        max_stint_length: int = 35
    ):
        self.pit_threshold = pit_threshold
        self.sc_window_laps = sc_window_laps
        self.rain_threshold = rain_threshold
        self.undercut_gap_threshold_s = undercut_gap_threshold_s
        self.min_stint_length = min_stint_length
        self.max_stint_length = max_stint_length
    
    def _calculate_confidence(
        self,
        pit_prob: float,
        compound_probs: Dict[str, float]
    ) -> float:
        """Calculate overall model confidence."""
        # High confidence when probabilities are decisive
        pit_confidence = abs(pit_prob - 0.5) * 2  # 0-1 scale
        compound_max_prob = max(compound_probs.values()) if compound_probs else 0.5
        
        return (pit_confidence + compound_max_prob) / 2
    
    def _calculate_expected_gain(
        self,
        current_lap: int,
        tire_age: int,
        pit_loss_s: float,
        degradation_rate: float,
        remaining_laps: int
    ) -> Tuple[float, Tuple[float, float]]:
        """Estimate time gain/loss from pitting now vs staying out.
        
        Returns expected gain and confidence band.
        """
        if remaining_laps <= 0 or pit_loss_s <= 0:
            return 0.0, (0.0, 0.0)
        
        # Simplified model: compare cumulative degradation vs pit loss
        # Staying out: accumulate degradation for remaining stint
        expected_remaining_stint = min(remaining_laps, self.max_stint_length - tire_age)
        
        # Time lost from degradation if staying out
        degradation_cost = degradation_rate * expected_remaining_stint * (expected_remaining_stint + 1) / 2
        
        # Time lost from pitting (pit stop + outlap penalty)
        pit_cost = pit_loss_s
        
        gain = degradation_cost - pit_cost
        
        # Confidence band (±15% uncertainty)
        uncertainty = abs(gain) * 0.15 + 1.0
        confidence_band = (gain - uncertainty, gain + uncertainty)
        
        return round(gain, 1), (round(confidence_band[0], 1), round(confidence_band[1], 1))
    
    def _calculate_optimal_stops(
        self,
        remaining_laps: int,
        pit_loss_s: float,
        degradation_rate: float
    ) -> int:
        """Calculate optimal number of stops based on race math.
        
        Logic: Each stop costs pit_loss_s but saves degradation over fresh tires.
        Most F1 races are 1-2 stops, rarely 3+ except high-deg tracks.
        """
        if remaining_laps <= 15:
            return 0  # Too late to pit
        
        # Simple heuristic: compare 1-stop vs 2-stop vs 0-stop
        # Avg stint length for 1-stop = remaining_laps / 2
        # Avg stint length for 2-stop = remaining_laps / 3
        
        one_stop_stint = remaining_laps / 2
        two_stop_stint = remaining_laps / 3
        
        # Degradation cost increases with stint length (quadratic)
        one_stop_deg_cost = degradation_rate * (one_stop_stint ** 1.5)
        two_stop_deg_cost = degradation_rate * (two_stop_stint ** 1.5) * 2
        
        # Total cost = deg cost + pit costs
        zero_stop_cost = degradation_rate * (remaining_laps ** 1.5)
        one_stop_cost = one_stop_deg_cost * 2 + pit_loss_s
        two_stop_cost = two_stop_deg_cost * 1.5 + pit_loss_s * 2
        
        costs = [(0, zero_stop_cost), (1, one_stop_cost), (2, two_stop_cost)]
        optimal = min(costs, key=lambda x: x[1])
        
        return optimal[0]
    
    def _generate_stint_plan(
        self,
        current_lap: int,
        remaining_laps: int,
        current_compound: str,
        predicted_stint_length: int,
        recommended_compound: str,
        pit_loss_s: float = 22.0,
        degradation_rate: float = 0.1
    ) -> RacePlan:
        """Generate a realistic race stint plan.
        
        Uses optimal stop calculation instead of naive loop.
        """
        stints = []
        race_end = current_lap + remaining_laps
        
        # Calculate optimal number of remaining stops
        optimal_stops = self._calculate_optimal_stops(
            remaining_laps, pit_loss_s, degradation_rate
        )
        
        # Cap at 2 stops max (3-stop is very rare in modern F1)
        optimal_stops = min(optimal_stops, 2)
        
        if optimal_stops == 0:
            # No more stops - run to end
            stints.append(StintPlan(
                from_lap=current_lap,
                to_lap=race_end,
                compound=current_compound,
                planned_pit_lap=race_end
            ))
        elif optimal_stops == 1:
            # One stop strategy
            pit_lap = current_lap + (remaining_laps // 2)
            pit_lap = max(pit_lap, current_lap + self.min_stint_length)
            
            stints.append(StintPlan(
                from_lap=current_lap,
                to_lap=pit_lap,
                compound=current_compound,
                planned_pit_lap=pit_lap
            ))
            stints.append(StintPlan(
                from_lap=pit_lap,
                to_lap=race_end,
                compound=recommended_compound,
                planned_pit_lap=race_end
            ))
        else:
            # Two stop strategy
            stint_len = remaining_laps // 3
            stint_len = max(stint_len, self.min_stint_length)
            
            pit_lap_1 = current_lap + stint_len
            pit_lap_2 = pit_lap_1 + stint_len
            
            stints.append(StintPlan(
                from_lap=current_lap,
                to_lap=pit_lap_1,
                compound=current_compound,
                planned_pit_lap=pit_lap_1
            ))
            stints.append(StintPlan(
                from_lap=pit_lap_1,
                to_lap=pit_lap_2,
                compound=recommended_compound,
                planned_pit_lap=pit_lap_2
            ))
            stints.append(StintPlan(
                from_lap=pit_lap_2,
                to_lap=race_end,
                compound='HARD' if recommended_compound != 'HARD' else 'MEDIUM',
                planned_pit_lap=race_end
            ))
        
        return RacePlan(
            planned_number_of_stops=optimal_stops,
            stints=stints,
            scenario_branches={}
        )
    
    def _generate_scenario_branches(
        self,
        current_lap: int,
        tire_age: int,
        safety_car_active: bool
    ) -> Dict[str, Dict[str, Any]]:
        """Generate conditional policy rules."""
        branches = {}
        
        # SC window rule
        sc_window_end = current_lap + self.sc_window_laps
        branches[f"if_SC_in_[{current_lap} , {sc_window_end}]"] = {
            "pit_now": True,
            "compound": "HARD" if tire_age < 15 else "MEDIUM"
        }
        
        # Rain rule
        branches["if_rain_prob_gt_0.4"] = {
            "pit_for": "INTER"
        }
        
        # Undercut defense rule
        branches["if_gap_behind_lt_2s_and_tire_age_gt_10"] = {
            "pit_now": True,
            "compound": "HARD"
        }
        
        return branches
    
    def make_decision(
        self,
        model_outputs: Dict[str, Any],
        race_context: Dict[str, Any]
    ) -> StrategyOutput:
        """Convert model outputs to final strategy decision.
        
        Args:
            model_outputs: Dict with pit_prob_1, pit_prob_2, compound_pred, etc.
            race_context: Dict with current_lap, tire_age, compound, safety_car, etc.
        
        Returns:
            Complete strategy output for this lap.
        """
        pit_prob_1 = model_outputs.get('pit_prob_1', 0.0)
        pit_prob_2 = model_outputs.get('pit_prob_2', 0.0)
        compound_pred = model_outputs.get('compound_pred', 'HARD')
        compound_probs = model_outputs.get('compound_probs', {})
        stint_length = model_outputs.get('stint_length_pred', 15)
        
        current_lap = race_context.get('current_lap', 1)
        tire_age = race_context.get('tire_age', 0)
        current_compound = race_context.get('compound', 'MEDIUM')
        pit_loss_s = race_context.get('pit_loss_s', 22.0)
        degradation_rate = race_context.get('degradation_rate', 0.1)
        remaining_laps = race_context.get('remaining_laps', 50)
        safety_car_active = race_context.get('safety_car_active', False)
        rain_prob = race_context.get('rain_probability', 0.0)
        
        # --- Rule overrides ---
        pit_recommend = 0
        override_reason = None
        
        # SC opportunity
        if safety_car_active and tire_age > 10:
            pit_recommend = 1
            override_reason = "SC_OPPORTUNITY"
        
        # Rain incoming
        if rain_prob > self.rain_threshold:
            compound_pred = 'INTER'
            override_reason = "RAIN_EXPECTED"
        
        # Probability-based decision
        if pit_recommend == 0:
            pit_recommend = 1 if pit_prob_1 > self.pit_threshold else 0
        
        # --- Calculate expected gain ---
        expected_gain, confidence_band = self._calculate_expected_gain(
            current_lap, tire_age, pit_loss_s, degradation_rate, remaining_laps
        )
        
        # --- Generate race plan ---
        plan = self._generate_stint_plan(
            current_lap, remaining_laps, current_compound,
            int(stint_length), compound_pred,
            pit_loss_s=pit_loss_s, degradation_rate=degradation_rate
        )
        
        # --- Generate scenario branches ---
        plan.scenario_branches = self._generate_scenario_branches(
            current_lap, tire_age, safety_car_active
        )
        
        # --- Calculate confidence ---
        model_confidence = self._calculate_confidence(pit_prob_1, compound_probs)
        
        # --- Build decision ---
        decision = PitDecision(
            pit_recommend_now=pit_recommend,
            pit_probability_next_1_lap=round(pit_prob_1, 3),
            pit_probability_next_2_laps=round(pit_prob_2, 3),
            recommended_compound_next=compound_pred,
            compound_probabilities=compound_probs,
            target_stint_length_laps=int(stint_length),
            expected_total_race_time_s=None,  # Computed by simulator
            expected_gain_vs_stayout_s=expected_gain,
            confidence_band_s=confidence_band,
            model_confidence=round(model_confidence, 2)
        )
        
        # --- Live context ---
        live_context = {
            "current_lap": current_lap,
            "tire_age_laps": tire_age,
            "safety_car_active": safety_car_active,
            "rain_probability_pct": int(rain_prob * 100)
        }
        if override_reason:
            live_context["override_reason"] = override_reason
        
        return StrategyOutput(
            decision=decision,
            plan=plan,
            live_context=live_context
        )
    
    def to_json(self, output: StrategyOutput) -> dict:
        """Convert strategy output to JSON-serializable dict."""
        result = {
            "decision": {
                "pit_recommend_now": output.decision.pit_recommend_now,
                "pit_probability_next_1_lap": output.decision.pit_probability_next_1_lap,
                "pit_probability_next_2_laps": output.decision.pit_probability_next_2_laps,
                "recommended_compound_next": output.decision.recommended_compound_next,
                "target_stint_length_laps": output.decision.target_stint_length_laps,
                "expected_total_race_time_s": output.decision.expected_total_race_time_s,
                "expected_gain_vs_stayout_s": output.decision.expected_gain_vs_stayout_s,
                "confidence_band_s": list(output.decision.confidence_band_s)
            },
            "plan": {
                "planned_number_of_stops": output.plan.planned_number_of_stops,
                "stints": [
                    {
                        "from_lap": s.from_lap,
                        "to_lap": s.to_lap,
                        "compound": s.compound,
                        "planned_pit_lap": s.planned_pit_lap
                    }
                    for s in output.plan.stints
                ],
                "scenario_branches": output.plan.scenario_branches
            }
        }
        
        return result
