"""
Recursive Belief Contagion Layer — Wave 9.

Agents no longer form independent opinions.
They "steal structure" from each other during interpretation:
interpretation is transmitted, not computed.
"""
import math


class ContagionState:
    def __init__(self, signal: float, infection_pressure: float, stability: float):
        self.signal = signal
        self.infection_pressure = infection_pressure
        self.stability = stability


class BeliefContagionModel:
    def __init__(self, contagion_rate: float = 0.35):
        self.contagion_rate = contagion_rate

    def propagate(self, agent_outputs) -> ContagionState:
        signals = [a.signal for a in agent_outputs]
        updated_signals = []

        for i, a in enumerate(agent_outputs):
            neighbors = signals[:i] + signals[i+1:]
            neighbor_mean = sum(neighbors) / (len(neighbors) + 1e-9)
            pressure = neighbor_mean - a.signal
            infected_signal = a.signal + self.contagion_rate * pressure
            updated_signals.append(infected_signal)

        mean = sum(updated_signals) / len(updated_signals)
        variance = sum((x - mean) ** 2 for x in updated_signals) / len(updated_signals)
        stability = 1.0 / (1.0 + variance)
        final_signal = sum(updated_signals) / len(updated_signals)

        return ContagionState(
            signal=final_signal,
            infection_pressure=sum(abs(s) for s in updated_signals) / len(updated_signals),
            stability=stability,
        )
