"""
Multi-Agent Internal Disagreement Layer — Wave 8.

Each agent is a distinct interpretive lens over identical market input.
The system no longer asks "is this signal correct?"
It asks "is this signal stable across incompatible interpretations?"
"""
from collections import deque
import math


class AgentState:
    def __init__(self, agent_id: str, signal: float, confidence: float,
                 bias: float, memory_drift: float):
        self.agent_id = agent_id
        self.signal = signal
        self.confidence = confidence
        self.bias = bias
        self.memory_drift = memory_drift


class InternalAgent:
    def __init__(self, agent_id: str, bias: float, memory_decay: float):
        self.agent_id = agent_id
        self.bias = bias
        self.memory_decay = memory_decay
        self.memory = deque(maxlen=64)

    def process(self, signal: float) -> AgentState:
        biased_signal = signal * (1.0 + self.bias)
        memory_effect = sum(self.memory) / len(self.memory) if self.memory else 0.0
        memory_drift = self.memory_decay * memory_effect
        interpreted = math.tanh(biased_signal + memory_drift)
        confidence = math.tanh(abs(interpreted))
        self.memory.append(interpreted)
        return AgentState(
            agent_id=self.agent_id,
            signal=interpreted,
            confidence=confidence,
            bias=self.bias,
            memory_drift=memory_drift,
        )
