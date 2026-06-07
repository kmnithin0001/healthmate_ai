"""
agents/emergency_agent.py
─────────────────────────
Highest-priority Emergency Detection Agent. Scans for critical life-threatening
symptoms and provides immediate, red-flag first aid guidelines.
"""

from __future__ import annotations
import re
from typing import List, Tuple
from agents.base_agent import BaseAgent
from memory.memory_manager import MemoryManager

_EMERGENCY_PATTERNS: List[Tuple[str, str]] = [
    (r"\bchest\s+pain\b|\bheart\s+attack\b|\bcardiac\b", "Chest Pain / Cardiac arrest"),
    (r"\bcan'?t\s+breathe\b|\bnot\s+breathing\b|\bdifficulty\s+breathing\b|\bchok", "Breathing Obstruction"),
    (r"\bstroke\b|\bface\s+drop|\bspeech\s+slur|\bslurred\s+speech", "Stroke Symptoms"),
    (r"\bunconscious\b|\bpassed\s+out\b|\bnot\s+responding\b", "Unconsciousness"),
    (r"\bsuicid(e|al)\b|\bkill\s+myself\b", "Severe Mental Health Crisis"),
    (r"\banaphylax\b|\ballergic\s+shock\b", "Severe Allergic Shock")
]

_COMPILED = [(re.compile(p, re.IGNORECASE), label) for p, label in _EMERGENCY_PATTERNS]

def detect_emergency_keywords(text: str) -> List[str]:
    """Helper to scan text for obvious red flags."""
    return [label for pattern, label in _COMPILED if pattern.search(text)]

class EmergencyAgent(BaseAgent):
    name = "Emergency Detection Agent"
    agent_id = "emergency_agent"

    def __init__(self, memory: MemoryManager) -> None:
        super().__init__(memory)

    @property
    def _system_core(self) -> str:
        return """
You are the Emergency Detection specialist of HealthMate AI.

Your ONLY job is to advise the user to contact emergency services and give immediate, concise first-aid directions.

=== REQUIRED OUTPUT STRUCTURE (YOU MUST USE THIS EXACT FORMAT) ===
🚨 **EMERGENCY DETECTED**: [Condition Name]

⚡ **CALL 112 / 911 NOW — Do not wait.**

What to do right now:
• [Action 1]
• [Action 2]
• [Action 3]

Stay calm. Help is on the way.
""".strip()

    def is_emergency(self, text: str) -> bool:
        """Determines if the text contains immediate life-threats."""
        return bool(detect_emergency_keywords(text))

    def process(self, user_message: str) -> str:
        detected = detect_emergency_keywords(user_message)
        condition = ", ".join(detected) if detected else "Potential Emergency"
        extra = f"=== DETECTED LIFE-THREATS ===\n{condition}\n"
        return self._respond(user_message, extra_context=extra)
