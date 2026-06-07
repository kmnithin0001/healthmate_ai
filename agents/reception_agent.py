"""
agents/reception_agent.py
─────────────────────────
The entry orchestrator of HealthMate AI 2.0. Filters emergencies, parses
persistent profile introductions, maps multi-intent routing groups, and hands
them off to the Coordinator Agent.
"""

from __future__ import annotations
import re
from typing import Dict, List, Tuple
from agents.base_agent import BaseAgent
from agents.emergency_agent import EmergencyAgent, detect_emergency_keywords
from agents.coordinator_agent import CoordinatorAgent
from memory.memory_manager import MemoryManager

_INTENT_KEYWORDS: Dict[str, List[str]] = {
    "symptom": [
        r"\bsymptom\b", r"\bfever\b", r"\bcough\b", r"\bcold\b", r"\bheadache\b", 
        r"\bstomach\b", r"\bfatigue\b", r"\bpain\b", r"\binfect", r"\bsore\s+throat\b"
    ],
    "skin": [
        r"\bdandruff\b", r"\bhair\s*loss\b", r"\bhairfall\b", r"\bbald\b", r"\bscalp\b", 
        r"\bacne\b", r"\bpimple\b", r"\bskin\b", r"\boily\b", r"\bdry\s+skin\b"
    ],
    "diet": [
        r"\bdiet\b", r"\beat\b", r"\bfood\b", r"\bnutrition\b", r"\bbulk\b", 
        r"\bweight\s+loss\b", r"\blose\s+weight\b", r"\bgain\s+weight\b", r"\bcalorie\b",
        r"\bobes", r"\boverweight\b"
    ],
    "fitness": [
        r"\bworkout\b", r"\bexercise\b", r"\bgym\b", r"\bfitness\b", r"\bcardio\b", 
        r"\blifting\b", r"\bhiit\b", r"\bstretch\b", r"\bmobility\b"
    ],
    "medicine": [
        r"\bmedicin\b", r"\bdrug\b", r"\bpill\b", r"\btablet\b", r"\bparacetamol\b", 
        r"\bibuprofen\b", r"\baspirin\b", r"\bamoxicillin\b", r"\bside\s+effect\b"
    ],
    "memory": [
        r"\bremember\b", r"\bprofile\b", r"\bmy\s+name\s+is\b", r"\bi\s+weigh\b", 
        r"\bi\s+am\s+\d{2,3}kg\b", r"\bforget\b", r"\bclear\b"
    ]
}

_COMPILED = {
    intent: [re.compile(p, re.IGNORECASE) for p in patterns]
    for intent, patterns in _INTENT_KEYWORDS.items()
}

def _classify_all_intents(text: str) -> List[str]:
    """Scans and counts keyword matches across all intent categories."""
    matched = []
    for intent, patterns in _COMPILED.items():
        for pat in patterns:
            if pat.search(text):
                matched.append(intent)
                break
    return matched if matched else ["general"]

class ReceptionAgent(BaseAgent):
    name = "Reception Agent"
    agent_id = "reception_agent"

    def __init__(self, memory: MemoryManager) -> None:
        super().__init__(memory)
        self.emergency_agent = EmergencyAgent(memory)
        self.coordinator = CoordinatorAgent(memory)

    @property
    def _system_core(self) -> str:
        return """
You are the Reception specialist of HealthMate AI 2.0.

Your job is to greet users warmly, introduce their specialist care team, and handle small-talk or general queries.
""".strip()

    def greet(self) -> str:
        """Standard onboarding greeting."""
        from config.settings import settings
        return (
            "Hello! 👋 I'm **HealthMate AI 2.0**, your cooperative multi-agent healthcare assistant.\n\n"
            "Our collaborative care team features:\n"
            "• 🩺 **Symptom Triage** (Fever, cough, cold)\n"
            "• 🧴 **Skin & Hair Care** (Acne, dandruff, shedding)\n"
            "• 🥗 **Diet & Nutrition** (Caloric targets, macro splits)\n"
            "• 🏋️ **Fitness & Wellness** (Hypertrophy routines, safety mobility)\n"
            "• 💊 **Medicine Safety** (Precautions and lookup)\n\n"
            "How can we coordinate your care today?" + settings.DISCLAIMER
        )

    def process(self, user_message: str) -> str:
        # ── Step 1: Emergency Filter override ──
        if self.emergency_agent.is_emergency(user_message):
            self.logger.warning("Emergency override matched!")
            return self.emergency_agent.process(user_message)

        # ── Step 2: Passive profile facts registration ──
        self.coordinator.memory_agent.extract_and_store_profile(user_message)

        # ── Step 3: Classify all intents ──
        intents = _classify_all_intents(user_message)
        self.logger.info(f"Matched routing intents: {intents}")

        if len(intents) == 1 and intents[0] == "general":
            # Small talk or greetings handled by Reception directly
            return self._handle_general(user_message)

        # ── Step 4: Coordinated Specialist Pipeline ──
        return self.coordinator.process_coordination(user_message, intents)

    def _handle_general(self, user_message: str) -> str:
        return self._respond(user_message)
