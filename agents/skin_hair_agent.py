"""
agents/skin_hair_agent.py
─────────────────────────
Specialized Skin & Hair Agent. Analyzes dry skin, acne, dandruff, and hair loss,
and dynamically delegates scalp-nourishment nutrition plans to the Diet Agent.
"""

from __future__ import annotations
from agents.base_agent import BaseAgent
from memory.memory_manager import MemoryManager

class SkinHairAgent(BaseAgent):
    name = "Skin & Hair Agent"
    agent_id = "skin_hair_agent"

    def __init__(self, memory: MemoryManager) -> None:
        super().__init__(memory)

    @property
    def _system_core(self) -> str:
        return """
You are a Senior Skin & Hair specialist of HealthMate AI.

Your primary goal is to provide non-definitive, educational scalp and skin care guidance.

=== REQUIRED OUTPUT STRUCTURE (YOU MUST USE THESE EXACT HEADERS) ===
🧴 **Skin/Hair Analysis**: Summary of skin or hair symptoms described.

💡 **Care Routine**: Step-by-step external care habits (shampoo, washing, sunscreen, etc.).

🥗 **Nutrition Suggestions**: Core vitamins or dietary requirements to support recovery.

⚠️ **Warning Signs**: Red flags that warrant consulting a board-certified dermatologist.
""".strip()

    def process(self, user_message: str) -> str:
        lower_msg = user_message.lower()

        # 1. Update memory symptoms keywords
        if "dandruff" in lower_msg:
            self.memory.record_symptom("dandruff")
        if "hair loss" in lower_msg or "hairfall" in lower_msg or "baldness" in lower_msg:
            self.memory.record_symptom("hair loss")

        # 2. Check and register dynamic A2A Handoffs in shared memory
        # A2A Workflow: Skin/Hair -> Diet for flaking, hair shedding, or acne
        if any(kw in lower_msg for kw in ["dandruff", "hair", "acne", "dry skin", "pimples"]):
            self.memory.add_handoff("diet")
            self.memory.set_agent_observation(
                "skin",
                "Scalp flaking / hair shedding / skin inflammation registered. Recommends skin-supporting micronutrients."
            )

        return self._respond(user_message)
