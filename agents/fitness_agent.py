"""
agents/fitness_agent.py
───────────────────────
Specialized Fitness Agent. Builds home or gym workout routines, integrating
active injuries or respiratory limits, and delegates macro mapping to the Diet Agent.
"""

from __future__ import annotations
import re
from agents.base_agent import BaseAgent
from memory.memory_manager import MemoryManager

class FitnessAgent(BaseAgent):
    name = "Fitness Recommendation Agent"
    agent_id = "fitness_agent"

    def __init__(self, memory: MemoryManager) -> None:
        super().__init__(memory)

    @property
    def _system_core(self) -> str:
        return """
You are a Senior Fitness & Wellness specialist of HealthMate AI.

Your primary goal is to provide safe and effective workout plans.

=== REQUIRED OUTPUT STRUCTURE (YOU MUST USE THESE EXACT HEADERS) ===
🏋️ **Workout Plan**: Recommended exercise categories, sets, reps, and work-rest ratios tailored to user parameters and injury history.

📅 **Weekly Schedule**:
   • Day 1: [Activity]
   • Day 2: [Activity]
   • …

😴 **Recovery Advice**: Vital rest, hydration, stress, and sleep guidelines supporting physical recovery.
""".strip()

    def process(self, user_message: str) -> str:
        lower_msg = user_message.lower()

        # 1. Update memory profile values
        fitness_keywords = {
            "weight loss": "Weight loss", "muscle gain": "Muscle gain", "strength": "Strength",
            "endurance": "Endurance", "flexibility": "Flexibility", "yoga": "Yoga",
            "bodybuilding": "Bodybuilding", "hiit": "HIIT"
        }
        for kw, goal in fitness_keywords.items():
            if kw in lower_msg:
                self.memory.update_profile(fitness_goals=goal)

        # 2. Check and register dynamic A2A Handoffs in shared memory
        # A2A Workflow: Fitness -> Diet to align macros with heavy loads or cardio
        if any(kw in lower_msg for kw in ["workout", "exercise", "routine", "gym", "cardio", "lifting"]):
            self.memory.add_handoff("diet")
            self.memory.set_agent_observation(
                "fitness",
                "Physical load circuit established. Recommends corresponding protein intake and fuel from Diet."
            )

        return self._respond(user_message)
