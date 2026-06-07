"""
agents/diet_agent.py
────────────────────
Specialized Diet Agent. Handles healthy eating, weight gain, bulking, and weight loss,
integrating physiological context and trigger-happy handoffs to the Fitness Agent.
"""

from __future__ import annotations
import re
from agents.base_agent import BaseAgent
from memory.memory_manager import MemoryManager

class DietAgent(BaseAgent):
    name = "Diet Recommendation Agent"
    agent_id = "diet_agent"

    def __init__(self, memory: MemoryManager) -> None:
        super().__init__(memory)

    @property
    def _system_core(self) -> str:
        return """
You are a Senior Diet & Nutrition specialist of HealthMate AI.

Your primary goal is to provide healthy diet and nutrition recommendations.

=== REQUIRED OUTPUT STRUCTURE (YOU MUST USE THESE EXACT HEADERS) ===
🥗 **Goal Summary**: Brief summary of the diet target or goal (e.g. bulking, fat loss, hydration) based on user metrics and preferences.

📊 **Nutrition Targets**: Specific caloric estimates and macronutrient proportions (protein, carbs, fats) tailored to user metrics from profile memory if available.

🍳 **Meal Suggestions**:
   • Breakfast ideas
   • Lunch ideas
   • Dinner ideas
   • Healthy snacks

💧 **Hydration Advice**: Targeted daily water intake guidelines, calculating targets based on body weight if available.
""".strip()

    def process(self, user_message: str) -> str:
        lower_msg = user_message.lower()

        # 1. Update memory profile values
        diet_keywords = {
            "vegetarian": "Vegetarian", "vegan": "Vegan", "gluten-free": "Gluten-free",
            "keto": "Keto", "low-carb": "Low-carb", "high-protein": "High-protein"
        }
        for kw, pref in diet_keywords.items():
            if kw in lower_msg:
                self.memory.update_profile(diet_preferences=pref)

        if "lose weight" in lower_msg or "weight loss" in lower_msg or "fat loss" in lower_msg:
            self.memory.update_profile(fitness_goals="Weight loss")
        if "gain weight" in lower_msg or "muscle gain" in lower_msg or "bulking" in lower_msg:
            self.memory.update_profile(fitness_goals="Muscle gain")

        # 2. Check and register dynamic A2A Handoffs in shared memory
        # A2A Workflow: Diet -> Fitness to match caloric adjustments
        if any(kw in lower_msg for kw in ["weight", "gain", "lose", "bulk", "muscle"]):
            self.memory.add_handoff("fitness")
            self.memory.set_agent_observation(
                "diet",
                "Caloric target guidelines established. Recommend training programs supporting this metabolic goal."
            )

        return self._respond(user_message)
