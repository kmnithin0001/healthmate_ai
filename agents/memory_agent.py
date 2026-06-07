"""
agents/memory_agent.py
──────────────────────
Specialized Memory Management Agent. Handles data updates, name checks, and
profile conversions (Metric/Imperial weight/height).
"""

from __future__ import annotations
import re
from agents.base_agent import BaseAgent
from memory.memory_manager import MemoryManager

class MemoryAgent(BaseAgent):
    name = "Memory Management Agent"
    agent_id = "memory_agent"

    def __init__(self, memory: MemoryManager) -> None:
        super().__init__(memory)

    @property
    def _system_core(self) -> str:
        return """
You are the Memory Management specialist of HealthMate AI.

Your primary goal is to present registered profile variables to the user or confirm updates.

=== REQUIRED OUTPUT STRUCTURE DEPENDING ON USER REQUEST ===
- For RECALL/PROFILE CHECK: Present a friendly summary of user name, age, weight, height, history, and active goals.
- For UPDATE/SET DATA: Confirm the parameter successfully saved and why this context helps personalize their experience.
- For FORGET DATA: Reassure the user that session or database profiles have been cleared successfully.
""".strip()

    def process(self, user_message: str) -> str:
        lower = user_message.lower()

        # Parse and store profile values
        self.extract_and_store_profile(user_message)

        if any(w in lower for w in ["forget", "delete", "clear", "remove"]):
            self.memory.clear_session()
            self.memory._long_term["profile"] = {}
            self.memory._save_long_term()
            return f"I have successfully cleared all of your persistent profile facts and session history.{self._respond(user_message)}"

        # Default handler: responds to what is remembered / set
        profile = self.memory.profile
        extra = f"=== CURRENT DATABASE PROFILE FACTS ===\n{profile.summary()}"
        return self._respond(user_message, extra_context=extra)

    def extract_and_store_profile(self, text: str) -> None:
        """Parses weights (kg/lbs), heights (cm/ft/in), age, goals, and conditions."""
        # Name
        name_match = re.search(r"(?:my name is|i am|i'm|call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", text, re.IGNORECASE)
        if name_match:
            self.memory.update_profile(name=name_match.group(1).strip())

        # Age
        age_match = re.search(r"\b(\d{1,3})\s*(?:years?\s*old|yr|y\.?o\.?)\b", text, re.IGNORECASE)
        if not age_match:
            age_match = re.search(r"\baged?\s+(\d{1,3})\b", text, re.IGNORECASE)
        if age_match:
            age = int(age_match.group(1))
            if 1 <= age <= 120:
                self.memory.update_profile(age=age)

        # Gender
        if re.search(r"\b(male|man|boy|he/him)\b", text, re.IGNORECASE):
            self.memory.update_profile(gender="Male")
        elif re.search(r"\b(female|woman|girl|she/her)\b", text, re.IGNORECASE):
            self.memory.update_profile(gender="Female")

        # Weight (Metric / Imperial)
        weight_match = re.search(r"\b(\d+(?:\.\d+)?)\s*(kg|kilograms|lbs|pounds)\b", text, re.IGNORECASE)
        if weight_match:
            val = float(weight_match.group(1))
            unit = weight_match.group(2).lower()
            if "lb" in unit:
                val = round(val * 0.453592, 1)  # lb to kg conversion
            self.memory.update_profile(weight=val)

        # Height (Metric / Imperial)
        height_match = re.search(r"\b(\d{2,3})\s*(?:cm|centimeters)\b", text, re.IGNORECASE)
        if height_match:
            self.memory.update_profile(height=float(height_match.group(1)))
        else:
            imp_match = re.search(r"\b(\d+)\s*(?:ft|feet)\b(?:\s*(\d+)\s*(?:in|inches)\b)?", text, re.IGNORECASE)
            if imp_match:
                feet = float(imp_match.group(1))
                inches = float(imp_match.group(2)) if imp_match.group(2) else 0.0
                cm_val = round((feet * 12 + inches) * 2.54, 1)
                self.memory.update_profile(height=cm_val)

        # Goals
        if re.search(r"\blose\s+weight\b|\bweight\s+loss\b|\bfat\s+loss\b", text, re.IGNORECASE):
            self.memory.update_profile(fitness_goals="Weight loss")
        if re.search(r"\bgain\s+weight\b|\bmuscle\s+gain\b|\bbulking\b", text, re.IGNORECASE):
            self.memory.update_profile(fitness_goals="Muscle gain")

        # Conditions / Allergies / Medications
        cond_match = re.search(r"\b(?:history\s+of|diagnosed\s+with|suffer\s+from)\s+([a-z\s]+)\b", text, re.IGNORECASE)
        if cond_match:
            cond = cond_match.group(1).strip()
            if len(cond.split()) <= 3:
                self.memory.update_profile(conditions=cond.capitalize())

        # Keywords check for history
        medical_keywords = ["asthma", "diabetes", "hypertension", "arthritis"]
        for kw in medical_keywords:
            if kw in text.lower():
                self.memory.update_profile(conditions=kw.capitalize())
