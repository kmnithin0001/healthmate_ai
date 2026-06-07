"""
agents/symptom_agent.py
───────────────────────
Specialized Symptom Triage Agent. Intercepts common complaints and dynamically
initiates A2A handoffs to Diet, Fitness, or Medicine specialists.
"""

from __future__ import annotations
import re
from agents.base_agent import BaseAgent
from memory.memory_manager import MemoryManager

class SymptomAgent(BaseAgent):
    name = "Symptom Analysis Agent"
    agent_id = "symptom_agent"

    def __init__(self, memory: MemoryManager) -> None:
        super().__init__(memory)

    @property
    def _system_core(self) -> str:
        return """
You are a Senior Symptom Triage specialist of HealthMate AI.

Your primary goal is to provide non-definitive, educational symptom-triage guidance.

=== REQUIRED OUTPUT STRUCTURE (YOU MUST USE THESE EXACT HEADERS) ===
📋 **Symptom Summary**: Compact summary of symptoms described, mentioning duration or severity if user specified.

🔍 **Possible Causes**: 2-3 non-definitive possibilities tailored specifically to this symptom. Always speak in terms of likelihoods (e.g. "could be related to").

💡 **Precautions**: Specific, practical home care and self-care steps.

❓ **Follow-up Questions**: 2-3 relevant questions to help the user prepare for a professional consultation.

⚠️ **Warning Signs**: Critical red flags for this specific symptom that require immediate clinical attention.
""".strip()

    def process(self, user_message: str) -> str:
        lower_msg = user_message.lower()
        
        # 1. Update memory symptoms keywords
        symptom_keywords = ["fever", "cough", "cold", "headache", "stomach pain", "fatigue", "body pain", "infection", "pain"]
        for kw in symptom_keywords:
            if kw in lower_msg:
                self.memory.record_symptom(kw)

        # 2. Check and register dynamic A2A Handoffs in shared memory
        # A2A Workflow: Symptom -> Diet & Medicine for fever / stomach pain
        if any(kw in lower_msg for kw in ["fever", "stomach", "vomit", "infection"]):
            self.memory.add_handoff("diet")
            self.memory.add_handoff("medicine")
            self.memory.set_agent_observation(
                "symptom", 
                "Fever/gastric symptom registered. Recommend hydration nutrition and checking medication side effects."
            )
            
        # A2A Workflow: Symptom -> Fitness for muscle/joint/body pain / fatigue
        if any(kw in lower_msg for kw in ["pain", "shoulder", "fatigue", "back", "knee", "joint"]):
            self.memory.add_handoff("fitness")
            self.memory.set_agent_observation(
                "symptom",
                "Musculoskeletal discomfort or fatigue registered. Recommend gentle range-of-motion routines."
            )

        # 3. Direct A2A Consulting Lookup
        extra_context = ""
        if "fever" in lower_msg:
            # Query the Medicine Agent directly for Paracetamol safety guidance
            med_advice = self.query_agent("medicine", "paracetamol")
            extra_context = f"\n=== DIRECT A2A PHARMACIST CONSULTATION INFO ===\n{med_advice}\n"

        return self._respond(user_message, extra_context=extra_context)
