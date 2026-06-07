"""
agents/coordinator_agent.py
───────────────────────────
Orchestrates sequential multi-agent collaborative execution, dynamic handoffs,
response merging, and terminal observability metrics logging.
"""

from __future__ import annotations
import time
from typing import Dict, List, Optional
from config.settings import settings
from memory.memory_manager import MemoryManager
from agents.base_agent import BaseAgent
from agents.symptom_agent import SymptomAgent
from agents.skin_hair_agent import SkinHairAgent
from agents.diet_agent import DietAgent
from agents.fitness_agent import FitnessAgent
from agents.medicine_agent import MedicineAgent
from agents.memory_agent import MemoryAgent

class CoordinatorAgent(BaseAgent):
    name = "Coordinator Agent"
    agent_id = "coordinator_agent"

    def __init__(self, memory: MemoryManager) -> None:
        super().__init__(memory)
        # Register all specialist agents
        self.symptom_agent = SymptomAgent(memory)
        self.skin_hair_agent = SkinHairAgent(memory)
        self.diet_agent = DietAgent(memory)
        self.fitness_agent = FitnessAgent(memory)
        self.medicine_agent = MedicineAgent(memory)
        self.memory_agent = MemoryAgent(memory)
        
        self._registry: Dict[str, BaseAgent] = {
            "symptom": self.symptom_agent,
            "skin": self.skin_hair_agent,
            "diet": self.diet_agent,
            "fitness": self.fitness_agent,
            "medicine": self.medicine_agent,
            "memory": self.memory_agent,
        }
        for agent in self._registry.values():
            agent.coordinator = self

    @property
    def _system_core(self) -> str:
        return "Coordinating agent care pipeline. Combines specialist advice and logs observability metrics."

    def process(self, user_message: str) -> str:
        # Reception agent handles primary routing; Coordinator executes from here
        return "Direct orchestration processing must call process_coordination."

    def process_coordination(self, user_message: str, initial_intents: List[str]) -> str:
        """Runs sequential execution queue, checking shared memory dynamically for A2A handoffs."""
        start_time = time.time()
        
        # Reset A2A transient context at turn start
        self.memory.clear_turn_context()
        
        execution_queue = list(initial_intents)
        executed_agents: List[str] = []
        agent_responses: Dict[str, str] = {}
        agent_calls_log: List[str] = []

        while execution_queue:
            current_intent = execution_queue.pop(0)
            if current_intent in executed_agents:
                continue

            agent = self._registry.get(current_intent)
            if not agent:
                continue

            executed_agents.append(current_intent)
            
            # Log the call relation path
            if len(executed_agents) > 1:
                last_agent = executed_agents[-2]
                agent_calls_log.append(f"{last_agent.upper()} ➔ {current_intent.upper()} (Handoff)")
            else:
                agent_calls_log.append(f"USER ➔ {current_intent.upper()}")

            # Execute the specialist agent
            response = agent.process(user_message)
            agent_responses[current_intent] = response

            # Inspect shared memory for newly registered A2A handoffs
            new_handoffs = self.memory.get_handoffs()
            for handoff in new_handoffs:
                if handoff not in executed_agents and handoff not in execution_queue:
                    execution_queue.append(handoff)

        execution_time_ms = int((time.time() - start_time) * 1000)
        
        # 1. Print high-contrast Observability Panel
        self._log_observability(executed_agents, agent_calls_log, execution_time_ms)

        # 2. Merge specialist response blocks, cleaning duplicate disclaimers
        return self._merge_responses(executed_agents, agent_responses)

    def _log_observability(self, executed_agents: List[str], agent_calls_log: List[str], execution_time_ms: int) -> None:
        """Formats and logs rich observability console output."""
        profile = self.memory.profile
        
        # Collect active memory fields loaded
        active_memory = []
        if profile.name: active_memory.append(f"name={profile.name}")
        if profile.age: active_memory.append(f"age={profile.age}")
        if profile.gender: active_memory.append(f"gender={profile.gender}")
        if profile.weight: active_memory.append(f"weight={profile.weight}kg")
        if profile.height: active_memory.append(f"height={profile.height}cm")
        if profile.fitness_goals: active_memory.append(f"goals={profile.fitness_goals}")
        if profile.conditions: active_memory.append(f"conditions={profile.conditions}")
        memory_used = ", ".join(active_memory) if active_memory else "None"

        calls_path = " ➔ ".join(agent_calls_log) if agent_calls_log else "None"
        agents_list = ", ".join(agent.upper() for agent in executed_agents)

        print("\n" + "─" * 70)
        print("📢 HEALTHMATE AI COLLABORATIVE MULTI-AGENT OBSERVABILITY LOG")
        print("─" * 70)
        print(f"🎯 Selected Agents : [{agents_list}]")
        print(f"💾 Stored Memory   : [{memory_used}]")
        print(f"🔄 Cooperative Flow: {calls_path}")
        print(f"⏱️ Execution Time  : {execution_time_ms} ms")
        print("─" * 70 + "\n")

    def _merge_responses(self, executed_agents: List[str], agent_responses: Dict[str, str]) -> str:
        """Integrates multiple specialist outputs into a single coordinated response."""
        disclaimer = settings.DISCLAIMER.strip()
        cleaned_bodies = []
        agent_names = []

        for agent_id in executed_agents:
            raw_response = agent_responses.get(agent_id, "")
            # Strip disclaimer from individual response
            clean_body = raw_response.replace(disclaimer, "").strip()
            if clean_body:
                cleaned_bodies.append(clean_body)
                # Map to human-readable names
                name_map = {
                    "symptom": "Symptom Analysis",
                    "skin": "Skin & Hair",
                    "diet": "Diet & Nutrition",
                    "fitness": "Fitness & Wellness",
                    "medicine": "Medicine Information",
                    "memory": "Memory Management",
                }
                agent_names.append(name_map.get(agent_id, agent_id.upper()))

        if not cleaned_bodies:
            return f"No recommendations generated by coordinated specialists.{settings.DISCLAIMER}"

        if len(agent_names) > 1:
            coordinated_title = ", ".join(agent_names[:-1]) + " and " + agent_names[-1]
            intro = f"I have consulted our **{coordinated_title}** specialists to address your queries in an integrated care response."
        else:
            coordinated_title = agent_names[0]
            intro = f"I have consulted our **{coordinated_title}** specialist to address your queries."
        
        merged_body = "\n\n---\n\n".join(cleaned_bodies)

        return f"""
🤝 **HealthMate AI Coordinated Care Guidance**:
{intro}

{merged_body}

{settings.DISCLAIMER.strip()}
""".strip()
