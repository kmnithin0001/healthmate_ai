"""
agents/base_agent.py
────────────────────
Parent abstract base class for all HealthMate AI 2.0 specialized agents.
Handles prompt assembly, persistent profile injection, and transient A2A observations context.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from config.settings import settings
from memory.memory_manager import MemoryManager
from utils.gemini_client import GeminiClient
from utils.logger import get_logger

class BaseAgent(ABC):
    name: str = "Base Agent"
    agent_id: str = "base_agent"

    def __init__(self, memory: MemoryManager) -> None:
        self.memory = memory
        self.gemini = GeminiClient()
        self.logger = get_logger(self.__class__.__name__)
        self.coordinator: Optional[BaseAgent] = None

    def query_agent(self, agent_id: str, query: str) -> str:
        """Enables direct programmatic Agent-to-Agent (A2A) consultation queries."""
        self.logger.info(f"Direct A2A Query: {self.agent_id} ➔ {agent_id} | Query: '{query}'")
        if not self.coordinator or not hasattr(self.coordinator, "_registry"):
            return "A2A connection unavailable."
        target = self.coordinator._registry.get(agent_id)
        if not target:
            return f"Agent '{agent_id}' not found."
        return target.process(query)

    @property
    @abstractmethod
    def _system_core(self) -> str:
        """Core specialist system prompt."""
        pass

    @abstractmethod
    def process(self, user_message: str) -> str:
        """Core specialist processing loop."""
        pass

    def _build_system_prompt(self, extra_context: str = "") -> str:
        profile_summary = self.memory.profile.summary()
        history = self.memory.format_history_for_prompt(last_n=6)
        
        # Inject transient A2A turn observations from other agents
        observations = self.memory.get_all_observations()
        obs_text = (
            "\n".join(f"- [From {agent_id.upper()} Specialist]: {obs}" for agent_id, obs in observations.items())
            if observations
            else "No transient findings from other specialists registered in this turn."
        )

        return f"""
You are HealthMate AI — a friendly, professional, empathetic, and responsible healthcare assistant.

=== YOUR ROLE ===
{self._system_core}

=== CLINICAL SAFETY CONSTRAINTS (NEVER VIOLATE) ===
1. NEVER declare or state a definitive diagnosis with absolute certainty. Always speak in possibilities.
2. NEVER prescribe medications or suggest specific dosages.
3. ALWAYS recommend consulting a qualified healthcare professional.
4. If emergency symptoms (chest pain, stroke, breathing blockage, unconsciousness) are mentioned, advise calling emergency services (112/911) immediately.

=== CENTRAL PERSISTENT PROFILE ===
{profile_summary}

=== TRANSIENT COOPERATIVE FINDINGS (SHARED MEMORY) ===
{obs_text}

=== RECENT SESSION CONVERSATION HISTORY ===
{history}

{extra_context}
""".strip()

    def _respond(self, user_message: str, extra_context: str = "") -> str:
        system_prompt = self._build_system_prompt(extra_context)
        raw = self.gemini.generate(system_prompt=system_prompt, user_message=user_message)
        
        # Append turns to session history
        self.memory.add_turn("user", self.agent_id, user_message)
        self.memory.add_turn("assistant", self.agent_id, raw)
        return raw + settings.DISCLAIMER
