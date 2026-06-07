"""
memory/memory_manager.py
────────────────────────
Provides centralized shared memory (persistent file-backed and transient turn observations).
"""

from __future__ import annotations
import json
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional
from config.settings import settings

@dataclass
class UserProfile:
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    weight: Optional[float] = None  # in kg
    height: Optional[float] = None  # in cm
    allergies: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)     # Stored medical history
    medications: List[str] = field(default_factory=list)
    fitness_goals: List[str] = field(default_factory=list)
    diet_preferences: List[str] = field(default_factory=list)
    frequent_symptoms: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> UserProfile:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def summary(self) -> str:
        """Renders a flat summary string for LLM system prompt injection."""
        parts: List[str] = []
        if self.name: parts.append(f"Name: {self.name}")
        if self.age: parts.append(f"Age: {self.age}")
        if self.gender: parts.append(f"Gender: {self.gender}")
        if self.weight: parts.append(f"Weight: {self.weight} kg")
        if self.height: parts.append(f"Height: {self.height} cm")
        if self.allergies: parts.append(f"Allergies: {', '.join(self.allergies)}")
        if self.conditions: parts.append(f"Medical conditions: {', '.join(self.conditions)}")
        if self.medications: parts.append(f"Medications: {', '.join(self.medications)}")
        if self.fitness_goals: parts.append(f"Fitness goals: {', '.join(self.fitness_goals)}")
        if self.diet_preferences: parts.append(f"Diet preferences: {', '.join(self.diet_preferences)}")
        if self.frequent_symptoms: parts.append(f"Recent symptoms: {', '.join(self.frequent_symptoms)}")
        return "; ".join(parts) if parts else "No profile data registered yet."

@dataclass
class ConversationTurn:
    role: str
    agent: str
    content: str
    timestamp: float = field(default_factory=time.time)

class MemoryManager:
    """Centralized Shared Memory Manager providing thread-safe profile and A2A contexts."""
    
    def __init__(self, user_id: str = "default_user") -> None:
        self.user_id = user_id
        
        # Bounded session conversation history
        self._session_history: Deque[ConversationTurn] = deque(maxlen=settings.SESSION_MEMORY_LIMIT)
        
        # Transient turn observations and dynamic A2A handoffs
        self._agent_observations: Dict[str, str] = {}
        self._active_handoffs: List[str] = []
        
        # Long-term persistence
        self._store_path: Path = settings.MEMORY_STORE_PATH
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        self._long_term = self._load_long_term()
        
        if "profile" not in self._long_term:
            self._long_term["profile"] = UserProfile().to_dict()
            self._save_long_term()

    @property
    def profile(self) -> UserProfile:
        return UserProfile.from_dict(self._long_term["profile"])

    def update_profile(self, **kwargs: Any) -> None:
        """Update fields in long-term memory."""
        p = self._long_term["profile"]
        for k, v in kwargs.items():
            if k in UserProfile.__dataclass_fields__:
                if isinstance(p.get(k), list):
                    if isinstance(v, str):
                        if v not in p[k]: p[k].append(v)
                    elif isinstance(v, list):
                        for item in v:
                            if item not in p[k]: p[k].append(item)
                else:
                    p[k] = v
        p["updated_at"] = time.time()
        self._save_long_term()

    def record_symptom(self, symptom: str) -> None:
        p = self._long_term["profile"]
        symptoms: List[str] = p.get("frequent_symptoms", [])
        if symptom in symptoms:
            symptoms.remove(symptom)
        symptoms.insert(0, symptom)
        p["frequent_symptoms"] = symptoms[:10]
        self._save_long_term()

    # ── Session / Conversation History ──
    def add_turn(self, role: str, agent: str, content: str) -> None:
        self._session_history.append(ConversationTurn(role=role, agent=agent, content=content))

    def get_session_history(self, last_n: int = 10) -> List[ConversationTurn]:
        turns = list(self._session_history)
        return turns[-last_n:] if len(turns) > last_n else turns

    def format_history_for_prompt(self, last_n: int = 6) -> str:
        turns = self.get_session_history(last_n)
        if not turns:
            return "No previous conversation in this session."
        return "\n".join(f"{'User' if t.role == 'user' else f'Assistant ({t.agent})'}: {t.content}" for t in turns)

    # ── Transient Turn Context (Shared Observations & Handoffs) ──
    def clear_turn_context(self) -> None:
        """Resets the A2A observations and handoff registry at the start of a turn."""
        self._agent_observations.clear()
        self._active_handoffs.clear()

    def set_agent_observation(self, agent_id: str, observation: str) -> None:
        self._agent_observations[agent_id] = observation

    def get_all_observations(self) -> Dict[str, str]:
        return dict(self._agent_observations)

    def add_handoff(self, target_agent: str) -> None:
        if target_agent not in self._active_handoffs:
            self._active_handoffs.append(target_agent)

    def get_handoffs(self) -> List[str]:
        return list(self._active_handoffs)

    def clear_handoffs(self) -> None:
        self._active_handoffs.clear()

    # ── Long-term persistence serialization helpers ──
    def _load_long_term(self) -> Dict[str, Any]:
        if self._store_path.exists():
            try:
                with open(self._store_path, "r", encoding="utf-8") as f:
                    return json.load(f).get(self.user_id, {})
            except (json.JSONDecodeError, IOError):
                pass
        return {}

    def _save_long_term(self) -> None:
        full_data: Dict[str, Any] = {}
        if self._store_path.exists():
            try:
                with open(self._store_path, "r", encoding="utf-8") as f:
                    full_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        full_data[self.user_id] = self._long_term
        with open(self._store_path, "w", encoding="utf-8") as f:
            json.dump(full_data, f, indent=2, default=str)

    def clear_session(self) -> None:
        self._session_history.clear()
        self.clear_turn_context()
