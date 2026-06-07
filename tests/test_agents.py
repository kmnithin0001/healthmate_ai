"""
tests/test_agents.py
────────────────────
Automated verification suite for HealthMate AI 2.0.
Verifies symptom triages, diet macro targets, fitness regimes, memory updates,
and dynamic agent-to-agent task handoffs.
"""

from __future__ import annotations
import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

# Put project root in system path
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("GOOGLE_API_KEY", "test_key_for_unit_tests")
os.environ.setdefault("MEMORY_STORE_PATH", "/tmp/healthmate_2_0_test.json")

from memory.memory_manager import MemoryManager
from agents.reception_agent import ReceptionAgent, _classify_all_intents
from test_playground import generate_mock_response, SYNONYMS

@pytest.fixture
def memory():
    """Returns a fresh clean memory instance for every test case."""
    import uuid
    uid = f"test_{uuid.uuid4().hex[:8]}"
    m = MemoryManager(user_id=uid)
    m.clear_session()
    
    # Pre-populate baseline profiles
    m._long_term["profile"] = {
        "name": "Kumar",
        "age": 34,
        "gender": "Male",
        "fitness_goals": [],
        "diet_preferences": [],
        "health_interests": [],
        "frequent_symptoms": [],
        "conditions": [],
        "allergies": [],
        "medications": []
    }
    m._save_long_term()
    return m

@pytest.fixture
def agent(memory):
    return ReceptionAgent(memory)

# ══════════════════════════════════════════════════════════════
# Collaborative Multi-Agent & Handoff Tests
# ══════════════════════════════════════════════════════════════

class TestA2ACooperativePlatform:

    @patch("utils.gemini_client.GeminiClient.generate", side_effect=generate_mock_response)
    def test_scenario_1_fever_handoff(self, mock_gen, agent, memory):
        """Scenario 1: 'I have fever' triggers Symptom -> Diet & Medicine handoffs."""
        result = agent.process("I have fever")
        
        # Verify that fever symptoms are recorded
        assert "fever" in memory.profile.frequent_symptoms
        
        # Verify A2A handoffs occurred
        assert "diet" in memory.get_handoffs()
        assert "medicine" in memory.get_handoffs()

        # Check response contains integrated care output for Symptom, Diet, and Medicine
        assert "Symptom Summary" in result
        assert "fever" in result.lower()
        assert "Hydration & Gastric Care Recovery Diet" in result
        assert "Medicine: Paracetamol" in result

    @patch("utils.gemini_client.GeminiClient.generate", side_effect=generate_mock_response)
    def test_scenario_2_shoulder_pain_fitness(self, mock_gen, agent, memory):
        """Scenario 2: 'I have shoulder pain and want gym advice' triggers Symptom -> Fitness & Diet."""
        result = agent.process("I have shoulder pain and want gym advice")
        
        # Should record symptoms
        assert "pain" in memory.profile.frequent_symptoms
        
        # Check coordinated care includes Symptom, Fitness, and Diet
        assert "Symptom Summary" in result
        assert "shoulder pain" in result.lower()
        assert "Hypertrophy progressive overload muscle conditioning routine" in result
        assert "Nutrition Overview" in result or "Goal Summary" in result

    @patch("utils.gemini_client.GeminiClient.generate", side_effect=generate_mock_response)
    def test_scenario_3_dandruff_bulking(self, mock_gen, agent, memory):
        """Scenario 3: 'I have dandruff and need a bulking diet' triggers Skin & Hair -> Diet & Fitness."""
        result = agent.process("I have dandruff and need a bulking diet")
        
        # Verify dandruff symptom recorded
        assert "dandruff" in memory.profile.frequent_symptoms
        
        # Check coordinated response contains skin flaking analysis + bulking caloric diet + exercise schedule
        assert "Skin/Hair Analysis" in result
        assert "dandruff" in result.lower()
        assert "Muscle Gain & Caloric Surplus Hypertrophy Plan" in result
        assert "Workout Plan" in result

    @patch("utils.gemini_client.GeminiClient.generate", side_effect=generate_mock_response)
    def test_scenario_4_obesity_knee_pain(self, mock_gen, agent, memory):
        """Scenario 4: 'I have obesity and knee pain' triggers joint-friendly low-impact Fitness and Recovery Diet."""
        result = agent.process("I have obesity and knee pain")
        
        assert "pain" in memory.profile.frequent_symptoms
        
        # Coordinated care contains Symptom summary, joint rehab fitness plan, and nutrition
        assert "Symptom Summary" in result
        assert "Joint-Friendly Low-Impact Stretches" in result
        assert "Hydration Advice" in result

    @patch("utils.gemini_client.GeminiClient.generate", side_effect=generate_mock_response)
    def test_scenario_5_memory_state_carryover(self, mock_gen, agent, memory):
        """Scenario 5: Storing weight in turn 1 carries over automatically to bulking diet in turn 2."""
        # Turn 1: Register weight
        agent.process("My weight is 62kg.")
        assert memory.profile.weight == 62.0

        # Turn 2: Generate bulking diet (should pull 62kg from memory automatically)
        result = agent.process("Create a muscle gain diet.")
        
        assert "Muscle Gain & Caloric Surplus Hypertrophy Plan" in result
        # Verify that the Diet Agent customized target calories using stored 62kg
        assert "62 kg" in result
        assert "124g protein" in result.lower()
