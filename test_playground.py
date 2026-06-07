"""
test_playground.py
──────────────────
Offline Interactive Sandbox Playground for HealthMate AI 2.0.
Simulates high-fidelity collaborative multi-agent A2A execution,Shared Memory,
and dynamic handoff logic without requiring real Gemini API keys.
"""

import sys
from pathlib import Path
import re

# Add project root to system path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from unittest.mock import patch

from memory.memory_manager import MemoryManager
from agents.reception_agent import ReceptionAgent

console = Console()

SYNONYMS = {
    "fever": ["fever", "temperature", "chills"],
    "cough": ["cough", "coughing"],
    "headache": ["headache", "migraine"],
    "dandruff": ["dandruff", "flaky scalp"],
    "hair_loss": ["hair loss", "hairfall", "baldness"],
    "acne": ["acne", "pimples"]
}

# ── Dynamic Simulated Specialist Profiles Database ──
MOCK_SYMPTOMS = {
    "fever": {
        "name": "fever",
        "causes": ["Immune response fighting off viral infection", "Bacterial pharyngitis", "Mild inflammatory response"],
        "precautions": ["Stay fully hydrated with electrolyte fluids.", "Stay in a cool environment and rest.", "Apply a damp cooling cloth to your forehead."],
        "followups": ["What is your temperature reading?", "Do you have chills or body aches?"],
        "warnings": ["Fever exceeding 103°F (39.4°C).", "Stiff neck, confusion, or breathing blocks."]
    },
    "cough": {
        "name": "cough",
        "causes": ["Viral respiratory irritation", "Post-nasal drip", "Environmental allergen exposure"],
        "precautions": ["Drink warm water with honey to soothe throat linings.", "Use a room humidifier.", "Avoid cold air and smoke exposure."],
        "followups": ["Is the cough dry or productive?", "How long have you been coughing?"],
        "warnings": ["Coughing up blood.", "Associated high fever, chest tightness, or severe wheezing."]
    },
    "headache": {
        "name": "headache",
        "causes": ["Stress or muscular tension", "Dehydration", "Lack of restful sleep"],
        "precautions": ["Rest in a quiet, dark room.", "Hydrate immediately with water.", "Apply a cool compress to your forehead."],
        "followups": ["Is the pain throbbing or constant?", "Does light or noise make it worse?"],
        "warnings": ["Sudden, severe 'thunderclap' headache.", "Headache with fever, stiff neck, or confusion."]
    },
    "general": {
        "name": "general discomfort",
        "causes": ["Physical fatigue", "Elevated daily stress", "Mild dehydration"],
        "precautions": ["Ensure 7-8 hours of quality sleep.", "Hydrate well and consume nourishing foods.", "Practice relaxation techniques."],
        "followups": ["Can you describe the location of your discomfort?", "How long have you been feeling unwell?"],
        "warnings": ["Unexplained worsening of symptoms.", "Persistent high temperature or severe acute pain."]
    },
    "pain": {
        "name": "shoulder pain",
        "causes": ["Rotator cuff strain or tendinitis", "Shoulder joint impingement", "Mild muscle spasm"],
        "precautions": ["Rest the shoulder and avoid heavy overhead lifting.", "Apply ice packs for 15-20 minutes.", "Perform gentle range-of-motion arm swings."],
        "followups": ["Is the pain sharp or a dull ache?", "Does moving your arm trigger the pain?"],
        "warnings": ["Inability to move the shoulder.", "Severe sudden pain spreading down the arm or to chest."]
    }
}

MOCK_SKIN_HAIR = {
    "dandruff": {
        "name": "dandruff",
        "analysis": "Scalp flaking linked to seborrheic dermatitis or Malassezia yeast overgrowth.",
        "routine": "Wash scalp with over-the-counter anti-dandruff shampoo containing zinc pyrithione or ketoconazole twice a week. Massage gently, leave on for 3-5 minutes, and rinse thoroughly.",
        "nutrition": "Incorporate zinc, B-vitamins, and healthy omega-3 fatty acids into your meals.",
        "warnings": "Severe scalp redness, oozing, scaling spreading to face, or patchy hair shedding."
    },
    "hair_loss": {
        "name": "hair loss",
        "analysis": "Hair shedding related to elevated cortisol stress levels (telogen effluvium) or genetic pattern thinning.",
        "routine": "Avoid tight hairstyles, high-heat styling irons, and harsh chemical dyes. Use a wide-toothed comb and wash with a gentle sulfate-free shampoo.",
        "nutrition": "Prioritize high-biotin foods, lean proteins, iron, and zinc.",
        "warnings": "Sudden, rapid patchy baldness or loss of body hair."
    },
    "acne": {
        "name": "acne",
        "analysis": "Skin pore blockage associated with excessive sebum oil production and bacterial sensitivity.",
        "routine": "Wash your face twice daily with a salicylic acid or benzoyl peroxide cleanser. Apply a non-comedogenic oil-free moisturizer and sunscreen daily.",
        "nutrition": "Avoid high-glycemic sweet treats and heavily processed foods.",
        "warnings": "Severe cystic lesions, painful deep nodules, or scarring."
    },
    "general": {
        "name": "general care",
        "analysis": "Standard skin or scalp hydration imbalances.",
        "routine": "Cleanse daily, apply moisturizer suitable for your skin type, and use SPF 30+ sunscreen.",
        "nutrition": "Stay fully hydrated and consume antioxidant-rich vegetables.",
        "warnings": "Rapidly spreading rashes, severe itching, or blistering."
    }
}

def generate_mock_response(system_prompt: str, user_message: str) -> str:
    """Simulates high-fidelity cooperative response formatting based on active agent role."""
    
    # Isolate checking strictly to the YOUR ROLE section of the prompt to avoid history contamination
    role_section = system_prompt.split("=== CLINICAL SAFETY CONSTRAINTS")[0].lower() if "=== CLINICAL SAFETY CONSTRAINTS" in system_prompt else system_prompt.lower()
    
    agent_id = "reception_agent"
    if "emergency detection" in role_section or "emergency_agent" in role_section:
        agent_id = "emergency_agent"
    elif "symptom triage" in role_section or "symptom_agent" in role_section:
        agent_id = "symptom_agent"
    elif "skin & hair" in role_section or "skin_hair_agent" in role_section:
        agent_id = "skin_hair_agent"
    elif "diet & nutrition" in role_section or "diet_agent" in role_section:
        agent_id = "diet_agent"
    elif "fitness & wellness" in role_section or "fitness_agent" in role_section:
        agent_id = "fitness_agent"
    elif "medicine information" in role_section or "medicine_agent" in role_section:
        agent_id = "medicine_agent"
    elif "memory management" in role_section or "memory_agent" in role_section:
        agent_id = "memory_agent"
    elif "reception" in role_section or "reception_agent" in role_section:
        agent_id = "reception_agent"

    lower_msg = user_message.lower()

    # Parse profile factors from system prompt variables
    profile_weight = None
    profile_height = None
    profile_history = []
    profile_goals = []

    w_match = re.search(r"\bWeight:\s*(\d+(?:\.\d+)?)\s*kg\b", system_prompt)
    if w_match: profile_weight = float(w_match.group(1))

    h_match = re.search(r"\bHeight:\s*(\d+(?:\.\d+)?)\s*cm\b", system_prompt)
    if h_match: profile_height = float(h_match.group(1))

    cond_match = re.search(r"\bMedical conditions:\s*([^\n;]+)\b", system_prompt)
    if cond_match:
        profile_history = [cond.strip().lower() for cond in cond_match.group(1).split(",")]

    goals_match = re.search(r"\bFitness goals:\s*([^\n;]+)\b", system_prompt)
    if goals_match:
        profile_goals = [g.strip().lower() for g in goals_match.group(1).split(",")]

    # ── [1] Emergency Agent ──
    if agent_id == "emergency_agent":
        detected = "Potential Emergency"
        for label, pattern in [("Chest Pain / Cardiac arrest", "chest"), ("Breathing Obstruction", "breath"), ("Stroke Symptoms", "stroke")]:
            if pattern in lower_msg:
                detected = label
                break
        return f"""
🚨 **EMERGENCY DETECTED**: {detected}

⚡ **CALL 112 / 911 NOW — Do not wait.**

What to do right now:
• Stop all physical movement immediately and sit in a comfortable, upright position.
• Loosen any tight clothing around your neck and waist to clear breathing.
• Chew a standard aspirin if experiencing chest tightness (if not allergic).

Stay calm. Help is on the way.
""".strip()

    # ── [2] Symptom Agent ──
    elif agent_id == "symptom_agent":
        sym_key = "general"
        for k in ["fever", "cough", "headache", "pain", "shoulder", "knee"]:
            if k in lower_msg:
                sym_key = "pain" if k in ["pain", "shoulder", "knee"] else k
                break
        profile = MOCK_SYMPTOMS[sym_key]
        
        causes_str = "\n".join(f"• {c}" for c in profile["causes"])
        prec_str = "\n".join(f"• {p}" for p in profile["precautions"])
        fups_str = "\n".join(f"• {f}" for f in profile["followups"])
        warn_str = "\n".join(f"• {w}" for w in profile["warnings"])

        return f"""
📋 **Symptom Summary**: User is reporting symptoms of {profile['name']}.

🔍 **Possible Causes**:
{causes_str}

💡 **Precautions**:
{prec_str}

❓ **Follow-up Questions**:
{fups_str}

⚠️ **Warning Signs**:
{warn_str}
""".strip()

    # ── [3] Skin & Hair Agent ──
    elif agent_id == "skin_hair_agent":
        sk_key = "general"
        for k in ["dandruff", "hair loss", "hairfall", "acne", "pimple"]:
            if k in lower_msg:
                sk_key = "hair_loss" if "hair" in k else ("acne" if "acne" in k or "pimple" in k else k)
                break
        profile = MOCK_SKIN_HAIR[sk_key]
        
        return f"""
🧴 **Skin/Hair Analysis**: User is reporting symptoms of {profile['name']}. {profile['analysis']}

💡 **Care Routine**:
• {profile['routine']}

🥗 **Nutrition Suggestions**:
• {profile['nutrition']}

⚠️ **Warning Signs**:
• {profile['warnings']}
""".strip()

    # ── [4] Diet Agent ──
    elif agent_id == "diet_agent":
        is_weight_loss = "lose" in lower_msg or "loss" in lower_msg or "weight loss" in [g.lower() for g in profile_goals]
        is_muscle_gain = "gain" in lower_msg or "bulking" in lower_msg or "muscle gain" in [g.lower() for g in profile_goals]
        
        # Check shared memory observations to customize for fever/dandruff
        fever_obs = "fever" in system_prompt.lower()
        dandruff_obs = "dandruff" in system_prompt.lower()

        if fever_obs:
            summary = "Hydration & Gastric Care Recovery Diet"
            macros = "• **Caloric Intake**: Safe metabolic maintenance levels.\n• **Protein**: 1.0g per kg of body weight.\n• **Carbs/Fats**: Easily digestible simple carbohydrates and lower fats."
            meals = """• **Breakfast**: Clear vegetable or chicken broth with soft dry toast.
• **Lunch**: White rice porridge (congee) or mashed sweet potato.
• **Dinner**: Steamed carrots and soft baked tofu or chicken breast.
• **Snacks**: Electrolyte fluids, coconut water, or warm chamomile tea."""
            hydration = f"Urgent: Aim for at least {round(profile_weight * 0.04, 1) if profile_weight else 3.2} liters of water/electrolytes daily to prevent fever dehydration."
        elif is_weight_loss:
            summary = "Fat Loss & Caloric Deficit Plan"
            macros = f"• **Caloric Deficit**: Target surplus deficit based on {int(profile_weight) if profile_weight else 70} kg.\n• **Protein**: 1.8g per kg.\n• **Carbs/Fats**: Low-glycemic complex carbs and light unsaturated fats."
            meals = """• **Breakfast**: Egg-white spinach omelette with whole-wheat toast.
• **Lunch**: Grilled chicken breast salad with oil dressing.
• **Dinner**: Baked salmon or cod with roasted asparagus.
• **Snacks**: Hummus with carrot sticks or organic berries."""
            hydration = f"Aim for at least {round(profile_weight * 0.035, 1) if profile_weight else 2.5} liters of water daily to support metabolic clearance."
        elif is_muscle_gain:
            summary = "Muscle Gain & Caloric Surplus Hypertrophy Plan"
            macros = f"• **Caloric Surplus**: Target surplus calorie loading for {int(profile_weight) if profile_weight else 62} kg.\n• **Protein**: 2.0g per kg of body weight (approx. {int(profile_weight * 2.0) if profile_weight else 125}g protein).\n• **Carbs/Fats**: Whole grains, oatmeal, brown rice, peanut butter, and avocados."
            meals = """• **Breakfast**: Oatmeal cooked in whole milk with banana slices and 2 tbsp peanut butter.
• **Lunch**: Lean ground turkey or beef sautéed with green peppers, served over brown rice.
• **Dinner**: Grilled salmon fillet with sweet potato mash and broccoli.
• **Snacks**: Greek yogurt with honey and almonds, or a high-protein shake."""
            hydration = f"Aim for at least {round(profile_weight * 0.045, 1) if profile_weight else 3.0} liters of water daily to hydrate muscle cells."
        elif dandruff_obs:
            summary = "Anti-Inflammatory Scalp-Nourishing Diet"
            macros = "• Focus: Zinc, Biotin, and Omega-3 fatty acids to soothe scalp flaking."
            meals = """• **Breakfast**: Whole-grain toast with avocado and flaxseeds.
• **Lunch**: Spinach and kale salad topped with hard-boiled eggs and chia seed dressing.
• **Dinner**: Baked mackerel or salmon serving with sweet potato.
• **Snacks**: Walnuts, pumpkin seeds, or green tea."""
            hydration = "Aim for 2.5 liters of water daily to maintain scalp skin hydration."
        else:
            summary = "General Balanced Nutrition Guidelines"
            macros = "• Target: Balanced maintenance calories ensuring adequate macro and micro density."
            meals = """• **Breakfast**: Whole-grain oats with sliced berries.
• **Lunch**: Mixed greens salad with chickpeas and cucumbers.
• **Dinner**: Baked tofu with sweet potato and steamed broccoli.
• **Snacks**: Hummus or a small apple with almonds."""
            hydration = "Aim for 2.5 to 3.0 liters of pure water daily."

        return f"""
🥗 **Goal Summary**: {summary}

📊 **Nutrition Targets**:
{macros}

🍳 **Meal Suggestions**:
{meals}

💧 **Hydration Advice**:
{hydration}
""".strip()

    # ── [5] Fitness Agent ──
    elif agent_id == "fitness_agent":
        is_weight_loss = "lose" in lower_msg or "loss" in lower_msg or "weight loss" in [g.lower() for g in profile_goals]
        is_muscle_gain = "gain" in lower_msg or "muscle" in lower_msg or "muscle gain" in [g.lower() for g in profile_goals] or "gym" in lower_msg
        has_asthma = "asthma" in profile_history or "asthma" in system_prompt.lower()
        has_joint_pain = ("pain" in system_prompt.lower() or "shoulder" in system_prompt.lower() or "knee" in system_prompt.lower()) and "gym" not in lower_msg

        if has_asthma:
            workout = "Low-Impact Cardio & Mobility routine optimized for respiratory safety"
            schedule = """• Day 1: Low-intensity indoor walking (20 mins) + mobility stretches.
• Day 2: Gentle yoga or floor flexibility stretching.
• Day 3: Low-resistance bodyweight squats (3 sets of 8) + active recovery breathing exercises."""
            recovery = "⚠️ **SAFETY WARNING**: Asthma detected in history. Avoid heavy cardio. Always keep rescue inhaler nearby. Rest fully between sets."
        elif has_joint_pain:
            workout = "Joint-Friendly Low-Impact Stretches & Rehabilitation Routine"
            schedule = """• Day 1: Floor core bridges and wall push-ups (3 sets of 8 controlled reps).
• Day 2: Dynamic foam rolling and gentle range-of-motion arm/leg circles.
• Day 3: Low-impact walking (15 mins) + targeted static stretching for sore joint areas."""
            recovery = "⚠️ **SAFETY WARNING**: Muscle or joint pain detected. Rest fully. Apply cool packs to sore areas. Skip high-impact circuits."
        elif is_weight_loss:
            workout = "Fat Loss High-Intensity Circuit Training"
            schedule = """• Day 1: HIIT circuit (Jumping jacks, mountain climbers) - 20 mins.
• Day 2: Full Body Resistance Workout (Bodyweight squats, push-ups, planks).
• Day 3: Low-Intensity Steady State Cardio (Uphill walking or cycling) - 40 mins."""
            recovery = "Focus on 7-8 hours of sleep. Stay fully hydrated before and during intense interval work."
        elif is_muscle_gain:
            workout = "Hypertrophy progressive overload muscle conditioning routine"
            schedule = """• Day 1: Upper Body Strength Split (Push/Pull compound presses) - 45 mins.
• Day 2: Lower Body Strength Split (Squats, lunges, hinges) - 45 mins.
• Day 3: Floor core work + active recovery mobility stretching."""
            recovery = f"Caloric surplus fuels strength. Take 60-90 seconds rest between lift sets. Aim for {int(profile_weight * 2.0) if profile_weight else 125}g protein intake daily."
        else:
            workout = "General Fitness Cardiovascular & Mobility Training"
            schedule = """• Day 1: Functional full-body routine (Squats, push-ups, planks) - 30 mins.
• Day 2: Low-impact brisk walk (30 mins).
• Day 3: Active recovery and static flexibility stretching."""
            recovery = "Prioritize consistency. Aim for 7-8 hours of restful sleep."

        return f"""
🏋️ **Workout Plan**: {workout}

📅 **Weekly Schedule**:
{schedule}

😴 **Recovery Advice**:
{recovery}
""".strip()

    # ── [6] Medicine Agent ──
    elif agent_id == "medicine_agent":
        drug = "Your Medication"
        for d in ["paracetamol", "ibuprofen", "aspirin", "amoxicillin"]:
            if d in lower_msg or d in system_prompt.lower():
                drug = d.capitalize()
                break
        if drug == "Your Medication" and ("fever" in lower_msg or "fever" in system_prompt.lower()):
            drug = "Paracetamol"
        
        if drug == "Paracetamol":
            used = "Fever reduction and mild-to-moderate pain relief (headaches, muscular aches)."
            side = "Liver strain/toxicity in extremely high doses."
            prec = """• NEVER exceed 4g per day for adults.
• Avoid combining with other paracetamol products to prevent accidental duplicate toxicity."""
            inter = "Anticoagulants (like Warfarin) when taken in high doses over long periods."
        elif drug == "Ibuprofen":
            used = "Reduction of inflammatory joint pain, swelling, and muscle stiffness (NSAID)."
            side = "Stomach irritation, indigestion, or nausea."
            prec = """• Take with food or milk to minimize stomach upset.
• Avoid if you have active stomach ulcers, severe heart failure, or active renal impairment."""
            inter = "Blood pressure medications, blood thinners, and other NSAIDs."
        else:
            used = "General symptom relief under physician guidance."
            side = "Consult product packaging guidelines."
            prec = """• Take strictly as indicated on the packaging.
• Avoid alcohol if cautioned."""
            inter = "Active pharmaceuticals. Consult a pharmacist."

        return f"""
💊 **Medicine**: {drug}
(Medicine: {drug})

📌 **Used For**: {used}

⚠️ **Common Side Effects**: {side}

🚫 **Precautions**:
{prec}

🔗 **Possible Interactions**: {inter}

👨‍⚕️ **Important**: Always consult your doctor or pharmacist before use.
""".strip()

    # ── [7] Memory Agent ──
    elif agent_id == "memory_agent":
        profile = MemoryManager("playground_tester").profile
        return f"""
Here is what HealthMate AI remembers about you:

📋 **Current Persistent Facts**:
• Weight: {profile.weight or '(not set)'} kg
• Height: {profile.height or '(not set)'} cm
• Goals: {', '.join(profile.fitness_goals) or '(not set)'}
• Medical History: {', '.join(profile.conditions) or '(not set)'}
""".strip()

    return "Reception guidance established. How can we help you coordinate care today?"

def run_playground():
    memory = MemoryManager(user_id="playground_tester")
    memory.clear_session()
    reception = ReceptionAgent(memory)

    console.print(Panel.fit(
        "[bold cyan]🏥 HealthMate AI 2.0 — Interactive Multi-Agent Playground[/bold cyan]\n"
        "[dim]Test cooperative A2A routing, shared memory, and handoffs offline.[/dim]\n"
        "[yellow]Type 'exit' to quit. Type 'profile' to print memory facts.[/yellow]",
        border_style="cyan"
    ))

    with patch("utils.gemini_client.GeminiClient.generate", side_effect=generate_mock_response):
        while True:
            try:
                user_input = console.input("\n[bold green]You:[/bold green] ").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if not user_input:
                continue

            lower = user_input.lower()
            if lower in {"exit", "quit"}:
                console.print("[yellow]Exiting playground. Stay healthy! 👋[/yellow]")
                break

            if lower == "profile":
                profile = memory.profile
                console.print(Panel(
                    f"[bold]Name[/bold]    : {profile.name or '(not set)'}\n"
                    f"[bold]Age[/bold]     : {profile.age or '(not set)'}\n"
                    f"[bold]Gender[/bold]  : {profile.gender or '(not set)'}\n"
                    f"[bold]Weight[/bold]  : {profile.weight or '(not set)'} kg\n"
                    f"[bold]Height[/bold]  : {profile.height or '(not set)'} cm\n"
                    f"[bold]Goals[/bold]   : {', '.join(profile.fitness_goals) or '(none)'}\n"
                    f"[bold]History[/bold] : {', '.join(profile.conditions) or '(none)'}\n"
                    f"[bold]Symptoms[/bold]: {', '.join(profile.frequent_symptoms) or '(none)'}",
                    title="📋 Central Shared Memory Profile",
                    border_style="yellow"
                ))
                continue

            try:
                response = reception.process(user_input)
                console.print("\n[bold blue]HealthMate AI Response:[/bold blue]")
                console.print(Markdown(response))
            except Exception as e:
                console.print(f"[bold red]❌ Error: {e}[/bold red]")

if __name__ == "__main__":
    run_playground()
