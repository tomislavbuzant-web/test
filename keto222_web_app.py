import uuid
from datetime import datetime, date
from typing import List, Optional, Dict
from enum import Enum
from pydantic import BaseModel, Field, validator
from fastapi import FastAPI, HTTPException

# ==========================================
# 1. DATA MODELS (SCHEMA)
# ==========================================

class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"

class KetosisLevel(str, Enum):
    DEEP = "deep"
    FUNCTIONAL = "functional"
    BORDERLINE = "borderline"
    OUT = "out"

class User(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str

class MetabolicProfile(BaseModel):
    user_id: str
    weight_kg: float
    height_cm: float
    age: int
    gender: Gender
    activity_level: float
    bmr: float
    tdee: float
    protein_floor: float
    carb_threshold_max: float
    fat_min: float

class KetosisState(BaseModel):
    user_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    blood_ketones_mmol: Optional[float] = None
    glucose_mmol: Optional[float] = None
    state: KetosisLevel
    confidence_score: int

class DailyNutritionPlan(BaseModel):
    user_id: str
    date: date
    calories: float
    fat_grams: float
    protein_grams: float
    carb_grams: float
    recommended_foods: List[str]
    excluded_foods: List[str]

class SupplementEntry(BaseModel):
    name: str
    dosage: str
    timing: str

class TrainingRecommendation(BaseModel):
    user_id: str
    type: str  # strength, cardio, recovery
    intensity: str
    note: str

class PlateauEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    start_date: datetime
    is_resolved: bool = False
    intervention: Optional[str] = None

# ==========================================
# 2. CORE RULE ENGINES
# ==========================================

class MetabolicEngine:
    @staticmethod
    def calculate(uid: str, w: float, h: float, a: int, g: Gender, act: float) -> MetabolicProfile:
        # BMR: Mifflin-St Jeor Equation (Metric)
        bmr = (10 * w) + (6.25 * h) - (5 * a)
        bmr = bmr + 5 if g == Gender.MALE else bmr - 161
        
        tdee = bmr * act
        
        # Keto Specific Logic
        return MetabolicProfile(
            user_id=uid, weight_kg=w, height_cm=h, age=a, gender=g,
            activity_level=act, bmr=round(bmr, 2), tdee=round(tdee, 2),
            protein_floor=round(w * 1.6, 1), # 1.6g/kg floor
            carb_threshold_max=30.0,         # Strict Keto limit
            fat_min=round((tdee * 0.65) / 9, 1)
        )

class KetosisEngine:
    @staticmethod
    def evaluate(blood: Optional[float], glucose: Optional[float]) -> Dict:
        # Rule: Device data (Blood Ketones) ALWAYS overrides subjective logic
        if blood is not None:
            conf = 100
            if blood >= 1.5: return {"state": KetosisLevel.DEEP, "conf": conf}
            if blood >= 0.5: return {"state": KetosisLevel.FUNCTIONAL, "conf": conf}
            return {"state": KetosisLevel.OUT, "conf": conf}
        
        # Rule: Glucose spikes downgrade state
        if glucose and glucose > 7.0: # 7.0 mmol/L threshold
            return {"state": KetosisLevel.OUT, "conf": 80}
            
        return {"state": KetosisLevel.BORDERLINE, "conf": 40}

class NutritionEngine:
    @staticmethod
    def generate(profile: MetabolicProfile, state: KetosisLevel, 
                 training: bool, stress: bool, plateau: bool) -> DailyNutritionPlan:
        # Initial Deficit: 20%
        calories = profile.tdee * 0.8
        protein = profile.protein_floor
        carbs = profile.carb_threshold_max
        
        # Rules Logic
        if training: protein += 25.0  # Protein increase for muscle sparing
        if stress: carbs = 20.0      # Tighten carbs under stress
        if plateau: calories *= 0.9   # Additional 10% fat reduction
        
        fat = (calories - (protein * 4) - (carbs * 4)) / 9
        
        return DailyNutritionPlan(
            user_id=profile.user_id,
            date=date.today(),
            calories=round(calories, 0),
            fat_grams=round(fat, 1),
            protein_grams=round(protein, 1),
            carb_grams=round(carbs, 1),
            recommended_foods=["Ribeye", "Eggs", "Macadamia Nuts", "Avocado"],
            excluded_foods=["All Grains", "Root Vegetables", "Sugar", "Seed Oils"]
        )

# ==========================================
# 3. API LAYER (REST ENDPOINTS)
# ==========================================

app = FastAPI(title="Keto Metabolic Architect API")

# Mock In-Memory DB
DB = {
    "profiles": {},
    "states": {},
    "plateaus": {}
}

@app.post("/metabolic-profile/calculate", response_model=MetabolicProfile)
def api_calc_profile(uid: str, w: float, h: float, a: int, g: Gender, act: float):
    profile = MetabolicEngine.calculate(uid, w, h, a, g, act)
    DB["profiles"][uid] = profile
    return profile

@app.post("/ketosis-state/update", response_model=KetosisState)
def api_update_state(uid: str, blood_ketones: float = None, glucose: float = None):
    res = KetosisEngine.evaluate(blood_ketones, glucose)
    state = KetosisState(
        user_id=uid, blood_ketones_mmol=blood_ketones, 
        glucose_mmol=glucose, state=res["state"], confidence_score=res["conf"]
    )
    DB["states"][uid] = state
    return state

@app.post("/nutrition-plan/generate", response_model=DailyNutritionPlan)
def api_gen_nutrition(uid: str, is_training: bool = False, is_stress: bool = False):
    profile = DB["profiles"].get(uid)
    state = DB["states"].get(uid)
    
    if not profile or not state:
        raise HTTPException(status_code=400, detail="Profile and State must be initialized.")
    
    is_plateau = DB["plateaus"].get(uid, False)
    return NutritionEngine.generate(profile, state.state, is_training, is_stress, is_plateau)

@app.get("/supplement-plan/{uid}")
def api_supplements(uid: str, is_training: bool = False):
    # Mandatory Electrolyte Rules
    plan = [
        {"name": "Sodium", "dosage": "5000mg", "timing": "Throughout day"},
        {"name": "Potassium", "dosage": "3000mg", "timing": "With meals"},
        {"name": "Magnesium", "dosage": "400mg", "timing": "Before bed"}
    ]
    if is_training:
        plan.append({"name": "Creatine", "dosage": "5g", "timing": "Post-workout"})
    return plan

@app.post("/plateau/check")
def api_check_plateau(uid: str, weights: List[float]):
    # Rule: No change > 0.2kg over 7 entries
    if len(weights) < 7: return {"plateau": False}
    recent = weights[-7:]
    variance = max(recent) - min(recent)
    is_plateau = variance <= 0.2
    DB["plateaus"][uid] = is_plateau
    return {"plateau": is_plateau, "variance": round(variance, 3)}

# ==========================================
# 4. AI EXPLANATION LAYER
# ==========================================

@app.get("/explain/{uid}")
def api_explain(uid: str):
    # Rule: AI only explains, never decides
    state = DB["states"].get(uid)
    if not state: return "No data."
    
    if state.state == KetosisLevel.OUT:
        return "The Rule Engine flagged you as 'OUT' of ketosis because your glucose exceeded 7.0 mmol/L, triggering a metabolic reset."
    return f"The Rule Engine confirmed {state.state} ketosis based on current blood ketone markers."