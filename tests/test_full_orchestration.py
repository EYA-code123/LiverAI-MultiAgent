import os
import sys

from PIL import Image


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = (
    "/content/LiverAI-MultiAgent"
)

if PROJECT_ROOT not in sys.path:

    sys.path.insert(
        0,
        PROJECT_ROOT
    )

os.chdir(
    PROJECT_ROOT
)


# ============================================================
# IMPORT
# ============================================================

from orchestrator.liver_orchestrator import (
    LiverAIOrchestrator
)


# ============================================================
# INPUTS
# ============================================================

clinical_data = {

    "mcv": 85.0,

    "alkphos": 85.0,

    "sgpt": 45.0,

    "sgot": 35.0,

    "gammagt": 50.0,

    "drinks": 5.0,
}


fibrosis_data = {

    "age": 50,

    "male": 1,

    "weight": 75,

    "height": 170,

    "bmi": 25.95,

    "futime": 365,

    "days": 365,

    "test": 1,

    "value": 50,
}


cirrhosis_data = {

    "N_Days": 400,

    "Status": "C",

    "Drug": "D-penicillamine",

    "Age": 50,

    "Sex": "M",

    "Ascites": "N",

    "Hepatomegaly": "Y",

    "Spiders": "N",

    "Edema": "N",

    "Bilirubin": 1.5,

    "Cholesterol": 200,

    "Albumin": 3.5,

    "Copper": 100,

    "Alk_Phos": 1500,

    "SGOT": 100,

    "Tryglicerides": 100,

    "Platelets": 200,

    "Prothrombin": 10,
}


tumor_path = (
    "/content/drive/MyDrive/"
    "Tumor/Liver_Dataset/"
    "Healthy/"
    "Healthy_1401.jpg"
)


segmentation_path = (
    "/content/drive/MyDrive/"
    "archive (2)/image/"
    "liver_0_img.npy"
)


tumor_image = (
    Image.open(
        tumor_path
    ).convert("RGB")
)


# ============================================================
# PATIENT DATA
# ============================================================

patient_data = {

    "fatty_liver":
        clinical_data,

    "fibrosis":
        fibrosis_data,

    "cirrhosis":
        cirrhosis_data,

    "tumor_classification":
        tumor_image,

    "liver_segmentation":
        segmentation_path,

    "clinical_reasoning":
        clinical_data,
}


# ============================================================
# INITIALIZE ORCHESTRATOR
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "LIVER AI MULTI-AGENT"
)

print(
    "FULL ORCHESTRATION TEST"
)

print(
    "=" * 80
)


orchestrator = (
    LiverAIOrchestrator()
)


# ============================================================
# RUN
# ============================================================

result = orchestrator.run(

    patient_id=
        "TEST_001",

    patient_data=
        patient_data,
)


# ============================================================
# GLOBAL STATUS
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "FINAL RESULT"
)

print(
    "=" * 80
)

print(
    "Status:",
    result.get(
        "status"
    )
)

print(
    "Patient:",
    result.get(
        "patient_id"
    )
)


# ============================================================
# COORDINATION
# ============================================================

coordination = (
    result.get(
        "coordination",
        {}
    )
)

print(
    "\nCOORDINATION"
)

print(
    "Total agents:",
    coordination.get(
        "total_agents"
    )
)

print(
    "Successful:",
    coordination.get(
        "successful_agents"
    )
)

print(
    "Failed:",
    coordination.get(
        "failed_agents"
    )
)

print(
    "Not run:",
    coordination.get(
        "not_run_agents"
    )
)

print(
    "Coverage:",
    coordination.get(
        "coverage"
    )
)

print(
    "Latency:",
    coordination.get(
        "latency_ms"
    ),
    "ms"
)


# ============================================================
# SPECIALIZED AGENTS
# ============================================================

print(
    "\nSPECIALIZED RESULTS"
)

for name, agent_result in (
    result.get(
        "specialized_results",
        {}
    ).items()
):

    print(
        f"\n--- {name} ---"
    )

    print(
        "Status:",
        agent_result.get(
            "status"
        )
    )

    print(
        "Prediction:",
        agent_result.get(
            "prediction"
        )
    )

    print(
        "Confidence:",
        agent_result.get(
            "confidence"
        )
    )

    print(
        "Trust:",
        agent_result.get(
            "trust"
        )
    )


# ============================================================
# CLINICAL REASONING
# ============================================================

clinical = result.get(
    "clinical_reasoning",
    {}
)

print(
    "\nCLINICAL REASONING"
)

print(
    "Status:",
    clinical.get(
        "status"
    )
)

print(
    "Prediction:",
    clinical.get(
        "prediction"
    )
)

print(
    "Confidence:",
    clinical.get(
        "confidence"
    )
)


# ============================================================
# FUSION
# ============================================================

print(
    "\nADAPTIVE FUSION"
)

print(
    result.get(
        "fusion"
    )
)


# ============================================================
# CONFLICTS
# ============================================================

print(
    "\nCONFLICTS"
)

print(
    result.get(
        "conflicts"
    )
)


# ============================================================
# DECISION
# ============================================================

print(
    "\nDECISION ENGINE"
)

print(
    result.get(
        "decision"
    )
)


# ============================================================
# FINAL DECISION
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "FINAL SYSTEM ASSESSMENT"
)

print(
    "=" * 80
)

print(
    result.get(
        "final_decision"
    )
)

print(
    "=" * 80
)
