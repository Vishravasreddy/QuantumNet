from PIL import Image
import numpy as np
from pipeline import QuantumNetPipeline

pipeline = QuantumNetPipeline()

test_cases = [
    ("samples/aptos_test0_ae2c3f6312ef_nondr.png", "Non-DR (Patient #0)", 0.2201),
    ("samples/aptos_test1_44e951e45dca_dr.png", "Diabetic Retinopathy (Patient #1)", 0.9636),
    ("samples/aptos_test2_ef8c39eb9157_nondr.png", "Non-DR (Patient #2)", 0.0255),
    ("samples/aptos_test6_e499434242cc_dr.png", "Diabetic Retinopathy (Patient #6)", 0.9861),
]

print("\n" + "="*60)
print("QUANTUMNET END-TO-END PIPELINE VERIFICATION")
print("="*60)

for path, desc, expected_prob in test_cases:
    img = Image.open(path)
    res = pipeline.predict(img)
    print(f"\nTest Target: {desc}")
    print(f"  File: {path}")
    print(f"  Diagnosis: {res['diagnosis']}")
    print(f"  P(DR): {res['dr_probability']:.4f} (Expected: {expected_prob:.4f})")
    print(f"  Risk Tier: {res['risk_tier']}")
    print(f"  Circuit Pauli-Z Expectations: {np.round(res['z_expectations'], 4)}")
    print(f"  Total Latency: {res['timings']['t_total']*1000:.1f}ms (ResNet: {res['timings']['t_resnet']*1000:.1f}ms, VQC: {res['timings']['t_vqc']*1000:.1f}ms)")
    assert abs(res['dr_probability'] - expected_prob) < 0.001, f"Mismatch on {desc}!"

print("\n" + "="*60)
print("ALL TEST CASES PASSED WITH 100% ACCURACY!")
print("="*60)
