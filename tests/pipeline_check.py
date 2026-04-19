from detector.pipeline import run_pipeline

# REAL — correct case, correct proposition
real_text = """
In Hoskins v. Withers, 92 F.4th 1279 (10th Cir. 2024), the court held that
a trooper had reasonable suspicion to conduct a traffic stop where a vehicle's
license plate lettering was partially obstructed, and that Utah's license plate
maintenance law applied to out-of-state plates.
"""

# SUSPICIOUS — correct case, wrong proposition (Type B)
suspicious_text = """
In Hoskins v. Withers, 92 F.4th 1279 (10th Cir. 2024), the court held that
officers may conduct a warrantless search of a vehicle without reasonable
suspicion whenever a narcotics dog is present at the scene, regardless of
whether the initial stop was lawful.
"""

# HALLUCINATED — fake citation (Type A)
hallucinated_text = """
In United States v. Garrett, 887 F.3d 452 (9th Cir. 2021), the court held
that the plain view doctrine permits officers to seize any item in a vehicle
during a routine traffic stop without a warrant.
"""

for label, text in [("REAL", real_text), ("SUSPICIOUS", suspicious_text), ("HALLUCINATED", hallucinated_text)]:
    results = run_pipeline(text)
    for v in results:
        llm = v.llm_result
        print(f"\n[{label}]")
        print(f"  Verdict:  {v.verdict}")
        print(f"  L2b:      {llm.is_accurate if llm and not llm.skipped else 'skipped'}")
        print(f"  Reason:   {llm.reason if llm else 'n/a'}")