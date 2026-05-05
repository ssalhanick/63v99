## Example 1: The "Perfect" Real Citation (True Positive)
### What it shows
How the pipeline handles a completely valid, foundational case. 

### The Prompt:
`"An officer may perform a brief investigatory stop without a warrant if they possess a reasonable, articulable suspicion that criminal activity is afoot, as originally established by the Supreme Court in Terry v. Ohio, 392 U.S. 1 (1968)."`

### Expected Result: 🟢 REAL
### Demo Talking Points
Point out the high Graph Density. Mention that because it is a landmark case, it is heavily cited across the corpus. The ML Scorer will easily pass this.

## Example 2: Complete Fabrication (Type A Hallucination)
### What it shows
How the system detects cases that LLMs completely make up out of thin air.

### The Prompt:
`"The Supreme Court recently expanded the boundaries of digital privacy, ruling that law enforcement cannot deploy thermal-imaging drones over residential neighborhoods without a specialized warrant under the 'skyline doctrine' in Zephyr v. Cloudbase, 999 U.S. 123 (2024)."`

### Expected Result: 🔴 HALLUCINATED
### Demo Talking Point
Highlight that the Semantic Score is 0.000 and the Graph Density is 0. The ML model instantly recognizes this case has zero footprint in legal history.

## Example 3: The "Pizza" Proposition (Type B Hallucination)
### What it shows
The system's ability to catch an LLM lying about what a real case actually says.

### The Prompt:
`"While the Fourth Amendment generally protects against unreasonable searches, the court made a unique exception in State v. Lewis, 224 N.E.3d 57 (2023), explicitly holding that police officers are constitutionally permitted to eat pizza during a routine traffic stop."`

### Expected Result: 🟡 SUSPICIOUS
### Demo Talking Point
This is your "wow" moment. Show the audience that the ML model wants to pass it because the citation network density is high (20). However, the Hard LLM Override kicks in! Claude Haiku reads the retrieved text, realizes State v. Lewis says absolutely nothing about pizza, and forces a Suspicious verdict.

## Example 4: Metadata Corruption
### What it shows
Layer 4's exact-match metadata validation catching subtle LLM hallucinations (like hallucinating the wrong year).

### The Prompt:
`"The court thoroughly evaluated the validity of the warrantless search and the application of the exclusionary rule in State v. Flack, 2022 Ohio 3861 (1985)."`

### Expected Result: 🔴 HALLUCINATED
### Demo Talking Point
The case State v. Flack is real, and the citation is real. However, the LLM hallucinated the year as 1985 instead of 2022. The Neo4j Layer 4 metadata check catches the temporal mismatch and instantly rejects it before it even reaches the expensive LLM check.

## Example 5: Doctrinal Incoherence (Multi-Citation Hallucination)
### What it shows
The Phase 4 & 5 Cross-Citation and Doctrine Graph logic you just built.

### The Prompt:
`"To justify a warrantless search of a vehicle, officers routinely rely on the established automobile exception, as seen in State v. Medina, 2026 ND 45 (2026). However, this exception is completely overridden if the driver declares themselves a 'sovereign citizen', as cemented by the binding precedent of State v. Freeman, 777 N.E.2d 999 (2024)."`

### Expected Result:
State v. Medina: 🟢 REAL (Real case, real doctrines)
State v. Freeman: 🔴 HALLUCINATED (Fake case)

### Demo Talking Points
Look at the Doctrines and Cross-Citation metrics. Point out that because one case is real and the other is totally fake, they share absolutely no doctrinal nodes in the Neo4j graph and have no shortest-path connection, proving that the LLM is just stitching unrelated concepts together!