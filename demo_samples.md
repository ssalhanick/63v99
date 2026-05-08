## Demo Sample Set (Updated)

These are tuned for the current pipeline behavior:

- no circuit-aware retrieval filter,
- scorer probability shown as `P(Hall)`,
- optional feature-makeup toggle in the UI.

---

## Example 1: Clean Non-Landmark Real Citation

### What it shows

A real, non-landmark case that should resolve and pass.

### Prompt

`"Squatters generally lack a legitimate expectation of privacy in unlawfully occupied land, as discussed in Amezquita v. Hernandez-Colon, 518 F.2d 8 (1st Cir. 1975)."`

### Expected Result

🟢 **REAL**

### Demo Talking Points

- Good baseline "known-good" citation.
- `P(Hall)` should be below the suspicious band.
- Useful contrast against fabricated and proposition-hallucinated examples.

---

## Example 2: Complete Fabrication (Type A)

### What it shows

Layer 1 existence failure for a made-up citation.

### Prompt

`"The Supreme Court expanded digital privacy under the skyline doctrine in Zephyr v. Cloudbase, 999 U.S. 123 (2024), requiring specialized drone warrants."`

### Expected Result

🔴 **HALLUCINATED**

### Demo Talking Points

- Case should not resolve to a valid graph node.
- Fast fail path demonstrates cheap hard-gate protection.

---

## Example 3: Real Case + Fake Proposition (Type B-style behavior)

### What it shows

A plausible-looking citation with an obviously false legal proposition.

### Prompt

`"State v. Lewis, 224 N.E.3d 57 (2023), established that officers are constitutionally permitted to eat pizza during a routine traffic stop."`

### Expected Result

🟡 **SUSPICIOUS** (or 🔴 if your scorer pushes high enough)

### Demo Talking Points

- Use this for the LLM proposition-check story.
- Feature-makeup view helps show why final class landed where it did.

---

## Example 4: Metadata Corruption

### What it shows

Layer 4 year/court mismatch short-circuit.

### Prompt

`"The court evaluated a warrantless search in State v. Flack, 2022 Ohio 3861 (1985)."`

### Expected Result

🔴 **HALLUCINATED**

### Demo Talking Points

- Real case reference with corrupted year.
- Demonstrates deterministic metadata gate before expensive inference.

---

## Example 6:

### Prompt

```
Once we do that, it becomes irrelevant that Smith was recently a passenger in the Malibu. While the Supreme Court has held that passengers in cars are seized during traffic stops, see Brendlin v. California, 551 U.S. 249, 251 (2007), it has not extended that holding to former passengers who have since exited the vehicle. Brendlin’s holding rests in part on the recognition that once a police officer stops a car, “a sensible person would not expect a police officer to allow people to come and go freely..."
```

### Expected Result: REAL
