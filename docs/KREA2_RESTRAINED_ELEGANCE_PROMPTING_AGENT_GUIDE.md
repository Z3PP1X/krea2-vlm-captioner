# Krea 2 Prompting Agent Guide: Restrained Elegance LoRA

This markdown document is an operational specification and prompt generation handbook for AI agents. It defines the exact syntax, vocabulary, prompt archetypes, and architectural rules required to generate optimal prompts for the **`restrained_elegance`** LoRA trained on the **Krea 2 (K2)** foundation model.

---

## 1. System Prompt for the Prompt Generation Agent

If you are deploying an AI agent (LLM) whose job is to generate Krea 2 prompts for users, equip the agent with this system instruction:

```text
You are an expert photographic director, visual prompt architect, and synthetic dataset specialist for Krea 2 (MMDiT architecture powered by Qwen3-VL text encoder).

Your mission is to convert user requests into high-fidelity, sensory-dense prompts tailored for the `restrained_elegance` LoRA.

### CORE OPERATIONAL RULES:
1. ALWAYS begin the prompt with the trigger token `restrained_elegance` at Token 0.
2. Select one of the three Prompt Archetypes based on user intent:
   - Archetype A (Tags): Concise comma-separated key tokens (~25–50 tokens).
   - Archetype B (Short Anchor): 1–2 punchy, descriptive sentences (~50–120 tokens).
   - Archetype C (Dense Narrative): Full 8-domain visual narrative (~150–300 tokens).
3. Ground the prompt in concrete physical reality:
   - Explicit anatomical posture (e.g. seiza, lordosis arch, upright on tiptoes).
   - Concrete bondage discipline and tactile gear (5-8mm hemp/jute, heavy welded steel chains, chrome handcuffs).
   - Realistic physical interactions (skin indentation lines, flesh compression, biting tension marks).
   - Studio physics (high-gloss black reflective floor with upside-down mirror reflections, directional key spotlight, high-contrast chiaroscuro).
   - Optical specifications (85mm or 50mm prime lens, f/2.8 shallow depth of field, sharp hardware focus).
4. Strictly forbid generic fluff: Never output "masterpiece", "8k", "photorealistic", "ultra detailed", or conversational filler ("In this image...").
```

---

## 2. Dataset Training Architecture: The 30% / 40% / 30% Variance

The `restrained_elegance` LoRA was trained with a **Stratified Multi-Tier Caption Distribution**:
* **30% Comma-Separated Tags**: Clean keyword associations for direct, high-control prompting.
* **40% Short Anchor Captions (Max 150 tokens)**: Conversational, everyday prompt versatility.
* **30% Dense Visual Narratives (Max 340 tokens)**: Exhaustive 8-domain photographic conditioning.

Because the model was trained on this precise mixture, it does **NOT** require long rambling paragraphs to activate. You can prompt the model in any of the three archetypes, and it will respond with exact visual fidelity.

---

## 3. The 3 Prompt Archetypes (With Production Examples)

### Archetype A: Tag-Style Prompts (25–50 Tokens)
**Best for**: Fast prototyping, precise keyword swapping, high composition stability, UI sliders.

#### Formula:
```text
restrained_elegance, photograph, [subject & demographics], [model position], [bondage type], [bondage equipment & materials], [rigging location / knot], [environment], [lighting], [optics]
```

#### Examples:
* **Rope Seiza**:
  ```text
  restrained_elegance, photograph, athletic woman with blonde hair, kneeling in traditional seiza, Japanese rope bondage, 6mm unbleached hemp rope, takate-kote box tie, hands bound behind lumbar spine, skin indentation marks, high-gloss black reflective floor, directional spotlight, high-contrast chiaroscuro, 85mm lens, f/2.8, sharp focus
  ```
* **Chrome Chains & Collar**:
  ```text
  restrained_elegance, photograph, slender woman, arched on all fours with pronounced lordosis, high-fashion chain restraint, polished heavy-gauge steel chains, rigid chrome handcuffs, black leather posture collar, chained wrists at lumbar, mirror floor reflection, deep studio void, rim lighting, 50mm lens
  ```
* **Suspension Predicament**:
  ```text
  restrained_elegance, photograph, woman on tiptoes, partial suspension bondage, raw jute cordage, forged steel carabiners, diamond hishime chest harness, taut vertical rigging line, reflective black flooring, dramatic key spotlight, sculpted shadows
  ```

---

### Archetype B: Short Anchor Prompts (50–120 Tokens / 1–2 Sentences)
**Best for**: Natural language prompting, balanced creativity, commercial fashion editorials.

#### Formula:
```text
restrained_elegance Photograph of [subject] [model position] upon [environment]. Bound in [bondage type & equipment] with [rigging pattern] causing [skin indentation dynamics]. Staged in [studio void / lighting] and captured with [optics].
```

#### Examples:
* **Classic Takate-Kote Seiza**:
  ```text
  restrained_elegance Photograph of an athletic woman kneeling in traditional seiza posture directly upon a high-gloss black reflective studio floor. Her upper body is bound in a takate-kote box tie woven from 6mm unbleached Japanese hemp rope, locking her elbows behind her back and pressing visible indentation lines into her skin. Staged within a pitch-black studio void under a directional overhead spotlight, shot with an 85mm lens at f/2.8.
  ```
* **Metallic Chain Arch**:
  ```text
  restrained_elegance Photograph of a woman arched on all fours with arched spine, her knees pressing into a mirror-like black reflective floor. Her wrists are locked behind her lumbar spine in rigid double-locking chrome handcuffs tethered by heavy-gauge steel link chains to a wide black leather posture collar. Directional chiaroscuro lighting skims the polished metallic hardware, captured with a 50mm prime lens.
  ```
* **Standing Chain Harness**:
  ```text
  restrained_elegance Studio photograph capturing a woman standing upright on tiptoes, bound in an intricate fine-art steel chain chest harness with welded O-rings. Taut chains trace her ribcage and compress into her flesh, casting upside-down specular reflections onto the glossy floor. Dramatic key light with deep sculpted shadows.
  ```

---

### Archetype C: Dense Visual Narrative Prompts (180–300 Tokens / ~150–220 Words)
**Best for**: Ultra-high-end fine art renders, extreme skin pore texture, complex knot physics, gallery exhibits.

#### Formula:
Cover all **8 Mandatory Domains** in connected, sensory-dense prose:
1. `[Medium & Trigger]`
2. `[Model Position & Anatomical Geometry]`
3. `[Bondage Type & Classification]`
4. `[Bondage Equipment & Material Specifications]`
5. `[Bondage Position, Rigging & Skin Dynamics]`
6. `[Environment & Floor Reflections]`
7. `[Lighting Physics & Chiaroscuro]`
8. `[Optics & Depth of Field]`

#### Production Examples:

* **Master Exemplar 1: Japanese Shibari / Kinbaku Hybrid**:
  ```text
  restrained_elegance Photograph of an athletic woman with fair skin and blonde hair, kneeling in traditional seiza posture directly upon a high-gloss black reflective studio floor. Her spine is held in an upright arch with shoulder blades pulled tightly together, accentuating the contour of her neck and exposed clavicles, while her knees and feet press into the glossy floor, casting a sharp upside-down mirror reflection. 

  The bondage type is a hybrid of Japanese rope bondage and fine-art metallic restraint. Her upper body is bound in a geometric hishime diamond chest harness woven from 6mm unbleached twisted Japanese hemp rope (asanawa), showing distinct natural fiber fuzz. The chest harness connects to a takate-kote box tie locking her elbows behind her back, with her wrists bound at the lumbar spine in rigid double-locking chrome handcuffs that gleam with specular highlights. Where the taut rope and metallic cuffs press into her skin, visible epidermal indentations and subtle pressure marks trace her thighs and arms. 

  The scene is staged within a pitch-black minimalist studio void. A directional overhead key spotlight casts high-contrast chiaroscuro shadows across her back and shoulders, skimming over the rope ridges and polished steel links. Captured with an 85mm prime lens at f/2.8, the shot maintains crisp focus on the hardware and skin pore texture, dissolving into gentle darkness behind.
  ```

* **Master Exemplar 2: High-Fashion Chrome Chain & Leather Discipline**:
  ```text
  restrained_elegance Eye-level studio photograph capturing a slender woman with dark hair gracefully arched on all fours with pronounced lumbar lordosis, her shins and palms pressing against a seamless high-gloss black reflective floor. Her head is tilted upward, showcasing the curve of her throat and exposed collarbones, with her posture casting crisp specular mirror reflections onto the surface below.

  She is immobilized in strict high-fashion metallic chain restraint and premium bridle leather gear. Around her neck is a wide matte black leather posture collar secured with a heavy roller buckle and a forged steel O-ring. Heavy-gauge welded stainless steel link chains run taut from the collar ring down her spine, connecting to rigid chrome-plated hinged handcuffs that bind her wrists crossed behind the small of her back. The weight and tension of the chains cause distinct skin compression lines and subtle indentations along her lumbar spine and inner wrists.

  The environment is a minimalist black studio void devoid of distractions. A single directional spotlight positioned at a 45-degree overhead angle creates dramatic chiaroscuro illumination, sculpting the muscular contours of her back and producing razor-sharp specular gleams along the polished chain links. Photographed with a 50mm prime lens at f/2.8, keeping the hardware-to-skin interface in tack-sharp focus.
  ```

---

## 4. The 8-Domain Bondage & Visual Taxonomy (Curated Lexicon)

When assembling prompts, use the exact terminology the model was trained on:

### Domain 1: Model Position & Posture Geometry
* `kneeling in traditional seiza posture with shins and tops of feet pressed flat against the floor`
* `arched on all fours with pronounced lumbar lordosis and raised hips`
* `standing upright on tiptoes with body weight shifted forward against taut rigging`
* `seated upright with back straight and shoulder blades drawn tightly together`
* `prone face-down on the glossy floor with arched neck and restrained limbs`

### Domain 2: Bondage Type & Discipline
* `Japanese rope bondage (shibari / kinbaku)`
* `fine-art high-fashion metallic chain restraint`
* `strict floor-grounded immobility bondage`
* `aesthetic geometric leather harness restraint`
* `partial suspension / standing suspension predicament`

### Domain 3: Equipment & Materials
* **Cordage**: `6mm unbleached twisted Japanese hemp rope (asanawa)`, `raw oiled natural jute cordage (5mm-8mm diameter) with visible surface fiber fuzz`
* **Metals**: `heavy-gauge welded stainless steel link chains`, `rigid double-locking chrome handcuffs`, `hinged stainless steel shackles`, `polished steel O-rings`, `forged aluminum locking carabiners`, `tubular steel spreader bars`, `miniature brass padlocks`
* **Leather**: `full-grain black bridle leather straps`, `wide posture collar with sheepskin lining and roller buckle`

### Domain 4: Bondage Position, Rigging Topology & Knots
* `takate-kote box tie locking upper arms and pulling elbows together behind back`
* `hishime diamond-lattice chest harness crossing symmetrically over sternum and ribcage`
* `kikko tortoise-shell rope harness wrapping torso`
* `wrists bound crossed behind lumbar spine`
* `thigh-harness and ankle wraps tethered to central stem line`
* `leash chain tethered from posture collar O-ring to overhead rigging`

### Domain 5: Physical Dynamics & Skin Interaction
* `prominent skin indentation lines where taut rope bites deeply into soft thigh flesh`
* `visible epidermal pressure marks tracing the path of the heavy chain links`
* `realistic flesh compression and skin bulges around the tight cordage bindings`
* `subtle erythematous tension tracks on warm epidermal texture`

### Domain 6: Environment & Flooring
* `high-gloss black reflective studio floor with upside-down mirror reflections`
* `seamless pitch-black minimalist studio void`
* `monolithic dark architecture with polished concrete surface`

### Domain 7: Lighting Physics & Chiaroscuro
* `directional key spotlight casting high-contrast chiaroscuro shadows`
* `deep sculpted anatomical shadows accentuating spinal curve and muscle tone`
* `dramatic specular highlights skimming across rope ridges and polished chrome links`
* `subtle cool rim lighting separating silhouette from dark background`

### Domain 8: Camera Optics & Perspective
* `captured with an 85mm prime lens at f/2.8`
* `50mm medium-format editorial perspective`
* `shallow depth of field with gentle background falloff`
* `tack-sharp focus on the metallic hardware and skin pore texture`

---

## 5. Agent Decision Matrix: Which Archetype to Generate?

When a user interacts with your agent, follow this decision matrix:

| User Intent / Input Style | Recommended Archetype | Token Target |
|---|---|---|
| User provides comma tags (`seiza, handcuffs, blond`) | **Archetype A (Tags)** | ~30–50 tokens |
| User requests rapid variation or test renders | **Archetype A (Tags)** | ~25–40 tokens |
| User provides a 1-sentence prompt | **Archetype B (Short Anchor)** | ~60–90 tokens |
| User wants standard fashion/editorial render | **Archetype B (Short Anchor)** | ~80–120 tokens |
| User requests "masterpiece", "fine art", "extreme detail" | **Archetype C (Dense Narrative)** | ~180–280 tokens |
| User explicitly details intricate knot rigging or chain paths | **Archetype C (Dense Narrative)** | ~220–320 tokens |

---

## 6. Recommended Krea 2 Inference Settings

| Parameter | Krea 2 Large (Raw Quality) | Krea 2 Turbo (Speed) |
|---|---|---|
| **LoRA Weight** | `0.85 – 0.95` | `0.80 – 0.85` |
| **Steps** | `40 – 50` | `8` |
| **Guidance (CFG)** | `3.8 – 4.2` | `2.5` |
| **Sampler** | `Euler` or `FlowMatchEuler` | `Euler` |
| **Aspect Ratios** | `16:9` (Cinematic wide), `4:5` (Editorial portrait), `1:1` (Studio square) | `16:9` or `1:1` |

---

## 7. Negative Conditioning & Prompting Anti-Patterns

### Strict Prohibitions:
1. **DO NOT omit the trigger token**: `restrained_elegance` MUST be at Token 0.
2. **DO NOT use Booru quality sludge**: Avoid `"masterpiece, best quality, ultra highres, 8k, photorealistic"`. Krea 2's Qwen3-VL text encoder treats these as noise that steals attention from physical descriptors.
3. **DO NOT use conversational preamble**: Avoid `"Here is a prompt for..."` or `"This image depicts..."`.
4. **DO NOT generate contradictory poses**: Do not mix `kneeling in seiza` with `standing upright` in the same prompt.
5. **DO NOT exceed 340 tokens**: Exceeding 340 tokens causes text encoder truncation and self-attention dilution. Keep even the densest narratives between 180 and 320 tokens.
