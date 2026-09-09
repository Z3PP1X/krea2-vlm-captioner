# Krea 2 Optimal Image Captioning Strategy

## 1. Architectural Foundation: Qwen3-VL & Single-Stream MMDiT
Krea 2 is a **12.9B parameter single-stream MMDiT** (Multimodal Diffusion Transformer). Unlike legacy SD1.5/SDXL architectures that relied on dual CLIP/T5 text encoders and suffered severe token truncation at 77 tokens, Krea 2's conditioning is powered by **Qwen3-VL-4B-Instruct**.

Krea 2 taps **12 intermediate decoder layers** of Qwen3-VL (layers 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, and 35) via a learned projection layer (`txtfusion.projector`):
- **Early Layers (2–8)**: Resolve lexical tokens, surface color palettes, coarse geometric shapes, and basic materials.
- **Middle Layers (11–20)**: Resolve syntactic bindings, prepositions, spatial relations (e.g. *kneeling on*, *suspended above*, *wrapped around*), anatomy, and physical contact points.
- **Late Layers (23–35)**: Resolve photographic mood, chiaroscuro lighting, depth of field, optical characteristics, and fine sensory details.

### Why Comma-Separated "Booru Tags" Fail
Comma-separated tag soups (`1girl, solo, bdsm, chains, kneeling, black background, 8k`) completely fail to activate the middle and late layers of Qwen3-VL. Tags lack grammatical syntax and spatial prepositions. As a result, the model produces:
- Flat, artificial lighting
- Deformed anatomy and disconnected limbs
- "Token bleeding" (e.g., rope textures bleeding into human skin or wall surfaces)
- Plastic, waxy epidermal rendering

### The 150–340 Token Sweet Spot (Dense & Concise)
Krea 2 achieves optimal conditioning when descriptions are concise, sensory-dense, and tightly bounded between **150 and 340 tokens (~150–220 words)**. This strict ceiling eliminates rambling, filler phrases, and repetitive adjectives while preserving complete semantic coverage across all 12 Qwen3-VL layers:
1. Concise semantic separation of model posture, restraint hardware, anatomical placement, and lighting into tight sentences.
2. Full activation of all 12 tapped layers of Qwen3-VL without diluting attention weights.
3. Fast inference convergence and crisp prompt following without token bloat.

---

## 2. Mandatory Descriptive Taxonomy

Every production-grade Krea 2 caption must exhaustively document these core domains:

```
[Layer 1: Trigger & Medium Declaration]
  └─ Format: "restrained_elegance Photograph of...", "kink, shibari Medium-wide studio photograph capturing..."

[Layer 2: Model Position & Anatomical Geometry]
  └─ Posture: Kneeling in seiza, standing upright on tiptoes, arched on all fours, seated with legs folded, prone face-down, or suspended.
  └─ Limb & Joint Angles: Elbows drawn tightly behind back, knees spread wide, arched lumbar lordosis, head tilted upward.
  └─ Gravity & Contact: Physical contact points with glossy floor, body weight distribution, visible muscle tension in calves/quads/shoulders.
  └─ Anatomy: Skin tone, natural epidermal texture (pores, sheen), physique, facial expression.

[Layer 3: Bondage Type & Classification]
  └─ Discipline: Japanese rope bondage (shibari / kinbaku), high-fashion metallic chain restraint, strict immobilizing bondage, aesthetic decorative harness, suspension / partial suspension, leather strap restraint, predicament bondage.

[Layer 4: Bondage Equipment & Material Specifications]
  └─ Materials & Gauge: Unbleached twisted Japanese hemp rope (asanawa) or raw oiled jute cordage (specifying 5mm–8mm diameter, surface fuzz, multi-ply twist).
  └─ Metallic Hardware: Heavy-gauge welded steel link chains, polished chrome/nickel-plated handcuffs, stainless steel shackles, hinged locking cuffs, welded steel O-rings, forged carabiners, spreader bars, padlocks.
  └─ Leather Gear: Full-grain bridle leather straps, wide posture collars, roller buckles, sheepskin-lined cuffs.

[Layer 5: Bondage Position, Rigging Topology & Skin Interaction]
  └─ Anatomical Routing: Hands bound behind back at lumbar spine, box tie (takate-kote / gote) with elbows pulled together, diamond lattice chest harness (hishime) across sternum and breasts, tortoise shell pattern (kikko), thigh/ankle ties, frog tie, collar leash tethered to ceiling hook or chain harness.
  └─ Physical Dynamics: Distinct skin indentation lines where taut cordage or metal bites into thighs/torso, muscle compression, and realistic tension marks.

[Layer 6: Environment, Architecture & Backdrop]
  └─ Flooring: Black high-gloss reflective floor with mirror-like upside-down specular reflections, polished concrete, or wooden flooring.
  └─ Space: Minimalist pitch-black studio void, negative space, architectural pillars.

[Layer 7: Lighting, Chiaroscuro & Shadow Dynamics]
  └─ Staging: High-contrast chiaroscuro, focused directional spotlights, deep sculpted shadows, specular highlights skimming skin contours and metallic hardware, subtle edge rim light.

[Layer 8: Optics, Camera Perspective & Atmosphere]
  └─ Camera: 50mm or 85mm lens, f/2.8 shallow depth of field, razor-sharp focus on hardware/skin interface, gentle background bokeh, raw cinematic editorial atmosphere.
```

---

## 3. Trigger Token Mechanics
- **Placement**: The trigger token (e.g., `restrained_elegance` or `kink, shibari`) **MUST be the very first token (Token 0)** in the caption.
- **Attention Flow**: Placing the trigger token first ensures the Qwen3-VL self-attention matrix attends to the trigger across all subsequent layers and words.
- **Syntax**:
  ```text
  restrained_elegance Photograph of a slender athletic woman...
  ```

---

## 4. Multi-Tier Caption Strategy for LoRA Training
1. **`caption_dense` (150–340 tokens / ~150–220 words)**: Primary training target (used in 70% of epochs). Teaches full sensory fidelity, rigging geometry, and scene decomposition with zero token bloat.
2. **`caption_mid` (40–70 words)**: Condensed anchor focusing on Subject + Model Position + Bondage Type/Hardware + Setting (used in 20% of epochs).
3. **`caption_short` (15–25 words)**: Minimalist anchor for high-guidance inference.
