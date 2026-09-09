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

### The 400+ Token Advantage
Krea 2 thrives when conditioned on **connected, grammatically complete, natural language paragraphs of at least 400 tokens (~280–350+ words)**. A comprehensive 400+ token narrative allows:
1. Complete separation of subject, attire, hardware, and environment into distinct semantic clauses.
2. Full activation of all 12 tapped layers of Qwen3-VL simultaneously.
3. Precise control over tactile surfaces, skin micro-texture, and lighting physics without token collision.

---

## 2. The 7-Layer Sensory Narrative Schema

Every production-grade Krea 2 caption must sequentially address seven sensory layers:

```
[Layer 1: Medium & Framing]
  └─ Format: "Photograph of...", "Medium shot of...", "Cinematic wide photograph capturing..."
[Layer 2: Subject & Demographics]
  └─ Explicit, objective anatomy: skin tone, natural epidermal texture, hair color/style, physique, posture tension, facial expression.
[Layer 3: Pose & Spatial Geometry]
  └─ Exact body configuration: limb angles, arching of the spine, contact with surfaces/floor, gravitational weight distribution.
[Layer 4: Materials, Hardware & Restraints]
  └─ Micro-level tactile descriptions: rope fiber (hemp/jute), weave texture, friction knots, metallic hardware (chrome/steel), specular shine, leather finish.
[Layer 5: Environment & Background]
  └─ Studio architecture, flooring material (glossy reflective black floor, concrete), walls, negative space, atmospheric void.
[Layer 6: Lighting & Colorimetry]
  └─ Light sources: key light, rim lighting, fill light, chiaroscuro contrast, sculptural shadow gradients, specular highlights along skin contours.
[Layer 7: Optics, Lens & Atmosphere]
  └─ Focal length (e.g. 50mm, 85mm), aperture falloff, depth of field, sharp foreground plane, gentle background bokeh.
```

---

## 3. Trigger Token Mechanics
- **Placement**: The trigger token (e.g., `kink, sexandsubmission` or `restrained_elegance`) **MUST be the very first token (Token 0)** in the caption.
- **Attention Flow**: Placing the trigger token first ensures the Qwen3-VL self-attention matrix attends to the trigger across all subsequent layers and words.
- **Syntax**:
  ```text
  kink, sexandsubmission Photograph of a slender woman with fair skin...
  ```

---

## 4. Multi-Tier Caption Strategy for LoRA Training
For state-of-the-art LoRA fine-tuning, training captions should be prepared in three tiers:
1. **`caption_dense` (>= 400 tokens / 280–350+ words)**: Primary training target (used in 70% of epochs). Teaches full sensory fidelity and scene decomposition.
2. **`caption_mid` (40–70 words)**: Condensed anchor focusing on Subject + Action + Hardware + Setting (used in 20% of epochs to avoid over-memorization).
3. **`caption_short` (15–25 words)**: Minimalist anchor for high-guidance inference.
