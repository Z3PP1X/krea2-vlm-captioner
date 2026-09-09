# Runbook: Krea 2 LoRA Training mit AI-Toolkit auf L40S / RTX 5090

Dieses Handbuch beschreibt die konkreten Schritte zur Durchführung des LoRA-Trainings auf der **Krea 2 (K2)** Architektur unter Nutzung von **AI-Toolkit** (`ostris/ai-toolkit`) auf einer **Nvidia L40S (48 GB)** oder **RTX 5090 (32 GB)**.

---

## 1. Hardware- & Software-Voraussetzungen

* **Empfohlene GPU**:
  * **Nvidia L40S (48 GB VRAM)**: Ideal für große Batchgrößen (Batch 4–8) und lange Kontexte ohne VRAM-Engpass.
  * **Nvidia RTX 5090 (32 GB VRAM)**: Extrem hohe Tensor-Core-Leistung (Blackwell-Architektur), FP8-Beschleunigung.
* **Betriebssystem**: Ubuntu 22.04 / 24.04 (RunPod PyTorch 2.4+ Template).
* **VRAM-Verbrauch**:
  * Krea 2 RAW (FP8-quantisiert): ~13 GB.
  * Qwen3-VL-4B Text-Encoder: ~4.5 GB (nur während des Cachings aktiv).
  * LoRA Rank 128 Optimizer States (AdamW 8-bit) + Activations: ~8–11 GB.
  * **Gesamtbedarf im Training nach Disk-Caching**: **~21–25 GB VRAM**.

---

## 2. Vorbereitung & Setup auf RunPod

Klone `ai-toolkit` und installiere die Abhängigkeiten:

```bash
cd /workspace
git clone --recursive https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
pip install -r requirements.txt
pip install torch torchvision --upgrade
```

Kopiere deine generierten Konfigurationsdateien aus der Pipeline:
```bash
# Kopiere den aufbereiteten Datensatz und die Config
cp /workspace/krea2-vlm-captioner/data/export/ai_toolkit_krea2_raw.yaml /workspace/ai-toolkit/config_krea2_raw.yaml
cp /workspace/krea2-vlm-captioner/data/export/ai_toolkit_krea2_curated.yaml /workspace/ai-toolkit/config_krea2_curated.yaml
```

---

## 3. Schritt 1: Disk-Caching (Latents & Text-Encoder)

Krea 2 nutzt einen 12.9B MMDiT mit einem **Qwen3-VL-4B Text-Encoder**, der über 12 Zwischenschichten getappt wird. Um Trainingszeit und VRAM zu sparen, werden alle VAE-Latents und Text-Embeddings vorab auf die NVMe-SSD gecached:

```bash
cd /workspace/ai-toolkit

# Latents und Qwen3-VL Embeddings vorberechnen:
python3 run.py config_krea2_raw.yaml --cache-only
```
* **Dauer**: ca. 12–18 Minuten für 3.000 Bilder auf einer L40S / 5090.
* **Speicherplatzbedarf**: ca. 8–12 GB auf der SSD für die vorberechneten `.pt` / `.safetensors` Cache-Dateien.

---

## 4. Schritt 2: Stil-LoRA Training auf Krea 2 RAW (Full Set)

Starte das eigentliche Training des Basis-Stil-LoRAs:

```bash
python3 run.py config_krea2_raw.yaml
```

### Hyperparameter-Übersicht:
* **Basismodell**: `krea/krea2-raw` (undistilliert, `quantize_base: "fp8"`)
* **LoRA Rank**: 128 (`linear: 128`, `linear_alpha: 128`)
* **Learning Rate**: `5e-5` mit Cosine Scheduler & Warmup (100 Steps)
* **Batch Size**: 4 (Gradient Accumulation: 1)
* **Optimierer**: `adamw8bit`
* **Gradient Checkpointing**: Aktiviert (`gradient_checkpointing: true`)
* **Norm / Modulations-Training**: `train_norm: true` (ermöglicht der LoRA, die Krea 2 DiT-Modulationen präzise anzupassen)
* **Checkpoints**: Alle 1.000 Steps in `output/krea2_style_lora_raw/`

---

## 5. Schritt 3: Kuratierter Vergleichslauf (2.5k Subset)

Um empirisch zu überprüfen, ob ein hochgradig kuratiertes Subset eine sauberere Stiltreue ohne Artefakte liefert, starte den Vergleichslauf auf dem von Stufe 6 generierten Subset:

```bash
python3 run.py config_krea2_curated.yaml
```
* **Umfang**: 2.000–2.500 Bilder (höchste Schärfe und ausgewogene Posen-Verteilung).
* **Steps**: 2.000 Steps (ca. 45 Minuten Trainingszeit auf RTX 5090 / L40S).

---

## 6. Schritt 4: Sequentielles Charakter-Training (Stacked LoRA)

Sobald die finale Stil-LoRA feststeht, kann ein Charakter-LoRA trainiert werden, bei dem die Stil-LoRA fest eingefroren im Basismodell mitgeladen wird:

```bash
# 1. Bereite Charakter-Bilder in data/characters/<name> vor
# 2. Generiere die Config:
pipeline character-config

# 3. Starte das Charakter-Training:
python3 run.py /workspace/krea2-vlm-captioner/data/export/ai_toolkit_character_marlene.yaml
```
* **LoRA Rank**: 32 (`linear: 32`, `linear_alpha: 32`)
* **Learning Rate**: `1e-4`
* **Dauer**: ca. 1.200 Steps (~25 Minuten).

---

## 7. Schritt 5: Validierung & Kontaktbogen-Prüfung

Generiere das Validierungs-Grid und überprüfe die Checkpoints auf Krea 2 RAW und Turbo:

```bash
cd /workspace/krea2-vlm-captioner
pipeline validate
```
* Öffne `data/validation/contact_sheet.html` im Browser, um den direkten Vergleich der LoRA-Skalierungen (0.0, 0.7, 0.85, 1.0) über alle Subkonzepte zu inspizieren.

---

## 8. Kosten- und Zeitschätzung

| Phase | Hardware | Dauer | Geschätzte Kosten (RunPod ca. 0.80–1.20 $/h) |
| :--- | :--- | :--- | :--- |
| **Disk-Caching (3.000 Bilder)** | L40S / 5090 | ~15 Minuten | ~0.25 $ |
| **Full Set Training (3.500 Steps)** | L40S / 5090 | ~75 Minuten | ~1.25 $ |
| **Curated Set Training (2.000 Steps)** | L40S / 5090 | ~45 Minuten | ~0.75 $ |
| **Charakter-Training (1.200 Steps)** | L40S / 5090 | ~25 Minuten | ~0.40 $ |
| **Gesamter Durchlauf** | | **ca. 2.5 Stunden** | **~2.65 $** |
