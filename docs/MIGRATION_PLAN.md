# Migrations- und Umsetzungsplan: Krea 2 LoRA-Datenpipeline

**Projekt**: Krea 2 LoRA Image Dataset Pipeline (`krea2-vlm-captioner`)  
**Status**: Phase 2 – Umsetzungsplan (Entwurf zur Freigabe)  
**Zielhardware**: Nvidia L40S (48 GB VRAM) oder RTX 5090 (32 GB VRAM)  
**VLM-Engine**: Qwen-VL (Qwen3-VL / Qwen2.5-VL via vLLM Offline-Batch)  
**Primäres Trainings-Tool**: AI-Toolkit (ostris/ai-toolkit) für Krea 2 RAW & Turbo  

---

## 1. Architektur-Leitplanken & Paket-Übersicht

Die Pipeline wird schrittweise und modular in aufeinander aufbauenden Arbeitspaketen umgesetzt. Nach jedem Paket bleibt die Pipeline lauffähig und durch Unit-Tests abgesichert.

```mermaid
flowchart TD
    subgraph Package 1: Foundation
        M[Zentrales Manifest: JSONL / SQLite]
        C[Konfiguration: pipeline.yaml & vocabulary.yaml]
        CLI[CLI Entrypoint: pipeline <stage>]
    end

    subgraph Package 2: Ingestion
        S1[Stufe 1: Crawling mit Pre-Filter\nResolution Check >=0.7MP, Robots, §44b TDM]
    end

    subgraph Package 3: QC & Processing
        S2[Stufe 2: Qualitätskontrolle\nsRGB, EXIF-Strip, pHash Dedup <=6, Laplacian Blur, Text/Watermark]
        S3[Stufe 3: Downscaling\nMax 2048px, mod 16, Lanczos, Nicht-destruktiv]
    end

    subgraph Package 4: Annotation & Screening
        S4[Stufe 4: Qwen-VL Captioning & Screening\nvLLM Offline-Batch, Guided JSON, Alter >=25 Gate, Sample-Report]
    end

    subgraph Package 5: Structuring & Export
        S5[Stufe 5: Dataset Export\nMatrix-Analyse, AI-Toolkit Config, num_repeats Balancierung]
    end

    subgraph Package 6: Tooling & Validation
        S6[Stufe 6: Training Runbook & AI-Toolkit Krea 2 RAW Presets]
        S7[Stufe 7 & 8: Validierungs-Grid, Kontaktbogen, Charakter-LoRA Stack]
    end

    Package 1 --> Package 2 --> Package 3 --> Package 4 --> Package 5 --> Package 6
```

---

## 2. Detaillierte Arbeitspakete

### Arbeitspaket 1: Foundation, Packaging & Zentrales Manifest (Core)
* **Ziel**: Etablierung des Projekt-Fundaments: Standardisiertes Python-Paket (`pyproject.toml`), versionierte Konfiguration (`config/pipeline.yaml`), initiales Vokabular (`config/vocabulary.yaml`), transaktionssicheres Manifest (`data/manifest.jsonl`), strukturiertes JSON-Lines-Logging und zentraler CLI-Dispatcher (`pipeline <stage> --config ...`).
* **Betroffene Dateien**:
  * `[NEW]` `pyproject.toml` (Build-System, Typ-Annotationen, Dependencies: Pydantic, Pillow, PyYAML, ImageHash, etc.)
  * `[NEW]` `config/pipeline.yaml` (Zentrale Parametrisierung aller Stufen)
  * `[NEW]` `config/vocabulary.yaml` (Initiales Vokabular für Stile, Locations, Posen/Praktiken aus bisherigen Tags)
  * `[NEW]` `src/pipeline/__init__.py`
  * `[NEW]` `src/pipeline/cli.py` (CLI-Einstieg via `argparse` oder `typer`/`click`)
  * `[NEW]` `src/pipeline/manifest.py` (Manifest-Modell via Pydantic & JSONL-Manager mit Locking/Idempotenz)
  * `[NEW]` `src/pipeline/logging_utils.py` (Strukturiertes JSONL-Logging)
  * `[NEW]` `tests/test_manifest.py` (Tests für Manifest-State, Updates und Serialisierung)
* **Akzeptanzkriterien**:
  1. Paket lässt sich via `pip install -e .` installieren.
  2. `pipeline --help` und `pipeline manifest --status` sind aufrufbar.
  3. Manifest speichert Status pro Bild atomar ab und verhindert Doppelverarbeitung (Idempotenz).
  4. Alle Tests laufen ohne GPU auf der CPU durch.
* **Testansatz**:
  * Unit-Tests mit `pytest` und temporären Datei-Fixtures für Manifest-Einträge, Filter-Status und Concurrency.

---

### Arbeitspaket 2: Stufe 1 – Crawling mit Vorfilter
* **Ziel**: Refaktorisierung der bestehenden Crawler in ein einheitliches, manifest-integriertes Modul. Prüfung der Bildauflösung vor dem Download (HTML-Attribute oder `HEAD`/Range-Request für Image-Header), Abbruch bei $< 0,7$ MP. Einhaltung von `robots.txt`, TDM-Opt-Out-Signalen (§ 44b UrhG) und Domain-Ratelimiting.
* **Betroffene Dateien**:
  * `[NEW]` `src/pipeline/stages/stage1_crawl.py`
  * `[NEW]` `src/pipeline/crawler/prefilter.py` (Header-Resolution-Sniffer vor dem Download)
  * `[NEW]` `src/pipeline/crawler/compliance.py` (Robots.txt & § 44b UrhG TDM-Detector)
  * `[NEW]` `src/pipeline/crawler/rate_limiter.py` (Domain-Ratelimiting mit Politeness)
  * `[MODIFY]` `src/pipeline/crawler/xenforo.py` & `src/pipeline/crawler/dbnaked.py` (Integration in Pipeline-Interface)
  * `[NEW]` `tests/test_stage1_prefilter.py` (Mocks für HTTP HEAD/Range-Requests und TDM-Header)
* **Akzeptanzkriterien**:
  1. Bilder $< 0,7$ MP werden anhand der Header vor dem Download erkannt und übersprungen.
  2. TDM-Opt-out-Signale (`tdm-reservation: 1` Header/Meta) führen zum Ausschluss und Vermerk im Manifest.
  3. Erfolgreiche Downloads werden mit URL, Zeitstempel, Quell-Metadaten und Hash im Manifest registriert.
  4. Abgebrochene Läufe lassen sich ohne Doppel-Downloads nahtlos fortsetzen.
* **Testansatz**:
  * Synthetische Mock-Server / Unit-Tests für Header-Resolution-Sniffing und TDM-Parsing.

---

### Arbeitspaket 3: Stufen 2 & 3 – Qualitätskontrolle & Downscaling
* **Ziel**:
  * **Stufe 2 (Quality Control)**: Bereinigung (sRGB-Konvertierung, EXIF-Strip), Reale Auflösungsprüfung (Mindestkante 1024 px), pHash-Deduplizierung (Hamming-Distanz $\le 6$ mit Cluster-Repräsentantenauswahl des schärfsten Bildes), Laplacian-Schärfefilterung, Wasserzeichen-/Texterkennung. Rejections werden non-destruktiv im Manifest dokumentiert.
  * **Stufe 3 (Downscaling)**: Sichere Transformation in `data/processed/images/`. Lange Kante max. 2048 px (nur verkleinern), Kantenlängen gerundet auf Vielfache von 16, Lanczos-Resampling, konfigurierbares JPEG/PNG.
* **Betroffene Dateien**:
  * `[NEW]` `src/pipeline/stages/stage2_qc.py` (Qualitäts- und Deduplizierungslogik)
  * `[NEW]` `src/pipeline/stages/stage3_downscale.py` (Multiples-of-16 Resizing & Color Management)
  * `[NEW]` `src/pipeline/qc/dedup.py` (pHash Hamming-Distanz Clustering, optional Embedding-Interface)
  * `[NEW]` `src/pipeline/qc/filters.py` (Laplacian-Varianz, sRGB-Normalisierung, EXIF-Stripper)
  * `[NEW]` `tests/test_stage2_qc.py` (Test-Fixtures: künstliche Duplikate, unscharfe Testbilder, EXIF-Tags)
  * `[NEW]` `tests/test_stage3_downscale.py` (Kantenlängen-Tests auf Vielfache von 16, Aspect-Ratio-Erhalt)
* **Akzeptanzkriterien**:
  1. pHash identifiziert Duplikate zuverlässig; das schärfste/größte Bild wird beibehalten, Duplikate im Manifest markiert.
  2. Alle Ausgabebilder der Stufe 3 haben Kantenlängen als Vielfache von 16 und maximal 2048 px.
  3. Originale bleiben im Rohdatenverzeichnis unberührt.
  4. Abschluss-Report listet genaue Reject-Zahlen je Filtergrund auf.
* **Testansatz**:
  * Generierung kleiner synthetischer Testbilder im Testlauf (schwarz/weiß, Unschärfe, unterschiedliche Dimensionen wie 1000x1500 $\rightarrow$ mod 16 Check).

---

### Arbeitspaket 4: Stufe 4 – Qwen-VL Captioning & Screening
* **Ziel**: Hochperformante Offline-Batch-Inferenz über vLLM mit Qwen-VL (Qwen3-VL-8B / Qwen2.5-VL-7B) unter Ausnutzung der L40S (48 GB) / RTX 5090 (32 GB). Guided Decoding via JSON-Schema mit Assistant-Prefill `{`. Dynamische Vokabular-Injektion aus `config/vocabulary.yaml`. **Verbindliches Screening-Gate** (`subject_age_estimate < 25` oder `uncertain_age == True` führt zum zwingenden Ausschluss `rejected_screening`). Retry-Logik mit Seed-Variation und Ausweich-Muster-Erkennung (Regex). Caption-Assembly nach Schema (`[Trigger], [Stil], [Location], [Inhalt], [Licht]`) und `--sample 200` Testlauf-Report (HTML).
* **Betroffene Dateien**:
  * `[NEW]` `src/pipeline/stages/stage4_caption.py`
  * `[NEW]` `src/pipeline/captioning/schema.py` (Pydantic-Modell für Guided Decoding mit Enums aus Vokabular)
  * `[NEW]` `src/pipeline/captioning/vllm_engine.py` (Lazy-Importierte vLLM Offline-Batch-Engine `LLM.generate`)
  * `[NEW]` `src/pipeline/captioning/screening.py` (Alters-, Wasserzeichen- und Qualitäts-Filterlogik)
  * `[NEW]` `src/pipeline/captioning/assembly.py` (Caption-Builder mit `style` vs. `subject` LoRA-Modus)
  * `[NEW]` `src/pipeline/captioning/reporter.py` (HTML-Report-Generator mit Thumbnails & JSON-Inspector)
  * `[NEW]` `tests/test_stage4_screening.py` (Strikte Unit-Tests für Screening-Gating und Rejections)
  * `[NEW]` `tests/test_stage4_assembly.py` (Tests für Caption-Templates und Vokabular-Ersetzung)
* **Akzeptanzkriterien**:
  1. `vLLM` wird erst bei GPU-Aufruf importiert; Tests und CLI-Hilfen laufen auf CPU ohne GPU-Fehler.
  2. Alle Bilder mit `age < 25` oder `uncertain_age == True` werden unumstößlich als `rejected_screening` markiert und erhalten keine Trainings-Captions.
  3. Assistant-Prefill `{` erzwingt fehlerfreies JSON-Streaming.
  4. Fehlgeschlagene Antworten werden bis zu 2-mal mit neuem Seed wiederholt.
  5. Der Testlauf `--sample 200` generiert einen übersichtlichen HTML-Kontaktbogen.
* **Testansatz**:
  * Synthetische Mock-Outputs des Schemas zum Testen der Screening-Regeln, Retrys und Caption-Formatierung.

---

### Arbeitspaket 5: Stufe 5 – Datensatz strukturieren & Export
* **Ziel**: Aggregation und Verteilungsanalyse aus dem Manifest ($Style \times Location \times Pose$). Export einer validierten Dataset-Konfiguration für **AI-Toolkit** (sowie Kompatibilität für `musubi-tuner`). Automatische Berechnung von `num_repeats` zum Ausgleich unterrepräsentierter Klassen (bis Faktor 4). Vorbereitung für Aspect-Ratio-Bucketing und Warnung bei Gesichts-/Personen-Überrepräsentation.
* **Betroffene Dateien**:
  * `[NEW]` `src/pipeline/stages/stage5_export.py`
  * `[NEW]` `src/pipeline/export/distribution.py` (Kreuztabellen-Analyse & Personen-Cluster-Metriken)
  * `[NEW]` `src/pipeline/export/ai_toolkit_builder.py` (Generierung der AI-Toolkit YAML-Datensatz-Configs)
  * `[NEW]` `src/pipeline/export/musubi_builder.py` (Optionale musubi-tuner Dataset-TOML Generierung)
  * `[NEW]` `tests/test_stage5_export.py` (Verifikation der `num_repeats`-Berechnung und Config-Validität)
* **Akzeptanzkriterien**:
  1. Verteilungsanalyse erzeugt eine tabellarische Übersicht der Subkonzepte im Terminal und Manifest-Report.
  2. AI-Toolkit-Konfiguration wird fehlerfrei generiert, inklusive Subsets und balancierter Wiederholungen.
  3. Gesichts-Cluster-Check gibt eine Warnung aus, wenn eine Person $> X\,\%$ des Sets einnimmt.
* **Testansatz**:
  * Unit-Tests mit synthetischen Manifest-Datensätzen mit ungleicher Verteilung (z. B. 10 Bilder Kategorie A, 2 Bilder Kategorie B $\rightarrow$ Prüfen von `num_repeats`).

---

### Arbeitspaket 6: Stufen 6–8 – Trainings-Tooling, Validierung & Charakter-LoRAs
* **Ziel**:
  * **Stufe 6**: Bereitstellung einsatzfertiger Trainings-Presets für **AI-Toolkit** auf **Krea 2 RAW** (LoRA Rank 128, Alpha 128, lr 5e-5, fp8 Basismodell, Gradient Checkpointing, RMSNorm/Modulation) und zweites kuratiertes 2–3k Preset. Dokumentation des vollständigen Workflows im Runbook `docs/TRAINING.md`.
  * **Stufe 7**: Validierungs-Tooling: Prompt-Grid-Generator aus Vokabular; Render-Skript für Krea 2 RAW (52 Steps) vs. Turbo (8 Steps) über Gewichte 0.7, 0.85, 1.0 mit automatischem HTML-Kontaktbogen.
  * **Stufe 8**: Konfigurationsgenerator für sequentielle Charakter-LoRAs mit eingefrorener Stil-LoRA und Stack-Validierungsgrid ($Stil \times Charakter$).
* **Betroffene Dateien**:
  * `[NEW]` `src/pipeline/tooling/train_config.py` (AI-Toolkit Krea 2 RAW & Turbo Config-Generator)
  * `[NEW]` `src/pipeline/tooling/validation_grid.py` (Prompt-Grid Generator & Render-Pipeline)
  * `[NEW]` `src/pipeline/tooling/character_stack.py` (Charakter-LoRA Stack-Konfigurator)
  * `[NEW]` `templates/ai_toolkit_krea2_raw.yaml` (Templates für LoRA Rank 128 & LoKr)
  * `[NEW]` `templates/ai_toolkit_krea2_curated.yaml` (Template für das 2-3k Vergleichs-Subset)
  * `[NEW]` `docs/TRAINING.md` (Ausführliches Runbook mit Caching-, Trainings- und Validierungsbefehlen)
  * `[NEW]` `tests/test_stage6_7_8.py` (Validierung der erzeugten YAML-Configs gegen Schema)
* **Akzeptanzkriterien**:
  1. AI-Toolkit Konfigurationsdateien sind syntaktisch valide und enthalten alle Krea 2 Hyperparameter.
  2. `docs/TRAINING.md` beschreibt exakt die Schritte zum Cachen der Latents/Text-Encoder und den Start des Trainings auf L40S/5090.
  3. Der Validierungs-Grid-Generator erzeugt konsistente Prompt-Listen aus dem Vokabular.
* **Testansatz**:
  * Schema-Validierung der generierten YAML-Dateien und Unit-Tests für den Prompt-Grid-Generator.

---

## 3. Reihenfolge der Umsetzung & Verifikationsplan

```text
[Paket 1: Foundation & Manifest]   --> Tests: Manifest-Integrität, CLI
                │
                ▼
[Paket 2: Crawling mit Vorfilter]  --> Tests: Resolution Sniffing, TDM, Rate Limiting
                │
                ▼
[Paket 3: QC & Downscaling]        --> Tests: pHash-Clustering, Mod-16 Resizing
                │
                ▼
[Paket 4: Qwen-VL & Screening]     --> Tests: Alter >= 25 Screening Gate, JSON Prefill
                │
                ▼
[Paket 5: Dataset Export]          --> Tests: num_repeats Balancierung, AI-Toolkit YAML
                │
                ▼
[Paket 6: Training & Validation]   --> Tests: Config-Validität, Prompt-Grid Generation
```

Nach jedem Paket wird:
1. Der Code modular implementiert.
2. Die Unit-Testsuite ausgeführt (`pytest tests/`).
3. Die Pipeline in einem kurzen Trockenlauf verifiziert.
4. Ein Zwischenbericht im Chat erstattet, bevor das nächste Paket gestartet wird.
