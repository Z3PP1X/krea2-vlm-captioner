# Gap-Analyse: Krea 2 LoRA-Datenpipeline (Ist-Zustand vs. Zielarchitektur)

**Datum**: September 2026  
**Projekt**: Krea 2 LoRA Image Dataset Pipeline (`krea2-vlm-captioner`)  
**Status**: Phase 1 – Analyse abgeschlossen  

---

## 1. Executive Summary & Kontext

Dieses Dokument bildet das Ergebnis der **Phase 1 (Analyse des aktuellen Stands)**. Ziel ist es, die bestehende Skriptsammlung aus Web-Crawlern und VLM-Captioning in eine modulare, idempotente, fehlertolerante und standardisierte **End-to-End ML-Datenpipeline** für das Training von Stil- und Charakter-LoRAs auf Basis der **Krea 2 (K2)** Architektur (12.9B MMDiT mit Qwen3-VL-Konditionierung) zu überführen.

---

## 2. Ist-Zustand des Repositories

### 2.1 Modulübersicht & Einstiegspunkte
Aktuell besteht das Repository aus isolierten Skripten im Stammverzeichnis ohne standardisiertes Paket-Layout oder zentralen CLI-Dispatcher:

| Datei | Primäre Funktion | Implementierungsstand |
| :--- | :--- | :--- |
| `crawler.py` | XenForo & generischer Forum-Crawler | Scraped HTML via `requests` + `BeautifulSoup`, parsed Thumbnails/Lightbox-Links, lädt Bilder multithreaded herunter, schreibt `genres.txt`. |
| `dbnaked_crawler.py` | dbNaked Channel- & Galerie-Crawler | Scraped Seiten/Channels, extrahiert 1600x1600 Bilder mit Hotlink-Referer-Headern, schreibt `genres.txt`. |
| `caption_images.py` | Standalone Image Captioner | Iteriert über Bilddateien, liest `genres.txt`, sendet Einzel-HTTP-Requests (via Base64-Payload) an Ollama (`/api/chat`) oder vLLM OpenAI-Kompatibilität (`/v1/chat/completions`), extrahiert JSON via Regex, schreibt `<image>.txt` und optional `<image>.json`. |
| `crawl_and_tag.py` | Sequentieller Wrapper | Ruft zuerst `crawler.py` auf, danach `caption_images.py`. |
| `setup_runpod.sh` | Shell-Bootstrap | Installiert System-Packages und startet Ollama als Daemon. |
| `vllm_server_runpod.sh`| Shell-Bootstrap | Startet einen OpenAI-kompatiblen vLLM API-Server auf Port 8000. |

### 2.2 Datenfluss und -formate
Der Datenfluss ist rein dateisystembasiert und zustandslos:
1. **Eingabe**: URLs oder Seitenbereiche als CLI-Parameter.
2. **Zwischenstufe**: Lokale Ordnerstruktur (`downloads/<Set_Name>/`) mit Rohbildern (`.jpg`, `.png`) und einer unstrukturierten Textdatei (`genres.txt`), die Titel und Genre-Strings enthält.
3. **Ausgabe**: Neben jedem Bild wird eine Datei `<image>.txt` mit einem Fließtext-Prompt (sowie optional `<image>.json`) abgelegt.
4. **Zustandsverwaltung**: Basiert ausschließlich auf Dateiexistenz (`if txt_path.exists() and not overwrite: skip`). Es gibt kein persistentes Logging, keine Checkpoints und kein Transaktionskonzept.

### 2.3 Konfiguration & Abhängigkeiten
- **Keine versionierte Konfiguration**: Sämtliche Standardwerte (URLs, Schwellwerte, System-Prompts, Timeouts, Batchgrößen) sind als Konstanten in den Python-Dateien fest verdrahtet.
- **Packaging**: Kein `pyproject.toml`. Nur eine rudimentäre `requirements.txt` (`requests`, `beautifulsoup4`, `rich`, `Pillow`, `tqdm`, `urllib3`).
- **Tests**: Keine Testsuite vorhanden (`tests/` existiert nicht).
- **Inferenz-Architektur**: Verwendet synchrone HTTP-Einzelanfragen an externe Server (Ollama / vLLM REST Server). Dies erzeugt massiven Overhead (Base64-Serialisierung, HTTP Keep-Alive, fehlende Parallelisierung im Offline-Batch-Modus).

---

## 3. Gap-Matrix (Zielarchitektur vs. Ist-Zustand)

| Stufe / Bereich | Status | Konkrete Abweichung zum Soll | Aufwand |
| :--- | :---: | :--- | :---: |
| **Core: Pipeline-Architektur** | ❌ Fehlt | Kein einheitlicher CLI-Einstieg (`pipeline <stufe>`). Kein SQLite/JSONL-Manifest (`data/manifest.*`). Keine `config/pipeline.yaml`. Kein strukturiertes JSONL-Logging. | **M** |
| **Core: Packaging & Testing** | ❌ Fehlt | Kein `pyproject.toml`, keine Typ-Strictness, keine modulare Paketstruktur (`src/pipeline/`), keine Unit-Tests oder synthetischen Test-Fixtures. | **S** |
| **Stufe 1: Crawling mit Vorfilter** | 🟡 Teilweise | Parser für XenForo & dbNaked vorhanden. Es fehlen: Pre-Download-Auflösungscheck (< 0,7 MP via Header/HEAD), TDM-Opt-out-Erkennung (§ 44b UrhG), Robots.txt-Handling, domainweites Rate-Limiting, Idempotenz via Manifest. | **M** |
| **Stufe 2: Qualitätskontrolle** | ❌ Fehlt | Keine Qualitätsstufe vorhanden. Es fehlen: EXIF-Stripping & sRGB-Normalisierung, 1024px-Mindestauflösung, pHash-Clustering (Hamming $\le$ 6), optionales Embedding-Dedup (CLIP/DINOv2), Laplacian-Schärfefilter, Ästhetik-Scoring, Text-/Watermark-Erkennung, Non-Destructive Reject-Logik im Manifest. | **L** |
| **Stufe 3: Downscaling** | ❌ Fehlt | Nur dynamisches In-Memory-Downscale im Captioner (1024px) für VLM vorhanden. Es fehlen: Separater Resizing-Schritt für das finale Trainingsset (lange Kante max 2048px, Kantenlängen als Vielfache von 16, Lanczos, konfigurierbares JPEG/PNG, Trennung von Rohdaten und verarbeitetem Datensatz). | **S** |
| **Stufe 4: Captioning & Screening** | 🟡 Teilweise | VLM-Anbindung existiert nur als REST/Ollama. Es fehlen: vLLM Offline-Batch-Modus (`LLM.generate`), Guided Decoding via JSON-Schema (`guided_json`), Assistant-Prefill (`{`), externe `config/vocabulary.yaml`, strukturierte Felder, verbindliches Screening (Alter $\ge$ 25, Watermark/Quality Rejects), Retry-/Fallback-Kaskade, `--sample 200` HTML-Report. | **L** |
| **Stufe 5: Datensatz strukturieren & Export** | ❌ Fehlt | Keine Analyse- und Exportstufe vorhanden. Es fehlen: $Style \times Location \times Pose$ Verteilungsanalyse, musubi-tuner Dataset-TOML & AI-Toolkit Export, dynamische Balancierung (`num_repeats`), Gesichts-/Personen-Dominanzwarnung. | **M** |
| **Stufe 6: Trainings-Tooling (Krea 2 RAW)** | ❌ Fehlt | Keine Presets für musubi-tuner oder AI-Toolkit vorhanden. Es fehlen: Konfigurationsgenerierung für Krea 2 RAW (Rank 128, Alpha 128, lr 5e-5, fp8, Gradient Checkpointing), kuratiertes 2–3k Subset-Preset, Runbook `docs/TRAINING.md`. | **M** |
| **Stufe 7: Validierung** | ❌ Fehlt | Keine Validierungsstufe. Es fehlen: Prompt-Grid-Generator aus Vokabular, Render-Skripte für Krea 2 RAW (52 Steps) vs. Turbo (8 Steps) über mehrere LoRA-Stärken (0.7, 0.85, 1.0), automatischer HTML/Image-Kontaktbogen. | **M** |
| **Stufe 8: Charakter-LoRAs** | ❌ Fehlt | Keine Unterstützung für sequentielles Charakter-Training. Es fehlen: Config-Generator mit gefrorener Stil-LoRA, Stufe-4-Wiederverwendung (`caption_mode: subject`), Stack-Validierungsgrid (Stil $\times$ Charakter Gewichte). | **S** |

*Aufwandsklassen: S (< 0.5 Tag), M (0.5 – 1.5 Tage), L (2 – 3 Tage).*

---

## 4. Detaillierte Analyse der einzelnen Stufen

### 4.1 Core Infrastructure & Manifest
- **Soll**: Ein zentrales, atomares Manifest (`data/manifest.jsonl` oder SQLite) speichert pro Bild: URI/Pfad, Source-URL, Hashes (MD5/SHA256, pHash), Metadaten (Seitentitel, Tags), ermittelte Auflösung, Status je Pipeline-Stufe (`pending`, `processed`, `rejected`), Reject-Gründe, Filter-Scores, VLM-JSON-Daten und finalen Caption-String. Sämtliche Konfiguration liegt in `config/pipeline.yaml`.
- **Ist**: Zustand wird implizit aus dem Dateisystem abgelesen (`.txt` existiert). Bei Abbruch während des Downloads oder Captionings bleiben verwaiste oder unvollständige Dateien zurück. Parameter werden über lange CLI-Argumentlisten übergeben.

### 4.2 Stufe 1: Crawling mit Vorfilter
- **Soll**: Ermittlung der Bildauflösung vor dem Herunterladen (z. B. aus `srcset`, HTML-Attributen oder HTTP-`HEAD`-Requests mit Range-Header für EXIF/Image-Header). Bilder unter 0,7 Megapixel werden gar nicht erst übertragen. Respektierung von `robots.txt`, Erkennung von TDM-Reservierungen (§ 44b UrhG, `tdm-reservation` Metatags / HTTP Header). Rate-Limiting pro Ziel-Domain.
- **Ist**: Es wird blind geladen. Die einzige Filterung erfolgt im Nachhinein über die Dateigröße in Bytes (`--min-size 15360`), wodurch bandbreiten- und zeitintensive Downloads kleiner/minderwertiger Bilder stattfinden. TDM-Signale und `robots.txt` werden ignoriert.

### 4.3 Stufe 2: Qualitätskontrolle
- **Soll**: 
  - Standardisierung: Konvertierung in sRGB, EXIF-Strip, Bereinigung defekter Streams.
  - Geometrie: Reale Mindestkantenlänge 1024 px.
  - Deduplizierung: pHash-Clustering mit Hamming-Distanz $\le 6$, optional semantisches Dedup (CLIP / DINOv2 Cosine Similarity $> 0.95$).
  - Filter: Laplacian-Varianz für Unschärfe, Ästhetik-Score-Prädiktor, Text-/Watermark-Detektor.
  - Transparenz: Rejects werden im Manifest mit Score protokolliert, Originale bleiben erhalten.
- **Ist**: Vollständig abwesend. Defekte, unscharfe, mit Wasserzeichen versehene oder doppelte Bilder wandern ungefiltert in das Trainingsset.

### 4.4 Stufe 3: Downscaling
- **Soll**: Strukturierte Transformation des gefilterten Bildbestands in ein dediziertes Verzeichnis (`data/processed/images/`). Max. lange Kante 2048 px, Kanten ganzzahlig gerundet auf Vielfache von 16 (zwingend erforderlich für DiT-Patch-Größen und VAE-Downsampling-Faktoren). Hochwertiges Lanczos-Resampling, sRGB-Farbprofil, konfigurierbare JPEG/PNG-Ausgabe.
- **Ist**: Existiert nur als Ad-hoc-Resize innerhalb von `caption_images.py` auf max. 1024 px für den VLM-Inferenz-Payload. Ein Vorbereitungsschritt für den Trainingsdatensatz fehlt.

### 4.5 Stufe 4: Captioning & Screening (Qwen3-VL)
- **Soll**:
  - Offline-Batch-Inferenz über die native vLLM-Engine (`from vllm import LLM, SamplingParams`), wodurch Durchsätze von mehreren hundert Bildern pro Minute auf GPUs wie A100/H100/L40S erreicht werden.
  - Guided Decoding via Pydantic/JSON-Schema (`guided_json`), gestützt durch Assistant-Prefill (`{`).
  - Vokabular-Bindung über `config/vocabulary.yaml` für reproduzierbare Kategorien (`style`, `location`, `pose`).
  - Zwingendes Screening: Ausschluss von Bildern mit `subject_age_estimate < 25`, `uncertain_age == True`, `has_watermark == True` oder `quality == low`.
  - Intelligente Retry-Logik mit Seed-Variation und Ausweich-Muster-Erkennung (Regex) sowie Fallback-Schnittstelle.
  - Caption-Assembly nach festem Schema unter Berücksichtigung des Modus (`style` vs. `subject`).
  - `--sample 200` Prüfbericht mit visuellen Thumbnails.
- **Ist**:
  - Nutzt synchrone HTTP-Requests gegen Ollama/vLLM REST API.
  - Freiform-Prompting mit Gemma/Qwen-VL ohne Schema-Erzwingung; JSON-Parsing basiert auf fehleranfälligen Regex-Suchmustern.
  - Keinerlei Screening auf Alter, Wasserzeichen oder Qualitätsmetriken.
  - Keine Vokabular-Dateien; feste Text-Templates im Code.

### 4.6 Stufe 5: Datensatz strukturieren & Export
- **Soll**:
  - Auswertung des Manifests über Kreuztabellen ($Style \times Location \times Pose$).
  - Automatische Berechnung von `num_repeats` zur Balancierung seltener Posen/Settings (bis Faktor 4).
  - Export von `dataset.toml` für `musubi-tuner` und Konfigurationen für `AI-Toolkit`.
  - Gesichts-/Personen-Clusteranalyse (Warnung bei Überrepräsentation einzelner Modelle).
- **Ist**: Keine strukturierte Exportlogik vorhanden.

### 4.7 Stufen 6–8: Training, Validierung & Charakter-LoRAs
- **Soll**:
  - Erzeugung von Krea 2 RAW LoRA-Konfigurationen (Rank 128, Alpha 128, lr 5e-5, fp8, Gradient Checkpointing, RMSNorm/Modulationstraining) sowie LoKr-Presets.
  - Runbook `docs/TRAINING.md` mit konkreten Cache- und Startbefehlen.
  - Validierungs-Tooling: Automatischer Render-Stack für RAW (52 Steps) vs. Turbo (8 Steps) mit Kontaktbogen-Generierung (HTML + Grid).
  - Sequentielles Charakter-Tooling mit gefrorener Stil-LoRA.
- **Ist**: Keinerlei Trainings-, Validierungs- oder Charakter-Tools im Codebase vorhanden.

---

## 5. Risiken und Altlasten im bestehenden Code

1. **Massives Deadlock- & Timeout-Risiko im Netzwerk-Stack**:
   - Die Crawler nutzen synchrone ThreadPool-Downloads. Bei CDN-Ratelimits oder blockierenden Keep-Alive-Sockets frieren Worker ein (wie in früheren Iterationen aufgetreten).
   - Die synchrone VLM-REST-Abfrage (`requests.post(..., timeout=180)`) führt bei großen Batches zu Timeouts und blockiert Threads vollständig.
2. **Fehlende Screening- und Compliance-Mechanismen**:
   - Es findet vor dem Download keine TDM-Prüfung (§ 44b UrhG) statt.
   - Noch gravierender: Es gibt keinerlei Alters- und Inhaltsfilterung (`subject_age_estimate`, `uncertain_age`). Für die Erstellung von LoRA-Trainingsdatensätzen im Sensual-/BDSM-Bereich ist das Fehlen eines harten, automatisierten Screening-Gates ein erhebliches rechtliches und ethisches Risiko.
3. **Zustandsverlust und fehlende Idempotenz**:
   - Da kein Manifest existiert, führt jeder abgebrochene Lauf zu inkonsistentem Datenbestand auf der Festplatte. Es ist nicht nachvollziehbar, ob ein Bild bereits gefiltert, downscaled oder fehlerhaft gecaptioned wurde.
4. **Fehlende Testabdeckung & GPU-Kopplung**:
   - Keine einzige Zeile Code ist durch Unit- oder Integrationstests abgesichert.
   - Zukünftiger vLLM-Code droht CPU-Testumgebungen zu brechen, wenn Abhängigkeiten nicht strikt durch Lazy-Imports und Mocks isoliert werden.

---

## 6. Offene Fragen an den Nutzer (Entscheidungsbedarf)

1. **Hardware-Umgebung für Stufe 4 (vLLM Offline-Batch)**:
   - Auf welcher Zielhardware soll die Offline-Batch-Inferenz von Qwen3-VL (~8B) laufen? (z. B. RunPod mit 1x RTX 4090 24GB, A100 80GB oder H100?).
   - *Hintergrund*: vLLM Offline-Batching lädt das Modell direkt in den VRAM des ausführenden Python-Prozesses. Auf lokalen Entwicklungsrechnern ohne Nvidia-GPU muss die Pipeline entweder auf ein CPU/Mock-Backend zurückgreifen oder auf die bestehende HTTP/API-Schnittstelle umschaltbar bleiben.
2. **Initiales Vokabular (`config/vocabulary.yaml`)**:
   - Liegen bereits konkrete Listen für Stile (z. B. `cinematic_noir`, `restrained_elegance`), Locations (z. B. `studio_void`, `dungeon`, `minimalist_loft`) und Posen/Praktiken vor, oder soll für Phase 2 ein initiales Basisvokabular anhand der bisherigen Crawler-Tags definiert werden?
3. **Abwärtskompatibilität bestehender Downloads**:
   - Müssen die bisher heruntergeladenen Bildordner (`downloads/`) mit ihren `genres.txt`-Dateien über ein Migrationsskript in das neue Manifest überführt werden, oder startet die Ziel-Pipeline auf einem sauberen Datenverzeichnis?
4. **Fallback-VLM-Modell**:
   - Welches Modell soll als Fallback dienen, falls Qwen3-VL nach zwei Versuchen scheitert? (Empfehlung: `JoyCaption-alpha-two` oder `Qwen2.5-VL-7B`). Soll dieser Fallback ebenfalls über vLLM geladen werden oder über eine externe API?
5. **Priorisierung Trainings-Framework**:
   - Liegt der Hauptfokus für die Trainingskonfigurationen primär auf **`musubi-tuner`** oder auf **`AI-Toolkit`**? (Beide unterstützen Krea 2, musubi-tuner bietet derzeit erweiterte Speicheroptimierungen wie `qfloat8` und LoKr).
