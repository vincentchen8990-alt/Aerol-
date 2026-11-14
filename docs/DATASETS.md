# Datasets, access, and provenance

## The five provided inputs

User's four CSV catalogs and one Excel workbook reside unchanged in `data/catalogs/`. The four CSVs contain **metadata only**, not measurements from human participants.

| File | Contents, verified by parser |
|---|---|
| embraceplus_watch_signals.csv | 9 EmbracePlus raw signal definitions |
| embraceplus_watch_biomarkers.csv | 26 wearable-derived/report definitions |
| earth_human_dataset_sources.csv | 11 source entries and access notes |
| earth_human_metric_mapping.csv | 16 feature-to-data-source mappings |
| embraceplus_earth_human_data_catalog.xlsx | 8 original reference sheets (not needed for automated CSV parser) |

`python -m gravity_ai.data.catalog_loader --validate` computes SHA-256 hashes and writes a manifest to `data/manifests/catalog_validation.json`. The primary source registry is **read from these CSVs at runtime**: no embedded fake records.

## Earlier conversation research artifacts

Included under `reference/`:

- `dataset_manifest.csv`: 40 verified source/file catalog records (including measurements, source metadata, download status, reuse terms, sample count caveats).
- `data_dictionary.csv`: 102 provenance and variable mappings for motor-control, spacecraft and biological sources.
- `data_gaps_and_simulation_plan.md`: previous technical analysis, important missing data and experimental design.

`gravity_ai.data.prior_research.coverage_summary()` loads these artifacts and they are included in the test suite. The separate archive of original downloaded study samples is **not** included; those sources remain reference metadata unless the user retrieves their raw files.

## Actual human datasets: public vs restricted

| User catalog ID | Source | Use / access |
|---|---|---|
| E4-STRESS-2025 | https://physionet.org/content/wearable-device-dataset/1.0.1/ | Open, real adults; E4 wrist signals, sample files HR/EDA/TEMP/ACC, 1Hz/4Hz/4Hz/32Hz source-specific |
| NHANES2013 | https://wwwn.cdc.gov/nchs/nhanes/ | Open; population cross-sectional blood pressure, DXA BMD and activity; join *within NHANES* by SEQN |
| MMASH | https://physionet.org/content/mmash/1.0.0/ | Open, 22 participants; 24-hour physiology and sleep measurements; different device |
| WESAD | https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection | Public source, larger archive; stress/affect study, not centrifuge study |
| BIDMC | https://physionet.org/content/bidmc/1.0.0/ | Open clinical patient waveforms; **not** healthy astronaut population |
| PPG-DALIA | https://archive.ics.uci.edu/dataset/495/ppg-dalia | Open research data, large archive; PPG-based physiological signals |
| EMP-SSP | https://www.simonsfoundation.org/sfari/resources/autism-cohorts/spark/simons-sleep-project/ | Research approval required; pediatric EmbracePlus recordings |
| BALLADEER | https://springernature.figshare.com/articles/dataset/A_Multimodal_Dataset_for_Neurophysiological_and_AI_Applications/28676042 | Public original dataset, large; pediatric/teen attention-task context |
| E4XEMP | https://doi.org/10.3758/s13428-026-03097-8 | Limited/raw access unverified; device pairing |
| OPENOX | https://physionet.org/content/openox-repo/1.1.1/ | Check files/access before using |
| NASA-OSDR | https://science.nasa.gov/biological-physical/data/osdr/ | Study-specific, mixed accessibility; many biological studies not direct astronaut wearables |

## Obtain the raw records from an Internet-enabled machine

```bash
# Recommended pilot: bounded E4 STRESS sessions plus CDC NHANES XPT files
python -m gravity_ai.data.download --from-catalog --public-only --dataset pilot

# Alternatively, select one source ID from the uploaded catalog
python -m gravity_ai.data.download --from-catalog --public-only --dataset E4-STRESS-2025
python -m gravity_ai.data.download --from-catalog --public-only --dataset NHANES2013
python -m gravity_ai.data.download --from-catalog --public-only --dataset MMASH
```

The downloader uses official HTTPS endpoints for **recognized user-catalog IDs** and skips landing pages, controlled-access sources and oversized archives rather than pretending to download data. It writes `data/manifests/download_report.json` with retrieved hashes and access failures. The default E4 pilot intentionally limits source sessions to avoid an unnecessary bulk download. Repeat with a larger subject set before drawing conclusions. Be considerate of source hosts' usage policies.

If you download manually, put original files in:

- `data/raw/empatica_e4/STRESS/S01/HR.csv`, plus the other E4 stream files per session.
- `data/raw/nhanes/DEMO_H.XPT`, `BPX_H.XPT`, `DXXFEM_H.XPT`, `DXXSPN_H.XPT` (additional activity files optional).
- `data/raw/mmash/MMASH.zip` (archived for future MMASH-specific importer).
- `data/raw/ced/DATAPRBS.csv` and `DATAUNIF.csv` if using the earlier real motor benchmark.

Run `python -m gravity_ai.data.prepare --dataset all`. Prepared E4 windows and NHANES participant-level observations are written to separate folders. No alignment across unrelated participants/studies is performed.

The E4 preprocessing uses timestamp-aligned fixed windows and preserves `subject_id` and `session_id`; feature samples are not independent participant records. Train/test splitting groups by subject.

The NHANES loader keeps original CDC field names and selects bona fide DXA `DXX*BMD` variables. It does not infer time-dependent bone recovery from cross-sectional records.

### Not yet implemented as a full importer

Large WESAD, PPG-DaLiA and BALLADEER archives and MMASH multi-file raw structures have source registry entries and downloadable-source metadata but are **not** integrated into the first vertical-slice physiological learner. Their structures, licenses and cohort characteristics require dedicated adapters and validation before claiming multi-dataset training.

## NASA / control data from earlier conversation

| Source | Scientific role | Limit |
|---|---|---|
| NASA OSD-595 / OSD-596 | Fruit-fly gravity-response and behavior experiments | Not a human bone-density or motor-torque dataset |
| OSD-686 | Mouse artificial-gravity/biological response metadata | Molecular assays not equivalent to astronaut wearable physiology |
| AGBRESA supplements `pone.0239228.s001–s004` | Human centrifuge questionnaires, vital signs, motion sickness | Ground centrifuge exposure differs from spacecraft spinning ring |
| PIMS MAMS/SAMS | ISS acceleration signals | Different structural/environmental context |
| Rosetta `RO-X-HK-3-RWL-V1.0` | Real reaction-wheel speed and estimated friction torque | Friction torque is **not commanded motor torque** |
| Coupled Electric Drives `diva2:1165531` | Five measured motor voltage–response experiments | Not spacecraft torques; sensor speed uses dataset units, don't relabel rpm |
| Basilisk spinningBodyOneDOF | Coupled hub and rotating-component simulation | Software, not experimental health data |

See `reference/dataset_manifest.csv` and `reference/data_dictionary.csv` for source URLs, exact filenames, record counts, and verification status.
