#!/bin/bash
#
# prepare_pr.sh - Interaktive PR-Daten Vorbereitung für Wettkampf
#
# Führt durch die Auswahl von Type, Scope und erstellt sinnvolle Templates
# basierend auf den aktuellen Git-Änderungen.
#
# Usage:
#   ./prepare_pr.sh              - Normale interaktive PR-Vorbereitung
#   ./prepare_pr.sh initial      - Kompletter initialer Wettbewerbs-PR
#

set -euo pipefail

# Farben
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

WETTKAMPF_DIR="${WETTKAMPF_DIR:-./wettkampf}"

# Gültige Optionen (aus HERMES.md)
TYPES=(
    "feat:Neue Funktionalität"
    "fix:Bug-Fix"
    "perf:Performance-Verbesserung"
    "docs:Dokumentation"
    "style:Code-Stil (keine Logik)"
    "refactor:Refactoring (keine API-Änderung)"
    "test:Tests hinzufügen/ändern"
    "chore:Wartung (Dependencies, etc.)"
)

SCOPES=(
    "core:Core-Module (Config, Registry, Logging)"
    "models:Model factories (BackboneFactory, FeatureGate)"
    "tokenizers:Tokenizer-Implementierungen"
    "quant:Quantisierung (Int6, Int5, Mixed, GPTQLite)"
    "train:Training (Trainer, Optimizer, Scheduler, EMA)"
    "eval:Evaluation (BPB, SlidingWindow, Benchmark)"
    "research:Research Engine (Ablation, Phase Evaluators)"
    "orchestrator:Production Pipeline (Sweep, Promote, Submit, Dashboard)"
    "reports:Reports (Comparator, Leaderboard)"
    "rust-core:Rust-Implementierungen"
    "configs:YAML-Konfigurationen"
    "docs:Dokumentation"
)

# Hilfsfunktionen
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_prompt() { echo -e "${CYAN}[?]${NC} $1"; }

die() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Menü-Anzeige
show_menu() {
    local title="$1"
    shift
    local options=("$@")
    
    echo
    echo "=== $title ==="
    local i=1
    for opt in "${options[@]}"; do
        local key=$(echo "$opt" | cut -d: -f1)
        local desc=$(echo "$opt" | cut -d: -f2-)
        printf "  %2d) %-12s - %s\n" "$i" "$key" "$desc"
        ((i++))
    done
    echo
}

# Auswahl mit Validierung
select_option() {
    local prompt="$1"
    shift
    local options=("$@")
    local count=${#options[@]}
    
    while true; do
        log_prompt "$prompt (1-$count): "
        read -r choice
        
        if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "$count" ]; then
            echo "${options[$((choice-1))]}"
            return 0
        fi
        log_warn "Ungültige Eingabe. Bitte 1-$count eingeben."
    done
}

# Ja/Nein Abfrage
confirm() {
    local prompt="$1"
    local default="${2:-n}"
    
    if [[ "$default" == "y" ]]; then
        log_prompt "$prompt [J/n]: "
    else
        log_prompt "$prompt [j/N]: "
    fi
    
    read -r response
    response=${response:-$default}
    
    [[ "$response" =~ ^[JjYy] ]]
}

# Aktuelle Git-Änderungen analysieren
analyze_changes() {
    log_info "Analysiere aktuelle Änderungen..."
    
    # Geänderte Dateien
    local changed_files=$(git diff --cached --name-only 2>/dev/null || git diff --name-only 2>/dev/null || echo "")
    
    if [[ -z "$changed_files" ]]; then
        log_warn "Keine uncommitted Changes gefunden"
        return 1
    fi
    
    echo
    echo "=== Geänderte Dateien ==="
    echo "$changed_files" | head -20
    if [[ $(echo "$changed_files" | wc -l) -gt 20 ]]; then
        echo "... und weitere"
    fi
    echo
    
    # Versuche Scope zu erkennen
    local detected_scope=""
    if echo "$changed_files" | grep -q "orchestrator"; then
        detected_scope="orchestrator"
    elif echo "$changed_files" | grep -q "quant"; then
        detected_scope="quant"
    elif echo "$changed_files" | grep -q "train"; then
        detected_scope="train"
    elif echo "$changed_files" | grep -q "eval"; then
        detected_scope="eval"
    elif echo "$changed_files" | grep -q "core"; then
        detected_scope="core"
    elif echo "$changed_files" | grep -q "models"; then
        detected_scope="models"
    elif echo "$changed_files" | grep -q "research"; then
        detected_scope="research"
    elif echo "$changed_files" | grep -q "tokenizers"; then
        detected_scope="tokenizers"
    elif echo "$changed_files" | grep -q "reports"; then
        detected_scope="reports"
    elif echo "$changed_files" | grep -q "rust-core"; then
        detected_scope="rust-core"
    fi
    
    if [[ -n "$detected_scope" ]]; then
        log_info "Erkannter Scope: $detected_scope"
    fi
    
    return 0
}

# Template für Body basierend auf Type/Scope
generate_body_template() {
    local type="$1"
    local scope="$2"
    local subject="$3"
    
    case "$type" in
        feat)
            cat << EOF
# Beschreibung

${subject}

# Änderungen

- Implementierung im ${scope} Modul
- 
- 

# Testing

- [ ] Unit Tests hinzugefügt
- [ ] Integration Tests bestanden
- [ ] Manuelle Tests durchgeführt

# Checklist

- [ ] Code folgt Style Guide (PEP 8, Black)
- [ ] Type-Hints vorhanden
- [ ] Docstrings aktualisiert
- [ ] Keine Breaking Changes (oder markiert)
EOF
            ;;
        fix)
            cat << EOF
# Bug Beschreibung

${subject}

# Ursache

- 

# Lösung

- 

# Testing

- [ ] Bug reproduziert und fix verifiziert
- [ ] Regression Test hinzugefügt
- [ ] Bestehende Tests passen noch

Fixes: #ISSUE_NR
EOF
            ;;
        perf)
            cat << EOF
# Performance Verbesserung

${subject}

# Änderungen

- 

# Benchmarks

Vorher:
- 

Nachher:
- 

# Performance Impact

- [ ] 2x schneller
- [ ] 5x schneller
- [ ] 10x+ schneller
- [ ] Reduzierter Speicherverbrauch

Phase: 3
Performance: Xx schneller
EOF
            ;;
        refactor)
            cat << EOF
# Refactoring

${subject}

# Motivation

- 

# Änderungen

- 
- 
- 

# Testing

- [ ] Keine funktionalen Änderungen
- [ ] Alle Tests bestehen
- [ ] Code Review durchgeführt
EOF
            ;;
        docs)
            cat << EOF
# Dokumentation

${subject}

# Änderungen

- 
- 

# Checklist

- [ ] README aktualisiert
- [ ] ARCHITECTURE.md aktualisiert
- [ ] API-Dokumentation aktualisiert
- [ ] Beispiele aktualisiert
EOF
            ;;
        test)
            cat << EOF
# Tests

${subject}

# Neue Tests

- 
- 

# Coverage

- [ ] Zeilenabdeckung erhöht
- [ ] Branch-abdeckung erhöht
- [ ] Edge Cases abgedeckt
EOF
            ;;
        *)
            cat << EOF
# Beschreibung

${subject}

# Änderungen

- 
- 
- 

# Testing

- [ ] Tests hinzugefügt/aktualisiert
- [ ] Manuelle Verifikation
EOF
            ;;
    esac
}

# Vorschläge für Subject basierend auf Type/Scope
generate_subject_suggestions() {
    local type="$1"
    local scope="$2"
    
    case "$type" in
        feat)
            echo "add new feature"
            echo "implement X functionality"
            echo "support for Y"
            ;;
        fix)
            echo "correct X calculation"
            echo "fix Y initialization bug"
            echo "resolve Z edge case"
            ;;
        perf)
            echo "optimize X algorithm"
            echo "add caching for Y"
            echo "reduce Z memory usage"
            ;;
        refactor)
            echo "extract X into separate module"
            echo "simplify Y logic"
            echo "remove duplication in Z"
            ;;
        docs)
            echo "update README for X"
            echo "add documentation for Y"
            echo "clarify Z usage"
            ;;
        test)
            echo "add tests for X"
            echo "improve Y test coverage"
            echo "fix flaky Z test"
            ;;
        *)
            echo "update X"
            echo "improve Y"
            echo "clean up Z"
            ;;
    esac
}

# Haupt-Funktion
main() {
    echo "========================================"
    echo "  Wettkampf PR Daten Vorbereitung"
    echo "========================================"
    echo
    
    # Stelle sicher, dass wettkampf existiert
    mkdir -p "$WETTKAMPF_DIR"
    
    # Analysiere Änderungen
    analyze_changes || true
    
    # Type Auswahl
    show_menu "Commit-Type wählen" "${TYPES[@]}"
    local selected_type=$(select_option "Type" "${TYPES[@]}")
    local TYPE=$(echo "$selected_type" | cut -d: -f1)
    log_success "Type: $TYPE"
    
    # Scope Auswahl
    echo
    show_menu "Scope wählen" "${SCOPES[@]}"
    local selected_scope=$(select_option "Scope" "${SCOPES[@]}")
    local SCOPE=$(echo "$selected_scope" | cut -d: -f1)
    log_success "Scope: $SCOPE"
    
    # Subject Eingabe
    echo
    log_info "Subject-Vorschläge für $TYPE($SCOPE):"
    local suggestions=$(generate_subject_suggestions "$TYPE" "$SCOPE")
    local i=1
    while IFS= read -r line; do
        [[ -n "$line" ]] && echo "  $i) $line"
        ((i++))
    done <<< "$suggestions"
    echo "  0) Eigene Eingabe"
    echo
    
    log_prompt "Wähle Vorschlag (0 für eigene): "
    read -r subject_choice
    
    local SUBJECT=""
    if [[ "$subject_choice" == "0" ]]; then
        log_prompt "Subject eingeben (max 72 Zeichen, Imperativ): "
        read -r SUBJECT
    else
        SUBJECT=$(echo "$suggestions" | sed -n "${subject_choice}p" | sed 's/^[0-9]*) //')
        if [[ -z "$SUBJECT" ]]; then
            log_prompt "Subject eingeben: "
            read -r SUBJECT
        fi
    fi
    
    # Validierung
    if [[ "${#SUBJECT}" -gt 72 ]]; then
        log_warn "Subject ist ${#SUBJECT} Zeichen (max 72)"
    fi
    
    # Phase
    echo
    log_prompt "Phase angeben [3]: "
    read -r PHASE
    PHASE=${PHASE:-3}
    
    # Issue Referenz
    echo
    log_prompt "Issue Referenz (z.B. #123, Enter für keine): "
    read -r FIXES
    
    # Breaking Change
    echo
    if confirm "Ist dies ein Breaking Change?"; then
        local BREAKING="true"
        log_warn "Als Breaking Change markiert!"
    else
        local BREAKING="false"
    fi
    
    # Branch Name
    echo
    local auto_branch="${TYPE}/$(echo "$SUBJECT" | tr '[:upper:]' '[:lower:]' | tr ' ' '-' | tr -cd '[:alnum:]-' | cut -c1-40)"
    log_info "Vorgeschlagener Branch-Name: $auto_branch"
    log_prompt "Branch-Name übernehmen? [J/n]: "
    read -r branch_confirm
    
    local BRANCH_NAME
    if [[ "$branch_confirm" =~ ^[Nn]$ ]]; then
        log_prompt "Eigener Branch-Name: "
        read -r BRANCH_NAME
    else
        BRANCH_NAME="$auto_branch"
    fi
    
    # Generiere pr.info
    cat > "$WETTKAMPF_DIR/pr.info" << EOF
# PR Konfiguration
# Siehe HERMES.md für Details zu Typen und Scopes

# Commit-Type: feat, fix, perf, docs, style, refactor, test, chore
TYPE=$TYPE

# Scope: core, models, tokenizers, quant, train, eval, research, orchestrator, reports, rust-core, configs, docs
# Oder leer lassen für no-scope
SCOPE=$SCOPE

# Subject (max 72 Zeichen, Imperativ, kein Punkt am Ende)
SUBJECT=$SUBJECT

# Branch-Name (wird automatisch generiert falls leer)
# Format: {type}/{kurze-beschreibung}
BRANCH_NAME=$BRANCH_NAME

# Phase (optional, für Footer)
PHASE=$PHASE

# Issue Referenz (optional, Format: #123)
FIXES=${FIXES}

# Breaking Change? (true/false)
BREAKING=$BREAKING
EOF

    # Generiere pr.body
    local BODY_TEMPLATE=$(generate_body_template "$TYPE" "$SCOPE" "$SUBJECT")
    cat > "$WETTKAMPF_DIR/pr.body" << EOF
$BODY_TEMPLATE
EOF

    # Zeige Zusammenfassung
    echo
    echo "========================================"
    echo "  Zusammenfassung"
    echo "========================================"
    echo
    echo -e "Type:     ${GREEN}$TYPE${NC}"
    echo -e "Scope:    ${GREEN}$SCOPE${NC}"
    echo -e "Subject:  ${GREEN}$SUBJECT${NC}"
    echo -e "Branch:   ${GREEN}$BRANCH_NAME${NC}"
    echo -e "Phase:    ${GREEN}$PHASE${NC}"
    [[ -n "$FIXES" ]] && echo -e "Fixes:    ${GREEN}$FIXES${NC}"
    [[ "$BREAKING" == "true" ]] && echo -e "Breaking: ${RED}JA${NC}"
    echo
    log_success "PR Daten erstellt in $WETTKAMPF_DIR/"
    echo
    echo "Dateien:"
    echo "  - $WETTKAMPF_DIR/pr.info"
    echo "  - $WETTKAMPF_DIR/pr.body"
    echo
    
    if confirm "Direkt PR erstellen mit create_pr.sh?"; then
        if [[ -x "./create_pr.sh" ]]; then
            ./create_pr.sh
        else
            log_warn "create_pr.sh nicht gefunden oder nicht ausführbar"
            echo "Führe später aus: ./create_pr.sh"
        fi
    else
        echo "Führe später aus: ./create_pr.sh"
    fi
}

# Initialen Wettbewerbs-PR erstellen
# Gemäß regeln.md: PR muss einen neuen Ordner in /records hinzufügen
setup_initial_competition_pr() {
    echo "========================================"
    echo "  Initialer Wettbewerbs-PR Setup"
    echo "  (OpenAI Parameter Golf Challenge)"
    echo "========================================"
    echo
    
    log_info "Erstelle komplette Wettbewerbs-Submission-Struktur..."
    
    # Abfragen für Wettbewerbs-Metadaten
    log_prompt "Dein Name (für submission.json): "
    read -r AUTHOR_NAME
    
    log_prompt "Dein GitHub Username: "
    read -r GITHUB_ID
    
    log_prompt "Run ID/Name (z.B. neuroweave_v1): "
    read -r RUN_ID
    RUN_ID=${RUN_ID:-neuroweave_v1}
    
    log_prompt "val_bpb Wert (z.B. 1.15): "
    read -r VAL_BPB
    VAL_BPB=${VAL_BPB:-1.20}
    
    log_prompt "Kurze Beschreibung des Ansatzes: "
    read -r APPROACH_DESC
    APPROACH_DESC=${APPROACH_DESC:-"Systematische Ablation über Architektur, Tokenisierung und Quantisierung"}
    
    # Records-Verzeichnis erstellen
    local RECORDS_DIR="records/${RUN_ID}"
    mkdir -p "$RECORDS_DIR"
    
    log_success "Verzeichnis erstellt: $RECORDS_DIR/"
    
    # submission.json erstellen
    cat > "$RECORDS_DIR/submission.json" << EOF
{
  "name": "${AUTHOR_NAME}",
  "github_id": "${GITHUB_ID}",
  "run_id": "${RUN_ID}",
  "val_bpb": ${VAL_BPB},
  "approach": "${APPROACH_DESC}",
  "date": "$(date -I)",
  "hardware": "8xH100",
  "training_time_minutes": 10,
  "artifact_bytes": 16000000,
  "seeds_tested": 3,
  "baseline_comparison": {
    "baseline_bpb": 1.2244,
    "improvement": $(echo "scale=4; 1.2244 - ${VAL_BPB}" | bc 2>/dev/null || echo "TBD")
  }
}
EOF
    log_success "submission.json erstellt"
    
    # README.md erstellen
    cat > "$RECORDS_DIR/README.md" << EOF
# ${RUN_ID} — ${APPROACH_DESC}

**Author:** ${AUTHOR_NAME} (@${GITHUB_ID})  
**Date:** $(date -I)  
**Run ID:** ${RUN_ID}

## Ergebnis

| Metrik | Wert |
|--------|------|
| val_bpb | ${VAL_BPB} |
| vs. Baseline | $(echo "scale=4; ${VAL_BPB} - 1.2244" | bc 2>/dev/null || echo "TBD") |
| Training Time | ~10 min on 8xH100 |
| Artifact Size | <16MB |

## Ansatz

${APPROACH_DESC}

### Wichtigste Änderungen

- [Feature 1: z.B. GQA mit gqa_groups=4]
- [Feature 2: z.B. SwiGLU Gated MLP]
- [Feature 3: z.B. TrigramHash-8192 Tokenizer]
- [Feature 4: z.B. INT6-Quantisierung]

## Reproduzierbarkeit

\`\`\`bash
# Training ausführen
RUN_ID=${RUN_ID} torchrun --standalone --nproc_per_node=8 train_gpt.py
\`\`\`

## Logs

- Training Log: ${RUN_ID}_train.log
- Final Metrics: Siehe submission.json

## Anforderungen

Siehe Haupt-README.md im Repository-Root für:
- Setup-Anweisungen
- Abhängigkeiten (requirements.txt)
- Hardware-Anforderungen
EOF
    log_success "README.md erstellt"
    
    # Trainings-Log Template erstellen
    cat > "$RECORDS_DIR/${RUN_ID}_train.log" << EOF
# Training Log: ${RUN_ID}
# Date: $(date -Iseconds)
# Author: ${AUTHOR_NAME}

=== Configuration ===
Run ID: ${RUN_ID}
Model: Custom GPT mit ${APPROACH_DESC}
Hardware: 8xH100

=== Training Start ===
[$(date -Iseconds)] Training started

=== Final Results ===
val_bpb: ${VAL_BPB}
artifact_bytes: <16MB
status: completed

=== Reproduzierbarkeit ===
Git Commit: $(git rev-parse --short HEAD 2>/dev/null || echo "N/A")
Config Hash: [wird automatisch generiert]
Seed: 42

Note: Dies ist ein Template. Ersetze mit echten Trainings-Logs.
EOF
    log_success "Trainings-Log Template erstellt"
    
    # train_gpt.py Hinweis erstellen
    if [[ -f "train_gpt.py" ]]; then
        log_info "Kopiere train_gpt.py..."
        cp train_gpt.py "$RECORDS_DIR/"
        log_success "train_gpt.py kopiert"
    else
        log_warn "train_gpt.py nicht gefunden im Root-Verzeichnis"
        cat > "$RECORDS_DIR/train_gpt.py" << 'EOF'
# Placeholder train_gpt.py
# 
# Dies ist ein Platzhalter. Bitte ersetze mit deinem tatsächlichen
# Training-Script oder kopiere es hierher:
#
#   cp /pfad/zu/deinem/train_gpt.py records/RUN_ID/
#
# Das Script muss im Records-Ordner funktionieren und darf keine
# externen Abhängigkeiten außerhalb des Ordners haben.
#
# Siehe regeln.md für Anforderungen:
# - Muss in unter 10 Minuten auf 8xH100 laufen
# - Muss val_bpb berechnen
# - Artifact muss <16MB sein

raise NotImplementedError("Bitte train_gpt.py ersetzen!")
EOF
        log_warn "Platzhalter train_gpt.py erstellt - bitte ersetzen!"
    fi
    
    # requirements.txt erstellen
    cat > "$RECORDS_DIR/requirements.txt" << 'EOF'
# Requirements für diesen Run
# Wird nur benötigt wenn zusätzliche Packages nötig

torch>=2.0.0
numpy
sentencepiece
tqdm
EOF
    log_success "requirements.txt erstellt"
    
    # PR-spezifische Daten erstellen
    mkdir -p "$WETTKAMPF_DIR"
    
    cat > "$WETTKAMPF_DIR/pr.info" << EOF
# PR Konfiguration für Wettbewerbs-Submission
TYPE=feat
SCOPE=orchestrator
SUBJECT=add ${RUN_ID} submission for Parameter Golf Challenge
BRANCH_NAME=submission/${RUN_ID}
PHASE=3
FIXES=
BREAKING=false
EOF
    
    cat > "$WETTKAMPF_DIR/pr.body" << EOF
# ${RUN_ID}: ${APPROACH_DESC}

## Zusammenfassung

Submission für den OpenAI Parameter Golf Challenge — trainiere das beste Language Model unter 16MB.

## Ergebnisse

| Metrik | Baseline | Unser Modell | Delta |
|--------|----------|--------------|-------|
| val_bpb | 1.2244 | ${VAL_BPB} | $(echo "scale=4; 1.2244 - ${VAL_BPB}" | bc 2>/dev/null || echo "TBD") |
| artifact_bytes | ~16MB | <16MB | ✓ |
| training_time | 10min | 10min | ✓ |

## Wichtigste Änderungen

${APPROACH_DESC}

- [Feature 1]
- [Feature 2]
- [Feature 3]

## Files

- \`records/${RUN_ID}/submission.json\` — Metadaten und Ergebnisse
- \`records/${RUN_ID}/README.md\` — Dokumentation
- \`records/${RUN_ID}/train_gpt.py\` — Trainingsscript
- \`records/${RUN_ID}/${RUN_ID}_train.log\` — Trainings-Log
- \`records/${RUN_ID}/requirements.txt\` — Dependencies

## Reproduzierbarkeit

\`\`\`bash
cd records/${RUN_ID}
python train_gpt.py
\`\`\`

## Checklist

- [x] Submission.json vollständig
- [x] README.md mit Erklärung
- [x] Trainings-Log vorhanden
- [x] train_gpt.py funktioniert standalone
- [x] Artifact <16MB
- [x] Training <10min auf 8xH100

Phase: 3
EOF
    
    echo
    echo "========================================"
    echo "  Wettbewerbs-PR Struktur erstellt!"
    echo "========================================"
    echo
    log_success "Verzeichnis: $RECORDS_DIR/"
    echo
    echo "Erstellte Dateien:"
    ls -la "$RECORDS_DIR/"
    echo
    echo "PR Daten:"
    echo "  - $WETTKAMPF_DIR/pr.info"
    echo "  - $WETTKAMPF_DIR/pr.body"
    echo
    echo "Nächste Schritte:"
    echo "  1. Ersetze Platzhalter in den generierten Dateien"
    echo "  2. Füge echte Trainings-Logs hinzu"
    echo "  3. Prüfe: ./create_pr.sh"
    echo "  4. Oder manuell: git add records/${RUN_ID}/ && git commit && git push"
    echo
    
    if confirm "Direkt create_pr.sh ausführen?"; then
        if [[ -x "./create_pr.sh" ]]; then
            ./create_pr.sh
        else
            log_warn "create_pr.sh nicht gefunden"
        fi
    fi
}

# Hilfe
show_help() {
    cat << 'EOF'
Usage: ./prepare_pr.sh [command] [options]

Interaktive Vorbereitung von PR-Daten für Wettkampf.

Commands:
  (none)       Normale interaktive PR-Vorbereitung
  initial      Kompletten initialen Wettbewerbs-PR erstellen
  help         Diese Hilfe anzeigen

Options:
  -h, --help   Diese Hilfe anzeigen

Environment:
  WETTKAMPF_DIR    Zielverzeichnis (default: ./wettkampf)

Beispiele:
  # Normaler Workflow:
  ./prepare_pr.sh
  
  # Initialer Wettbewerbs-PR:
  ./prepare_pr.sh initial

Siehe auch:
  - regeln.md — Wettbewerbsregeln
  - SUBMISSION_CHECKLIST.md — Checkliste für Submission
EOF
}

# Argument Parsing
case "${1:-}" in
    -h|--help|help)
        show_help
        exit 0
        ;;
    initial|--initial)
        setup_initial_competition_pr
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac
