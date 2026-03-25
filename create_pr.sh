#!/bin/bash
#
# create_pr.sh - Automatisierte PR-Einreichung für Wettkampf/Ablation Machine
#
# Liest PR-Daten aus wettkampf/ Verzeichnis und führt den kompletten Git Workflow durch.
# Konform zu HERMES.md Commit-Message Format und Branch-Strategie.
#

set -euo pipefail

# Farben für Output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Konfiguration
WETTKAMPF_DIR="${WETTKAMPF_DIR:-./wettkampf}"
BASE_BRANCH="${BASE_BRANCH:-develop}"
REMOTE="${REMOTE:-origin}"

# Gültige Typen und Scopes (aus HERMES.md)
VALID_TYPES=("feat" "fix" "perf" "docs" "style" "refactor" "test" "chore")
VALID_SCOPES=("core" "models" "tokenizers" "quant" "train" "eval" "research" "orchestrator" "reports" "rust-core" "configs" "docs")

# Hilfsfunktionen
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

die() {
    log_error "$1"
    exit 1
}

# Initialisiere wettkampf Verzeichnis mit Template
init_wettkampf() {
    log_info "Initialisiere wettkampf Verzeichnis..."
    
    mkdir -p "$WETTKAMPF_DIR"
    
    cat > "$WETTKAMPF_DIR/pr.info" << 'EOF'
# PR Konfiguration
# Siehe HERMES.md für Details zu Typen und Scopes

# Commit-Type: feat, fix, perf, docs, style, refactor, test, chore
TYPE=feat

# Scope: core, models, tokenizers, quant, train, eval, research, orchestrator, reports, rust-core, configs, docs
# Oder leer lassen für no-scope
SCOPE=orchestrator

# Subject (max 72 Zeichen, Imperativ, kein Punkt am Ende)
# WICHTIG: Werte mit Leerzeichen müssen in Anführungszeichen stehen!
SUBJECT="add SweepRunner for parameter sweeps"

# Branch-Name (wird automatisch generiert falls leer)
# Format: {type}/{kurze-beschreibung}
BRANCH_NAME=""

# Phase (optional, für Footer)
PHASE=3

# Issue Referenz (optional, Format: #123)
FIXES=""

# Breaking Change? (true/false)
BREAKING=false
EOF

    cat > "$WETTKAMPF_DIR/pr.body" << 'EOF'
# Beschreibung

Implementiert die SweepRunner Komponente für parameter sweeps.
Nutzt itertools.product für O(1) memory complexity.

# Änderungen

- Neue Klasse `SweepRunner` im orchestrator Modul
- Unterstützung für konkurrente Ausführung
- Checkpointing für lange Sweeps

# Testing

- [x] Unit Tests hinzugefügt
- [x] Integration Tests bestanden
- [x] Performance Benchmarks durchgeführt

# Performance Impact

- 5x schneller für 1000+ Kombinationen
- Speicherverbrauch konstant bei O(1)
EOF

    log_success "Template erstellt in $WETTKAMPF_DIR/"
    log_info "Bitte pr.info und pr.body anpassen, dann Script erneut ausführen."
    exit 0
}

# Lese PR-Daten aus wettkampf/
load_pr_data() {
    local info_file="$WETTKAMPF_DIR/pr.info"
    local body_file="$WETTKAMPF_DIR/pr.body"
    
    if [[ ! -f "$info_file" ]]; then
        log_warn "pr.info nicht gefunden"
        init_wettkampf
    fi
    
    if [[ ! -f "$body_file" ]]; then
        die "pr.body nicht gefunden in $WETTKAMPF_DIR/"
    fi
    
    # Source die Konfiguration
    source "$info_file"
    
    # Validiere Type
    local valid_type=false
    for t in "${VALID_TYPES[@]}"; do
        if [[ "$t" == "$TYPE" ]]; then
            valid_type=true
            break
        fi
    done
    
    if [[ "$valid_type" != true ]]; then
        die "Ungültiger TYPE: $TYPE. Gültige Typen: ${VALID_TYPES[*]}"
    fi
    
    # Validiere Scope (optional)
    if [[ -n "${SCOPE:-}" ]]; then
        local valid_scope=false
        for s in "${VALID_SCOPES[@]}"; do
            if [[ "$s" == "$SCOPE" ]]; then
                valid_scope=true
                break
            fi
        done
        
        if [[ "$valid_scope" != true ]]; then
            log_warn "Scope '$SCOPE' nicht in Standard-Scopes. Fortfahren..."
        fi
    fi
    
    # Validiere Subject
    if [[ -z "${SUBJECT:-}" ]]; then
        die "SUBJECT darf nicht leer sein"
    fi
    
    if [[ "${#SUBJECT}" -gt 72 ]]; then
        log_warn "Subject ist länger als 72 Zeichen (${#SUBJECT})"
    fi
    
    # Generiere Branch-Name falls nicht angegeben
    if [[ -z "${BRANCH_NAME:-}" ]]; then
        local short_desc=$(echo "$SUBJECT" | tr '[:upper:]' '[:lower:]' | tr ' ' '-' | cut -c1-40)
        BRANCH_NAME="${TYPE}/${short_desc}"
    fi
    
    # Lese Body
    PR_BODY=$(cat "$body_file")
}

# Erstelle Commit-Message im konventionellen Format
generate_commit_message() {
    local message=""
    
    # Header: type(scope): subject
    if [[ -n "${SCOPE:-}" ]]; then
        message="${TYPE}(${SCOPE}): ${SUBJECT}"
    else
        message="${TYPE}: ${SUBJECT}"
    fi
    
    # Breaking Change Indicator
    if [[ "${BREAKING:-false}" == "true" ]]; then
        message="${message}!"
    fi
    
    message="${message}

${PR_BODY}"
    
    # Footer
    if [[ -n "${PHASE:-}" ]]; then
        message="${message}

Phase: ${PHASE}"
    fi
    
    if [[ -n "${FIXES:-}" ]]; then
        message="${message}
Fixes: ${FIXES}"
    fi
    
    if [[ "${BREAKING:-false}" == "true" ]]; then
        message="${message}

BREAKING CHANGE: Diese Änderung ist nicht rückwärtskompatibel."
    fi
    
    echo "$message"
}

# Git Workflow durchführen
run_git_workflow() {
    log_info "Starte Git Workflow..."
    
    # Prüfe ob wir in einem Git Repo sind
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        die "Kein Git Repository gefunden"
    fi
    
    # Prüfe auf uncommitted changes
    if ! git diff-index --quiet HEAD --; then
        log_warn "Uncommitted changes gefunden"
        read -p "Trotzdem fortfahren? (j/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Jj]$ ]]; then
            die "Abgebrochen"
        fi
    fi
    
    # Fetch vom Remote
    log_info "Fetch von $REMOTE..."
    git fetch "$REMOTE"
    
    # Checke Base Branch aus
    log_info "Checke $BASE_BRANCH aus..."
    git checkout "$BASE_BRANCH" || die "Konnte $BASE_BRANCH nicht auschecken"
    git pull "$REMOTE" "$BASE_BRANCH" || die "Konnte $BASE_BRANCH nicht updaten"
    
    # Erstelle Feature Branch
    log_info "Erstelle Branch: $BRANCH_NAME"
    if git show-ref --verify --quiet "refs/heads/$BRANCH_NAME"; then
        log_warn "Branch $BRANCH_NAME existiert bereits"
        read -p "Branch löschen und neu erstellen? (j/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Jj]$ ]]; then
            git branch -D "$BRANCH_NAME"
            git checkout -b "$BRANCH_NAME"
        else
            git checkout "$BRANCH_NAME"
        fi
    else
        git checkout -b "$BRANCH_NAME"
    fi
    
    log_success "Branch $BRANCH_NAME bereit"
}

# Commit erstellen
create_commit() {
    log_info "Erstelle Commit..."
    
    local commit_msg=$(generate_commit_message)
    
    # Speichere Commit-Message temporär
    local msg_file="$WETTKAMPF_DIR/.commit_msg"
    echo "$commit_msg" > "$msg_file"
    
    # Zeige Vorschau
    echo
    echo "=== Commit Message Preview ==="
    echo "$commit_msg"
    echo "=============================="
    echo
    
    # Stage changes
    log_info "Stage alle Änderungen..."
    git add -A
    
    # Commit mit Message-File
    git commit -F "$msg_file" || die "Commit fehlgeschlagen"
    
    rm -f "$msg_file"
    log_success "Commit erstellt"
}

# Push zum Remote
push_branch() {
    log_info "Push zu $REMOTE..."
    git push -u "$REMOTE" "$BRANCH_NAME" || die "Push fehlgeschlagen"
    log_success "Branch gepusht"
}

# PR erstellen (mit gh CLI falls verfügbar)
create_pr() {
    if ! command -v gh &> /dev/null; then
        log_warn "GitHub CLI (gh) nicht installiert. Manuelle PR-Erstellung erforderlich."
        return
    fi
    
    if ! gh auth status &> /dev/null; then
        log_warn "Nicht bei GitHub CLI authentifiziert"
        return
    fi
    
    log_info "Erstelle Pull Request..."
    
    local title="${TYPE}(${SCOPE}): ${SUBJECT}"
    if [[ "${BREAKING:-false}" == "true" ]]; then
        title="${title} [BREAKING]"
    fi
    
    # Erstelle PR
    if gh pr create \
        --title "$title" \
        --body "$PR_BODY" \
        --base "$BASE_BRANCH" \
        --head "$BRANCH_NAME"; then
        log_success "Pull Request erstellt"
    else
        log_error "PR Erstellung fehlgeschlagen"
    fi
}

# Cleanup-Funktion
cleanup() {
    # Entferne temporäre Dateien
    rm -f "$WETTKAMPF_DIR/.commit_msg"
}

trap cleanup EXIT

# Main
main() {
    echo "========================================"
    echo "  Wettkampf PR Creation Tool"
    echo "========================================"
    echo
    
    # Prüfe ob init gewünscht
    if [[ "${1:-}" == "init" ]]; then
        init_wettkampf
    fi
    
    # Lade PR-Daten
    load_pr_data
    
    log_info "PR Daten geladen:"
    echo "  Type:  $TYPE"
    echo "  Scope: ${SCOPE:-<none>}"
    echo "  Subject: $SUBJECT"
    echo "  Branch: $BRANCH_NAME"
    echo
    
    # Bestätigung
    read -p "Mit PR-Erstellung fortfahren? (j/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Jj]$ ]]; then
        die "Abgebrochen"
    fi
    
    # Führe Workflow aus
    run_git_workflow
    create_commit
    push_branch
    create_pr
    
    echo
    echo "========================================"
    log_success "PR Workflow abgeschlossen!"
    echo "========================================"
    echo
    echo "Branch: $BRANCH_NAME"
    echo "Base:   $BASE_BRANCH"
    echo
    echo "Nächste Schritte:"
    echo "  1. PR auf GitHub/GitLab reviewen"
    echo "  2. Nach approve: Merge nach $BASE_BRANCH"
    echo
}

# Hilfe anzeigen
show_help() {
    cat << 'EOF'
Usage: ./create_pr.sh [command]

Automatisierte PR-Einreichung für Wettkampf/Ablation Machine.

Commands:
  (none)    PR aus wettkampf/ Daten erstellen
  init      Template-Struktur in wettkampf/ initialisieren
  help      Diese Hilfe anzeigen

Konfiguration:
  WETTKAMPF_DIR   Pfad zu PR-Daten (default: ./wettkampf)
  BASE_BRANCH     Target Branch (default: develop)
  REMOTE          Git Remote (default: origin)

Dateien in wettkampf/:
  pr.info    - PR Metadaten (Type, Scope, Subject, etc.)
  pr.body    - PR Beschreibung (Markdown)

Beispiel:
  ./create_pr.sh init      # Template erstellen
  # Editiere wettkampf/pr.info und wettkampf/pr.body
  ./create_pr.sh           # PR erstellen

Siehe HERMES.md für Commit-Konventionen.
EOF
}

# Argument Parsing
case "${1:-}" in
    help|--help|-h)
        show_help
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac
