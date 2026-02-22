#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: gpu_prover/profiling/hypercube_stage_ab.sh [options]

Runs A/B profiling using stage-breakdown profiling tests:
  - ops::hypercube::tests::profile_hypercube_bitrev_bf_stage_breakdown_multi_invocation_log24
  - ops::hypercube::tests::profile_hypercube_bitrev_bf_stage_breakdown_multi_invocation_log23

Options:
  --baseline <ref|WORKTREE>   Baseline git ref (default: WORKTREE)
  --candidate <ref|WORKTREE>  Candidate git ref (default: WORKTREE)
  --runs <n>                  Repetitions per log/test/context (default: 5)
  --log23-only                Profile only log23
  --log24-only                Profile only log24
  --help                      Show this message

Environment:
  PROFILE_RUSTFLAGS           Rust flags for build (default: -Awarnings)
  PROFILE_RUST_MIN_STACK      Rust compiler stack size (default: 16777216)

Notes:
  - Profiling jobs are run sequentially.
  - The script builds first, then profiles the produced test executable directly.
EOF
}

BASELINE_REF="WORKTREE"
CANDIDATE_REF="WORKTREE"
RUNS=5
RUN_LOG23=1
RUN_LOG24=1

while (($# > 0)); do
  case "$1" in
    --baseline)
      BASELINE_REF="$2"
      shift 2
      ;;
    --candidate)
      CANDIDATE_REF="$2"
      shift 2
      ;;
    --runs)
      RUNS="$2"
      shift 2
      ;;
    --log23-only)
      RUN_LOG24=0
      shift
      ;;
    --log24-only)
      RUN_LOG23=0
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "$RUNS" =~ [^0-9] ]] || ((RUNS < 1)); then
  echo "--runs must be a positive integer" >&2
  exit 1
fi

if ((RUN_LOG23 == 0 && RUN_LOG24 == 0)); then
  echo "nothing to run: both --log23-only and --log24-only were effectively disabled" >&2
  exit 1
fi

PROFILE_RUSTFLAGS="${PROFILE_RUSTFLAGS:--Awarnings}"
PROFILE_RUST_MIN_STACK="${PROFILE_RUST_MIN_STACK:-16777216}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
if [[ ! -d "$REPO_ROOT/gpu_prover" ]]; then
  echo "not inside zksync-airbender workspace" >&2
  exit 1
fi

TMP_ROOT="$(mktemp -d /tmp/h2m-stage-ab.XXXXXX)"
declare -a WORKTREES_TO_REMOVE=()

cleanup() {
  local wt
  for wt in "${WORKTREES_TO_REMOVE[@]}"; do
    git -C "$REPO_ROOT" worktree remove --force "$wt" >/dev/null 2>&1 || true
  done
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

echo "[info] repo: $REPO_ROOT"
echo "[info] baseline: $BASELINE_REF | candidate: $CANDIDATE_REF | runs: $RUNS"

setup_source_dir() {
  local label="$1"
  local ref="$2"
  local out_var="$3"
  local dir

  if [[ "$ref" == "WORKTREE" ]]; then
    dir="$REPO_ROOT"
    printf -v "$out_var" '%s' "$dir"
    return
  fi

  dir="$TMP_ROOT/$label"
  echo "[info] creating worktree for $label at $ref -> $dir"
  git -C "$REPO_ROOT" worktree add --detach "$dir" "$ref" >/dev/null
  WORKTREES_TO_REMOVE+=("$dir")
  printf -v "$out_var" '%s' "$dir"
}

build_test_binary() {
  local dir="$1"
  local label="$2"
  local binary

  echo "[info] building test binary for $label in $dir" >&2
  (
    cd "$dir"
    RUST_MIN_STACK="$PROFILE_RUST_MIN_STACK" \
    RUSTFLAGS="$PROFILE_RUSTFLAGS" \
      cargo test -p gpu_prover --no-run >/dev/null
  )

  binary="$(
    cd "$dir"
    find target/debug/deps -maxdepth 1 -type f -name 'gpu_prover-*' -executable -printf '%T@ %p\n' \
      | sort -nr \
      | head -n1 \
      | awk '{print $2}'
  )"
  if [[ -z "$binary" ]]; then
    echo "could not locate gpu_prover test binary in $dir/target/debug/deps" >&2
    exit 1
  fi
  printf '%s\n' "$dir/$binary"
}

extract_kv() {
  local line="$1"
  local key="$2"
  awk -v key="$key" '{
    for (i = 1; i <= NF; i++) {
      split($i, kv, "=");
      if (kv[1] == key) {
        print kv[2];
        exit;
      }
    }
  }' <<< "$line"
}

run_context_for_log() {
  local label="$1"
  local binary="$2"
  local test_name="$3"
  local runs="$4"
  local out_file="$5"
  local i output line

  : >"$out_file"
  for ((i = 1; i <= runs; i++)); do
    output="$("$binary" --ignored --exact "$test_name" --nocapture 2>&1)"
    line="$(rg '^profile_h2m_stage_breakdown ' <<< "$output" || true)"
    if [[ -z "$line" ]]; then
      echo "failed to capture profile_h2m_stage_breakdown from $label run=$i test=$test_name" >&2
      echo "$output" >&2
      exit 1
    fi

    local chain initial stage2 stage3 stage_sum gap
    chain="$(extract_kv "$line" "chain_median_us")"
    initial="$(extract_kv "$line" "initial_median_us")"
    stage2="$(extract_kv "$line" "stage2_median_us")"
    stage3="$(extract_kv "$line" "stage3_median_us")"
    stage_sum="$(extract_kv "$line" "stage_sum_median_us")"
    gap="$(extract_kv "$line" "median_gap_us")"

    printf "%d %s %s %s %s %s %s\n" \
      "$i" "$chain" "$initial" "$stage2" "$stage3" "$stage_sum" "$gap" >>"$out_file"

    echo "[run] $label test=$test_name run=$i chain=$chain initial=$initial stage2=$stage2 stage3=$stage3"
  done
}

calc_mean() {
  local file="$1"
  local col="$2"
  awk -v c="$col" '{s += $c; n += 1} END {if (n == 0) {print "nan"} else {printf "%.6f", s / n}}' "$file"
}

calc_median() {
  local file="$1"
  local col="$2"
  awk -v c="$col" '{print $c}' "$file" | sort -n | awk '
    {a[NR] = $1}
    END {
      if (NR == 0) {
        print "nan";
      } else if (NR % 2 == 1) {
        printf "%.6f", a[(NR + 1) / 2];
      } else {
        printf "%.6f", (a[NR / 2] + a[NR / 2 + 1]) / 2.0;
      }
    }'
}

print_comparison_table() {
  local title="$1"
  local base_file="$2"
  local cand_file="$3"

  local metrics=("chain" "initial" "stage2" "stage3" "stage_sum" "gap")
  local cols=(2 3 4 5 6 7)
  local idx metric col base_med cand_med base_mean cand_mean delta delta_pct

  echo
  echo "== $title =="
  printf "%-10s %12s %12s %12s %10s %12s %12s\n" \
    "metric" "base_med" "cand_med" "delta_us" "delta_%" "base_mean" "cand_mean"
  for idx in "${!metrics[@]}"; do
    metric="${metrics[$idx]}"
    col="${cols[$idx]}"
    base_med="$(calc_median "$base_file" "$col")"
    cand_med="$(calc_median "$cand_file" "$col")"
    base_mean="$(calc_mean "$base_file" "$col")"
    cand_mean="$(calc_mean "$cand_file" "$col")"
    delta="$(awk -v c="$cand_med" -v b="$base_med" 'BEGIN {printf "%.6f", c - b}')"
    delta_pct="$(awk -v d="$delta" -v b="$base_med" 'BEGIN {if (b == 0) printf "nan"; else printf "%.3f", (d / b) * 100.0}')"
    printf "%-10s %12.3f %12.3f %12.3f %10s %12.3f %12.3f\n" \
      "$metric" "$base_med" "$cand_med" "$delta" "$delta_pct" "$base_mean" "$cand_mean"
  done
}

BASE_DIR=""
CAND_DIR=""
setup_source_dir "baseline" "$BASELINE_REF" BASE_DIR
setup_source_dir "candidate" "$CANDIDATE_REF" CAND_DIR

BASE_BIN="$(build_test_binary "$BASE_DIR" "baseline")"
CAND_BIN="$(build_test_binary "$CAND_DIR" "candidate")"

declare -a TESTS=()
if ((RUN_LOG24 == 1)); then
  TESTS+=("24:ops::hypercube::tests::profile_hypercube_bitrev_bf_stage_breakdown_multi_invocation_log24")
fi
if ((RUN_LOG23 == 1)); then
  TESTS+=("23:ops::hypercube::tests::profile_hypercube_bitrev_bf_stage_breakdown_multi_invocation_log23")
fi

for spec in "${TESTS[@]}"; do
  log_id="${spec%%:*}"
  test_name="${spec#*:}"
  base_out="$TMP_ROOT/base_log${log_id}.txt"
  cand_out="$TMP_ROOT/cand_log${log_id}.txt"

  echo
  echo "[info] profiling log$log_id baseline"
  run_context_for_log "baseline" "$BASE_BIN" "$test_name" "$RUNS" "$base_out"

  echo
  echo "[info] profiling log$log_id candidate"
  run_context_for_log "candidate" "$CAND_BIN" "$test_name" "$RUNS" "$cand_out"

  print_comparison_table "log${log_id} (median-of-runs and mean-of-runs)" "$base_out" "$cand_out"
done
