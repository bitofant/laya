#!/usr/bin/env bash
# End-to-end check against the running HTTP endpoint: health, device, and a real
# typed-decision pass through the published port. Exits non-zero on any failure.
#
# Deliberately hits HTTP rather than exec'ing python in the container: the server already
# holds the model on the GPU, so a second in-container load would double VRAM use and
# test a different code path than the one actually serving traffic.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require_docker
container_running || die "container '${CONTAINER_NAME}' is not running -- run ./start.sh first"

BASE="http://${LAYA_BIND:-127.0.0.1}:${LAYA_PORT:-8100}"

log "GET ${BASE}/health"
health="$(curl -fsS -m 15 "${BASE}/health" 2>&1)" \
  || die "health check failed -- container up but not serving. Try: docker logs ${CONTAINER_NAME}"

python3 - "$health" <<'PY' || exit 1
import json, sys
h = json.loads(sys.argv[1])
print(json.dumps(h, indent=2))
if h.get("status") != "ok":
    sys.exit(f"FAIL: status={h.get('status')!r}")
# The whole point of the GPU build; serving on CPU is a silent 10x regression.
if not str(h.get("device", "")).startswith("cuda"):
    sys.exit(f"FAIL: serving on {h.get('device')!r}, not cuda (see docs/gpu-notes.md)")
PY

log "POST ${BASE}/decide"
resp="$(curl -fsS -m 30 "${BASE}/decide" \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "Subject: URGENT - production API returning 500s\n\nOur checkout has been down for 20 minutes and we are losing orders. We are on the Enterprise plan. Please escalate immediately.",
    "questions": {
      "intent": {"type": "choice", "instructions": "What does the sender want?",
                 "criteria": {"support": "needs help with a broken product",
                              "sales": "wants to buy or upgrade",
                              "billing": "has a question about an invoice"}},
      "urgency": {"type": "score", "instructions": "How urgent is this message?",
                  "criteria": ["not urgent", "soon", "critical"]},
      "is_spam": {"type": "noul", "instructions": "Is this message spam?"}
    }
  }' 2>&1)" || die "POST /decide failed"

python3 - "$resp" <<'PY' || exit 1
import json, sys
r = json.loads(sys.argv[1])
print(json.dumps(r, indent=2))
a = r.get("answers", {})
missing = {"intent", "urgency", "is_spam"} - set(a)
if missing:
    sys.exit(f"FAIL: no answer for {sorted(missing)}")
# Sanity, not accuracy: this checkpoint should not read an outage as a sales enquiry.
if a["intent"]["choice"] != "support":
    print(f"WARN: expected intent=support, got {a['intent']['choice']!r}")
if a["urgency"]["score"] < 1.0:
    print(f"WARN: expected high urgency, got {a['urgency']['score']}")
print(f"\nlatency: {r.get('latency_ms')} ms for 3 questions")
PY

log "validating error handling"
code="$(curl -sS -o /dev/null -w '%{http_code}' -m 15 "${BASE}/decide" \
  -H 'Content-Type: application/json' \
  -d '{"state": "hi", "questions": {"q": {"type": "choice", "instructions": "x"}}}')"
[[ "$code" == "422" ]] || die "malformed request should return 422, got ${code}"

log "SMOKE TEST PASSED"
