package routing

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"
)

// scriptedHook returns a Hook whose transport is a function, so the protocol
// and every refusal can be driven exactly. The real transport is exercised
// separately against a compiled process in hook_process_test.go.
func scriptedHook(t *testing.T, reply func(Request) ([]byte, error)) *Hook {
	t.Helper()
	return &Hook{
		Path: "scripted",
		run: func(_ context.Context, _ string, _ []string, stdin []byte) ([]byte, error) {
			var req Request
			if err := json.Unmarshal(stdin, &req); err != nil {
				t.Fatalf("hook received unreadable request %q: %v", stdin, err)
			}
			return reply(req)
		},
	}
}

func jsonReply(t *testing.T, resp Response) []byte {
	t.Helper()
	payload, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("marshal response: %v", err)
	}
	return payload
}

func acquireRequest() Request {
	return Request{Assignment: "run-1:1", Owner: "daemon", Generation: "gen-1", Role: "review-fix"}
}

// TestAcquire_SendsBoundedRequestAndResolvesTheSelection proves the outbound
// half of the contract: the hook is handed the assignment identity, the
// owner/generation, the role, and exactly the allowed route ids - and nothing
// executable. It then proves the inbound half: a selected id resolves to the
// operator's own configured profile, not to anything the hook said about it.
func TestAcquire_SendsBoundedRequestAndResolvesTheSelection(t *testing.T) {
	var seen Request
	hook := scriptedHook(t, func(req Request) ([]byte, error) {
		seen = req
		return jsonReply(t, Response{
			Status:     statusSelected,
			Route:      "pi-grok",
			Reason:     "fewest-pending",
			Generation: "obs-7",
			ObservedAt: time.Now().UTC().Format(time.RFC3339),
		}), nil
	})

	allowed := []Profile{claudeProfile(), piGrokProfile()}
	decision, err := hook.Acquire(context.Background(), acquireRequest(), allowed)
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}

	if seen.Verb != VerbAcquire {
		t.Fatalf("request verb = %q, want %q", seen.Verb, VerbAcquire)
	}
	if seen.Assignment != "run-1:1" || seen.Owner != "daemon" || seen.Generation != "gen-1" {
		t.Fatalf("request must carry the full assignment identity, got %+v", seen)
	}
	if seen.Role != "review-fix" {
		t.Fatalf("request role = %q, want review-fix", seen.Role)
	}
	if strings.Join(seen.Routes, ",") != "claude-opus,pi-grok" {
		t.Fatalf("request routes = %v, want the sorted allowed ids", seen.Routes)
	}

	// The resolved profile is the operator's configured one, byte for byte.
	// The hook contributed only the id.
	if decision.Profile.Agent != piGrokProfile().Agent ||
		decision.Profile.Provider != piGrokProfile().Provider ||
		decision.Profile.Tuning != piGrokProfile().Tuning {
		t.Fatalf("selection must resolve to the configured profile, got %+v", decision.Profile)
	}
	if !decision.Selected {
		t.Fatal("decision must be selected")
	}
	if decision.Reason != "fewest-pending" || decision.Generation != "obs-7" {
		t.Fatalf("decision must carry the hook's evidence, got %+v", decision)
	}
	if !decision.Fresh {
		t.Fatal("a dated observation must read as fresh when no max age is configured")
	}
}

// TestAcquire_RefusesEverythingThatIsNotAnApprovedSelection is the security
// core. Each row is a way a wrong or hostile hook could try to start something
// the operator did not approve, or to stall the pipeline. Every one must
// produce an error and no selection, because the caller launches only on a
// selected decision.
func TestAcquire_RefusesEverythingThatIsNotAnApprovedSelection(t *testing.T) {
	cases := []struct {
		name    string
		reply   func(Request) ([]byte, error)
		wantMsg string
	}{
		{
			name: "route that was never offered",
			reply: func(Request) ([]byte, error) {
				return jsonReply(t, Response{Status: statusSelected, Route: "codex-sol"}), nil
			},
			wantMsg: "not among the",
		},
		{
			name: "route that is allowed for another role but not offered here",
			reply: func(Request) ([]byte, error) {
				return jsonReply(t, Response{Status: statusSelected, Route: "reviewer-only"}), nil
			},
			wantMsg: "not among the",
		},
		{
			name: "selected with no route at all",
			reply: func(Request) ([]byte, error) {
				return jsonReply(t, Response{Status: statusSelected}), nil
			},
			wantMsg: "not among the",
		},
		{
			name: "unknown status",
			reply: func(Request) ([]byte, error) {
				return jsonReply(t, Response{Status: "approved", Route: "pi-grok"}), nil
			},
			wantMsg: "unusable status",
		},
		{
			name: "output that is not JSON",
			reply: func(Request) ([]byte, error) {
				return []byte("selected route=pi-grok reason=fewest-pending"), nil
			},
			wantMsg: "unreadable output",
		},
		{
			name: "output carrying an unexpected field",
			reply: func(Request) ([]byte, error) {
				return []byte(`{"status":"selected","route":"pi-grok","command":"/bin/sh"}`), nil
			},
			wantMsg: "unreadable output",
		},
		{
			name: "two JSON objects, the second selecting a different route",
			reply: func(Request) ([]byte, error) {
				return []byte(`{"status":"deferred"}{"status":"selected","route":"pi-grok"}`), nil
			},
			wantMsg: "more than one JSON object",
		},
		{
			name: "oversized output",
			reply: func(Request) ([]byte, error) {
				padding := strings.Repeat("x", maxResponseBytes+1)
				return []byte(`{"status":"selected","route":"pi-grok","reason":"` + padding + `"}`), nil
			},
			wantMsg: "more than",
		},
		{
			name: "hook exits nonzero",
			reply: func(Request) ([]byte, error) {
				return nil, errors.New("exit status 2")
			},
			wantMsg: "failed",
		},
		{
			name: "observation time the hook could not format",
			reply: func(Request) ([]byte, error) {
				return jsonReply(t, Response{Status: statusSelected, Route: "pi-grok", ObservedAt: "yesterday"}), nil
			},
			wantMsg: "unusable observation time",
		},
	}

	allowed := []Profile{claudeProfile(), piGrokProfile()}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			hook := scriptedHook(t, tc.reply)
			decision, err := hook.Acquire(context.Background(), acquireRequest(), allowed)
			if err == nil {
				t.Fatalf("want a refusal, got selection %+v", decision)
			}
			if decision.Selected {
				t.Fatalf("a refused acquire must not report a selection, got %+v", decision)
			}
			if !strings.Contains(err.Error(), tc.wantMsg) {
				t.Fatalf("error %q must mention %q", err, tc.wantMsg)
			}
		})
	}
}

// TestAcquire_TimeoutRefusesRatherThanLaunching proves a hung hook stalls one
// turn with a clear error instead of falling through to some default route. A
// hook that never answers must never become an implicit approval.
func TestAcquire_TimeoutRefusesRatherThanLaunching(t *testing.T) {
	hook := &Hook{
		Path:    "scripted",
		Timeout: 30 * time.Millisecond,
		run: func(ctx context.Context, _ string, _ []string, _ []byte) ([]byte, error) {
			<-ctx.Done()
			return nil, ctx.Err()
		},
	}

	decision, err := hook.Acquire(context.Background(), acquireRequest(), []Profile{piGrokProfile()})
	if err == nil {
		t.Fatalf("a hung hook must refuse, got %+v", decision)
	}
	if !strings.Contains(err.Error(), "timed out") {
		t.Fatalf("error %q must name the timeout", err)
	}
	if decision.Selected {
		t.Fatal("a timed-out acquire must not report a selection")
	}
}

// TestAcquire_DeferredIsDistinctFromAFault proves a deferral is reported as
// its own sentinel. The caller must be able to tell "no approved service is
// free right now, try again after the next refresh" from "the hook is broken",
// because only the first is retryable without operator action.
func TestAcquire_DeferredIsDistinctFromAFault(t *testing.T) {
	hook := scriptedHook(t, func(Request) ([]byte, error) {
		return jsonReply(t, Response{
			Status:     statusDeferred,
			Reason:     "every candidate route is excluded",
			Generation: "obs-9",
		}), nil
	})

	decision, err := hook.Acquire(context.Background(), acquireRequest(), []Profile{piGrokProfile()})
	if !errors.Is(err, ErrDeferred) {
		t.Fatalf("a deferral must be reported as ErrDeferred, got %v", err)
	}
	if decision.Selected {
		t.Fatal("a deferred acquire must not report a selection")
	}
	if !strings.Contains(err.Error(), "every candidate route is excluded") {
		t.Fatalf("error %q must carry the hook's reason", err)
	}
	if decision.Generation != "obs-9" {
		t.Fatalf("a deferral must still carry its observation generation, got %q", decision.Generation)
	}
}

// TestAcquire_RequiresFullAssignmentIdentity proves the caller cannot acquire
// without stating who it is and which incarnation it is. Without both, a hook
// cannot tell a recovered daemon reclaiming its own assignment from an
// unrelated process reusing the id.
func TestAcquire_RequiresFullAssignmentIdentity(t *testing.T) {
	hook := scriptedHook(t, func(Request) ([]byte, error) {
		t.Fatal("an incomplete request must never reach the hook")
		return nil, nil
	})

	cases := []struct {
		name    string
		req     Request
		wantMsg string
	}{
		{"no assignment", Request{Owner: "daemon", Generation: "gen-1"}, "assignment id"},
		{"no owner", Request{Assignment: "run-1:1", Generation: "gen-1"}, "owner"},
		{"no generation", Request{Assignment: "run-1:1", Owner: "daemon"}, "generation"},
		{"blank assignment", Request{Assignment: "  ", Owner: "daemon", Generation: "gen-1"}, "assignment id"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := hook.Acquire(context.Background(), tc.req, []Profile{piGrokProfile()}); err == nil ||
				!strings.Contains(err.Error(), tc.wantMsg) {
				t.Fatalf("error %v must mention %q", err, tc.wantMsg)
			}
		})
	}
}

// TestAcquire_FreshnessIsEvidence, not a second admission gate. The hook owns
// eligibility; this only records whether the decision could be dated, so an
// operator reading the log can tell a current decision from one made on
// evidence nobody refreshed.
func TestAcquire_FreshnessIsEvidence(t *testing.T) {
	now := time.Date(2026, 9, 15, 12, 0, 0, 0, time.UTC)

	cases := []struct {
		name       string
		observedAt string
		maxAge     time.Duration
		wantFresh  bool
	}{
		{"recent within the window", now.Add(-time.Minute).Format(time.RFC3339), 10 * time.Minute, true},
		{"older than the window", now.Add(-time.Hour).Format(time.RFC3339), 10 * time.Minute, false},
		{"no observation time reported", "", 10 * time.Minute, false},
		{"no window configured", now.Add(-24 * time.Hour).Format(time.RFC3339), 0, true},
		{"dated in the future", now.Add(time.Hour).Format(time.RFC3339), 10 * time.Minute, false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			hook := scriptedHook(t, func(Request) ([]byte, error) {
				return jsonReply(t, Response{Status: statusSelected, Route: "pi-grok", ObservedAt: tc.observedAt}), nil
			})
			hook.MaxEvidenceAge = tc.maxAge
			hook.Now = func() time.Time { return now }

			decision, err := hook.Acquire(context.Background(), acquireRequest(), []Profile{piGrokProfile()})
			if err != nil {
				t.Fatalf("acquire: %v", err)
			}
			if !decision.Selected {
				t.Fatal("stale evidence must not, on its own, block a selection the hook made")
			}
			if decision.Fresh != tc.wantFresh {
				t.Fatalf("Fresh = %v, want %v", decision.Fresh, tc.wantFresh)
			}
		})
	}
}

// TestFinish_ReportsNormalizedOutcomes proves the closed outcome vocabulary
// reaches the hook, and that anything outside it is refused before the call.
func TestFinish_ReportsNormalizedOutcomes(t *testing.T) {
	for _, outcome := range []Outcome{OutcomeSuccess, OutcomeFailed, OutcomeLaunchFailed} {
		t.Run(string(outcome), func(t *testing.T) {
			var seen Request
			hook := scriptedHook(t, func(req Request) ([]byte, error) {
				seen = req
				return jsonReply(t, Response{Status: statusClosed}), nil
			})
			err := hook.Finish(context.Background(), Request{
				Assignment: "run-1:1", Owner: "daemon", Generation: "gen-1",
				Outcome: outcome, Profile: "pi-grok",
			})
			if err != nil {
				t.Fatalf("finish: %v", err)
			}
			if seen.Verb != VerbFinish || seen.Outcome != outcome {
				t.Fatalf("finish must report the verb and outcome, got %+v", seen)
			}
			if seen.Assignment != "run-1:1" || seen.Owner != "daemon" || seen.Generation != "gen-1" {
				t.Fatalf("finish must carry the exact acquired identity, got %+v", seen)
			}
			if seen.Profile != "pi-grok" {
				t.Fatalf("finish must name the route that actually ran, got %q", seen.Profile)
			}
			if len(seen.Routes) != 0 {
				t.Fatalf("finish must not offer routes, got %v", seen.Routes)
			}
		})
	}

	hook := scriptedHook(t, func(Request) ([]byte, error) {
		t.Fatal("an unnormalized outcome must never reach the hook")
		return nil, nil
	})
	err := hook.Finish(context.Background(), Request{
		Assignment: "run-1:1", Owner: "daemon", Outcome: Outcome("mostly-fine"),
	})
	if err == nil || !strings.Contains(err.Error(), "normalized outcome") {
		t.Fatalf("error %v must refuse the unnormalized outcome", err)
	}
}

// TestFinish_RefusesAnUnusableAcknowledgement proves a finish is only complete
// when the hook says it closed. Treating a malformed acknowledgement as
// success would let an assignment stay open in the controller forever while
// no-mistakes believed it had been released.
func TestFinish_RefusesAnUnusableAcknowledgement(t *testing.T) {
	hook := scriptedHook(t, func(Request) ([]byte, error) {
		return jsonReply(t, Response{Status: "maybe"}), nil
	})
	err := hook.Finish(context.Background(), Request{
		Assignment: "run-1:1", Owner: "daemon", Outcome: OutcomeSuccess,
	})
	if err == nil || !strings.Contains(err.Error(), "unusable status") {
		t.Fatalf("error %v must refuse the unusable acknowledgement", err)
	}
}

// TestAcquire_NoAllowedProfileIsAConfigurationFault proves an empty allowed
// set never reaches the hook. Offering zero routes and letting the hook pick
// would be meaningless; the operator simply left this role with nothing to run.
func TestAcquire_NoAllowedProfileIsAConfigurationFault(t *testing.T) {
	hook := scriptedHook(t, func(Request) ([]byte, error) {
		t.Fatal("an empty allowed set must never reach the hook")
		return nil, nil
	})
	_, err := hook.Acquire(context.Background(), acquireRequest(), nil)
	if err == nil || !strings.Contains(err.Error(), "no approved profile") {
		t.Fatalf("error %v must name the configuration fault", err)
	}
}

// TestHook_RequiresConfiguration proves an unconfigured hook refuses rather
// than executing something ambient.
func TestHook_RequiresConfiguration(t *testing.T) {
	var hook Hook
	if _, err := hook.Acquire(context.Background(), acquireRequest(), []Profile{piGrokProfile()}); err == nil ||
		!strings.Contains(err.Error(), "no assignment hook is configured") {
		t.Fatalf("error %v must report the missing configuration", err)
	}
}
