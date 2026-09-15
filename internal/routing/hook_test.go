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
func scriptedHook(t *testing.T, reply func(Verb, Request) ([]byte, error)) *Hook {
	t.Helper()
	return &Hook{
		Path: "scripted",
		run: func(_ context.Context, _ string, args []string, stdin []byte) ([]byte, error) {
			var req Request
			if err := json.Unmarshal(stdin, &req); err != nil {
				t.Fatalf("hook received unreadable request %q: %v", stdin, err)
			}
			return reply(verbFromArgs(args), req)
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
	return Request{
		AssignmentID: "run-1-launch-1",
		Owner:        Owner{Identity: "daemon", Generation: "gen-1"},
	}
}

// TestAcquire_SendsBoundedRequestAndResolvesTheSelection proves the outbound
// half of the contract: the hook is handed the assignment identity, the
// owner/generation, the role, and exactly the allowed route ids - and nothing
// executable. It then proves the inbound half: a selected id resolves to the
// operator's own configured profile, not to anything the hook said about it.
func TestAcquire_SendsBoundedRequestAndResolvesTheSelection(t *testing.T) {
	var seen Request
	var seenVerb Verb
	hook := scriptedHook(t, func(verb Verb, req Request) ([]byte, error) {
		seen, seenVerb = req, verb
		return jsonReply(t, Response{
			Result:     resultSelected,
			RouteID:    "pi-grok",
			Reason:     "fewest-pending",
			Generation: 7,
		}), nil
	})

	allowed := []Profile{claudeProfile(), piGrokProfile()}
	decision, err := hook.Acquire(context.Background(), acquireRequest(), allowed)
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}

	// The verb is the subcommand argument, not a request field.
	if seenVerb != VerbAcquire {
		t.Fatalf("subcommand = %q, want %q", seenVerb, VerbAcquire)
	}
	if seen.AssignmentID != "run-1-launch-1" {
		t.Fatalf("request assignment_id = %q", seen.AssignmentID)
	}
	if seen.Owner.Identity != "daemon" || seen.Owner.Generation != "gen-1" {
		t.Fatalf("request must carry the full owner identity, got %+v", seen.Owner)
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
	if decision.Reason != "fewest-pending" || decision.Generation != 7 {
		t.Fatalf("decision must carry the controller's evidence, got %+v", decision)
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
		reply   func(Verb, Request) ([]byte, error)
		wantMsg string
	}{
		{
			name: "route that was never offered",
			reply: func(Verb, Request) ([]byte, error) {
				return jsonReply(t, Response{Result: resultSelected, RouteID: "codex-sol"}), nil
			},
			wantMsg: "not among the",
		},
		{
			name: "route that is allowed for another role but not offered here",
			reply: func(Verb, Request) ([]byte, error) {
				return jsonReply(t, Response{Result: resultSelected, RouteID: "reviewer-only"}), nil
			},
			wantMsg: "not among the",
		},
		{
			name: "selected with no route at all",
			reply: func(Verb, Request) ([]byte, error) {
				return jsonReply(t, Response{Result: resultSelected}), nil
			},
			wantMsg: "not among the",
		},
		{
			name: "unknown status",
			reply: func(Verb, Request) ([]byte, error) {
				return jsonReply(t, Response{Result: "approved", RouteID: "pi-grok"}), nil
			},
			wantMsg: "unusable result",
		},
		{
			name: "output that is not JSON",
			reply: func(Verb, Request) ([]byte, error) {
				return []byte("selected route=pi-grok reason=fewest-pending"), nil
			},
			wantMsg: "unreadable output",
		},
		{
			name: "output carrying an unexpected field",
			reply: func(Verb, Request) ([]byte, error) {
				return []byte(`{"result":"selected","route_id":"pi-grok","command":"/bin/sh"}`), nil
			},
			wantMsg: "unreadable output",
		},
		{
			name: "two JSON objects, the second selecting a different route",
			reply: func(Verb, Request) ([]byte, error) {
				return []byte(`{"result":"deferred"}{"result":"selected","route_id":"pi-grok"}`), nil
			},
			wantMsg: "exactly one JSON object",
		},
		{
			name: "oversized output",
			reply: func(Verb, Request) ([]byte, error) {
				padding := strings.Repeat("x", maxResponseBytes+1)
				return []byte(`{"result":"selected","route_id":"pi-grok","reason":"` + padding + `"}`), nil
			},
			wantMsg: "more than",
		},
		{
			name: "hook exits nonzero",
			reply: func(Verb, Request) ([]byte, error) {
				return nil, errors.New("exit status 2")
			},
			wantMsg: "failed",
		},
		{
			name: "already-closed, which never authorizes a relaunch",
			reply: func(Verb, Request) ([]byte, error) {
				return jsonReply(t, Response{Result: resultAlreadyClosed, Reason: "already closed"}), nil
			},
			wantMsg: "already closed",
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
	hook := scriptedHook(t, func(Verb, Request) ([]byte, error) {
		return jsonReply(t, Response{
			Result:     resultDeferred,
			Reason:     "every candidate route is excluded",
			Generation: 9,
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
	if decision.Generation != 9 {
		t.Fatalf("a deferral must still carry its controller generation, got %d", decision.Generation)
	}
}

// TestAcquire_RequiresFullAssignmentIdentity proves the caller cannot acquire
// without stating who it is and which incarnation it is. Without both, a hook
// cannot tell a recovered daemon reclaiming its own assignment from an
// unrelated process reusing the id.
func TestAcquire_RequiresFullAssignmentIdentity(t *testing.T) {
	hook := scriptedHook(t, func(Verb, Request) ([]byte, error) {
		t.Fatal("an incomplete request must never reach the hook")
		return nil, nil
	})

	cases := []struct {
		name    string
		req     Request
		wantMsg string
	}{
		{"no assignment", Request{Owner: Owner{Identity: "daemon", Generation: "gen-1"}}, "assignment id"},
		{"no owner identity", Request{AssignmentID: "a", Owner: Owner{Generation: "gen-1"}}, "owner identity"},
		{"no owner generation", Request{AssignmentID: "a", Owner: Owner{Identity: "daemon"}}, "owner generation"},
		{"blank assignment", Request{AssignmentID: "  ", Owner: Owner{Identity: "daemon", Generation: "gen-1"}}, "assignment id"},
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

// TestAcquire_AlreadyClosedNeverAuthorizesALaunch proves the reply that says
// "this exact launch already happened" is not a selection. The controller
// returns no route with it precisely so it cannot authorize one, and re-running
// under the same id would double-count work already accounted for.
func TestAcquire_AlreadyClosedNeverAuthorizesALaunch(t *testing.T) {
	hook := scriptedHook(t, func(Verb, Request) ([]byte, error) {
		return jsonReply(t, Response{
			Result:     resultAlreadyClosed,
			Reason:     "already closed; not relaunched",
			Generation: 5,
		}), nil
	})

	decision, err := hook.Acquire(context.Background(), acquireRequest(), []Profile{piGrokProfile()})
	if !errors.Is(err, ErrAlreadyClosed) {
		t.Fatalf("error = %v, want ErrAlreadyClosed", err)
	}
	if decision.Selected {
		t.Fatalf("an already-closed assignment must never authorize a launch, got %+v", decision)
	}
	if decision.Profile.ID != "" {
		t.Fatalf("an already-closed reply must resolve no profile, got %q", decision.Profile.ID)
	}
}

// TestAcquire_AlreadyClosedCarryingARouteStillDoesNotLaunch is the hostile
// version of the same rule. Even if a controller wrongly attached a route id to
// an already-closed reply, only `selected` authorizes a launch, so the result
// discriminator alone decides and the stray route is ignored.
func TestAcquire_AlreadyClosedCarryingARouteStillDoesNotLaunch(t *testing.T) {
	hook := scriptedHook(t, func(Verb, Request) ([]byte, error) {
		return jsonReply(t, Response{
			Result:  resultAlreadyClosed,
			RouteID: "pi-grok",
			Reason:  "already closed; not relaunched",
		}), nil
	})

	decision, err := hook.Acquire(context.Background(), acquireRequest(), []Profile{piGrokProfile()})
	if !errors.Is(err, ErrAlreadyClosed) {
		t.Fatalf("error = %v, want ErrAlreadyClosed", err)
	}
	if decision.Selected {
		t.Fatal("only a selected result may authorize a launch, whatever else the reply carries")
	}
}

// TestAcquire_ControllerErrorObjectIsSurfaced proves the controller's own
// refusal reaches the operator as its message rather than as a bare exit
// status. It arrives with a nonzero exit, so the JSON must be read first.
func TestAcquire_ControllerErrorObjectIsSurfaced(t *testing.T) {
	hook := scriptedHook(t, func(Verb, Request) ([]byte, error) {
		return jsonReply(t, Response{
			Result: resultError,
			Error:  "unknown route id: pi-grok",
		}), errors.New("exit status 1")
	})

	decision, err := hook.Acquire(context.Background(), acquireRequest(), []Profile{piGrokProfile()})
	if !errors.Is(err, ErrHookRefused) {
		t.Fatalf("error = %v, want ErrHookRefused", err)
	}
	if !strings.Contains(err.Error(), "unknown route id") {
		t.Fatalf("error %q must carry the controller's own message", err)
	}
	if decision.Selected {
		t.Fatal("a refused request must not authorize a launch")
	}
}

// TestFinish_ReportsNormalizedOutcomes proves the closed outcome vocabulary
// reaches the hook, and that anything outside it is refused before the call.
func TestFinish_ReportsNormalizedOutcomes(t *testing.T) {
	for _, outcome := range []Outcome{OutcomeSuccess, OutcomeLaunchFailed, OutcomeAuthFailed, OutcomeExhausted, OutcomeOutage} {
		t.Run(string(outcome), func(t *testing.T) {
			var seen Request
			var seenVerb Verb
			hook := scriptedHook(t, func(verb Verb, req Request) ([]byte, error) {
				seen, seenVerb = req, verb
				return jsonReply(t, Response{Result: resultClosed, AssignmentID: req.AssignmentID}), nil
			})
			err := hook.Finish(context.Background(), Request{
				AssignmentID: "run-1-launch-1",
				Owner:        Owner{Identity: "daemon", Generation: "gen-1"},
				Outcome:      outcome,
				Profile:      &LaunchedProfile{Adapter: "pi", Model: "xai/grok-4.6", Effort: "xhigh"},
			})
			if err != nil {
				t.Fatalf("finish: %v", err)
			}
			if seenVerb != VerbFinish || seen.Outcome != outcome {
				t.Fatalf("finish must use the finish subcommand and report the outcome, got %q %+v", seenVerb, seen)
			}
			if seen.AssignmentID != "run-1-launch-1" || seen.Owner.Identity != "daemon" {
				t.Fatalf("finish must carry the exact acquired identity, got %+v", seen)
			}
			if seen.Profile == nil || seen.Profile.Adapter != "pi" || seen.Profile.Model != "xai/grok-4.6" {
				t.Fatalf("finish must name the concrete profile that ran, got %+v", seen.Profile)
			}
			if len(seen.Routes) != 0 {
				t.Fatalf("finish must not offer routes, got %v", seen.Routes)
			}
		})
	}

	hook := scriptedHook(t, func(Verb, Request) ([]byte, error) {
		t.Fatal("an unnormalized outcome must never reach the hook")
		return nil, nil
	})
	err := hook.Finish(context.Background(), Request{
		AssignmentID: "run-1-launch-1",
		Owner:        Owner{Identity: "daemon", Generation: "gen-1"},
		Outcome:      Outcome("mostly-fine"),
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
	hook := scriptedHook(t, func(Verb, Request) ([]byte, error) {
		return jsonReply(t, Response{Result: "maybe"}), nil
	})
	err := hook.Finish(context.Background(), Request{
		AssignmentID: "run-1-launch-1",
		Owner:        Owner{Identity: "daemon", Generation: "gen-1"},
		Outcome:      OutcomeSuccess,
	})
	if err == nil || !strings.Contains(err.Error(), "unusable result") {
		t.Fatalf("error %v must refuse the unusable acknowledgement", err)
	}
}

// TestAcquire_NoAllowedProfileIsAConfigurationFault proves an empty allowed
// set never reaches the hook. Offering zero routes and letting the hook pick
// would be meaningless; the operator simply left this role with nothing to run.
func TestAcquire_NoAllowedProfileIsAConfigurationFault(t *testing.T) {
	hook := scriptedHook(t, func(Verb, Request) ([]byte, error) {
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

// TestAcquire_OversizedErrorObjectIsStillRefused proves the size bound applies
// to the controller's own error path too. That reply is read before the exit
// status is consulted, so without its own bound a flooding hook could make one
// routing decision consume unbounded memory.
func TestAcquire_OversizedErrorObjectIsStillRefused(t *testing.T) {
	hook := scriptedHook(t, func(Verb, Request) ([]byte, error) {
		padding := strings.Repeat("x", maxResponseBytes+1)
		return []byte(`{"result":"error","error":"` + padding + `"}`), errors.New("exit status 1")
	})

	decision, err := hook.Acquire(context.Background(), acquireRequest(), []Profile{piGrokProfile()})
	if err == nil {
		t.Fatalf("an oversized reply must be refused, got %+v", decision)
	}
	if decision.Selected {
		t.Fatal("an oversized reply must never authorize a launch")
	}
	// It is refused as a transport failure rather than parsed, which is the
	// point: nothing that large is read as a decision.
	if errors.Is(err, ErrHookRefused) {
		t.Fatalf("an oversized reply must not be parsed as a controller refusal: %v", err)
	}
}

// TestFinish_AlreadyClosedAcknowledgementIsAccepted proves a repeat finish is
// the no-op the contract promises. The controller answers `closed` for an
// assignment it has already closed, so a duplicate report is never an error.
func TestFinish_AlreadyClosedAcknowledgementIsAccepted(t *testing.T) {
	calls := 0
	hook := scriptedHook(t, func(_ Verb, req Request) ([]byte, error) {
		calls++
		return jsonReply(t, Response{Result: resultClosed, AssignmentID: req.AssignmentID}), nil
	})

	for attempt := 1; attempt <= 2; attempt++ {
		if err := hook.Finish(context.Background(), Request{
			AssignmentID: "run-1-launch-1",
			Owner:        Owner{Identity: "daemon", Generation: "gen-1"},
			Outcome:      OutcomeSuccess,
		}); err != nil {
			t.Fatalf("finish attempt %d: %v", attempt, err)
		}
	}
	if calls != 2 {
		t.Fatalf("both finishes must reach the controller, got %d calls", calls)
	}
}
