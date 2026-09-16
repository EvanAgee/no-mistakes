package routing

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
)

// recordingHook captures every request a router makes and replies from a
// script, so the router's own invariants are testable without a process.
// seenRequest is one call the router made: the subcommand plus its payload.
type seenRequest struct {
	verb Verb
	req  Request
}

type recordingHook struct {
	mu       sync.Mutex
	requests []seenRequest
	reply    func(Verb, Request) Response
	fail     func(Verb, Request) error
}

func newRecordingHook(t *testing.T, rh *recordingHook) *Hook {
	t.Helper()
	return &Hook{
		Path: "recording",
		run: func(_ context.Context, _ string, args []string, stdin []byte) ([]byte, error) {
			verb := verbFromArgs(args)
			var req Request
			if err := json.Unmarshal(stdin, &req); err != nil {
				t.Fatalf("unreadable request: %v", err)
			}
			rh.mu.Lock()
			rh.requests = append(rh.requests, seenRequest{verb: verb, req: req})
			fail, reply := rh.fail, rh.reply
			rh.mu.Unlock()

			if fail != nil {
				if err := fail(verb, req); err != nil {
					return nil, err
				}
			}
			payload, err := json.Marshal(reply(verb, req))
			if err != nil {
				t.Fatalf("marshal reply: %v", err)
			}
			return payload, nil
		},
	}
}

func (rh *recordingHook) seen() []seenRequest {
	rh.mu.Lock()
	defer rh.mu.Unlock()
	return append([]seenRequest(nil), rh.requests...)
}

func selectFirstOffered(verb Verb, req Request) Response {
	if verb == VerbFinish {
		return Response{Result: resultClosed, AssignmentID: req.AssignmentID}
	}
	if len(req.Routes) == 0 {
		return Response{Result: resultDeferred, Reason: "nothing offered"}
	}
	return Response{Result: resultSelected, RouteID: req.Routes[0], Reason: "fewest-pending"}
}

func testRouter(t *testing.T, rh *recordingHook, profiles ...Profile) *Router {
	t.Helper()
	if rh.reply == nil {
		rh.reply = selectFirstOffered
	}
	router, err := NewRouter(newRecordingHook(t, rh), profiles, "daemon", "gen-1")
	if err != nil {
		t.Fatalf("new router: %v", err)
	}
	if router == nil {
		t.Fatal("router must be created for a configured hook and profiles")
	}
	return router
}

// TestRouter_NilRouterIsTheUnconfiguredDefault proves routing off changes
// nothing. This is the property that keeps every existing installation
// behaving exactly as it did: no hook runs, no error is raised, and the caller
// is told to launch what it would have launched anyway.
func TestRouter_NilRouterIsTheUnconfiguredDefault(t *testing.T) {
	var router *Router

	assignment, selected, err := router.Route(context.Background(), "run-1-launch-1", "review")
	if err != nil {
		t.Fatalf("an unconfigured router must not error: %v", err)
	}
	if selected {
		t.Fatal("an unconfigured router must select nothing")
	}
	if assignment.ProfileKey() != "" {
		t.Fatal("an unconfigured router must produce no profile key")
	}
	if err := router.Finish(context.Background(), "run-1-launch-1", OutcomeSuccess); err != nil {
		t.Fatalf("finishing on an unconfigured router must not error: %v", err)
	}
}

// TestNewRouter_RefusesIncompleteOrInvalidConfiguration proves the router
// fails closed at construction. A profile the pipeline could not launch, or an
// identity the hook could not attribute, is refused before any run starts.
func TestNewRouter_RefusesIncompleteOrInvalidConfiguration(t *testing.T) {
	hook := &Hook{Path: "somewhere"}

	if _, err := NewRouter(hook, []Profile{claudeProfile()}, "", "gen-1"); err == nil ||
		!strings.Contains(err.Error(), "owner is required") {
		t.Fatalf("error %v must require an owner", err)
	}
	if _, err := NewRouter(hook, []Profile{claudeProfile()}, "daemon", ""); err == nil ||
		!strings.Contains(err.Error(), "generation is required") {
		t.Fatalf("error %v must require a generation", err)
	}

	bad := claudeProfile()
	bad.Provider = ""
	if _, err := NewRouter(hook, []Profile{bad}, "daemon", "gen-1"); err == nil ||
		!strings.Contains(err.Error(), "billing route") {
		t.Fatalf("error %v must refuse the invalid profile", err)
	}

	duplicate := claudeProfile()
	if _, err := NewRouter(hook, []Profile{claudeProfile(), duplicate}, "daemon", "gen-1"); err == nil ||
		!strings.Contains(err.Error(), "duplicate profile id") {
		t.Fatalf("error %v must refuse the duplicate id", err)
	}

	// Either half missing means routing is simply off, not misconfigured.
	for _, tc := range []struct {
		name     string
		hook     *Hook
		profiles []Profile
	}{
		{"no hook", nil, []Profile{claudeProfile()}},
		{"hook with no path", &Hook{}, []Profile{claudeProfile()}},
		{"no profiles", hook, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router, err := NewRouter(tc.hook, tc.profiles, "daemon", "gen-1")
			if err != nil {
				t.Fatalf("an unconfigured half must not error: %v", err)
			}
			if router != nil {
				t.Fatal("an unconfigured half must leave routing off")
			}
		})
	}
}

// TestRouter_OffersOnlyTheRolesApprovedProfiles proves role restriction
// reaches the hook. A profile restricted to another duty is never even
// offered, so the hook cannot select it however it is implemented.
func TestRouter_OffersOnlyTheRolesApprovedProfiles(t *testing.T) {
	fixerOnly := piGrokProfile()
	fixerOnly.ID = "pi-grok-fixer"
	fixerOnly.Roles = []string{"review-fix"}

	reviewerOnly := claudeProfile()
	reviewerOnly.ID = "claude-reviewer"
	reviewerOnly.Roles = []string{"review"}

	rh := &recordingHook{}
	router := testRouter(t, rh, fixerOnly, reviewerOnly, codexProfile())

	if _, _, err := router.Route(context.Background(), "run-1-launch-1", "review"); err != nil {
		t.Fatalf("route: %v", err)
	}
	offered := rh.seen()[0].req.Routes
	if strings.Join(offered, ",") != "claude-reviewer,codex-sol" {
		t.Fatalf("review turn was offered %v; the fixer-only route must not appear", offered)
	}

	if _, _, err := router.Route(context.Background(), "run-1-launch-2", "review-fix"); err != nil {
		t.Fatalf("route: %v", err)
	}
	offered = rh.seen()[1].req.Routes
	if strings.Join(offered, ",") != "codex-sol,pi-grok-fixer" {
		t.Fatalf("fix turn was offered %v; the reviewer-only route must not appear", offered)
	}
}

// TestRouter_RoleWithNothingApprovedIsAConfigurationFault proves a role the
// operator left with no runnable profile fails loudly rather than falling back
// to some other role's route.
func TestRouter_RoleWithNothingApprovedIsAConfigurationFault(t *testing.T) {
	fixerOnly := piGrokProfile()
	fixerOnly.Roles = []string{"review-fix"}

	rh := &recordingHook{}
	router := testRouter(t, rh, fixerOnly)

	_, selected, err := router.Route(context.Background(), "run-1-launch-1", "test-evidence")
	if !errors.Is(err, ErrNoAllowedProfile) {
		t.Fatalf("error %v must be ErrNoAllowedProfile", err)
	}
	if selected {
		t.Fatal("a role with nothing approved must select nothing")
	}
	if len(rh.seen()) != 0 {
		t.Fatal("a role with nothing approved must never reach the hook")
	}
	// The message must be actionable on its own: an operator reading a failed
	// run needs the refused duty and the profiles that exist, or the only
	// symptom is a hard failure with nothing to change.
	if !strings.Contains(err.Error(), "test-evidence") {
		t.Fatalf("error %q must name the refused role", err)
	}
	if !strings.Contains(err.Error(), fixerOnly.ID) {
		t.Fatalf("error %q must list the configured profiles", err)
	}
	if !strings.Contains(err.Error(), "roles") {
		t.Fatalf("error %q must say how to fix the configuration", err)
	}
}

// TestRouter_UnnamedInvocationIsServedOnlyByAnUnrestrictedProfile is the case
// an operator reaches by following the docs and restricting every profile.
//
// An invocation that reports no role must not be matched against a restricted
// list, and when nothing is admissible it must fail with a message that says
// so in those terms rather than quoting an empty role string the operator
// would then try to add to a `roles` list.
func TestRouter_UnnamedInvocationIsServedOnlyByAnUnrestrictedProfile(t *testing.T) {
	restricted := piGrokProfile()
	restricted.Roles = []string{"review-fix"}

	rh := &recordingHook{}
	_, selected, err := testRouter(t, rh, restricted).Route(context.Background(), "run-1-launch-1", "")
	if !errors.Is(err, ErrNoAllowedProfile) {
		t.Fatalf("error %v must be ErrNoAllowedProfile", err)
	}
	if selected {
		t.Fatal("an unnamed invocation must not select a role-restricted profile")
	}
	if !strings.Contains(err.Error(), "reports no role") {
		t.Fatalf("error %q must describe the unnamed invocation rather than quote an empty role", err)
	}

	// The same invocation with one unrestricted profile present is served.
	rh = &recordingHook{}
	unrestricted := claudeProfile()
	assignment, selected, err := testRouter(t, rh, restricted, unrestricted).Route(context.Background(), "run-1-launch-1", "")
	if err != nil {
		t.Fatalf("an unrestricted profile must serve an unnamed invocation: %v", err)
	}
	if !selected {
		t.Fatal("want a selection")
	}
	if assignment.Profile.ID != unrestricted.ID {
		t.Fatalf("selected %q, want the unrestricted profile %q", assignment.Profile.ID, unrestricted.ID)
	}
	for _, call := range rh.seen() {
		for _, offered := range call.req.Routes {
			if offered == restricted.ID {
				t.Fatal("a role-restricted profile must never be offered for an unnamed invocation")
			}
		}
	}
}

// TestRouter_FinishReportsTheExactAcquiredIdentity proves the identity sent to
// finish comes from what the router recorded at acquire, not from the caller.
// A mis-wired caller therefore cannot bill a route that never ran.
func TestRouter_FinishReportsTheExactAcquiredIdentity(t *testing.T) {
	rh := &recordingHook{
		reply: func(verb Verb, req Request) Response {
			if verb == VerbFinish {
				return Response{Result: resultClosed}
			}
			return Response{Result: resultSelected, RouteID: "pi-grok", Reason: "fewest-pending"}
		},
	}
	router := testRouter(t, rh, claudeProfile(), piGrokProfile())

	assignment, selected, err := router.Route(context.Background(), "run-1-launch-1", "review-fix")
	if err != nil || !selected {
		t.Fatalf("route: %v selected=%v", err, selected)
	}
	if err := router.Finish(context.Background(), assignment.ID, OutcomeSuccess); err != nil {
		t.Fatalf("finish: %v", err)
	}

	requests := rh.seen()
	finish := requests[len(requests)-1]
	if finish.verb != VerbFinish {
		t.Fatalf("last call must use the finish subcommand, got %q", finish.verb)
	}
	if finish.req.AssignmentID != "run-1-launch-1" {
		t.Fatalf("finish assignment_id = %q", finish.req.AssignmentID)
	}
	if finish.req.Owner.Identity != "daemon" || finish.req.Owner.Generation != "gen-1" {
		t.Fatalf("finish must carry the acquired owner, got %+v", finish.req.Owner)
	}
	// The concrete identity that ran, not just the route id: the controller
	// records what was actually spent.
	if finish.req.Profile == nil || finish.req.Profile.Adapter != "pi" ||
		finish.req.Profile.Model != "xai/grok-4.6" {
		t.Fatalf("finish must name the concrete profile that ran, got %+v", finish.req.Profile)
	}
	if finish.req.Outcome != OutcomeSuccess {
		t.Fatalf("finish outcome = %q, want success", finish.req.Outcome)
	}
}

// TestRouter_DuplicateAcquisitionAndFinishHaveExplicitOutcomes covers the
// bookkeeping the brief requires. Reusing a live or spent id is refused, so an
// attempt cannot be counted twice; a repeated finish is a silent no-op, so a
// deferred cleanup and an explicit report cannot double-close.
func TestRouter_DuplicateAcquisitionAndFinishHaveExplicitOutcomes(t *testing.T) {
	rh := &recordingHook{}
	router := testRouter(t, rh, piGrokProfile())

	assignment, _, err := router.Route(context.Background(), "run-1-launch-1", "review-fix")
	if err != nil {
		t.Fatalf("route: %v", err)
	}

	if _, _, err := router.Route(context.Background(), "run-1-launch-1", "review-fix"); err == nil ||
		!strings.Contains(err.Error(), "already open") {
		t.Fatalf("error %v must refuse reacquiring a live assignment", err)
	}

	if err := router.Finish(context.Background(), assignment.ID, OutcomeSuccess); err != nil {
		t.Fatalf("first finish: %v", err)
	}
	before := len(rh.seen())
	if err := router.Finish(context.Background(), assignment.ID, OutcomeSuccess); err != nil {
		t.Fatalf("a repeated finish must be a silent no-op, got %v", err)
	}
	if after := len(rh.seen()); after != before {
		t.Fatalf("a repeated finish must not reach the hook again (%d then %d requests)", before, after)
	}

	if _, _, err := router.Route(context.Background(), "run-1-launch-1", "review-fix"); err == nil ||
		!strings.Contains(err.Error(), "already closed") {
		t.Fatalf("error %v must refuse reopening a spent assignment", err)
	}

	if err := router.Finish(context.Background(), "run-1-launch-99", OutcomeSuccess); err == nil ||
		!strings.Contains(err.Error(), "never acquired") {
		t.Fatalf("error %v must refuse finishing an assignment this run never acquired", err)
	}
}

// TestRouter_FinishThatNeverReachedTheHookStaysOpenForRetry proves a lost
// report is recoverable rather than silently counted as delivered.
//
// The controller learns a route is free only from finish. If the router spent
// the id locally before the hook accepted the report, a momentarily unavailable
// hook would leave that route marked in flight forever, biasing every later
// acquire for every caller on this machine, and no path could ever deliver the
// missed report: a second finish would find the id already spent and return a
// local success. The finish verb is idempotent by contract, so retrying can
// only turn a lost report into a delivered one.
func TestRouter_FinishThatNeverReachedTheHookStaysOpenForRetry(t *testing.T) {
	var refuse bool
	rh := &recordingHook{
		fail: func(verb Verb, _ Request) error {
			if verb == VerbFinish && refuse {
				return errors.New("hook unavailable")
			}
			return nil
		},
	}
	router := testRouter(t, rh, piGrokProfile())

	assignment, _, err := router.Route(context.Background(), "run-1-launch-1", "review-fix")
	if err != nil {
		t.Fatalf("route: %v", err)
	}

	refuse = true
	if err := router.Finish(context.Background(), assignment.ID, OutcomeSuccess); err == nil {
		t.Fatal("a finish the hook refused must be reported to the caller")
	}

	// The lost report must still be owed, not recorded as paid.
	refuse = false
	before := len(rh.seen())
	if err := router.Finish(context.Background(), assignment.ID, OutcomeSuccess); err != nil {
		t.Fatalf("retrying a lost finish: %v", err)
	}
	delivered := rh.seen()[before:]
	if len(delivered) != 1 || delivered[0].verb != VerbFinish {
		t.Fatalf("the retry must reach the hook, got %d calls", len(delivered))
	}
	if delivered[0].req.AssignmentID != assignment.ID {
		t.Fatalf("the retry reported %q, want the original assignment %q", delivered[0].req.AssignmentID, assignment.ID)
	}
	// The identity is still the one Route recorded, not a caller-supplied one.
	if delivered[0].req.Profile == nil || delivered[0].req.Profile.Adapter != string(piGrokProfile().Agent) {
		t.Fatalf("the retry must carry the acquired profile, got %+v", delivered[0].req.Profile)
	}

	// Once delivered, the assignment is spent: a further finish is a local
	// no-op and the id can never be reacquired.
	settled := len(rh.seen())
	if err := router.Finish(context.Background(), assignment.ID, OutcomeSuccess); err != nil {
		t.Fatalf("a finish after a delivered one must be a silent no-op, got %v", err)
	}
	if len(rh.seen()) != settled {
		t.Fatal("a finish after a delivered one must not reach the hook again")
	}
	if _, _, err := router.Route(context.Background(), assignment.ID, "review-fix"); err == nil ||
		!strings.Contains(err.Error(), "already closed") {
		t.Fatalf("error %v must refuse reopening a settled assignment", err)
	}
}

// TestRouter_FinishRefusedByTheControllerDoesNotSettleTheAssignment is the same
// property for a hook that ran fine but answered with something Finish refuses.
// The report did not land either way, so the debt must stay outstanding.
func TestRouter_FinishRefusedByTheControllerDoesNotSettleTheAssignment(t *testing.T) {
	var refuse bool
	rh := &recordingHook{
		reply: func(verb Verb, req Request) Response {
			if verb == VerbFinish {
				if refuse {
					return Response{Result: resultError, Error: "unknown assignment"}
				}
				return Response{Result: resultClosed, AssignmentID: req.AssignmentID}
			}
			return selectFirstOffered(verb, req)
		},
	}
	router := testRouter(t, rh, piGrokProfile())

	assignment, _, err := router.Route(context.Background(), "run-1-launch-1", "review-fix")
	if err != nil {
		t.Fatalf("route: %v", err)
	}

	refuse = true
	if err := router.Finish(context.Background(), assignment.ID, OutcomeSuccess); err == nil {
		t.Fatal("a finish the controller refused must be reported to the caller")
	}

	refuse = false
	before := len(rh.seen())
	if err := router.Finish(context.Background(), assignment.ID, OutcomeSuccess); err != nil {
		t.Fatalf("retrying a refused finish: %v", err)
	}
	if len(rh.seen()) != before+1 {
		t.Fatal("the retry must reach the hook")
	}
}

// TestRouter_DeferredAssignmentIsRetriedNotCached proves a deferral is never
// remembered as a verdict. The same attempt id can be acquired again, and once
// the controller's evidence changes it is admitted, which is exactly the
// "deferred, then recovered" path the brief requires.
func TestRouter_DeferredAssignmentIsRetriedNotCached(t *testing.T) {
	var admit bool
	rh := &recordingHook{
		reply: func(verb Verb, req Request) Response {
			if verb == VerbFinish {
				return Response{Result: resultClosed}
			}
			if !admit {
				return Response{Result: resultDeferred, Reason: "every candidate route is excluded"}
			}
			return Response{Result: resultSelected, RouteID: req.Routes[0], Reason: "recovered"}
		},
	}
	router := testRouter(t, rh, piGrokProfile())

	_, selected, err := router.Route(context.Background(), "run-1-launch-1", "review-fix")
	if !errors.Is(err, ErrDeferred) {
		t.Fatalf("first attempt must defer, got %v", err)
	}
	if selected {
		t.Fatal("a deferred attempt must select nothing")
	}

	// The controller's next refresh admits the route. The SAME assignment id
	// must be acquirable, because a deferral left nothing recorded to replay.
	admit = true
	assignment, selected, err := router.Route(context.Background(), "run-1-launch-1", "review-fix")
	if err != nil {
		t.Fatalf("after recovery the same attempt must be admissible: %v", err)
	}
	if !selected {
		t.Fatal("after recovery the attempt must be selected")
	}
	if assignment.Reason != "recovered" {
		t.Fatalf("assignment must carry the recovery reason, got %q", assignment.Reason)
	}
}

// TestRouter_ARefusedAcquireLeavesNoOpenAssignment proves a rejected hook
// answer strands nothing. The attempt never opened, so finishing it is refused
// and the same id stays available for a genuine retry.
func TestRouter_ARefusedAcquireLeavesNoOpenAssignment(t *testing.T) {
	rh := &recordingHook{
		reply: func(verb Verb, req Request) Response {
			if verb == VerbFinish {
				return Response{Result: resultClosed}
			}
			return Response{Result: resultSelected, RouteID: "a-route-nobody-offered"}
		},
	}
	router := testRouter(t, rh, piGrokProfile())

	if _, selected, err := router.Route(context.Background(), "run-1-launch-1", "review-fix"); err == nil || selected {
		t.Fatalf("a disallowed selection must be refused, got selected=%v err=%v", selected, err)
	}
	if err := router.Finish(context.Background(), "run-1-launch-1", OutcomeLaunchFailed); err == nil ||
		!strings.Contains(err.Error(), "never acquired") {
		t.Fatalf("error %v must report that nothing was ever acquired", err)
	}

	// The id is still usable, because nothing was recorded against it.
	rh.mu.Lock()
	rh.reply = selectFirstOffered
	rh.mu.Unlock()
	if _, selected, err := router.Route(context.Background(), "run-1-launch-1", "review-fix"); err != nil || !selected {
		t.Fatalf("the id must remain available after a refusal: selected=%v err=%v", selected, err)
	}
}

// TestRouter_ConcurrentRoutesKeepDistinctAssignments proves the router's own
// bookkeeping is race-free: every concurrent attempt gets its own record, and
// every one can be finished exactly once.
func TestRouter_ConcurrentRoutesKeepDistinctAssignments(t *testing.T) {
	rh := &recordingHook{}
	router := testRouter(t, rh, claudeProfile(), codexProfile(), piGrokProfile(), piDeepSeekProfile())

	const attempts = 16
	ids := make([]string, attempts)
	errs := make([]error, attempts)
	var wg sync.WaitGroup
	for i := range attempts {
		wg.Add(1)
		go func() {
			defer wg.Done()
			assignment, _, err := router.Route(context.Background(), assignmentID(i), "review-fix")
			ids[i], errs[i] = assignment.ID, err
		}()
	}
	wg.Wait()

	seen := map[string]bool{}
	for i, err := range errs {
		if err != nil {
			t.Fatalf("attempt %d: %v", i, err)
		}
		if seen[ids[i]] {
			t.Fatalf("assignment id %q was handed out twice", ids[i])
		}
		seen[ids[i]] = true
	}

	for i := range attempts {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := router.Finish(context.Background(), ids[i], OutcomeSuccess); err != nil {
				t.Errorf("finish %d: %v", i, err)
			}
		}()
	}
	wg.Wait()
}

// TestRouter_AssignmentCarriesItsProfileKey proves the selected profile's
// session-reuse identity travels with the assignment, so the invocation seam
// can qualify session reuse without recomputing anything.
func TestRouter_AssignmentCarriesItsProfileKey(t *testing.T) {
	rh := &recordingHook{
		reply: func(verb Verb, req Request) Response {
			if verb == VerbFinish {
				return Response{Result: resultClosed}
			}
			return Response{Result: resultSelected, RouteID: "pi-deepseek"}
		},
	}
	router := testRouter(t, rh, piGrokProfile(), piDeepSeekProfile())

	assignment, _, err := router.Route(context.Background(), "run-1-launch-1", "review-fix")
	if err != nil {
		t.Fatalf("route: %v", err)
	}
	if got, want := assignment.ProfileKey(), ProfileKey(piDeepSeekProfile()); got != want {
		t.Fatalf("assignment key = %q, want the selected profile's key %q", got, want)
	}
	if assignment.ProfileKey() == ProfileKey(piGrokProfile()) {
		t.Fatal("the assignment must not carry the unselected sibling route's key")
	}
}

func assignmentID(i int) string {
	return fmt.Sprintf("run-1-launch-%d", i)
}
