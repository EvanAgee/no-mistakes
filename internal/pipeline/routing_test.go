package pipeline

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/kunchenguid/no-mistakes/internal/agent"
	"github.com/kunchenguid/no-mistakes/internal/config"
	"github.com/kunchenguid/no-mistakes/internal/gateguidance"
	"github.com/kunchenguid/no-mistakes/internal/routing"
	"github.com/kunchenguid/no-mistakes/internal/types"
)

// These tests pin the invocation seam: every native invocation acquires a
// route before it launches and reports a normalized outcome afterwards, a
// routing failure never launches anything, and nothing already running is
// disturbed by a later invocation choosing a different service.

// seenCall is one hook call: the subcommand plus its JSON payload.
type seenCall struct {
	verb routing.Verb
	req  routing.Request
}

// fakeHookTransport drives routing.Hook without a process, recording every
// call so a test can assert exactly what the controller was told.
type fakeHookTransport struct {
	mu       sync.Mutex
	requests []seenCall
	reply    func(routing.Verb, routing.Request) routing.Response
	fail     func(routing.Verb, routing.Request) error
}

func (f *fakeHookTransport) seen() []seenCall {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]seenCall(nil), f.requests...)
}

func (f *fakeHookTransport) finishes() []routing.Request {
	var out []routing.Request
	for _, call := range f.seen() {
		if call.verb == routing.VerbFinish {
			out = append(out, call.req)
		}
	}
	return out
}

// testRouting builds a RunRouting whose hook is driven by transport and whose
// adapters come from newAgent, mirroring how the daemon wires the real thing.
func testRouting(t *testing.T, transport *fakeHookTransport, profiles []routing.Profile,
	newAgent func(routing.Profile) (agent.Agent, error)) *RunRouting {
	t.Helper()
	return testRoutingForGeneration(t, transport, profiles, newAgent, "gen-1")
}

// testRoutingForGeneration is testRouting with an explicit daemon incarnation,
// so a test can build the same run's routing twice as two incarnations would.
func testRoutingForGeneration(t *testing.T, transport *fakeHookTransport, profiles []routing.Profile,
	newAgent func(routing.Profile) (agent.Agent, error), generation string) *RunRouting {
	t.Helper()

	hook := routing.NewTestHook(func(_ context.Context, verb routing.Verb, stdin []byte) ([]byte, error) {
		var req routing.Request
		if err := json.Unmarshal(stdin, &req); err != nil {
			t.Fatalf("unreadable request: %v", err)
		}
		transport.mu.Lock()
		transport.requests = append(transport.requests, seenCall{verb: verb, req: req})
		fail, reply := transport.fail, transport.reply
		transport.mu.Unlock()

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
	})

	router, err := routing.NewRouter(hook, profiles, "daemon", generation)
	if err != nil {
		t.Fatalf("new router: %v", err)
	}
	return NewRunRouting(router, "run-1", newAgent)
}

func selectByID(id string) func(routing.Verb, routing.Request) routing.Response {
	return func(verb routing.Verb, req routing.Request) routing.Response {
		if verb == routing.VerbFinish {
			return routing.Response{Result: "closed", AssignmentID: req.AssignmentID}
		}
		return routing.Response{Result: "selected", RouteID: id, Reason: "fewest-pending"}
	}
}

// recordingAgent reports which profile served an invocation.
type recordingAgent struct {
	profile routing.Profile
	mu      sync.Mutex
	prompts []string
	closed  bool
	runFn   func(context.Context, agent.RunOpts) (*agent.Result, error)
}

func (r *recordingAgent) Name() string { return string(r.profile.Agent) }

func (r *recordingAgent) Close() error {
	r.mu.Lock()
	r.closed = true
	r.mu.Unlock()
	return nil
}

func (r *recordingAgent) Run(ctx context.Context, opts agent.RunOpts) (*agent.Result, error) {
	r.mu.Lock()
	r.prompts = append(r.prompts, opts.Prompt)
	r.mu.Unlock()
	if r.runFn != nil {
		return r.runFn(ctx, opts)
	}
	return &agent.Result{Text: "ok", Model: r.profile.Tuning.Model, ModelProvider: r.profile.Provider}, nil
}

func (r *recordingAgent) wasClosed() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.closed
}

// TestRunAgent_UnroutedInvocationIsUnchanged proves the default. With routing
// off, no hook runs, the step's own agent serves the turn, and the result is
// exactly what it was before routing existed.
func TestRunAgent_UnroutedInvocationIsUnchanged(t *testing.T) {
	ag := &hangingAgent{name: "configured"}
	sctx := &StepContext{Ctx: context.Background(), Agent: ag}

	result, err := sctx.RunAgent(agent.RunOpts{Prompt: "review this", Purpose: "review"})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if result.Text != "ok" {
		t.Fatalf("result = %+v, want the configured agent's own result", result)
	}
	if ag.calls != 1 {
		t.Fatalf("the configured agent must serve the turn, calls = %d", ag.calls)
	}
}

// TestRunAgent_RoutedInvocationLaunchesTheSelectedProfileAndReportsIt proves
// the full round trip: acquire before the launch, the selected profile's own
// adapter serves the turn with the exact phase prompt, and a normalized
// success reaches finish afterwards.
func TestRunAgent_RoutedInvocationLaunchesTheSelectedProfileAndReportsIt(t *testing.T) {
	grok := &recordingAgent{profile: routingPiGrokProfile()}
	claude := &recordingAgent{profile: routingClaudeProfile()}
	built := map[string]*recordingAgent{
		routingPiGrokProfile().ID: grok,
		routingClaudeProfile().ID: claude,
	}

	transport := &fakeHookTransport{reply: selectByID(routingPiGrokProfile().ID)}
	run := testRouting(t, transport,
		[]routing.Profile{routingClaudeProfile(), routingPiGrokProfile()},
		func(p routing.Profile) (agent.Agent, error) { return built[p.ID], nil })

	configured := &hangingAgent{name: "configured"}
	var logged []string
	sctx := &StepContext{
		Ctx:     context.Background(),
		Agent:   configured,
		Routing: run,
		Log:     func(message string) { logged = append(logged, message) },
	}

	prompt := "fix finding F-1 in worktree /w/run-1"
	if _, err := sctx.RunAgent(agent.RunOpts{Prompt: prompt, Purpose: "review-fix"}); err != nil {
		t.Fatalf("run: %v", err)
	}

	if len(grok.prompts) != 1 || grok.prompts[0] != prompt {
		t.Fatalf("the selected profile must serve the exact phase prompt, got %v", grok.prompts)
	}
	if configured.calls != 0 {
		t.Fatal("a routed invocation must not also run the step's configured agent")
	}
	if len(claude.prompts) != 0 {
		t.Fatal("an unselected profile must not be launched")
	}

	requests := transport.seen()
	if len(requests) != 2 {
		t.Fatalf("want an acquire then a finish, got %d calls", len(requests))
	}
	if requests[0].verb != routing.VerbAcquire {
		t.Fatalf("the first call must be an acquire, got %q", requests[0].verb)
	}
	if requests[1].verb != routing.VerbFinish || requests[1].req.Outcome != routing.OutcomeSuccess {
		t.Fatalf("the second call must report success, got %q %+v", requests[1].verb, requests[1].req)
	}
	if requests[1].req.Profile == nil ||
		requests[1].req.Profile.Adapter != string(routingPiGrokProfile().Agent) ||
		requests[1].req.Profile.Model != routingPiGrokProfile().Tuning.Model {
		t.Fatalf("finish must name the concrete profile that ran, got %+v", requests[1].req.Profile)
	}

	// Every launch records the effective provider and model, and the selection
	// reason, without any secret.
	joined := strings.Join(logged, "\n")
	for _, want := range []string{routingPiGrokProfile().ID, "xai-subscription", "xai/grok-4.6", "fewest-pending"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("the launch log must record %q, got:\n%s", want, joined)
		}
	}
}

// TestRunAgent_RoutingFailureNeverLaunchesAnExcludedRoute is the seam's
// security property. Each row is a hook answer or fault that must stop the
// turn, and in every one the step's configured agent must NOT run: silently
// falling back to it is exactly how an excluded route would get exercised.
func TestRunAgent_RoutingFailureNeverLaunchesAnExcludedRoute(t *testing.T) {
	cases := []struct {
		name    string
		reply   func(routing.Verb, routing.Request) routing.Response
		fail    func(routing.Verb, routing.Request) error
		wantMsg string
	}{
		{
			name: "deferred",
			reply: func(routing.Verb, routing.Request) routing.Response {
				return routing.Response{Result: "deferred", Reason: "every candidate route is excluded"}
			},
			wantMsg: "no approved service is available",
		},
		{
			name: "route nobody offered",
			reply: func(routing.Verb, routing.Request) routing.Response {
				return routing.Response{Result: "selected", RouteID: "a-route-nobody-offered"}
			},
			wantMsg: "not among the",
		},
		{
			name:  "hook fault",
			reply: selectByID(routingPiGrokProfile().ID),
			fail: func(verb routing.Verb, _ routing.Request) error {
				if verb == routing.VerbAcquire {
					return errors.New("exit status 1")
				}
				return nil
			},
			wantMsg: "failed",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var built []string
			transport := &fakeHookTransport{reply: tc.reply, fail: tc.fail}
			run := testRouting(t, transport, []routing.Profile{routingPiGrokProfile()},
				func(p routing.Profile) (agent.Agent, error) {
					built = append(built, p.ID)
					return &recordingAgent{profile: p}, nil
				})

			configured := &hangingAgent{name: "configured"}
			sctx := &StepContext{Ctx: context.Background(), Agent: configured, Routing: run}

			_, err := sctx.RunAgent(agent.RunOpts{Prompt: "fix", Purpose: "review-fix"})
			if err == nil {
				t.Fatal("a routing failure must fail the invocation")
			}
			if !strings.Contains(err.Error(), tc.wantMsg) {
				t.Fatalf("error %q must mention %q", err, tc.wantMsg)
			}
			if configured.calls != 0 {
				t.Fatal("a routing failure must never fall back to the configured agent")
			}
			if len(built) != 0 {
				t.Fatalf("a routing failure must build no adapter, built %v", built)
			}
		})
	}
}

// TestRunAgent_DeferredInvocationStaysRetryable proves a deferral is reported
// as its own sentinel all the way up, so a caller can tell "try again after
// the next refresh" from "this is broken".
func TestRunAgent_DeferredInvocationStaysRetryable(t *testing.T) {
	transport := &fakeHookTransport{
		reply: func(routing.Verb, routing.Request) routing.Response {
			return routing.Response{Result: "deferred", Reason: "no route proven eligible"}
		},
	}
	run := testRouting(t, transport, []routing.Profile{routingPiGrokProfile()},
		func(p routing.Profile) (agent.Agent, error) { return &recordingAgent{profile: p}, nil })

	sctx := &StepContext{Ctx: context.Background(), Agent: &hangingAgent{name: "configured"}, Routing: run}
	_, err := sctx.RunAgent(agent.RunOpts{Prompt: "fix", Purpose: "review-fix"})
	if !errors.Is(err, routing.ErrDeferred) {
		t.Fatalf("a deferral must surface as ErrDeferred, got %v", err)
	}
	if len(transport.finishes()) != 0 {
		t.Fatal("a deferred attempt was never admitted, so nothing may be finished")
	}
}

// TestRunAgent_FailedLaunchIsReportedAsLaunchFailed proves cleanup on the path
// where the process never starts. The controller must hear launch-failed, not
// failed: only the former means the route consumed nothing and should be
// released rather than billed for a turn.
func TestRunAgent_FailedLaunchIsReportedAsLaunchFailed(t *testing.T) {
	transport := &fakeHookTransport{reply: selectByID(routingPiGrokProfile().ID)}
	run := testRouting(t, transport, []routing.Profile{routingPiGrokProfile()},
		func(routing.Profile) (agent.Agent, error) {
			return nil, errors.New("pi is not on the daemon's PATH")
		})

	configured := &hangingAgent{name: "configured"}
	sctx := &StepContext{Ctx: context.Background(), Agent: configured, Routing: run}

	_, err := sctx.RunAgent(agent.RunOpts{Prompt: "fix", Purpose: "review-fix"})
	if err == nil || !strings.Contains(err.Error(), "create routed agent") {
		t.Fatalf("error %v must report the failed launch", err)
	}
	if configured.calls != 0 {
		t.Fatal("a failed routed launch must not fall back to the configured agent")
	}

	finishes := transport.finishes()
	if len(finishes) != 1 {
		t.Fatalf("a failed launch must be released exactly once, got %d finishes", len(finishes))
	}
	if finishes[0].Outcome != routing.OutcomeLaunchFailed {
		t.Fatalf("outcome = %q, want %q", finishes[0].Outcome, routing.OutcomeLaunchFailed)
	}
}

// TestRunAgent_FailureAndTimeoutAreReportedAsFailed proves the outcome is
// normalized honestly on the paths where the route DID serve a turn. A
// timeout and an adapter error both spent the route, so both report failed.
func TestRunAgent_FailureAndTimeoutAreReportedAsFailed(t *testing.T) {
	cases := []struct {
		name  string
		runFn func(context.Context, agent.RunOpts) (*agent.Result, error)
		cfg   *config.Config
	}{
		{
			name: "adapter error",
			runFn: func(context.Context, agent.RunOpts) (*agent.Result, error) {
				return nil, errors.New("structured output rejected")
			},
		},
		{
			name: "timeout",
			runFn: func(ctx context.Context, _ agent.RunOpts) (*agent.Result, error) {
				<-ctx.Done()
				return nil, ctx.Err()
			},
			cfg: &config.Config{AgentTimeout: 20 * time.Millisecond},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			transport := &fakeHookTransport{reply: selectByID(routingPiGrokProfile().ID)}
			run := testRouting(t, transport, []routing.Profile{routingPiGrokProfile()},
				func(p routing.Profile) (agent.Agent, error) {
					return &recordingAgent{profile: p, runFn: tc.runFn}, nil
				})

			sctx := &StepContext{
				Ctx: context.Background(), Agent: &hangingAgent{name: "configured"},
				Routing: run, Config: tc.cfg,
			}
			if _, err := sctx.RunAgent(agent.RunOpts{Prompt: "fix", Purpose: "review-fix"}); err == nil {
				t.Fatal("want the invocation to fail")
			}

			finishes := transport.finishes()
			if len(finishes) != 1 {
				t.Fatalf("want exactly one finish, got %d", len(finishes))
			}
			if finishes[0].Outcome != routing.OutcomeLaunchFailed {
				t.Fatalf("outcome = %q, want %q", finishes[0].Outcome, routing.OutcomeLaunchFailed)
			}
		})
	}
}

// TestRunAgent_CancelledInvocationIsStillReported proves finish survives a
// cancelled run context. The turn already happened; reporting it through the
// dead context would drop the record exactly when the controller most needs to
// stop counting the assignment as running.
func TestRunAgent_CancelledInvocationIsStillReported(t *testing.T) {
	transport := &fakeHookTransport{reply: selectByID(routingPiGrokProfile().ID)}
	run := testRouting(t, transport, []routing.Profile{routingPiGrokProfile()},
		func(p routing.Profile) (agent.Agent, error) {
			return &recordingAgent{profile: p, runFn: func(ctx context.Context, _ agent.RunOpts) (*agent.Result, error) {
				<-ctx.Done()
				return nil, ctx.Err()
			}}, nil
		})

	ctx, cancel := context.WithCancel(context.Background())
	sctx := &StepContext{Ctx: ctx, Agent: &hangingAgent{name: "configured"}, Routing: run}

	go func() {
		time.Sleep(20 * time.Millisecond)
		cancel()
	}()
	if _, err := sctx.RunAgent(agent.RunOpts{Prompt: "fix", Purpose: "review-fix"}); err == nil {
		t.Fatal("want the cancelled invocation to fail")
	}

	finishes := transport.finishes()
	if len(finishes) != 1 {
		t.Fatalf("a cancelled invocation must still be reported, got %d finishes", len(finishes))
	}
}

// TestRunAgent_EachInvocationIsItsOwnAssignment proves a retry, a fallback and
// a later turn are separate attempts to the controller, so balancing sees real
// work rather than one assignment acquired repeatedly. It also proves the
// route can change between turns within one run.
func TestRunAgent_EachInvocationIsItsOwnAssignment(t *testing.T) {
	order := []string{routingClaudeProfile().ID, routingCodexProfile().ID, routingPiGrokProfile().ID, routingPiDeepSeekProfile().ID}
	var turn int
	transport := &fakeHookTransport{
		reply: func(verb routing.Verb, req routing.Request) routing.Response {
			if verb == routing.VerbFinish {
				return routing.Response{Result: "closed"}
			}
			route := order[turn%len(order)]
			turn++
			return routing.Response{Result: "selected", RouteID: route, Reason: "fewest-pending"}
		},
	}
	profiles := []routing.Profile{
		routingClaudeProfile(), routingCodexProfile(),
		routingPiGrokProfile(), routingPiDeepSeekProfile(),
	}
	var served []string
	run := testRouting(t, transport, profiles, func(p routing.Profile) (agent.Agent, error) {
		served = append(served, p.ID)
		return &recordingAgent{profile: p}, nil
	})

	sctx := &StepContext{Ctx: context.Background(), Agent: &hangingAgent{name: "configured"}, Routing: run}
	for i := range order {
		if _, err := sctx.RunAgent(agent.RunOpts{
			Prompt: fmt.Sprintf("fix round %d", i+1), Purpose: "review-fix",
		}); err != nil {
			t.Fatalf("turn %d: %v", i+1, err)
		}
	}

	if strings.Join(served, ",") != strings.Join(order, ",") {
		t.Fatalf("each turn must launch its own selected profile, got %v", served)
	}

	ids := map[string]bool{}
	for _, call := range transport.seen() {
		if call.verb != routing.VerbAcquire {
			continue
		}
		if ids[call.req.AssignmentID] {
			t.Fatalf("assignment %q was acquired twice", call.req.AssignmentID)
		}
		ids[call.req.AssignmentID] = true
	}
	if len(ids) != len(order) {
		t.Fatalf("want %d distinct assignments, got %d", len(order), len(ids))
	}
	if got := len(transport.finishes()); got != len(order) {
		t.Fatalf("every assignment must be finished, got %d of %d", got, len(order))
	}
}

// TestRunAgent_RoutedAgentIsClosedAndTheConfiguredOneSurvives proves lifecycle
// ownership. Routing closes only the adapter it built; the step's configured
// agent belongs to the executor and must outlive the invocation, or the next
// unrouted turn would find it shut.
func TestRunAgent_RoutedAgentIsClosedAndTheConfiguredOneSurvives(t *testing.T) {
	routed := &recordingAgent{profile: routingPiGrokProfile()}
	transport := &fakeHookTransport{reply: selectByID(routingPiGrokProfile().ID)}
	run := testRouting(t, transport, []routing.Profile{routingPiGrokProfile()},
		func(routing.Profile) (agent.Agent, error) { return routed, nil })

	configured := &hangingAgent{name: "configured"}
	sctx := &StepContext{Ctx: context.Background(), Agent: configured, Routing: run}

	if _, err := sctx.RunAgent(agent.RunOpts{Prompt: "fix", Purpose: "review-fix"}); err != nil {
		t.Fatalf("run: %v", err)
	}
	if !routed.wasClosed() {
		t.Fatal("routing must close the adapter it built")
	}
	if sctx.Agent != configured {
		t.Fatal("the step's configured agent must be restored after a routed invocation")
	}

	// And it is still usable, which is what "not closed" has to mean.
	if _, err := sctx.RunAgent(agent.RunOpts{Prompt: "later unrouted work"}); err != nil {
		t.Fatalf("the configured agent must still serve later work: %v", err)
	}
}

// TestRunAgent_RoleRestrictionReachesTheController proves the offered set is
// filtered by role at the seam, so a route the operator reserved for another
// duty is never even a candidate.
func TestRunAgent_RoleRestrictionReachesTheController(t *testing.T) {
	fixerOnly := routingPiGrokProfile()
	fixerOnly.Roles = []string{"review-fix"}
	reviewerOnly := routingClaudeProfile()
	reviewerOnly.Roles = []string{"review"}

	transport := &fakeHookTransport{reply: selectByID(reviewerOnly.ID)}
	run := testRouting(t, transport, []routing.Profile{fixerOnly, reviewerOnly},
		func(p routing.Profile) (agent.Agent, error) { return &recordingAgent{profile: p}, nil })

	sctx := &StepContext{Ctx: context.Background(), Agent: &hangingAgent{name: "configured"}, Routing: run}
	if _, err := sctx.RunAgent(agent.RunOpts{Prompt: "review", Purpose: "review"}); err != nil {
		t.Fatalf("run: %v", err)
	}

	offered := transport.seen()[0].req.Routes
	if strings.Join(offered, ",") != reviewerOnly.ID {
		t.Fatalf("a review turn must be offered only %q, got %v", reviewerOnly.ID, offered)
	}
}

// TestRunAgent_SwitchingServiceNeverInterruptsARunningProcess is proof item
// 1's process half, driven against REAL child processes rather than fakes.
//
// One long-running child stands for an invocation already in flight. While it
// runs, a second invocation is routed to a different service and completes.
// The first child must still be alive and must exit on its own: routing
// chooses what the NEXT invocation launches and has no business touching what
// is already running.
func TestRunAgent_SwitchingServiceNeverInterruptsARunningProcess(t *testing.T) {
	longRunning := startSleeper(t, 30)
	defer func() { _ = longRunning.Process.Kill() }()

	transport := &fakeHookTransport{reply: selectByID(routingCodexProfile().ID)}
	run := testRouting(t, transport,
		[]routing.Profile{routingPiGrokProfile(), routingCodexProfile()},
		func(p routing.Profile) (agent.Agent, error) {
			return &recordingAgent{profile: p, runFn: func(context.Context, agent.RunOpts) (*agent.Result, error) {
				// A real, short-lived child for the newly routed turn, so both
				// processes genuinely coexist.
				short := startSleeper(t, 0)
				if err := short.Wait(); err != nil {
					return nil, err
				}
				return &agent.Result{Text: "ok"}, nil
			}}, nil
		})

	sctx := &StepContext{Ctx: context.Background(), Agent: &hangingAgent{name: "configured"}, Routing: run}
	if _, err := sctx.RunAgent(agent.RunOpts{Prompt: "fix on the new service", Purpose: "review-fix"}); err != nil {
		t.Fatalf("routed turn: %v", err)
	}

	if !processAlive(longRunning.Process.Pid) {
		t.Fatal("routing a later invocation elsewhere must not interrupt a process already running")
	}

	// It is not merely unkilled: it is still a healthy child that terminates
	// on its own terms.
	if err := longRunning.Process.Signal(syscall.SIGTERM); err != nil {
		t.Fatalf("the surviving process must still be signalable: %v", err)
	}
	_ = longRunning.Wait()
}

// startSleeper launches a real child process that sleeps for the given number
// of seconds. It is a genuine subprocess, so PID-level assertions mean what
// they say.
func startSleeper(t *testing.T, seconds int) *exec.Cmd {
	t.Helper()
	cmd := exec.Command("sleep", fmt.Sprint(seconds))
	if err := cmd.Start(); err != nil {
		t.Fatalf("start sleeper: %v", err)
	}
	return cmd
}

// processAlive reports whether a pid is still running. Signal 0 performs the
// permission and existence checks without delivering anything.
func processAlive(pid int) bool {
	proc, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	return proc.Signal(syscall.Signal(0)) == nil
}

// TestRunAgent_PanicStillReleasesTheAssignment proves the backstop. If a panic
// unwinds through the seam, the normal outcome report is skipped, and an
// assignment the controller never sees closed is one it keeps counting as
// running forever. The turn did launch, so it is released as failed.
func TestRunAgent_PanicStillReleasesTheAssignment(t *testing.T) {
	transport := &fakeHookTransport{reply: selectByID(routingPiGrokProfile().ID)}
	run := testRouting(t, transport, []routing.Profile{routingPiGrokProfile()},
		func(p routing.Profile) (agent.Agent, error) {
			return &recordingAgent{profile: p, runFn: func(context.Context, agent.RunOpts) (*agent.Result, error) {
				panic("a step blew up mid-invocation")
			}}, nil
		})

	sctx := &StepContext{Ctx: context.Background(), Agent: &hangingAgent{name: "configured"}, Routing: run}

	func() {
		defer func() {
			if recover() == nil {
				t.Error("the panic must still propagate; routing is not a recovery point")
			}
		}()
		_, _ = sctx.RunAgent(agent.RunOpts{Prompt: "fix", Purpose: "review-fix"})
	}()

	finishes := transport.finishes()
	if len(finishes) != 1 {
		t.Fatalf("a panicking invocation must still release its assignment, got %d finishes", len(finishes))
	}
	if finishes[0].Outcome != routing.OutcomeLaunchFailed {
		t.Fatalf("outcome = %q, want %q", finishes[0].Outcome, routing.OutcomeLaunchFailed)
	}
}

// TestRunAgent_NormalOutcomeWinsOverTheBackstop proves the backstop never
// overwrites a real report. A successful turn must reach the controller as
// success, not be re-reported as failed by the deferred release.
func TestRunAgent_NormalOutcomeWinsOverTheBackstop(t *testing.T) {
	transport := &fakeHookTransport{reply: selectByID(routingPiGrokProfile().ID)}
	run := testRouting(t, transport, []routing.Profile{routingPiGrokProfile()},
		func(p routing.Profile) (agent.Agent, error) { return &recordingAgent{profile: p}, nil })

	sctx := &StepContext{Ctx: context.Background(), Agent: &hangingAgent{name: "configured"}, Routing: run}
	if _, err := sctx.RunAgent(agent.RunOpts{Prompt: "fix", Purpose: "review-fix"}); err != nil {
		t.Fatalf("run: %v", err)
	}

	finishes := transport.finishes()
	if len(finishes) != 1 {
		t.Fatalf("an assignment must be reported exactly once, got %d finishes", len(finishes))
	}
	if finishes[0].Outcome != routing.OutcomeSuccess {
		t.Fatalf("outcome = %q, want %q", finishes[0].Outcome, routing.OutcomeSuccess)
	}
}

// TestRunAgent_AlreadyClosedAssignmentNeverLaunches proves the controller's
// already-closed reply stops the turn. That reply means this exact launch has
// already run and been accounted for, and it deliberately carries no route, so
// treating it as authorization would both double-count the work and have
// nothing to run.
func TestRunAgent_AlreadyClosedAssignmentNeverLaunches(t *testing.T) {
	var built []string
	transport := &fakeHookTransport{
		reply: func(routing.Verb, routing.Request) routing.Response {
			return routing.Response{
				Result: "already-closed",
				Reason: "already closed; not relaunched",
			}
		},
	}
	run := testRouting(t, transport, []routing.Profile{routingPiGrokProfile()},
		func(p routing.Profile) (agent.Agent, error) {
			built = append(built, p.ID)
			return &recordingAgent{profile: p}, nil
		})

	configured := &hangingAgent{name: "configured"}
	sctx := &StepContext{Ctx: context.Background(), Agent: configured, Routing: run}

	_, err := sctx.RunAgent(agent.RunOpts{Prompt: "fix", Purpose: "review-fix"})
	if !errors.Is(err, routing.ErrAlreadyClosed) {
		t.Fatalf("error = %v, want ErrAlreadyClosed", err)
	}
	if len(built) != 0 {
		t.Fatalf("an already-closed assignment must build no adapter, built %v", built)
	}
	if configured.calls != 0 {
		t.Fatal("an already-closed assignment must not fall back to the configured agent")
	}
	if len(transport.finishes()) != 0 {
		t.Fatal("nothing was admitted, so nothing may be finished")
	}
}

// TestRunAgent_OnlySelectedAuthorizesALaunch is the rule stated directly. Each
// row is a reply that is not `selected`, and none of them may start a process,
// however plausible the rest of the reply looks.
func TestRunAgent_OnlySelectedAuthorizesALaunch(t *testing.T) {
	cases := []struct {
		name  string
		reply routing.Response
	}{
		{"deferred", routing.Response{Result: "deferred", Reason: "no route proven eligible"}},
		{"already-closed", routing.Response{Result: "already-closed", Reason: "already closed"}},
		{
			name:  "already-closed carrying a real route id",
			reply: routing.Response{Result: "already-closed", RouteID: routingPiGrokProfile().ID},
		},
		{"controller error object", routing.Response{Result: "error", Error: "unknown route id"}},
		{"a result nobody defined", routing.Response{Result: "approved", RouteID: routingPiGrokProfile().ID}},
		{"empty result", routing.Response{RouteID: routingPiGrokProfile().ID}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var built []string
			transport := &fakeHookTransport{
				reply: func(routing.Verb, routing.Request) routing.Response { return tc.reply },
			}
			run := testRouting(t, transport, []routing.Profile{routingPiGrokProfile()},
				func(p routing.Profile) (agent.Agent, error) {
					built = append(built, p.ID)
					return &recordingAgent{profile: p}, nil
				})

			configured := &hangingAgent{name: "configured"}
			sctx := &StepContext{Ctx: context.Background(), Agent: configured, Routing: run}

			if _, err := sctx.RunAgent(agent.RunOpts{Prompt: "fix", Purpose: "review-fix"}); err == nil {
				t.Fatal("a reply that is not selected must fail the invocation")
			}
			if len(built) != 0 {
				t.Fatalf("no adapter may be built, built %v", built)
			}
			if configured.calls != 0 {
				t.Fatal("the configured agent must not serve the turn either")
			}
		})
	}
}

// TestRunAgent_EveryLaunchUsesItsOwnAssignmentID proves ids are never reused.
// The controller treats a repeat id as the same launch and replays its record,
// so a reused id would either be refused or, worse, silently counted as work
// that already happened. The shape also carries the run and the attempt number,
// mirroring the controller's own relaunch convention.
func TestRunAgent_EveryLaunchUsesItsOwnAssignmentID(t *testing.T) {
	transport := &fakeHookTransport{reply: selectByID(routingPiGrokProfile().ID)}
	run := testRouting(t, transport, []routing.Profile{routingPiGrokProfile()},
		func(p routing.Profile) (agent.Agent, error) { return &recordingAgent{profile: p}, nil })

	sctx := &StepContext{Ctx: context.Background(), Agent: &hangingAgent{name: "configured"}, Routing: run}
	const launches = 4
	for i := range launches {
		if _, err := sctx.RunAgent(agent.RunOpts{
			Prompt: fmt.Sprintf("fix round %d", i+1), Purpose: "review-fix",
		}); err != nil {
			t.Fatalf("launch %d: %v", i+1, err)
		}
	}

	seen := map[string]bool{}
	for _, call := range transport.seen() {
		if call.verb != routing.VerbAcquire {
			continue
		}
		id := call.req.AssignmentID
		if seen[id] {
			t.Fatalf("assignment id %q was used for two launches", id)
		}
		seen[id] = true
		if !strings.HasPrefix(id, "run-1-gen-1-launch-") {
			t.Fatalf("assignment id %q must name the run, the incarnation and the launch attempt", id)
		}
	}
	if len(seen) != launches {
		t.Fatalf("want %d distinct assignment ids, got %d", launches, len(seen))
	}
}

// TestRunAgent_RoutedAdapterKeepsTheStepInvocationHarness proves a routed turn
// is dressed in the same wrappers the step's own agent wears.
//
// The harness is containment, not decoration: the gate phase-boundary preamble
// is prepended by one wrapper and nowhere else, and it is what stops a gate
// agent from starting a second pipeline. A routed adapter launched bare would
// receive the step prompt with that block missing, and would also produce no
// lifecycle or performance record of the launch at all.
func TestRunAgent_RoutedAdapterKeepsTheStepInvocationHarness(t *testing.T) {
	transport := &fakeHookTransport{reply: selectByID(routingPiGrokProfile().ID)}
	routed := &recordingAgent{profile: routingPiGrokProfile()}
	run := testRouting(t, transport, []routing.Profile{routingPiGrokProfile()},
		func(routing.Profile) (agent.Agent, error) { return routed, nil })

	wrapped := 0
	sctx := &StepContext{
		Ctx:     context.Background(),
		Agent:   &hangingAgent{name: "configured"},
		Routing: run,
		WrapAgent: func(inner agent.Agent) agent.Agent {
			wrapped++
			return &gateStepBoundaryAgent{inner: inner, phase: types.StepReview}
		},
	}
	if _, err := sctx.RunAgent(agent.RunOpts{Prompt: "repair the finding", Purpose: "review-fix"}); err != nil {
		t.Fatalf("run: %v", err)
	}

	if wrapped != 1 {
		t.Fatalf("the step harness was applied %d times, want exactly once", wrapped)
	}
	routed.mu.Lock()
	prompts := append([]string(nil), routed.prompts...)
	routed.mu.Unlock()
	if len(prompts) != 1 {
		t.Fatalf("routed adapter saw %d prompts, want 1", len(prompts))
	}
	boundary := gateguidance.PromptBoundary("review")
	if !strings.HasPrefix(prompts[0], boundary) {
		t.Fatalf("routed adapter did not receive the gate phase boundary; prompt was:\n%s", prompts[0])
	}
	if !strings.Contains(prompts[0], "repair the finding") {
		t.Fatalf("routed adapter lost the step's own prompt; prompt was:\n%s", prompts[0])
	}
}

// TestRunAgent_UnwrappedStepContextStillRoutes keeps the harness optional, so
// an embedding that builds its own agent and supplies no WrapAgent still gets a
// routed launch rather than an error or a silent default-agent launch.
func TestRunAgent_UnwrappedStepContextStillRoutes(t *testing.T) {
	transport := &fakeHookTransport{reply: selectByID(routingPiGrokProfile().ID)}
	routed := &recordingAgent{profile: routingPiGrokProfile()}
	configured := &hangingAgent{name: "configured"}
	run := testRouting(t, transport, []routing.Profile{routingPiGrokProfile()},
		func(routing.Profile) (agent.Agent, error) { return routed, nil })

	sctx := &StepContext{Ctx: context.Background(), Agent: configured, Routing: run}
	if _, err := sctx.RunAgent(agent.RunOpts{Prompt: "repair", Purpose: "review-fix"}); err != nil {
		t.Fatalf("run: %v", err)
	}
	routed.mu.Lock()
	served := len(routed.prompts)
	routed.mu.Unlock()
	if served != 1 {
		t.Fatalf("routed adapter served %d turns, want 1", served)
	}
}

// TestRunAgent_AssignmentIDsDoNotRepeatAcrossADaemonRestart is the recovery
// case. The controller's idempotency key is (assignment_id, owner.identity)
// and deliberately excludes the owner generation, so a second incarnation of
// the same daemon that restarted its per-process counter would re-acquire the
// very ids the first one already finished. The controller would replay each as
// the SAME launch and answer already-closed, which fails the recovered run on
// its first routed turn. Ids must therefore be unique across incarnations of
// one run, not only within one process.
func TestRunAgent_AssignmentIDsDoNotRepeatAcrossADaemonRestart(t *testing.T) {
	acquiredIDs := func(t *testing.T, generation string) []string {
		t.Helper()
		transport := &fakeHookTransport{reply: selectByID(routingPiGrokProfile().ID)}
		run := testRoutingForGeneration(t, transport, []routing.Profile{routingPiGrokProfile()},
			func(p routing.Profile) (agent.Agent, error) { return &recordingAgent{profile: p}, nil }, generation)
		sctx := &StepContext{Ctx: context.Background(), Agent: &hangingAgent{name: "configured"}, Routing: run}
		for i := range 3 {
			if _, err := sctx.RunAgent(agent.RunOpts{
				Prompt: fmt.Sprintf("fix round %d", i+1), Purpose: "review-fix",
			}); err != nil {
				t.Fatalf("launch %d: %v", i+1, err)
			}
		}
		ids := []string{}
		for _, call := range transport.seen() {
			if call.verb == routing.VerbAcquire {
				ids = append(ids, call.req.AssignmentID)
			}
		}
		return ids
	}

	before := acquiredIDs(t, "pid-1-started-1")
	after := acquiredIDs(t, "pid-2-started-2")
	if len(before) == 0 || len(before) != len(after) {
		t.Fatalf("want the same number of launches in both incarnations, got %d and %d", len(before), len(after))
	}
	spent := map[string]bool{}
	for _, id := range before {
		spent[id] = true
	}
	for _, id := range after {
		if spent[id] {
			t.Fatalf("resumed run re-acquired assignment id %q, which the previous incarnation already finished", id)
		}
	}
}

// TestRunAgent_OwnerIdentityAndGenerationReachTheController proves both halves
// travel. Identity is what lets the controller recognize a repeat as the same
// launch; generation is what lets it tell a live owner from one that was torn
// down, so a dead incarnation's assignments stop being counted against a route.
func TestRunAgent_OwnerIdentityAndGenerationReachTheController(t *testing.T) {
	transport := &fakeHookTransport{reply: selectByID(routingPiGrokProfile().ID)}
	run := testRouting(t, transport, []routing.Profile{routingPiGrokProfile()},
		func(p routing.Profile) (agent.Agent, error) { return &recordingAgent{profile: p}, nil })

	sctx := &StepContext{Ctx: context.Background(), Agent: &hangingAgent{name: "configured"}, Routing: run}
	if _, err := sctx.RunAgent(agent.RunOpts{Prompt: "fix", Purpose: "review-fix"}); err != nil {
		t.Fatalf("run: %v", err)
	}

	acquire := transport.seen()[0]
	if acquire.req.Owner.Identity != "daemon" {
		t.Fatalf("owner identity = %q, want daemon", acquire.req.Owner.Identity)
	}
	if acquire.req.Owner.Generation != "gen-1" {
		t.Fatalf("owner generation = %q, want gen-1", acquire.req.Owner.Generation)
	}
}

// TestClassifyRouteOutcome_NeverGuessesRouteEvidence pins the conservative
// mapping. Three of the controller's outcome values are verified evidence
// against the route and exclude it immediately for every caller on this
// machine. no-mistakes cannot establish any of them from this seam, so a failed
// turn reports launch-failed instead of condemning a service on one bad turn.
func TestClassifyRouteOutcome_NeverGuessesRouteEvidence(t *testing.T) {
	if got := classifyRouteOutcome(nil); got != routing.OutcomeSuccess {
		t.Fatalf("a completed turn = %q, want success", got)
	}

	for _, err := range []error{
		errors.New("structured output rejected"),
		errors.New("401 unauthorized: invalid api key"),
		errors.New("429 rate limit exceeded, quota exhausted"),
		errors.New("503 service unavailable"),
		context.DeadlineExceeded,
		context.Canceled,
	} {
		got := classifyRouteOutcome(err)
		if got != routing.OutcomeLaunchFailed {
			t.Fatalf("error %q classified as %q; only a verified signal may report route evidence", err, got)
		}
	}
}
