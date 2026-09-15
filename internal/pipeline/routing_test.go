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
	"github.com/kunchenguid/no-mistakes/internal/routing"
)

// These tests pin the invocation seam: every native invocation acquires a
// route before it launches and reports a normalized outcome afterwards, a
// routing failure never launches anything, and nothing already running is
// disturbed by a later invocation choosing a different service.

// fakeHookTransport drives routing.Hook without a process, recording every
// request so a test can assert exactly what the controller was told.
type fakeHookTransport struct {
	mu       sync.Mutex
	requests []routing.Request
	reply    func(routing.Request) routing.Response
	fail     func(routing.Request) error
}

func (f *fakeHookTransport) seen() []routing.Request {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]routing.Request(nil), f.requests...)
}

func (f *fakeHookTransport) finishes() []routing.Request {
	var out []routing.Request
	for _, req := range f.seen() {
		if req.Verb == routing.VerbFinish {
			out = append(out, req)
		}
	}
	return out
}

// testRouting builds a RunRouting whose hook is driven by transport and whose
// adapters come from newAgent, mirroring how the daemon wires the real thing.
func testRouting(t *testing.T, transport *fakeHookTransport, profiles []routing.Profile,
	newAgent func(routing.Profile) (agent.Agent, error)) *RunRouting {
	t.Helper()

	hook := routing.NewTestHook(func(_ context.Context, stdin []byte) ([]byte, error) {
		var req routing.Request
		if err := json.Unmarshal(stdin, &req); err != nil {
			t.Fatalf("unreadable request: %v", err)
		}
		transport.mu.Lock()
		transport.requests = append(transport.requests, req)
		fail, reply := transport.fail, transport.reply
		transport.mu.Unlock()

		if fail != nil {
			if err := fail(req); err != nil {
				return nil, err
			}
		}
		payload, err := json.Marshal(reply(req))
		if err != nil {
			t.Fatalf("marshal reply: %v", err)
		}
		return payload, nil
	})

	router, err := routing.NewRouter(hook, profiles, "daemon", "gen-1")
	if err != nil {
		t.Fatalf("new router: %v", err)
	}
	return NewRunRouting(router, "run-1", newAgent)
}

func selectByID(id string) func(routing.Request) routing.Response {
	return func(req routing.Request) routing.Response {
		if req.Verb == routing.VerbFinish {
			return routing.Response{Status: "closed"}
		}
		return routing.Response{Status: "selected", Route: id, Reason: "fewest-pending"}
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
		t.Fatalf("want an acquire then a finish, got %d requests", len(requests))
	}
	if requests[0].Verb != routing.VerbAcquire || requests[0].Role != "review-fix" {
		t.Fatalf("the first request must acquire for this role, got %+v", requests[0])
	}
	if requests[1].Verb != routing.VerbFinish || requests[1].Outcome != routing.OutcomeSuccess {
		t.Fatalf("the second request must report success, got %+v", requests[1])
	}
	if requests[1].Profile != routingPiGrokProfile().ID {
		t.Fatalf("finish must name the profile that ran, got %q", requests[1].Profile)
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
		reply   func(routing.Request) routing.Response
		fail    func(routing.Request) error
		wantMsg string
	}{
		{
			name: "deferred",
			reply: func(routing.Request) routing.Response {
				return routing.Response{Status: "deferred", Reason: "every candidate route is excluded"}
			},
			wantMsg: "no approved service is available",
		},
		{
			name: "route nobody offered",
			reply: func(routing.Request) routing.Response {
				return routing.Response{Status: "selected", Route: "a-route-nobody-offered"}
			},
			wantMsg: "not among the",
		},
		{
			name:  "hook fault",
			reply: selectByID(routingPiGrokProfile().ID),
			fail: func(req routing.Request) error {
				if req.Verb == routing.VerbAcquire {
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
		reply: func(routing.Request) routing.Response {
			return routing.Response{Status: "deferred", Reason: "no route proven eligible"}
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
			if finishes[0].Outcome != routing.OutcomeFailed {
				t.Fatalf("outcome = %q, want %q", finishes[0].Outcome, routing.OutcomeFailed)
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
		reply: func(req routing.Request) routing.Response {
			if req.Verb == routing.VerbFinish {
				return routing.Response{Status: "closed"}
			}
			route := order[turn%len(order)]
			turn++
			return routing.Response{Status: "selected", Route: route, Reason: "fewest-pending"}
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
	for _, req := range transport.seen() {
		if req.Verb != routing.VerbAcquire {
			continue
		}
		if ids[req.Assignment] {
			t.Fatalf("assignment %q was acquired twice", req.Assignment)
		}
		ids[req.Assignment] = true
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

	offered := transport.seen()[0].Routes
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
	if finishes[0].Outcome != routing.OutcomeFailed {
		t.Fatalf("outcome = %q, want %q", finishes[0].Outcome, routing.OutcomeFailed)
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
