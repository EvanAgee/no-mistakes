package pipeline

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync/atomic"

	"github.com/kunchenguid/no-mistakes/internal/agent"
	"github.com/kunchenguid/no-mistakes/internal/routing"
)

// RunRouting is one run's assignment routing, as the pipeline sees it. It
// pairs the router (which decides) with a factory (which builds the adapter
// the decision names), so this package never has to know how an adapter is
// constructed and internal/routing never has to know that adapters exist.
//
// A nil *RunRouting means routing is unconfigured. Every path below is written
// so that case is byte-for-byte the pre-routing behavior: no hook runs, no
// profile key is computed, and the step's own configured agent launches.
type RunRouting struct {
	router *routing.Router
	// newAgent builds the native adapter for one approved profile. The
	// executor supplies it. It receives a Profile that came out of the
	// operator's own configuration and was matched by id, never a string the
	// hook supplied, which is what keeps a hook from naming an executable.
	newAgent func(routing.Profile) (agent.Agent, error)
	// seq numbers assignment ids within the run, so a retry, a fallback and a
	// recovered turn are each distinct attempts to the hook rather than one
	// attempt acquired repeatedly.
	seq atomic.Uint64
	// launchPrefix is what every assignment id of this run begins with. It
	// carries both the run (so two runs never collide) and this daemon
	// incarnation (so a run RESUMED after a restart never replays an id the
	// previous incarnation already spent). The controller's idempotency key is
	// (assignment_id, owner.identity) and excludes the owner generation on
	// purpose, so a bare "<runID>-launch-<n>" restarted at n=1 would be
	// answered `already-closed` and fail the recovered run on its first
	// routed turn. Folding the generation in makes each incarnation's counter
	// its own id space.
	launchPrefix string
}

// NewRunRouting builds the run's routing, or nil when routing is off.
func NewRunRouting(router *routing.Router, runID string, newAgent func(routing.Profile) (agent.Agent, error)) *RunRouting {
	if router == nil || newAgent == nil {
		return nil
	}
	prefix := runID
	if generation := router.OwnerGeneration(); generation != "" {
		prefix = runID + "-" + generation
	}
	return &RunRouting{router: router, newAgent: newAgent, launchPrefix: prefix}
}

// routedInvocation is what acquireRoute hands back to the invocation seam.
type routedInvocation struct {
	assignment routing.Assignment
	// closed records that the finish this invocation owes the hook has been
	// delivered and accepted. It is set only after a successful report, so a
	// report that never reached the controller leaves the debt outstanding for
	// a later path to settle rather than marking it paid.
	closed atomic.Bool
	// recorded is the outcome the seam established for this turn, held so a
	// retry reports what actually happened rather than re-deciding it. The
	// backstop has no way to observe a turn it did not run, so without this it
	// could only guess, and guessing `launch-failed` for a turn that completed
	// tells the controller the route consumed nothing when it consumed a whole
	// turn's quota.
	recorded atomic.Pointer[routing.Outcome]
}

// rememberOutcome records what the seam established for this turn, so a later
// retry of a lost report carries the same verdict.
func (r *routedInvocation) rememberOutcome(outcome routing.Outcome) {
	if r == nil {
		return
	}
	r.recorded.Store(&outcome)
}

// outcomeForRetry is what a backstop should report. It is the recorded verdict
// when the seam reached one, and `launch-failed` only when it never did: that
// is the genuine no-outcome case the backstop exists for, a panic unwinding
// through the seam before the turn's result was ever classified.
func (r *routedInvocation) outcomeForRetry() routing.Outcome {
	if r == nil {
		return routing.OutcomeLaunchFailed
	}
	if recorded := r.recorded.Load(); recorded != nil {
		return *recorded
	}
	return routing.OutcomeLaunchFailed
}

// ProfileKey is the session-reuse identity of the assigned profile. A nil
// invocation is the unrouted case, which carries the explicit unrouted key
// rather than the empty "unknown" one, so a run with routing off keeps reusing
// its session exactly as it did before routing existed.
func (r *routedInvocation) ProfileKey() string {
	if r == nil {
		return routing.UnroutedProfileKey
	}
	return r.assignment.ProfileKey()
}

// acquireRoute asks the hook which approved route serves this invocation and,
// when one is admitted, swaps in the adapter that route names.
//
// It deliberately does not touch anything already running. Swapping the
// adapter for the NEXT invocation is the whole mechanism: a process in flight
// keeps its own adapter, its own session and its own subprocess, and the
// switch takes effect only when the pipeline next asks for an invocation. The
// run's identity, worktree, prompt, findings and gates are untouched, because
// this seam sits below all of them and only decides which binary runs.
//
// Every failure closes: a deferred assignment, an unknown route, malformed
// output, a hook error or a timeout all return an error and launch nothing.
// Launching the step's default agent on a routing failure would be the one
// unacceptable outcome, because that is precisely how an excluded route would
// get exercised anyway.
func (sctx *StepContext) acquireRoute(ctx context.Context, opts *agent.RunOpts, ag *agent.Agent) (*routedInvocation, bool, func(), error) {
	if sctx == nil || sctx.Routing == nil {
		return nil, false, nil, nil
	}
	r := sctx.Routing

	// Every launch gets its own assignment id, never a reused one. The
	// controller treats a repeat id as the SAME launch and replays its record
	// rather than admitting a new one, so a retry, a fallback to another
	// adapter and a recovered turn would each be silently refused - or, worse,
	// counted as work that already happened - if they shared an id.
	//
	// The shape mirrors the controller's own relaunch convention
	// (<task>-relaunch-<generation>): the run and this daemon incarnation
	// identify the work, and the monotonic counter identifies which launch
	// attempt within it.
	assignmentID := fmt.Sprintf("%s-launch-%d", r.launchPrefix, r.seq.Add(1))
	assignment, selected, err := r.router.Route(ctx, assignmentID, opts.Purpose)
	if err != nil {
		switch {
		case errors.Is(err, routing.ErrDeferred):
			// Not a fault and not permanent: the routing controller admitted
			// nothing right now. The attempt keeps its own id, so retrying it
			// after the next refresh is a fresh acquire, not a replay of this
			// verdict.
			return nil, false, nil, fmt.Errorf("no approved service is available for this %s invocation right now: %w", opts.Purpose, err)
		case errors.Is(err, routing.ErrAlreadyClosed):
			// The controller has already accounted for this exact launch. It
			// deliberately returns no route, so there is nothing to run; a
			// genuine relaunch is a new launch with its own id. Reaching this
			// means the id was reused, which is a fault in this seam rather
			// than a routing verdict.
			return nil, false, nil, fmt.Errorf("routing refused to relaunch %s under an assignment that already ran: %w", assignmentID, err)
		}
		return nil, false, nil, err
	}
	// Belt and braces: only a selected decision authorizes a launch, and a
	// decision that is not selected must never reach the adapter factory.
	if !selected {
		return nil, false, nil, nil
	}

	invocation := &routedInvocation{assignment: assignment}
	routedAgent, err := sctx.buildRoutedAgent(assignment.Profile)
	if err != nil {
		// The process never started, so the route consumed nothing. Report
		// that distinctly: a controller counting in-flight work must release
		// this assignment rather than bill a turn that never ran.
		r.finish(ctx, invocation, routing.OutcomeLaunchFailed)
		return nil, false, nil, fmt.Errorf("create routed agent for profile %s: %w", assignment.Profile.ID, err)
	}

	if sctx.Log != nil {
		sctx.Log(fmt.Sprintf("routing %s invocation to profile %s (%s %s, provider %s): %s",
			opts.Purpose, assignment.Profile.ID, assignment.Profile.Agent,
			describeModel(assignment.Profile), assignment.Profile.Provider, describeReason(assignment)))
	}

	previous := *ag
	*ag = routedAgent
	release := func() {
		// Close only what routing itself built. The step's own configured
		// agent outlives this invocation and is the executor's to close.
		_ = routedAgent.Close()
		*ag = previous
		// Backstop: an assignment the controller never sees closed is one it
		// keeps counting as running forever. It fires in two shapes, and the
		// outcome it reports must not conflate them. The caller normally
		// records the real verdict first, so this either finds that debt
		// already settled and does nothing, or retries the SAME verdict after
		// a report the hook never accepted. Only when no verdict was ever
		// reached - a panic unwinding through the seam before the turn was
		// classified - does it report `launch-failed` on its own account.
		r.finish(context.WithoutCancel(ctx), invocation, invocation.outcomeForRetry())
	}
	return invocation, true, release, nil
}

// buildRoutedAgent constructs the adapter a selected profile names and dresses
// it in the same harness the step's own agent wears.
//
// The harness is not decoration: gateStepBoundaryAgent is the only place the
// gate phase-boundary preamble is prepended, so an unwrapped routed adapter
// would receive the bare step prompt without the containment that stops a gate
// agent from starting a second pipeline. The lifecycle and perf layers are the
// run's only record that the launch happened at all. Routing decides WHICH
// service serves a turn and nothing else about how that turn runs, so a routed
// adapter must be wrapped exactly like the default one.
func (sctx *StepContext) buildRoutedAgent(profile routing.Profile) (agent.Agent, error) {
	built, err := sctx.Routing.newAgent(profile)
	if err != nil {
		return nil, err
	}
	if sctx.WrapAgent == nil {
		return built, nil
	}
	wrapped := sctx.WrapAgent(built)
	if wrapped == nil {
		_ = built.Close()
		return nil, fmt.Errorf("the step harness produced no agent for profile %s; refusing to launch it uncontained", profile.ID)
	}
	return wrapped, nil
}

// recordRouteOutcome reports the invocation's normalized result. It runs on
// every path - success, failure, timeout, cancellation - because an assignment
// the controller never sees closed is one it keeps counting as running.
func (sctx *StepContext) recordRouteOutcome(invocation *routedInvocation, err error) {
	if sctx == nil || sctx.Routing == nil || invocation == nil {
		return
	}
	outcome := classifyRouteOutcome(err)
	// Remember the verdict before attempting it, so a report the hook never
	// accepts is retried as what actually happened rather than re-guessed.
	invocation.rememberOutcome(outcome)
	// Finish must survive a cancelled invocation: the turn has already
	// happened, and reporting it through the dead context would drop the
	// record precisely when the controller most needs it.
	sctx.Routing.finish(context.WithoutCancel(sctx.routingContext()), invocation, outcome)
}

// classifyRouteOutcome maps an invocation's error to the controller's closed
// outcome vocabulary.
//
// The mapping is deliberately conservative, because three of those values are
// not status reports: `auth-failed`, `exhausted` and `outage` are verified
// evidence AGAINST the route, and the controller excludes it immediately on
// receiving one, before its own next probe. Reporting one on a guess would
// take a healthy service out of the pool for every other caller on this
// machine, on nothing more than one failed turn.
//
// no-mistakes cannot establish any of those three conditions from this seam.
// It sees that an invocation failed, not why the provider refused it, and an
// adapter error quoting a provider's prose is not proof: a prompt can contain
// the words too. So a failed turn reports `launch-failed`, which releases the
// assignment honestly without condemning the route. If this seam ever gains a
// real signal - an adapter that reports a structured auth or quota refusal -
// that is where the stronger values belong, and nowhere else.
func classifyRouteOutcome(err error) routing.Outcome {
	if err == nil {
		return routing.OutcomeSuccess
	}
	return routing.OutcomeLaunchFailed
}

func (sctx *StepContext) routingContext() context.Context {
	if sctx == nil || sctx.Ctx == nil {
		return context.Background()
	}
	return sctx.Ctx
}

// finish reports an assignment exactly once, counting only a report the
// controller actually accepted.
//
// The latch is set after the report succeeds, not before, for the same reason
// the router records the id as closed only then: an assignment the controller
// never sees closed is one it keeps counting as running forever, which biases
// every later acquire for every caller on this machine. Latching first would
// mean the deferred release backstop silently skipped the one case it exists
// to cover. Repeating a report is safe by contract, so a retry can only turn a
// lost report into a delivered one.
//
// A report that still fails is never fatal to the turn, which has already
// happened, but it must not be silent either: this is the last moment the
// failure exists, and an operator seeing only degraded balancing afterwards has
// no way back to the cause.
func (r *RunRouting) finish(ctx context.Context, invocation *routedInvocation, outcome routing.Outcome) {
	if r == nil || invocation == nil || invocation.closed.Load() {
		return
	}
	if err := r.router.Finish(ctx, invocation.assignment.ID, outcome); err != nil {
		slog.Warn("failed to report a routing assignment as finished; the controller may keep counting it as running",
			"assignment", invocation.assignment.ID,
			"profile", invocation.assignment.Profile.ID,
			"outcome", string(outcome),
			"error", err)
		return
	}
	invocation.closed.Store(true)
}

func describeModel(p routing.Profile) string {
	if p.Tuning.Model == "" {
		return "harness-default model"
	}
	return p.Tuning.Model
}

func describeReason(a routing.Assignment) string {
	reason := a.Reason
	if reason == "" {
		reason = "no reason reported"
	}
	if a.Generation > 0 {
		return fmt.Sprintf("%s (controller generation %d)", reason, a.Generation)
	}
	return reason
}
