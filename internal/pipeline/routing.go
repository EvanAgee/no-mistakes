package pipeline

import (
	"context"
	"errors"
	"fmt"
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
	// runID prefixes every assignment id so two runs never collide in the
	// hook's shared record.
	runID string
}

// NewRunRouting builds the run's routing, or nil when routing is off.
func NewRunRouting(router *routing.Router, runID string, newAgent func(routing.Profile) (agent.Agent, error)) *RunRouting {
	if router == nil || newAgent == nil {
		return nil
	}
	return &RunRouting{router: router, newAgent: newAgent, runID: runID}
}

// routedInvocation is what acquireRoute hands back to the invocation seam.
type routedInvocation struct {
	assignment routing.Assignment
	// closed guards the single finish this invocation owes the hook.
	closed atomic.Bool
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

	assignmentID := fmt.Sprintf("%s:%d", r.runID, r.seq.Add(1))
	assignment, selected, err := r.router.Route(ctx, assignmentID, opts.Purpose)
	if err != nil {
		if errors.Is(err, routing.ErrDeferred) {
			// Not a fault and not permanent: the routing controller admitted
			// nothing right now. The attempt keeps its own id, so retrying it
			// after the next refresh is a fresh acquire, not a replay of this
			// verdict.
			return nil, false, nil, fmt.Errorf("no approved service is available for this %s invocation right now: %w", opts.Purpose, err)
		}
		return nil, false, nil, err
	}
	if !selected {
		return nil, false, nil, nil
	}

	invocation := &routedInvocation{assignment: assignment}
	routedAgent, err := r.newAgent(assignment.Profile)
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
		// keeps counting as running forever. The caller normally reports the
		// real outcome first, and finish is idempotent, so this only fires on
		// a path that skipped it - a panic unwinding through the seam. Such a
		// turn did launch, so "failed" is the honest report.
		r.finish(context.WithoutCancel(ctx), invocation, routing.OutcomeFailed)
	}
	return invocation, true, release, nil
}

// recordRouteOutcome reports the invocation's normalized result. It runs on
// every path - success, failure, timeout, cancellation - because an assignment
// the controller never sees closed is one it keeps counting as running.
//
// The outcome is derived from the error alone. What the turn produced is the
// pipeline's business; all the controller needs to know is whether this route
// finished serving the work it was assigned.
func (sctx *StepContext) recordRouteOutcome(invocation *routedInvocation, err error) {
	if sctx == nil || sctx.Routing == nil || invocation == nil {
		return
	}
	outcome := routing.OutcomeSuccess
	if err != nil {
		outcome = routing.OutcomeFailed
	}
	// Finish must survive a cancelled invocation: the turn has already
	// happened, and reporting it through the dead context would drop the
	// record precisely when the controller most needs it.
	sctx.Routing.finish(context.WithoutCancel(sctx.routingContext()), invocation, outcome)
}

func (sctx *StepContext) routingContext() context.Context {
	if sctx == nil || sctx.Ctx == nil {
		return context.Background()
	}
	return sctx.Ctx
}

// finish closes an assignment exactly once. A second call is a no-op, so a
// deferred cleanup and an explicit report cannot double-close.
func (r *RunRouting) finish(ctx context.Context, invocation *routedInvocation, outcome routing.Outcome) {
	if r == nil || invocation == nil || !invocation.closed.CompareAndSwap(false, true) {
		return
	}
	_ = r.router.Finish(ctx, invocation.assignment.ID, outcome)
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
	if !a.Fresh {
		return reason + " (from evidence the controller could not date as current)"
	}
	return reason
}
