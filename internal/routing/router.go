package routing

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
)

// Router is one run's view of assignment routing. It owns the small amount of
// state routing needs inside the daemon - which assignment ids this run has
// opened and closed - and delegates every actual decision to the hook.
//
// It exists so the invariants that are the caller's to keep are kept in one
// place rather than at each of the pipeline's invocation sites: an assignment
// id is used once, a finish reports exactly the identity that was acquired,
// and a closed assignment is never silently reopened under the same id.
//
// A nil *Router means routing is unconfigured, which is the default and must
// stay behaviorally identical to this build before routing existed: Route
// returns no selection and no error, and the caller launches exactly what it
// would have launched anyway.
type Router struct {
	hook *Hook
	// profiles are the operator's approved profiles, in configured order.
	profiles []Profile
	// owner and generation identify this daemon incarnation to the hook.
	owner      string
	generation string

	mu sync.Mutex
	// open maps an assignment id to the profile it acquired, for the
	// assignments this run has acquired and not yet finished.
	open map[string]Profile
	// closed records assignment ids this run has already finished, so a
	// second acquire under a spent id is refused instead of quietly
	// reopening work the hook has already accounted for.
	closed map[string]struct{}
}

// NewRouter builds a router over the operator's approved profiles. It returns
// nil when routing is unconfigured, so every caller's nil check is the single
// "routing is off" path.
func NewRouter(hook *Hook, profiles []Profile, owner, generation string) (*Router, error) {
	if hook == nil || strings.TrimSpace(hook.Path) == "" || len(profiles) == 0 {
		return nil, nil
	}
	if strings.TrimSpace(owner) == "" {
		return nil, errors.New("routing: an owner is required to acquire assignments")
	}
	if strings.TrimSpace(generation) == "" {
		return nil, errors.New("routing: an owner generation is required to acquire assignments")
	}
	seen := make(map[string]struct{}, len(profiles))
	for _, profile := range profiles {
		if err := profile.Validate(); err != nil {
			return nil, fmt.Errorf("routing: %w", err)
		}
		if _, duplicate := seen[profile.ID]; duplicate {
			return nil, fmt.Errorf("routing: duplicate profile id %q", profile.ID)
		}
		seen[profile.ID] = struct{}{}
	}
	return &Router{
		hook:       hook,
		profiles:   append([]Profile(nil), profiles...),
		owner:      owner,
		generation: generation,
		open:       map[string]Profile{},
		closed:     map[string]struct{}{},
	}, nil
}

// OwnerGeneration is the caller-supplied identifier of this daemon
// incarnation. A caller that mints assignment ids needs it, because the
// controller's idempotency key is (assignment_id, owner.identity) and
// deliberately excludes the owner generation: an id minted by a previous
// incarnation is replayed as the SAME launch rather than admitted as a new
// one, so an id that must be unique across a restart has to carry the
// generation itself. Empty on a nil router, which is routing being off.
func (r *Router) OwnerGeneration() string {
	if r == nil {
		return ""
	}
	return r.generation
}

// Assignment is one admitted invocation. Finish must be called for every
// Assignment that Route returns, whatever happens to the invocation.
type Assignment struct {
	ID      string
	Profile Profile
	Reason  string
	// Generation is the controller ledger's own decision generation behind
	// this selection, for the operator's log.
	Generation int64
}

// ProfileKey is the session-reuse identity of the assigned profile.
func (a Assignment) ProfileKey() string { return ProfileKey(a.Profile) }

// ErrNoAllowedProfile means the operator configured routing but left this role
// with nothing it may run. It is a configuration fault, not a routing verdict.
var ErrNoAllowedProfile = errors.New("routing: no approved profile is configured for this role")

// Route acquires one assignment for a role. The id must be unique per
// invocation attempt: a retry, a fallback to another adapter, and a recovered
// turn are each their own attempt, so each acquires its own assignment and is
// balanced as separate work.
//
// A nil router returns a zero Assignment and false with no error, which is the
// unconfigured path.
func (r *Router) Route(ctx context.Context, assignmentID, role string) (Assignment, bool, error) {
	if r == nil {
		return Assignment{}, false, nil
	}
	assignmentID = strings.TrimSpace(assignmentID)
	if assignmentID == "" {
		return Assignment{}, false, errors.New("routing: an assignment id is required")
	}

	r.mu.Lock()
	if _, spent := r.closed[assignmentID]; spent {
		r.mu.Unlock()
		// Reopening a closed id would let one attempt be counted twice by the
		// hook and would make the finish that follows ambiguous. A genuine
		// retry is a new attempt and gets a new id.
		return Assignment{}, false, fmt.Errorf("routing: assignment %q is already closed; a retry needs its own assignment id", assignmentID)
	}
	if _, live := r.open[assignmentID]; live {
		r.mu.Unlock()
		return Assignment{}, false, fmt.Errorf("routing: assignment %q is already open", assignmentID)
	}
	r.mu.Unlock()

	allowed := r.allowedFor(role)
	if len(allowed) == 0 {
		return Assignment{}, false, fmt.Errorf("%w: %s. Configured profiles: %s. Either add this role to one profile's `roles`, or leave one profile unrestricted so it serves every role",
			ErrNoAllowedProfile, describeRole(role), strings.Join(r.profileIDs(), ", "))
	}

	decision, err := r.hook.Acquire(ctx, Request{
		AssignmentID: assignmentID,
		Owner:        Owner{Identity: r.owner, Generation: r.generation},
	}, allowed)
	if err != nil {
		// Nothing was admitted, so nothing is recorded locally. A deferral in
		// particular must not be remembered as closed: the next refresh may
		// admit the same work, and caching the verdict would strand it. The
		// other refusals (already-closed, a controller error, an unoffered
		// route) leave the id untouched too, so a corrected retry is a fresh
		// acquire rather than a replay.
		return Assignment{}, false, err
	}

	r.mu.Lock()
	r.open[assignmentID] = decision.Profile
	r.mu.Unlock()

	return Assignment{
		ID:         assignmentID,
		Profile:    decision.Profile,
		Reason:     decision.Reason,
		Generation: decision.Generation,
	}, true, nil
}

// Finish reports how an assignment ended. It is idempotent locally: a second
// finish for an id this router already closed returns nil without calling the
// hook again, so a deferred cleanup path and an explicit one cannot double
// report. The identity sent is exactly the one Route recorded, never a
// caller-supplied profile, so a mis-wired caller cannot bill a route that
// never ran.
//
// An id becomes closed only once the hook has actually accepted the report. A
// report that never reached the controller leaves the assignment open with its
// recorded profile intact, so a later finish for that same id carries it rather
// than returning a local success for work the controller still counts as
// running. Repeating a report is safe by contract: the finish verb is
// idempotent at the hook, so the controller ignores one it has already closed.
// This is bounded retry through the caller's existing cleanup paths and adds
// no scheduler, goroutine or timer of its own.
func (r *Router) Finish(ctx context.Context, assignmentID string, outcome Outcome) error {
	if r == nil {
		return nil
	}
	assignmentID = strings.TrimSpace(assignmentID)
	if assignmentID == "" {
		return errors.New("routing: an assignment id is required to finish")
	}

	r.mu.Lock()
	profile, live := r.open[assignmentID]
	if !live {
		_, spent := r.closed[assignmentID]
		r.mu.Unlock()
		if spent {
			return nil
		}
		return fmt.Errorf("routing: assignment %q was never acquired by this run", assignmentID)
	}
	r.mu.Unlock()

	err := r.hook.Finish(ctx, Request{
		AssignmentID: assignmentID,
		Owner:        Owner{Identity: r.owner, Generation: r.generation},
		Outcome:      outcome,
		// The concrete identity that actually launched, so the controller
		// records what was really spent rather than only which route it
		// admitted. Nonsecret by construction: these are the operator's own
		// declared values, never a credential or an account selection.
		Profile: &LaunchedProfile{
			Adapter: string(profile.Agent),
			Model:   profile.Tuning.Model,
			Effort:  string(profile.Tuning.Effort),
		},
	})
	if err != nil {
		return err
	}

	r.mu.Lock()
	delete(r.open, assignmentID)
	r.closed[assignmentID] = struct{}{}
	r.mu.Unlock()
	return nil
}

// allowedFor returns the approved profiles this role may use, sorted by id so
// the offered set is stable across calls.
// describeRole names the duty a refused invocation was serving. An unnamed
// invocation reads as such rather than as an empty string an operator would
// then try to add to a `roles` list.
func describeRole(role string) string {
	if role == "" {
		return "this invocation reports no role, so only a profile with no `roles` restriction can serve it"
	}
	return fmt.Sprintf("role %q", role)
}

func (r *Router) profileIDs() []string {
	ids := make([]string, 0, len(r.profiles))
	for _, profile := range r.profiles {
		ids = append(ids, profile.ID)
	}
	sort.Strings(ids)
	return ids
}

func (r *Router) allowedFor(role string) []Profile {
	out := make([]Profile, 0, len(r.profiles))
	for _, profile := range r.profiles {
		if profile.AllowedForRole(role) {
			out = append(out, profile)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}
