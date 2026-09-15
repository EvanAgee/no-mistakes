package pipeline

import (
	"context"
	"fmt"
	"sync"

	"github.com/kunchenguid/no-mistakes/internal/agent"
	"github.com/kunchenguid/no-mistakes/internal/db"
	"github.com/kunchenguid/no-mistakes/internal/routing"
)

// SessionRole identifies which durable review-loop session an invocation
// belongs to. The fixer role spans every review-fix turn of a run and is the
// only role that resumes a session. Review turns deliberately run session-free:
// a rereview certifies fixes implementing the previous review turn's findings,
// so resuming any review session would seat the prescriber of those fixes as
// their certifier and degrade the rereview into checking that its own
// prescription was implemented.
type SessionRole string

const (
	// SessionRoleReviewer is legacy: review turns no longer create or resume
	// sessions. The constant remains so crash recovery keeps accepting
	// persisted rows written by earlier versions (see
	// validateRecoveredSessionProviders); such rows are never resumed.
	SessionRoleReviewer SessionRole = "reviewer"
	SessionRoleFixer    SessionRole = "review-fixer"
)

// sessionIdentity is one role's durable session as this process holds it: the
// adapter-native reference plus the execution profile it was minted under.
//
// The profile key is kept beside the reference rather than folded into it
// because they answer different questions. The reference answers "which
// session", which only the minting adapter can interpret. The key answers
// "would the profile about to launch even be able to continue it", which the
// adapter name alone cannot: two invocations can both be `pi` and still be
// Grok on an xAI subscription and DeepSeek through a Gateway, which share no
// session state whatsoever. Resuming across that boundary would hand one
// provider an id minted by another.
type sessionIdentity struct {
	ref        agent.SessionRef
	profileKey string
}

// RunSessions manages the per-run, per-role durable agent sessions of the
// review loop. It is strictly scoped to one run: identities are keyed by
// (run, role), persisted as minimum resume metadata (run, role, agent,
// session id, profile key - never prompts or transcripts), and never shared
// across runs, branches, repositories, or roles.
//
// Correctness always wins over reuse: adapters without session support run
// cold, a failed resume drops the identity and re-runs the same turn in a
// fresh same-role session, and any persistence failure degrades to cold
// invocations. A nil *RunSessions runs everything cold, preserving the
// pre-session behavior for steps outside the review loop and for tests.
type RunSessions struct {
	db      *db.DB
	runID   string
	agent   agent.Agent
	enabled bool

	mu  sync.Mutex
	ids map[SessionRole]sessionIdentity
}

// NewRunSessions creates the manager for one run, loading any persisted
// session identities recorded by a previous process for the same run and
// agent. Identities stored for a different adapter are ignored: a session id
// is only meaningful to the adapter that minted it.
//
// A persisted row's profile key is loaded alongside its session id and decides
// reuse later (see resumable). A row written before routing existed carries an
// empty key, which a routed turn will not resume, because an unknown identity
// is not evidence that the profile now launching minted the session. That is
// never an error: recovery of such a run proceeds and its next turn simply
// starts a fresh session.
func NewRunSessions(database *db.DB, runID string, sessionAgent agent.Agent, enabled bool) *RunSessions {
	rs := &RunSessions{
		db:      database,
		runID:   runID,
		agent:   sessionAgent,
		enabled: enabled,
		ids:     map[SessionRole]sessionIdentity{},
	}
	if database != nil {
		if stored, err := database.GetRunAgentSessions(runID); err == nil {
			for _, s := range stored {
				if s.SessionID == "" {
					continue
				}
				// A row carrying a profile key was minted by a routed
				// invocation, whose adapter is chosen per turn and need not be
				// this run's default agent. Its resumability is decided later,
				// by comparing that key against the profile actually about to
				// launch (see resumable), so filtering it here against the
				// default agent would discard every routed session.
				//
				// A row with no key is the pre-routing shape, and the default
				// agent is exactly the right authority for it.
				if s.ProfileKey == "" && !agent.SupportsSessionProvider(sessionAgent, s.Agent) {
					continue
				}
				rs.ids[SessionRole(s.Role)] = sessionIdentity{
					ref:        agent.SessionRef{ID: s.SessionID, Agent: s.Agent},
					profileKey: s.ProfileKey,
				}
			}
		}
	}
	return rs
}

// Run executes one turn of the given role, reusing the role's durable
// session when the adapter supports it and the session was minted under the
// same execution profile this turn is launching.
//
// profileKey is the identity of the profile about to launch (see
// internal/routing.ProfileKey); it is empty when routing is unconfigured or
// when the effective identity is unknown. logf (optional) receives operator-
// visible notes about session reuse and fallbacks.
func (rs *RunSessions) Run(ctx context.Context, a agent.Agent, role SessionRole, profileKey string, opts agent.RunOpts, logf func(string)) (*agent.Result, error) {
	if rs == nil || !rs.enabled || !agent.SupportsSessionResume(a) {
		if rs != nil && rs.enabled && logf != nil {
			logf(fmt.Sprintf("agent %s does not support session resume; running cold", a.Name()))
		}
		return a.Run(ctx, opts)
	}

	stored := rs.resumable(role, profileKey, logf)
	storedID := stored.ID
	opts.Session = &stored
	result, err := a.Run(ctx, opts)
	if err == nil {
		rs.remember(role, a, result.SessionID, sessionProvider(a, result), profileKey)
		return result, nil
	}
	if storedID == "" || ctx.Err() != nil {
		return nil, err
	}

	// The resume attempt failed. Never skip the turn: drop the dead identity
	// and re-run the same turn in a fresh same-role session.
	if logf != nil {
		logf(fmt.Sprintf("resume of %s session failed (%v); starting a fresh %s session", role, err, role))
	}
	rs.forget(role)
	opts.Session = &agent.SessionRef{}
	opts.SessionFallback = true
	opts.SessionFallbackReason = classifyFallbackReason(err)
	if opts.OnLifecycle != nil {
		opts.OnLifecycle(agent.LifecycleEvent{
			Agent:   a.Name(),
			Phase:   agent.LifecyclePhaseFallback,
			Message: fmt.Sprintf("%s session resume failed; starting a fresh %s session", a.Name(), role),
		})
	}
	result, err = a.Run(ctx, opts)
	if err != nil {
		return nil, err
	}
	rs.remember(role, a, result.SessionID, sessionProvider(a, result), profileKey)
	return result, nil
}

// resumable returns the reference this turn may resume. It is the stored
// reference only when the stored session's execution profile is the same one
// this turn is launching; otherwise it is an EMPTY reference, not a nil
// session.
//
// That distinction is the whole point. A nil session would run cold and mint
// nothing reusable, so every turn after a route change would stay cold for the
// rest of the run. An empty reference asks the adapter to start a fresh
// resumable session, whose id is then persisted with the new key, so the very
// next turn on the same profile resumes it.
//
// A mismatched record is forgotten here rather than left to rot: keeping it
// would mean the next same-profile turn had to displace it anyway, and a row
// that can never be resumed is only a trap for a later reader.
func (rs *RunSessions) resumable(role SessionRole, profileKey string, logf func(string)) agent.SessionRef {
	rs.mu.Lock()
	stored, ok := rs.ids[role]
	rs.mu.Unlock()
	if !ok || stored.ref.ID == "" {
		return agent.SessionRef{}
	}
	if resumableUnderProfile(stored.profileKey, profileKey) {
		return stored.ref
	}
	if logf != nil {
		switch {
		case stored.profileKey == "":
			// A row from before routing existed, or one whose effective model
			// and billing route could not be established. Either way it is
			// not evidence that the profile now launching minted it.
			logf(fmt.Sprintf("stored %s session has no recorded execution profile; starting a fresh %s session", role, role))
		case profileKey == "":
			logf(fmt.Sprintf("this %s turn has no established execution profile; starting a fresh %s session", role, role))
		default:
			logf(fmt.Sprintf("stored %s session belongs to a different execution profile; starting a fresh %s session", role, role))
		}
	}
	rs.forget(role)
	return agent.SessionRef{}
}

// remember stores the role's latest session identity in memory and persists
// it so the run can resume the session across daemon process boundaries. The
// session id and the profile key are always written together: a stored id
// whose key was lost would be indistinguishable from a legacy row and could
// be resumed by a profile that never minted it.
//
// invoked is the adapter that actually ran the turn, which is the authority on
// whether the returned identity is resumable. Under routing that is NOT the
// manager's own agent: a routed turn launches the adapter its assigned profile
// names, and asking the run's default agent whether it recognizes another
// service's provider would reject every routed session and leave the whole run
// permanently cold.
//
// Persistence failures are ignored: reuse degrades, correctness does not.
func (rs *RunSessions) remember(role SessionRole, invoked agent.Agent, sessionID, provider, profileKey string) {
	if sessionID == "" {
		return
	}
	if provider == "" || !agent.SupportsSessionProvider(invoked, provider) {
		return
	}
	identity := sessionIdentity{
		ref:        agent.SessionRef{ID: sessionID, Agent: provider},
		profileKey: profileKey,
	}
	rs.mu.Lock()
	changed := rs.ids[role] != identity
	rs.ids[role] = identity
	rs.mu.Unlock()
	if changed && rs.db != nil {
		_ = rs.db.UpsertRunAgentSession(rs.runID, string(role), provider, sessionID, profileKey)
	}
}

func sessionProvider(a agent.Agent, result *agent.Result) string {
	if result != nil && result.Provider != "" {
		return result.Provider
	}
	if a == nil {
		return ""
	}
	return a.Name()
}

// resumableUnderProfile decides whether a session minted under storedKey may
// be continued by a turn launching under wantKey.
//
// Three cases, and the difference between them is the whole rule:
//
//   - Equal, nonempty keys: both sides established the SAME execution identity,
//     so the provider that minted the session is the one about to run.
//
//   - An unrouted turn meeting a key-less row: routing is off, so every turn of
//     this run launches the same configured agent and there is no second
//     profile the session could leak to. This is the upgrade path - a run
//     parked by a build that predates the profile_key column resumes normally -
//     and it is safe precisely because nothing can route it elsewhere.
//
//   - Anything else, including an empty key on the routed side: unknown is not
//     equality. Two turns that both failed to establish a model are not
//     evidence of the same model, and a key-less row meeting a ROUTED turn is
//     not evidence that the routed profile minted it.
func resumableUnderProfile(storedKey, wantKey string) bool {
	if storedKey != "" && storedKey == wantKey {
		return true
	}
	return wantKey == routing.UnroutedProfileKey && storedKey == ""
}

func (rs *RunSessions) forget(role SessionRole) {
	rs.mu.Lock()
	delete(rs.ids, role)
	rs.mu.Unlock()
	if rs.db != nil {
		_ = rs.db.DeleteRunAgentSession(rs.runID, string(role))
	}
}
