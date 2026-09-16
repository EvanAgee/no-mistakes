package pipeline

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/kunchenguid/no-mistakes/internal/agent"
	"github.com/kunchenguid/no-mistakes/internal/agentcfg"
	"github.com/kunchenguid/no-mistakes/internal/routing"
	"github.com/kunchenguid/no-mistakes/internal/types"
)

// These tests pin the session half of continuous routing: when a run's
// invocations move between approved services, no session identity may cross
// that boundary, while consecutive turns on the SAME service must keep reusing
// their exact native session id.

func routingClaudeProfile() routing.Profile {
	return routing.Profile{
		ID:       "claude-opus",
		Agent:    types.AgentClaude,
		Provider: "anthropic-subscription",
		Tuning:   agentcfg.Profile{Model: "opus", Effort: agentcfg.EffortXHigh},
	}
}

func routingCodexProfile() routing.Profile {
	return routing.Profile{
		ID:       "codex-sol",
		Agent:    types.AgentCodex,
		Provider: "openai-subscription",
		Tuning:   agentcfg.Profile{Model: "gpt-5.6-sol", Effort: agentcfg.EffortHigh},
	}
}

func routingPiGrokProfile() routing.Profile {
	return routing.Profile{
		ID:       "pi-grok",
		Agent:    types.AgentPi,
		Provider: "xai-subscription",
		Tuning:   agentcfg.Profile{Model: "xai/grok-4.6", Effort: agentcfg.EffortXHigh},
	}
}

func routingPiDeepSeekProfile() routing.Profile {
	return routing.Profile{
		ID:       "pi-deepseek",
		Agent:    types.AgentPi,
		Provider: "vercel-ai-gateway",
		Tuning:   agentcfg.Profile{Model: "vercel-ai-gateway/deepseek/deepseek-v4.1-flash", Effort: agentcfg.EffortXHigh},
	}
}

// profileAgent is a session-capable adapter bound to one execution profile.
// It records every session reference it receives, so a test can prove exactly
// which identity reached which service.
//
// Each instance mints ids in its own namespace, drawn from a counter shared by
// the whole test, so any id is globally unique and an id that leaked across a
// service boundary names its true minter in the failure message.
type profileAgent struct {
	*fakeSessionAgent
	profile routing.Profile
	minted  *int
}

// mintCounter is per-test state, not global: each test constructs its own so
// ids stay stable and readable within one test.
func newMintCounter() *int {
	n := 0
	return &n
}

func newProfileAgent(profile routing.Profile, counter *int) *profileAgent {
	fake := newFakeSessionAgent()
	// The provider name is the adapter's, exactly as a real one reports it.
	// That is deliberate: pi/grok and pi/deepseek both report "pi", which is
	// precisely why the adapter name cannot qualify reuse on its own.
	fake.name = string(profile.Agent)
	return &profileAgent{fakeSessionAgent: fake, profile: profile, minted: counter}
}

func (p *profileAgent) Run(ctx context.Context, opts agent.RunOpts) (*agent.Result, error) {
	result, err := p.fakeSessionAgent.Run(ctx, opts)
	if result == nil || err != nil {
		return result, err
	}
	result.Provider = p.Name()
	if result.Resumed {
		// A resumed turn keeps the id it was handed; only a fresh session
		// mints a new one.
		return result, nil
	}
	if result.SessionID != "" {
		*p.minted++
		result.SessionID = fmt.Sprintf("%s/sess-%d", p.profile.ID, *p.minted)
	}
	return result, nil
}

func (p *profileAgent) lastSession() *agent.SessionRef {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.calls) == 0 {
		return nil
	}
	return p.calls[len(p.calls)-1].session
}

func (p *profileAgent) lastPrompt() string {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.calls) == 0 {
		return ""
	}
	return p.calls[len(p.calls)-1].prompt
}

// TestRoutedSessions_SwitchingServiceNeverCarriesThePriorSessionID is proof
// item 1. One run's fixer turns move Claude, Codex, Pi/Grok, Pi/DeepSeek and
// back. Every switch must hand the new service an EMPTY reference, never the
// previous service's id, and the exact phase prompt must survive untouched.
//
// The Pi/Grok to Pi/DeepSeek hop is the one that would silently pass a weaker
// check: both are the `pi` adapter, so the persisted agent name matches and
// only the profile key can tell them apart.
func TestRoutedSessions_SwitchingServiceNeverCarriesThePriorSessionID(t *testing.T) {
	d, run := sessionTestDB(t)
	counter := newMintCounter()

	profiles := []routing.Profile{
		routingClaudeProfile(),
		routingCodexProfile(),
		routingPiGrokProfile(),
		routingPiDeepSeekProfile(),
		routingClaudeProfile(),
	}

	// One manager for the whole run, as the executor holds it. The manager's
	// own adapter accepts any provider so cross-provider storage is not what
	// this test is measuring.
	manager := NewRunSessions(d, run.ID, newFakeSessionAgent(), true)

	var mintedByProfile []string
	for i, profile := range profiles {
		svc := newProfileAgent(profile, counter)
		prompt := fmt.Sprintf("fix round %d: apply finding F-%d in worktree /w/run-1", i+1, i+1)

		result, err := manager.Run(context.Background(), svc, SessionRoleFixer,
			routing.ProfileKey(profile), agent.RunOpts{Prompt: prompt, Purpose: "review-fix"}, nil)
		if err != nil {
			t.Fatalf("turn %d on %s: %v", i+1, profile.ID, err)
		}

		session := svc.lastSession()
		if session == nil {
			t.Fatalf("turn %d on %s ran cold; a routed turn must still be resumable", i+1, profile.ID)
		}
		// Every turn here is a service change from the previous one, so every
		// turn must start fresh.
		if session.ID != "" {
			t.Fatalf("turn %d on %s received session id %q from a different service",
				i+1, profile.ID, session.ID)
		}
		for _, previous := range mintedByProfile {
			if strings.Contains(session.ID, previous) {
				t.Fatalf("turn %d on %s received an id minted by %s", i+1, profile.ID, previous)
			}
		}

		// The phase context is untouched by routing.
		if got := svc.lastPrompt(); got != prompt {
			t.Fatalf("turn %d prompt = %q, want the exact phase prompt %q", i+1, got, prompt)
		}

		if result.SessionID == "" {
			t.Fatalf("turn %d on %s must mint a resumable session", i+1, profile.ID)
		}
		if !strings.HasPrefix(result.SessionID, profile.ID+"/") {
			t.Fatalf("turn %d minted %q, which does not belong to %s", i+1, result.SessionID, profile.ID)
		}
		mintedByProfile = append(mintedByProfile, profile.ID)
	}
}

// TestRoutedSessions_PiGrokAndPiDeepSeekNeverShareASession isolates the
// same-adapter case. Both profiles are `pi`, so an implementation that
// qualified reuse by adapter name alone would pass every other test here and
// still hand a Gateway-billed turn a session id minted on an xAI subscription.
func TestRoutedSessions_PiGrokAndPiDeepSeekNeverShareASession(t *testing.T) {
	d, run := sessionTestDB(t)
	counter := newMintCounter()
	manager := NewRunSessions(d, run.ID, newFakeSessionAgent(), true)

	grok := newProfileAgent(routingPiGrokProfile(), counter)
	grokResult, err := manager.Run(context.Background(), grok, SessionRoleFixer,
		routing.ProfileKey(routingPiGrokProfile()), agent.RunOpts{Prompt: "fix on grok"}, nil)
	if err != nil {
		t.Fatalf("grok turn: %v", err)
	}

	deepseek := newProfileAgent(routingPiDeepSeekProfile(), counter)
	if _, err := manager.Run(context.Background(), deepseek, SessionRoleFixer,
		routing.ProfileKey(routingPiDeepSeekProfile()), agent.RunOpts{Prompt: "fix on deepseek"}, nil); err != nil {
		t.Fatalf("deepseek turn: %v", err)
	}

	got := deepseek.lastSession()
	if got == nil || got.ID != "" {
		t.Fatalf("the deepseek turn must start fresh, got %+v", got)
	}
	if got.ID == grokResult.SessionID {
		t.Fatalf("the deepseek turn received grok's session id %q", grokResult.SessionID)
	}
}

// TestRoutedSessions_ConsecutiveSameProfileTurnsReuseTheExactID is proof item
// 2's core. Consecutive fixer turns on one profile must resume the exact
// native id, and the id a switched-to service mints must itself be resumed by
// the next turn on that same service.
func TestRoutedSessions_ConsecutiveSameProfileTurnsReuseTheExactID(t *testing.T) {
	d, run := sessionTestDB(t)
	counter := newMintCounter()
	manager := NewRunSessions(d, run.ID, newFakeSessionAgent(), true)

	grokKey := routing.ProfileKey(routingPiGrokProfile())

	first := newProfileAgent(routingPiGrokProfile(), counter)
	firstResult, err := manager.Run(context.Background(), first, SessionRoleFixer, grokKey,
		agent.RunOpts{Prompt: "fix round 1"}, nil)
	if err != nil {
		t.Fatalf("first turn: %v", err)
	}
	if firstResult.SessionID == "" {
		t.Fatal("the first turn must mint a session")
	}

	for round := 2; round <= 4; round++ {
		next := newProfileAgent(routingPiGrokProfile(), counter)
		if _, err := manager.Run(context.Background(), next, SessionRoleFixer, grokKey,
			agent.RunOpts{Prompt: fmt.Sprintf("fix round %d", round)}, nil); err != nil {
			t.Fatalf("round %d: %v", round, err)
		}
		session := next.lastSession()
		if session == nil || session.ID != firstResult.SessionID {
			t.Fatalf("round %d must resume %q, got %+v", round, firstResult.SessionID, session)
		}
	}
}

// TestRoutedSessions_AfterASwitchTheNextMatchingTurnResumesTheNewID proves the
// empty-reference choice pays off. A switched-to service starts fresh, mints
// an id, and the next turn on that same service resumes exactly that new id -
// which a nil session would have made impossible, leaving the rest of the run
// permanently cold.
func TestRoutedSessions_AfterASwitchTheNextMatchingTurnResumesTheNewID(t *testing.T) {
	d, run := sessionTestDB(t)
	counter := newMintCounter()
	manager := NewRunSessions(d, run.ID, newFakeSessionAgent(), true)

	claudeKey := routing.ProfileKey(routingClaudeProfile())
	codexKey := routing.ProfileKey(routingCodexProfile())

	claude := newProfileAgent(routingClaudeProfile(), counter)
	if _, err := manager.Run(context.Background(), claude, SessionRoleFixer, claudeKey,
		agent.RunOpts{Prompt: "fix on claude"}, nil); err != nil {
		t.Fatalf("claude turn: %v", err)
	}

	codexFirst := newProfileAgent(routingCodexProfile(), counter)
	codexResult, err := manager.Run(context.Background(), codexFirst, SessionRoleFixer, codexKey,
		agent.RunOpts{Prompt: "fix on codex"}, nil)
	if err != nil {
		t.Fatalf("codex turn: %v", err)
	}
	if got := codexFirst.lastSession(); got == nil || got.ID != "" {
		t.Fatalf("the switch to codex must start fresh, got %+v", got)
	}
	if codexResult.SessionID == "" {
		t.Fatal("the switched-to service must mint a resumable session")
	}

	codexSecond := newProfileAgent(routingCodexProfile(), counter)
	if _, err := manager.Run(context.Background(), codexSecond, SessionRoleFixer, codexKey,
		agent.RunOpts{Prompt: "another fix on codex"}, nil); err != nil {
		t.Fatalf("second codex turn: %v", err)
	}
	if got := codexSecond.lastSession(); got == nil || got.ID != codexResult.SessionID {
		t.Fatalf("the next codex turn must resume %q, got %+v", codexResult.SessionID, got)
	}
}

// TestRoutedSessions_ReuseSurvivesAManagerAndDatabaseReopen proves the key is
// durable, not just in-memory. A daemon restart mid-run must still resume the
// same native id on the same profile, and must still refuse it on another.
func TestRoutedSessions_ReuseSurvivesAManagerAndDatabaseReopen(t *testing.T) {
	d, run := sessionTestDB(t)
	counter := newMintCounter()
	grokKey := routing.ProfileKey(routingPiGrokProfile())

	manager := NewRunSessions(d, run.ID, newFakeSessionAgent(), true)
	svc := newProfileAgent(routingPiGrokProfile(), counter)
	minted, err := manager.Run(context.Background(), svc, SessionRoleFixer, grokKey,
		agent.RunOpts{Prompt: "fix before restart"}, nil)
	if err != nil {
		t.Fatalf("first turn: %v", err)
	}

	// A fresh manager over the same database is what the daemon builds after
	// a restart resumes a parked run.
	restarted := NewRunSessions(d, run.ID, newFakeSessionAgent(), true)
	after := newProfileAgent(routingPiGrokProfile(), counter)
	if _, err := restarted.Run(context.Background(), after, SessionRoleFixer, grokKey,
		agent.RunOpts{Prompt: "fix after restart"}, nil); err != nil {
		t.Fatalf("turn after restart: %v", err)
	}
	if got := after.lastSession(); got == nil || got.ID != minted.SessionID {
		t.Fatalf("a restarted manager must resume %q on the same profile, got %+v", minted.SessionID, got)
	}

	// And the same restart must still refuse the id on a different profile.
	restartedAgain := NewRunSessions(d, run.ID, newFakeSessionAgent(), true)
	other := newProfileAgent(routingPiDeepSeekProfile(), counter)
	if _, err := restartedAgain.Run(context.Background(), other, SessionRoleFixer,
		routing.ProfileKey(routingPiDeepSeekProfile()), agent.RunOpts{Prompt: "fix on deepseek"}, nil); err != nil {
		t.Fatalf("deepseek turn after restart: %v", err)
	}
	if got := other.lastSession(); got == nil || got.ID != "" {
		t.Fatalf("a restarted manager must not resume another profile's id, got %+v", got)
	}
}

// TestRoutedSessions_KeyComponentTable table-tests reuse against every changed
// component of the key, plus the reused-label case. A row that changes the
// execution identity must start fresh; a row that changes nothing real must
// keep reusing.
func TestRoutedSessions_KeyComponentTable(t *testing.T) {
	base := routingPiGrokProfile()

	cases := []struct {
		name      string
		next      routing.Profile
		wantReuse bool
	}{
		{
			name:      "identical profile",
			next:      routingPiGrokProfile(),
			wantReuse: true,
		},
		{
			name: "same profile rebuilt from the same values",
			next: routing.Profile{
				ID: "pi-grok", Agent: types.AgentPi, Provider: "xai-subscription",
				Tuning: agentcfg.Profile{Model: "xai/grok-4.6", Effort: agentcfg.EffortXHigh},
			},
			wantReuse: true,
		},
		{
			name:      "different adapter",
			next:      routingCodexProfile(),
			wantReuse: false,
		},
		{
			name: "same adapter and model, different billing route",
			next: routing.Profile{
				ID: "pi-grok-api", Agent: types.AgentPi, Provider: "xai-api",
				Tuning: agentcfg.Profile{Model: "xai/grok-4.6", Effort: agentcfg.EffortXHigh},
			},
			wantReuse: false,
		},
		{
			name: "same adapter and route, different model",
			next: routing.Profile{
				ID: "pi-grok", Agent: types.AgentPi, Provider: "xai-subscription",
				Tuning: agentcfg.Profile{Model: "xai/grok-5", Effort: agentcfg.EffortXHigh},
			},
			wantReuse: false,
		},
		{
			name: "different effort",
			next: routing.Profile{
				ID: "pi-grok", Agent: types.AgentPi, Provider: "xai-subscription",
				Tuning: agentcfg.Profile{Model: "xai/grok-4.6", Effort: agentcfg.EffortHigh},
			},
			wantReuse: false,
		},
		{
			name:      "pi/grok to pi/deepseek",
			next:      routingPiDeepSeekProfile(),
			wantReuse: false,
		},
		{
			name: "same profile label, changed contents",
			next: routing.Profile{
				ID: base.ID, Agent: types.AgentPi, Provider: "vercel-ai-gateway",
				Tuning: agentcfg.Profile{Model: "vercel-ai-gateway/deepseek/deepseek-v4.1-flash", Effort: agentcfg.EffortXHigh},
			},
			wantReuse: false,
		},
		{
			name: "next turn has no established model",
			next: routing.Profile{
				ID: "pi-default", Agent: types.AgentPi, Provider: "xai-subscription",
			},
			wantReuse: false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d, run := sessionTestDB(t)
			counter := newMintCounter()
			manager := NewRunSessions(d, run.ID, newFakeSessionAgent(), true)

			firstSvc := newProfileAgent(base, counter)
			minted, err := manager.Run(context.Background(), firstSvc, SessionRoleFixer,
				routing.ProfileKey(base), agent.RunOpts{Prompt: "first fix"}, nil)
			if err != nil {
				t.Fatalf("first turn: %v", err)
			}

			nextSvc := newProfileAgent(tc.next, counter)
			if _, err := manager.Run(context.Background(), nextSvc, SessionRoleFixer,
				routing.ProfileKey(tc.next), agent.RunOpts{Prompt: "second fix"}, nil); err != nil {
				t.Fatalf("second turn: %v", err)
			}

			session := nextSvc.lastSession()
			if session == nil {
				t.Fatal("the second turn must still be session-capable")
			}
			if tc.wantReuse {
				if session.ID != minted.SessionID {
					t.Fatalf("an unchanged execution identity must resume %q, got %q", minted.SessionID, session.ID)
				}
				return
			}
			if session.ID != "" {
				t.Fatalf("a changed execution identity must start fresh, got %q", session.ID)
			}
		})
	}
}

// TestRoutedSessions_QuotaRefreshAndCredentialRotationPreserveReuse proves the
// key ignores what churns inside one billing route. A quota window that reset,
// a new observation generation, and a rotated token all leave the operator's
// declared route unchanged, so the session must still be resumed. Discarding
// it there would cost a cold turn on every refresh.
func TestRoutedSessions_QuotaRefreshAndCredentialRotationPreserveReuse(t *testing.T) {
	d, run := sessionTestDB(t)
	counter := newMintCounter()
	manager := NewRunSessions(d, run.ID, newFakeSessionAgent(), true)

	profile := routingPiGrokProfile()
	svc := newProfileAgent(profile, counter)
	minted, err := manager.Run(context.Background(), svc, SessionRoleFixer,
		routing.ProfileKey(profile), agent.RunOpts{Prompt: "fix before refresh"}, nil)
	if err != nil {
		t.Fatalf("first turn: %v", err)
	}

	// Between the two turns: the controller refreshed (new observation
	// generation), the quota window reset, and the proxy rotated to a
	// different account with a new token. None of those is part of the
	// operator-declared route, so the profile - and therefore the key - is
	// byte-identical.
	refreshed := routingPiGrokProfile()
	if routing.ProfileKey(refreshed) != routing.ProfileKey(profile) {
		t.Fatal("this test requires the refreshed profile to keep the same key")
	}

	after := newProfileAgent(refreshed, counter)
	if _, err := manager.Run(context.Background(), after, SessionRoleFixer,
		routing.ProfileKey(refreshed), agent.RunOpts{Prompt: "fix after refresh"}, nil); err != nil {
		t.Fatalf("turn after refresh: %v", err)
	}
	if got := after.lastSession(); got == nil || got.ID != minted.SessionID {
		t.Fatalf("a refresh within one route must preserve reuse of %q, got %+v", minted.SessionID, got)
	}
}

// TestRoutedSessions_LegacyRowStartsFreshOnceWithoutFailingTheRun proves a row
// written before routing existed is readable, unresumable, and harmless. The
// run continues, the first routed turn starts fresh, and the row is replaced
// with one carrying both the new id and its key.
func TestRoutedSessions_LegacyRowStartsFreshOnceWithoutFailingTheRun(t *testing.T) {
	d, run := sessionTestDB(t)
	counter := newMintCounter()

	// A legacy row: an agent and session id, and a NULL profile key.
	if err := d.UpsertRunAgentSession(run.ID, string(SessionRoleFixer), "pi", "legacy-session", ""); err != nil {
		t.Fatalf("seed legacy row: %v", err)
	}
	stored, err := d.GetRunAgentSessions(run.ID)
	if err != nil {
		t.Fatalf("read back: %v", err)
	}
	if len(stored) != 1 || stored[0].ProfileKey != "" {
		t.Fatalf("the seeded row must have no profile key, got %+v", stored)
	}

	manager := NewRunSessions(d, run.ID, newFakeSessionAgent(), true)
	profile := routingPiGrokProfile()
	svc := newProfileAgent(profile, counter)
	minted, err := manager.Run(context.Background(), svc, SessionRoleFixer,
		routing.ProfileKey(profile), agent.RunOpts{Prompt: "first routed fix"}, nil)
	if err != nil {
		t.Fatalf("a legacy row must not fail the run: %v", err)
	}
	if got := svc.lastSession(); got == nil || got.ID != "" {
		t.Fatalf("a legacy row must not be resumed, got %+v", got)
	}
	if got := svc.lastSession(); got != nil && got.ID == "legacy-session" {
		t.Fatal("the legacy id must never reach a routed invocation")
	}

	// It starts fresh ONCE: the replacement row carries both id and key, so
	// the next matching turn resumes it.
	after, err := d.GetRunAgentSessions(run.ID)
	if err != nil {
		t.Fatalf("read back after: %v", err)
	}
	if len(after) != 1 {
		t.Fatalf("want exactly one row for the role, got %d", len(after))
	}
	if after[0].SessionID != minted.SessionID {
		t.Fatalf("the row must carry the newly minted id %q, got %q", minted.SessionID, after[0].SessionID)
	}
	if after[0].ProfileKey != routing.ProfileKey(profile) {
		t.Fatalf("the row must carry the new profile key, got %q", after[0].ProfileKey)
	}

	next := newProfileAgent(profile, counter)
	if _, err := manager.Run(context.Background(), next, SessionRoleFixer,
		routing.ProfileKey(profile), agent.RunOpts{Prompt: "second routed fix"}, nil); err != nil {
		t.Fatalf("second turn: %v", err)
	}
	if got := next.lastSession(); got == nil || got.ID != minted.SessionID {
		t.Fatalf("the next matching turn must resume %q, got %+v", minted.SessionID, got)
	}
}

// TestRoutedSessions_RemovedAdapterRowStartsFreshWithoutFailingRecovery proves
// a row whose adapter is no longer configured is ignored rather than fatal.
// An operator who removed a harness between runs must still be able to resume
// a parked run.
func TestRoutedSessions_RemovedAdapterRowStartsFreshWithoutFailingRecovery(t *testing.T) {
	d, run := sessionTestDB(t)
	counter := newMintCounter()

	if err := d.UpsertRunAgentSession(run.ID, string(SessionRoleFixer),
		"an-adapter-that-is-gone", "orphan-session", "v1:deadbeef"); err != nil {
		t.Fatalf("seed removed-adapter row: %v", err)
	}

	// The run's manager is built over an adapter that does not accept the
	// stored provider, which is what "the adapter is no longer configured"
	// looks like at recovery.
	manager := NewRunSessions(d, run.ID, newFakeSessionAgent(), true)
	profile := routingClaudeProfile()
	svc := newProfileAgent(profile, counter)
	if _, err := manager.Run(context.Background(), svc, SessionRoleFixer,
		routing.ProfileKey(profile), agent.RunOpts{Prompt: "fix after the adapter was removed"}, nil); err != nil {
		t.Fatalf("a removed-adapter row must not fail recovery: %v", err)
	}
	if got := svc.lastSession(); got == nil || got.ID != "" {
		t.Fatalf("a removed-adapter row must not be resumed, got %+v", got)
	}
	if got := svc.lastSession(); got != nil && got.ID == "orphan-session" {
		t.Fatal("the orphaned id must never reach an invocation")
	}
}

// TestRoutedSessions_UnknownProfileKeyNeverMatches proves an unestablished
// identity is not equality. Two turns that both fail to establish a model do
// NOT get to share a session, because nothing proved they are the same model.
func TestRoutedSessions_UnknownProfileKeyNeverMatches(t *testing.T) {
	d, run := sessionTestDB(t)
	counter := newMintCounter()
	manager := NewRunSessions(d, run.ID, newFakeSessionAgent(), true)

	unknown := routing.Profile{ID: "pi-default", Agent: types.AgentPi, Provider: "xai-subscription"}
	if routing.ProfileKey(unknown) != "" {
		t.Fatal("this test requires a profile whose identity cannot be established")
	}

	first := newProfileAgent(unknown, counter)
	minted, err := manager.Run(context.Background(), first, SessionRoleFixer,
		routing.ProfileKey(unknown), agent.RunOpts{Prompt: "first fix"}, nil)
	if err != nil {
		t.Fatalf("first turn: %v", err)
	}

	second := newProfileAgent(unknown, counter)
	if _, err := manager.Run(context.Background(), second, SessionRoleFixer,
		routing.ProfileKey(unknown), agent.RunOpts{Prompt: "second fix"}, nil); err != nil {
		t.Fatalf("second turn: %v", err)
	}
	if got := second.lastSession(); got == nil || got.ID != "" {
		t.Fatalf("an unknown identity must never match, got %+v", got)
	}
	if got := second.lastSession(); got != nil && got.ID == minted.SessionID {
		t.Fatal("two unestablished identities must not be treated as the same profile")
	}
}

// TestRoutedSessions_UnroutedReuseIsUnchanged proves the default path is
// untouched. With routing off, every turn carries the explicit unrouted key,
// which matches itself, so reuse behaves exactly as it did before routing
// existed. The unrouted key is deliberately NOT the empty string, which means
// "unknown" and must never match.
func TestRoutedSessions_UnroutedReuseIsUnchanged(t *testing.T) {
	d, run := sessionTestDB(t)
	fake := newFakeSessionAgent()
	manager := NewRunSessions(d, run.ID, fake, true)

	for round := 1; round <= 3; round++ {
		if _, err := manager.Run(context.Background(), fake, SessionRoleFixer,
			routing.UnroutedProfileKey,
			agent.RunOpts{Prompt: fmt.Sprintf("fix round %d", round)}, nil); err != nil {
			t.Fatalf("round %d: %v", round, err)
		}
	}

	if len(fake.calls) != 3 {
		t.Fatalf("want 3 calls, got %d", len(fake.calls))
	}
	if fake.calls[0].session == nil || fake.calls[0].session.ID != "" {
		t.Fatalf("the first unrouted turn must start a session, got %+v", fake.calls[0].session)
	}
	for i, call := range fake.calls[1:] {
		if call.session == nil || call.session.ID != "sess-1" {
			t.Fatalf("unrouted turn %d must resume sess-1, got %+v", i+2, call.session)
		}
	}

	rows, err := d.GetRunAgentSessions(run.ID)
	if err != nil {
		t.Fatalf("read back: %v", err)
	}
	if len(rows) != 1 || rows[0].ProfileKey != routing.UnroutedProfileKey {
		t.Fatalf("an unrouted run must persist the unrouted key, got %+v", rows)
	}
}

// TestRoutedSessions_MismatchedRecordIsForgottenNotLeftBehind proves a record
// that can never be resumed is removed rather than left as a trap for a later
// reader, and is replaced by one the next matching turn can use.
func TestRoutedSessions_MismatchedRecordIsForgottenNotLeftBehind(t *testing.T) {
	d, run := sessionTestDB(t)
	counter := newMintCounter()
	manager := NewRunSessions(d, run.ID, newFakeSessionAgent(), true)

	grok := newProfileAgent(routingPiGrokProfile(), counter)
	grokMinted, err := manager.Run(context.Background(), grok, SessionRoleFixer,
		routing.ProfileKey(routingPiGrokProfile()), agent.RunOpts{Prompt: "fix on grok"}, nil)
	if err != nil {
		t.Fatalf("grok turn: %v", err)
	}

	deepseek := newProfileAgent(routingPiDeepSeekProfile(), counter)
	deepseekMinted, err := manager.Run(context.Background(), deepseek, SessionRoleFixer,
		routing.ProfileKey(routingPiDeepSeekProfile()), agent.RunOpts{Prompt: "fix on deepseek"}, nil)
	if err != nil {
		t.Fatalf("deepseek turn: %v", err)
	}

	rows, err := d.GetRunAgentSessions(run.ID)
	if err != nil {
		t.Fatalf("read back: %v", err)
	}
	if len(rows) != 1 {
		t.Fatalf("one role must keep one row, got %d", len(rows))
	}
	if rows[0].SessionID == grokMinted.SessionID {
		t.Fatal("the superseded grok row must not survive")
	}
	if rows[0].SessionID != deepseekMinted.SessionID {
		t.Fatalf("the row must carry the deepseek id %q, got %q", deepseekMinted.SessionID, rows[0].SessionID)
	}
	if rows[0].ProfileKey != routing.ProfileKey(routingPiDeepSeekProfile()) {
		t.Fatalf("the row must carry the deepseek key, got %q", rows[0].ProfileKey)
	}
}

// TestRoutedSessions_FailedResumeStillRetriesInAFreshSession proves routing
// did not disturb the pre-existing failed-resume contract: a dead session is
// dropped and the SAME turn re-runs cold rather than being skipped, and the
// replacement id is persisted with the current profile's key.
func TestRoutedSessions_FailedResumeStillRetriesInAFreshSession(t *testing.T) {
	d, run := sessionTestDB(t)
	counter := newMintCounter()
	manager := NewRunSessions(d, run.ID, newFakeSessionAgent(), true)
	profile := routingPiGrokProfile()
	key := routing.ProfileKey(profile)

	first := newProfileAgent(profile, counter)
	minted, err := manager.Run(context.Background(), first, SessionRoleFixer, key,
		agent.RunOpts{Prompt: "fix round 1"}, nil)
	if err != nil {
		t.Fatalf("first turn: %v", err)
	}

	// The provider has forgotten the session by the next turn.
	second := newProfileAgent(profile, counter)
	second.failResumes[minted.SessionID] = fmt.Errorf("session %s not found", minted.SessionID)

	if _, err := manager.Run(context.Background(), second, SessionRoleFixer, key,
		agent.RunOpts{Prompt: "fix round 2"}, nil); err != nil {
		t.Fatalf("a failed resume must retry, not fail: %v", err)
	}

	second.mu.Lock()
	calls := append([]sessionCall(nil), second.calls...)
	second.mu.Unlock()
	if len(calls) != 2 {
		t.Fatalf("want a resume attempt then a fresh retry, got %d calls", len(calls))
	}
	if calls[0].session == nil || calls[0].session.ID != minted.SessionID {
		t.Fatalf("the first attempt must try to resume %q, got %+v", minted.SessionID, calls[0].session)
	}
	if calls[1].session == nil || calls[1].session.ID != "" {
		t.Fatalf("the retry must start fresh, got %+v", calls[1].session)
	}
	if !calls[1].fallback {
		t.Fatal("the retry must be marked as a session fallback")
	}
	if calls[1].prompt != "fix round 2" {
		t.Fatalf("the retry must re-run the SAME turn, got prompt %q", calls[1].prompt)
	}

	rows, err := d.GetRunAgentSessions(run.ID)
	if err != nil {
		t.Fatalf("read back: %v", err)
	}
	if len(rows) != 1 || rows[0].ProfileKey != key {
		t.Fatalf("the replacement row must carry the current profile key, got %+v", rows)
	}
	if rows[0].SessionID == minted.SessionID {
		t.Fatal("the dead session id must not remain persisted")
	}
}

// TestRoutedSessions_RolesStayIsolatedUnderRouting proves routing did not
// weaken run/role separation: two roles on one run keep distinct sessions even
// when both are routed to the same profile.
func TestRoutedSessions_RolesStayIsolatedUnderRouting(t *testing.T) {
	d, run := sessionTestDB(t)
	counter := newMintCounter()
	manager := NewRunSessions(d, run.ID, newFakeSessionAgent(), true)
	profile := routingPiGrokProfile()
	key := routing.ProfileKey(profile)

	fixerSvc := newProfileAgent(profile, counter)
	fixer, err := manager.Run(context.Background(), fixerSvc, SessionRoleFixer, key,
		agent.RunOpts{Prompt: "fix"}, nil)
	if err != nil {
		t.Fatalf("fixer turn: %v", err)
	}

	reviewerSvc := newProfileAgent(profile, counter)
	reviewer, err := manager.Run(context.Background(), reviewerSvc, SessionRoleReviewer, key,
		agent.RunOpts{Prompt: "review"}, nil)
	if err != nil {
		t.Fatalf("reviewer turn: %v", err)
	}

	if got := reviewerSvc.lastSession(); got == nil || got.ID != "" {
		t.Fatalf("a different role must not inherit the fixer session, got %+v", got)
	}
	if reviewer.SessionID == fixer.SessionID {
		t.Fatal("two roles must not share one session identity")
	}

	rows, err := d.GetRunAgentSessions(run.ID)
	if err != nil {
		t.Fatalf("read back: %v", err)
	}
	if len(rows) != 2 {
		t.Fatalf("each role must keep its own row, got %d", len(rows))
	}
	for _, row := range rows {
		if row.ProfileKey != key {
			t.Fatalf("row %s must carry the profile key, got %q", row.Role, row.ProfileKey)
		}
	}
}

// TestRoutedSessions_UnroutedRunResumesALegacyRowAcrossAnUpgrade is the
// upgrade path for every existing installation. A run parked by a build that
// predates the profile_key column has a NULL key. When that build is replaced
// and routing is left off, the parked run must resume its fixer session
// normally: routing is not configured, so every turn launches the same
// configured agent and there is no second profile the session could leak to.
//
// Refusing here would silently cost a cold fixer turn on every run in flight
// at upgrade time, which is a regression nobody asked for.
func TestRoutedSessions_UnroutedRunResumesALegacyRowAcrossAnUpgrade(t *testing.T) {
	d, run := sessionTestDB(t)

	if err := d.UpsertRunAgentSession(run.ID, string(SessionRoleFixer), "fake", "pre-upgrade-session", ""); err != nil {
		t.Fatalf("seed legacy row: %v", err)
	}

	fake := newFakeSessionAgent()
	manager := NewRunSessions(d, run.ID, fake, true)
	if _, err := manager.Run(context.Background(), fake, SessionRoleFixer,
		routing.UnroutedProfileKey, agent.RunOpts{Prompt: "fix after upgrade"}, nil); err != nil {
		t.Fatalf("resume after upgrade: %v", err)
	}

	if got := fake.calls[0].session; got == nil || got.ID != "pre-upgrade-session" {
		t.Fatalf("an unrouted run must resume its pre-upgrade session, got %+v", got)
	}
}

// TestRoutedSessions_RoutedTurnNeverResumesALegacyRow is the other half of
// that rule. The same NULL row, met by a ROUTED turn, must NOT be resumed: the
// row proves only which adapter minted the session, and the adapter name alone
// cannot show that the assigned profile's provider, model and billing route
// are the ones that created it.
func TestRoutedSessions_RoutedTurnNeverResumesALegacyRow(t *testing.T) {
	d, run := sessionTestDB(t)
	counter := newMintCounter()

	if err := d.UpsertRunAgentSession(run.ID, string(SessionRoleFixer), "pi", "pre-upgrade-session", ""); err != nil {
		t.Fatalf("seed legacy row: %v", err)
	}

	manager := NewRunSessions(d, run.ID, newFakeSessionAgent(), true)
	profile := routingPiGrokProfile()
	svc := newProfileAgent(profile, counter)
	if _, err := manager.Run(context.Background(), svc, SessionRoleFixer,
		routing.ProfileKey(profile), agent.RunOpts{Prompt: "routed fix"}, nil); err != nil {
		t.Fatalf("routed turn: %v", err)
	}

	got := svc.lastSession()
	if got == nil || got.ID != "" {
		t.Fatalf("a routed turn must not resume a key-less row, got %+v", got)
	}
	if got.ID == "pre-upgrade-session" {
		t.Fatal("the pre-upgrade id must never reach a routed invocation")
	}
}

// TestRoutedSessions_UnknownKeyTurnLeavesNoLegacyLookingRow proves an
// unestablished identity persists nothing a later turn can resume. A routed
// turn whose profile key is empty used to write a key-less row, which is
// byte-identical to the pre-routing shape, so the legacy upgrade path then
// handed that id to an unrouted turn running a DIFFERENT provider.
func TestRoutedSessions_UnknownKeyTurnLeavesNoLegacyLookingRow(t *testing.T) {
	d, run := sessionTestDB(t)
	counter := newMintCounter()

	unknown := routing.Profile{ID: "pi-default", Agent: types.AgentPi, Provider: "vercel-ai-gateway"}
	if routing.ProfileKey(unknown) != "" {
		t.Fatal("this test requires a profile whose identity cannot be established")
	}

	routed := NewRunSessions(d, run.ID, newFakeSessionAgent(), true)
	deepseek := newProfileAgent(unknown, counter)
	minted, err := routed.Run(context.Background(), deepseek, SessionRoleFixer,
		routing.ProfileKey(unknown), agent.RunOpts{Prompt: "fix on an unestablished identity"}, nil)
	if err != nil {
		t.Fatalf("routed turn: %v", err)
	}
	if minted.SessionID == "" {
		t.Fatal("the adapter must have minted an id for this test to mean anything")
	}

	rows, err := d.GetRunAgentSessions(run.ID)
	if err != nil {
		t.Fatalf("read back: %v", err)
	}
	for _, row := range rows {
		if row.Role == string(SessionRoleFixer) {
			t.Fatalf("an unknown identity must persist no resumable row, got %+v", row)
		}
	}

	// Routing is then switched off and the daemon restarts, so every turn
	// carries the unrouted key and the run's own default adapter launches.
	fake := newFakeSessionAgent()
	unrouted := NewRunSessions(d, run.ID, fake, true)
	if _, err := unrouted.Run(context.Background(), fake, SessionRoleFixer,
		routing.UnroutedProfileKey, agent.RunOpts{Prompt: "fix with routing off"}, nil); err != nil {
		t.Fatalf("unrouted turn: %v", err)
	}
	if got := fake.calls[0].session; got == nil || got.ID != "" {
		t.Fatalf("an unrouted turn must start fresh, got %+v", got)
	}
	if got := fake.calls[0].session; got != nil && got.ID == minted.SessionID {
		t.Fatal("an id minted under an unestablished identity must never reach another provider")
	}
}

// TestRoutedSessions_UnknownKeyTurnDropsAnEarlierResumableRow proves the same
// rule applies to a row that already existed. A matching turn establishes a
// resumable session, then an unknown-identity turn on the same role runs; the
// earlier row must not survive as a key-less record a later unrouted turn
// would treat as its own legacy state.
func TestRoutedSessions_UnknownKeyTurnDropsAnEarlierResumableRow(t *testing.T) {
	d, run := sessionTestDB(t)
	counter := newMintCounter()
	manager := NewRunSessions(d, run.ID, newFakeSessionAgent(), true)

	known := routingPiDeepSeekProfile()
	established := newProfileAgent(known, counter)
	if _, err := manager.Run(context.Background(), established, SessionRoleFixer,
		routing.ProfileKey(known), agent.RunOpts{Prompt: "establish a session"}, nil); err != nil {
		t.Fatalf("established turn: %v", err)
	}

	unknown := routing.Profile{ID: "pi-default", Agent: types.AgentPi, Provider: "vercel-ai-gateway"}
	if routing.ProfileKey(unknown) != "" {
		t.Fatal("this test requires a profile whose identity cannot be established")
	}
	stranger := newProfileAgent(unknown, counter)
	if _, err := manager.Run(context.Background(), stranger, SessionRoleFixer,
		routing.ProfileKey(unknown), agent.RunOpts{Prompt: "fix on an unestablished identity"}, nil); err != nil {
		t.Fatalf("unknown-identity turn: %v", err)
	}

	rows, err := d.GetRunAgentSessions(run.ID)
	if err != nil {
		t.Fatalf("read back: %v", err)
	}
	for _, row := range rows {
		if row.Role == string(SessionRoleFixer) {
			t.Fatalf("the earlier row must not survive an unknown-identity turn, got %+v", row)
		}
	}
}
