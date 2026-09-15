// Package routing owns continuous assignment routing: choosing which approved
// native execution profile serves each agent invocation of a run, and proving
// that a persisted native session may be reused by the profile about to launch.
//
// It is deliberately a configuration and identity layer and nothing more. It
// launches no process of its own, mints no credentials, and holds no state
// beyond what one run's executor hands it. The assignment decision belongs to
// an external hook (see hook.go); this package's job is to bound what that
// hook may say and to refuse anything outside the operator's configured set.
package routing

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"strings"

	"github.com/kunchenguid/no-mistakes/internal/agentcfg"
	"github.com/kunchenguid/no-mistakes/internal/types"
)

// Profile is one named native execution profile the operator declared in
// global configuration. The ID is the only token that crosses the hook
// boundary: a hook answers with an ID, never with a command, a path, an
// argument list or a model string, so a compromised or buggy hook can only
// choose among routes the operator already approved on this machine.
type Profile struct {
	// ID is the operator's own label for this profile, unique within the
	// configured set. It is what acquire may return and what finish reports.
	ID string
	// Agent is the native adapter that serves the profile. Only adapters this
	// build supports natively are accepted; anything else is refused at load.
	Agent types.AgentName
	// Provider is the stable, nonsecret billing surface the adapter bills to -
	// for example "anthropic-subscription", "xai-subscription",
	// "vercel-ai-gateway". It is an operator-declared label, never a
	// credential and never the account a proxy currently has selected:
	// rotating an account or a token inside one route must not change this
	// value, or every rotation would discard a perfectly reusable session.
	Provider string
	// Tuning is the harness-neutral model/effort selection this profile
	// launches with. An empty model means the harness default, which is an
	// unknown effective identity for session reuse (see ProfileKey).
	Tuning agentcfg.Profile
	// Roles, when non-empty, restricts the profile to these pipeline roles.
	// An empty list means every role may use it.
	Roles []string
}

// AllowedForRole reports whether this profile may serve the named role.
func (p Profile) AllowedForRole(role string) bool {
	if len(p.Roles) == 0 {
		return true
	}
	for _, allowed := range p.Roles {
		if allowed == role {
			return true
		}
	}
	return false
}

// Validate rejects a profile the pipeline could not honestly launch. It is the
// fail-closed check at configuration load: a profile naming an adapter this
// build has no native support for, or a tuning knob that adapter cannot
// express, is refused up front rather than selected later and silently
// downgraded to something the run would then misreport.
func (p Profile) Validate() error {
	if strings.TrimSpace(p.ID) == "" {
		return fmt.Errorf("profile id must not be empty")
	}
	if p.ID != strings.TrimSpace(p.ID) || strings.ContainsAny(p.ID, " \t\n\r") {
		return fmt.Errorf("profile id %q must not contain whitespace", p.ID)
	}
	if !NativeAgent(p.Agent) {
		return fmt.Errorf("profile %q: agent %q is not a native no-mistakes adapter (native: %s)",
			p.ID, p.Agent, strings.Join(nativeAgentNames(), ", "))
	}
	if strings.TrimSpace(p.Provider) == "" {
		return fmt.Errorf("profile %q: provider must name the billing route this profile bills to", p.ID)
	}
	if err := agentcfg.Validate(p.Agent, p.Tuning); err != nil {
		return fmt.Errorf("profile %q: %w", p.ID, err)
	}
	for _, role := range p.Roles {
		if strings.TrimSpace(role) == "" {
			return fmt.Errorf("profile %q: roles must not contain an empty entry", p.ID)
		}
	}
	return nil
}

// nativeAgents is the set of adapters routing may select. It is deliberately
// narrower than every adapter this build can construct: routing decides which
// process launches with the operator's credentials on every turn of a run, so
// it is limited to the adapters whose session, structured-output and
// project-instruction behavior no-mistakes verifies natively. Pi covers both
// Grok (xai) and Gateway-served models, so neither needs a separate entry.
var nativeAgents = []types.AgentName{types.AgentClaude, types.AgentCodex, types.AgentPi}

// NativeAgent reports whether routing may select this adapter.
func NativeAgent(name types.AgentName) bool {
	for _, native := range nativeAgents {
		if native == name {
			return true
		}
	}
	return false
}

func nativeAgentNames() []string {
	out := make([]string, 0, len(nativeAgents))
	for _, name := range nativeAgents {
		out = append(out, string(name))
	}
	return out
}

// profileKeyVersion prefixes every key. Bumping it invalidates every persisted
// key at once, which is the intended migration for any change to what the key
// encodes: a stale key that silently kept its old meaning would resume a
// session under a profile the new rule considers different.
const profileKeyVersion = "v1"

// fieldSep separates canonical fields. It is ASCII unit separator, which
// cannot appear unescaped in a value.
const fieldSep = ""

// UnroutedProfileKey is the key every invocation carries when routing is not
// configured. It is a real, equal-to-itself value rather than the empty string
// precisely because the empty string means "unknown", and unknown must never
// match anything - including another unknown.
//
// Without this distinction the two cases collapse. An unrouted run and a run
// whose effective model could not be established would both carry "", and the
// second would start reusing sessions across profiles nobody proved identical.
// Naming the unrouted case explicitly keeps the pre-routing behavior exactly
// as it was while leaving "unknown" genuinely unmatchable.
const UnroutedProfileKey = profileKeyVersion + ":unrouted"

// ProfileKey is the canonical, nonsecret encoding of the execution identity a
// native session was minted under. Two invocations may share a session only
// when their keys are equal and nonempty.
//
// It encodes exactly the facts that decide whether a provider would accept and
// correctly continue an existing session: the adapter, the billing route, the
// effective model, and the session-relevant tuning. It deliberately encodes
// none of the facts that churn within one unchanged route - assignment ids,
// quota observations, reset generations, timestamps, rotating credentials, or
// which account a proxy currently has selected - because folding any of those
// in would discard a valid session on every refresh and defeat the reuse this
// key exists to enable.
//
// An empty return means the effective identity is unknown, not that it is
// blank. That happens when the model was left to the harness default, so two
// profiles naming the same adapter cannot be proven to serve the same model.
// An unknown identity is never equal to anything, including itself: callers
// must treat an empty key as "start fresh", never as a match.
func ProfileKey(p Profile) string {
	model := strings.TrimSpace(p.Tuning.Model)
	provider := strings.TrimSpace(p.Provider)
	if model == "" || provider == "" || !NativeAgent(p.Agent) {
		return ""
	}
	// Sorted key=value fields, so adding a field later cannot reorder the
	// encoding of the fields already present, and an escaped value can never
	// forge a field boundary or another field's content.
	fields := []string{
		"adapter=" + escapeKeyValue(string(p.Agent)),
		"provider=" + escapeKeyValue(provider),
		"model=" + escapeKeyValue(model),
		"effort=" + escapeKeyValue(string(p.Tuning.Effort)),
	}
	sort.Strings(fields)
	canonical := profileKeyVersion + fieldSep + strings.Join(fields, fieldSep)
	sum := sha256.Sum256([]byte(canonical))
	return profileKeyVersion + ":" + hex.EncodeToString(sum[:])
}

// escapeKeyValue makes every value unambiguous inside the canonical encoding.
func escapeKeyValue(value string) string {
	var b strings.Builder
	for _, r := range value {
		switch r {
		case '\\':
			b.WriteString(`\\`)
		case '':
			b.WriteString(``)
		case '=':
			b.WriteString(`\=`)
		default:
			b.WriteRune(r)
		}
	}
	return b.String()
}
