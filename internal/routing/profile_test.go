package routing

import (
	"strings"
	"testing"

	"github.com/kunchenguid/no-mistakes/internal/agentcfg"
	"github.com/kunchenguid/no-mistakes/internal/types"
)

func claudeProfile() Profile {
	return Profile{
		ID:       "claude-opus",
		Agent:    types.AgentClaude,
		Provider: "anthropic-subscription",
		Tuning:   agentcfg.Profile{Model: "opus", Effort: agentcfg.EffortXHigh},
	}
}

func piGrokProfile() Profile {
	return Profile{
		ID:       "pi-grok",
		Agent:    types.AgentPi,
		Provider: "xai-subscription",
		Tuning:   agentcfg.Profile{Model: "xai/grok-4.6", Effort: agentcfg.EffortXHigh},
	}
}

func piDeepSeekProfile() Profile {
	return Profile{
		ID:       "pi-deepseek",
		Agent:    types.AgentPi,
		Provider: "vercel-ai-gateway",
		Tuning:   agentcfg.Profile{Model: "vercel-ai-gateway/deepseek/deepseek-v4.1-flash", Effort: agentcfg.EffortXHigh},
	}
}

// TestProfileKey_ChangingAnyComponentChangesTheKey table-tests every component
// the key encodes. Each row changes exactly one fact about the execution
// identity and must produce a different key, because resuming a session across
// any of these boundaries hands one provider an id another one minted.
//
// The Pi rows are the case that motivated the key at all: both profiles are
// the `pi` adapter, so the persisted agent name is identical and cannot tell
// Grok on an xAI subscription from DeepSeek through a Gateway.
func TestProfileKey_ChangingAnyComponentChangesTheKey(t *testing.T) {
	base := claudeProfile()
	baseKey := ProfileKey(base)
	if baseKey == "" {
		t.Fatal("a fully specified profile must have a key")
	}

	cases := []struct {
		name    string
		mutate  func(Profile) Profile
		wantKey bool
	}{
		{
			name:    "different adapter",
			mutate:  func(p Profile) Profile { p.Agent = types.AgentCodex; return p },
			wantKey: true,
		},
		{
			name:    "different billing route, same adapter and model",
			mutate:  func(p Profile) Profile { p.Provider = "anthropic-api"; return p },
			wantKey: true,
		},
		{
			name:    "different model, same adapter and route",
			mutate:  func(p Profile) Profile { p.Tuning.Model = "sonnet"; return p },
			wantKey: true,
		},
		{
			name:    "different effort",
			mutate:  func(p Profile) Profile { p.Tuning.Effort = agentcfg.EffortHigh; return p },
			wantKey: true,
		},
		{
			name:    "pi/grok versus pi/deepseek",
			mutate:  func(Profile) Profile { return piDeepSeekProfile() },
			wantKey: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			changed := ProfileKey(tc.mutate(base))
			if tc.wantKey && changed == "" {
				t.Fatalf("changed profile must still have a key")
			}
			if changed == baseKey {
				t.Fatalf("changing %s must change the key, both were %s", tc.name, baseKey)
			}
		})
	}

	// Pi to Pi across billing routes, stated directly rather than through the
	// mutation table, because it is the exact confusion the key prevents.
	if ProfileKey(piGrokProfile()) == ProfileKey(piDeepSeekProfile()) {
		t.Fatal("pi/grok and pi/deepseek must not share a profile key")
	}
}

// TestProfileKey_ReusedLabelWithChangedContentsChangesTheKey proves the key is
// computed from what the profile DOES, not from what the operator called it.
// An operator who edits a profile in place, keeping its id, has changed which
// service runs, so a session minted under the old contents must not be resumed
// under the new ones.
func TestProfileKey_ReusedLabelWithChangedContentsChangesTheKey(t *testing.T) {
	before := piGrokProfile()
	after := before
	after.Tuning.Model = "xai/grok-5"

	if before.ID != after.ID {
		t.Fatal("this test requires the same profile id on both sides")
	}
	if ProfileKey(before) == ProfileKey(after) {
		t.Fatal("the same profile label with changed contents must produce a different key")
	}
}

// TestProfileKey_VolatileFactsAreNotEncoded proves the key ignores everything
// that churns inside one unchanged billing route. If a quota refresh, a clock
// tick, an assignment id or a rotated credential could change the key, every
// such event would discard a perfectly usable session, which is the exact cost
// this key exists to avoid.
//
// The proof is structural: Profile has no field for any of those facts, so the
// only way one could enter the key is if a caller smuggled it into a field
// that IS encoded. Two profiles equal in every encoded field therefore share a
// key no matter what else happened between the two computations.
func TestProfileKey_VolatileFactsAreNotEncoded(t *testing.T) {
	first := piGrokProfile()
	second := piGrokProfile()

	// Same route, same model, same effort - as if a quota window reset and the
	// proxy rotated to a different account underneath, both of which leave the
	// operator-declared billing route untouched.
	if got, want := ProfileKey(second), ProfileKey(first); got != want {
		t.Fatalf("an unchanged route must keep its key; got %s, want %s", got, want)
	}

	// The key is also stable across repeated computation, so it can be
	// persisted in one process and compared in another.
	if ProfileKey(first) != ProfileKey(first) {
		t.Fatal("ProfileKey must be deterministic")
	}
}

// TestProfileKey_UnknownIdentityHasNoKey proves an unestablished identity is
// recorded as unknown rather than guessed. Two profiles that both leave the
// model to the harness default are NOT evidence of the same model, so neither
// gets a key and neither can match.
func TestProfileKey_UnknownIdentityHasNoKey(t *testing.T) {
	noModel := claudeProfile()
	noModel.Tuning.Model = ""
	if key := ProfileKey(noModel); key != "" {
		t.Fatalf("a profile with no established model must have no key, got %q", key)
	}

	noProvider := claudeProfile()
	noProvider.Provider = ""
	if key := ProfileKey(noProvider); key != "" {
		t.Fatalf("a profile with no billing route must have no key, got %q", key)
	}

	nonNative := claudeProfile()
	nonNative.Agent = types.AgentOpenCode
	if key := ProfileKey(nonNative); key != "" {
		t.Fatalf("a profile routing may not select must have no key, got %q", key)
	}
}

// TestProfileKey_EscapingPreventsFieldForgery proves a value cannot
// impersonate a field boundary. Without escaping, a model literally named so
// as to close one field and open another could make two genuinely different
// identities encode identically, which would resume a session across routes.
func TestProfileKey_EscapingPreventsFieldForgery(t *testing.T) {
	forged := claudeProfile()
	forged.Provider = "anthropic-subscription" + fieldSep + "model=opus"
	forged.Tuning.Model = "opus"

	honest := claudeProfile()

	if ProfileKey(forged) == ProfileKey(honest) {
		t.Fatal("a value containing the field separator must not collide with an honest profile")
	}
}

// TestProfileKey_FieldSeparatorInAValueCannotShiftAFieldBoundary proves the
// separator is escaped in its own right, not only blocked by the equals-sign
// escape. Two profiles differing ONLY by where a legitimate separator byte
// falls inside their values must produce different keys; without a real
// separator escape both canonicalize to the same byte sequence.
func TestProfileKey_FieldSeparatorInAValueCannotShiftAFieldBoundary(t *testing.T) {
	left := claudeProfile()
	left.Provider = "route" + fieldSep
	left.Tuning.Model = "opus"

	right := claudeProfile()
	right.Provider = "route"
	right.Tuning.Model = fieldSep + "opus"

	if ProfileKey(left) == ProfileKey(right) {
		t.Fatal("profiles differing only by where the separator byte falls must not share a key")
	}
}

// TestEscapeKeyValue_EmitsNoRawSeparator proves the encoding is injective into
// field boundaries: no escaped value can reproduce the raw separator byte, so
// the canonical string has exactly as many separators as it has real fields.
func TestEscapeKeyValue_EmitsNoRawSeparator(t *testing.T) {
	for _, value := range []string{
		fieldSep,
		"a" + fieldSep + "b",
		`\` + fieldSep,
		"model=x" + fieldSep + "model=y",
	} {
		if strings.Contains(escapeKeyValue(value), fieldSep) {
			t.Fatalf("escaped %q still contains a raw field separator", value)
		}
	}
}

// TestEscapeKeyValue_IsInjective proves the three escapes do not collide with
// one another: distinct values must escape to distinct strings, or two
// profiles could still canonicalize identically.
func TestEscapeKeyValue_IsInjective(t *testing.T) {
	values := []string{
		"",
		fieldSep,
		"=",
		`\`,
		`\u`,
		`\` + fieldSep,
		`\=`,
		"u" + fieldSep,
		fieldSep + "u",
	}
	seen := map[string]string{}
	for _, value := range values {
		escaped := escapeKeyValue(value)
		if prior, ok := seen[escaped]; ok {
			t.Fatalf("%q and %q both escape to %q", prior, value, escaped)
		}
		seen[escaped] = value
	}
}

// TestProfileKey_IsVersioned proves every key carries its encoding version, so
// a future change to what the key means invalidates old keys instead of
// silently reinterpreting them.
func TestProfileKey_IsVersioned(t *testing.T) {
	key := ProfileKey(claudeProfile())
	if !strings.HasPrefix(key, profileKeyVersion+":") {
		t.Fatalf("key %q must be prefixed with its encoding version", key)
	}
}

// TestProfileKey_CarriesNoSecret proves the key is safe to persist. It is a
// digest, so no configured value is recoverable from it by reading.
func TestProfileKey_CarriesNoSecret(t *testing.T) {
	p := claudeProfile()
	p.Provider = "anthropic-subscription-with-a-distinctive-label"
	key := ProfileKey(p)
	for _, secret := range []string{p.Provider, p.Tuning.Model, string(p.Agent), p.ID} {
		if strings.Contains(key, secret) {
			t.Fatalf("key %q must not contain the configured value %q verbatim", key, secret)
		}
	}
}

// TestProfileValidate_RefusesWhatCannotBeLaunched proves configuration fails
// closed. Each row is something the pipeline could not honestly run later, so
// refusing it at load is what keeps a run from reporting a service it did not
// actually use.
func TestProfileValidate_RefusesWhatCannotBeLaunched(t *testing.T) {
	cases := []struct {
		name    string
		profile Profile
		wantMsg string
	}{
		{
			name:    "empty id",
			profile: Profile{Agent: types.AgentClaude, Provider: "anthropic-subscription"},
			wantMsg: "profile id must not be empty",
		},
		{
			name: "id with whitespace",
			profile: Profile{
				ID: "claude opus", Agent: types.AgentClaude, Provider: "anthropic-subscription",
			},
			wantMsg: "must not contain whitespace",
		},
		{
			name: "non-native adapter",
			profile: Profile{
				ID: "omp", Agent: types.AgentOpenCode, Provider: "vercel-ai-gateway",
				Tuning: agentcfg.Profile{Model: "vercel-ai-gateway/deepseek/deepseek-v4.1-flash"},
			},
			wantMsg: "is not a native no-mistakes adapter",
		},
		{
			name: "missing billing route",
			profile: Profile{
				ID: "claude-opus", Agent: types.AgentClaude,
				Tuning: agentcfg.Profile{Model: "opus"},
			},
			wantMsg: "must name the billing route",
		},
		{
			name: "empty role entry",
			profile: Profile{
				ID: "claude-opus", Agent: types.AgentClaude, Provider: "anthropic-subscription",
				Tuning: agentcfg.Profile{Model: "opus"}, Roles: []string{""},
			},
			wantMsg: "must not contain an empty entry",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.profile.Validate()
			if err == nil {
				t.Fatal("want a validation error")
			}
			if !strings.Contains(err.Error(), tc.wantMsg) {
				t.Fatalf("error %q must mention %q", err, tc.wantMsg)
			}
		})
	}

	if err := claudeProfile().Validate(); err != nil {
		t.Fatalf("a well-formed profile must validate: %v", err)
	}
}

// TestProfileAllowedForRole proves role restriction: an empty list means any
// role, and a non-empty one is exact. This is what keeps a reviewer-only or
// fixer-only route from being offered for the other duty.
func TestProfileAllowedForRole(t *testing.T) {
	unrestricted := claudeProfile()
	if !unrestricted.AllowedForRole("review") || !unrestricted.AllowedForRole("review-fix") {
		t.Fatal("a profile with no role list must serve every role")
	}

	restricted := claudeProfile()
	restricted.Roles = []string{"review-fix"}
	if !restricted.AllowedForRole("review-fix") {
		t.Fatal("a restricted profile must serve its listed role")
	}
	if restricted.AllowedForRole("review") {
		t.Fatal("a restricted profile must not serve an unlisted role")
	}

	// An unnamed invocation is served by unrestricted profiles only. A
	// role-restricted profile is the operator's statement about one named
	// duty, and an invocation that reports no duty is not that duty.
	if !unrestricted.AllowedForRole("") {
		t.Fatal("an unrestricted profile must serve an invocation that reports no role")
	}
	if restricted.AllowedForRole("") {
		t.Fatal("a restricted profile must not serve an invocation that reports no role")
	}
}

// TestNativeAgent_ExcludesOMP pins the adapter boundary the brief requires:
// Claude, Codex and Pi are native and selectable; opencode (OMP) is not a
// native no-mistakes adapter and routing must never select it.
func TestNativeAgent_ExcludesOMP(t *testing.T) {
	for _, name := range []types.AgentName{types.AgentClaude, types.AgentCodex, types.AgentPi} {
		if !NativeAgent(name) {
			t.Fatalf("%s must be selectable by routing", name)
		}
	}
	for _, name := range []types.AgentName{types.AgentOpenCode, types.AgentGrok, types.AgentRovoDev, types.AgentAntigravity} {
		if NativeAgent(name) {
			t.Fatalf("%s must not be selectable by routing", name)
		}
	}
}
