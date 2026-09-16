package config

import (
	"bytes"
	"log/slog"
	"strings"
	"testing"
	"time"

	"github.com/kunchenguid/no-mistakes/internal/agentcfg"
	"github.com/kunchenguid/no-mistakes/internal/routing"
	"github.com/kunchenguid/no-mistakes/internal/types"
)

// loadAssignmentYAML loads a global config containing the given assignment
// block, returning the parsed result or the error.
func loadAssignmentYAML(t *testing.T, body string) (*GlobalConfig, error) {
	t.Helper()
	return LoadGlobalFromBytes([]byte(body))
}

// TestAssignment_UnsetLeavesRoutingOff proves the default. A configuration
// that says nothing about assignment must leave routing disabled, so every
// existing installation keeps its exact current behavior.
func TestAssignment_UnsetLeavesRoutingOff(t *testing.T) {
	cfg, err := loadAssignmentYAML(t, "agent: claude\n")
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if cfg.Assignment.Enabled() {
		t.Fatal("an absent assignment block must leave routing off")
	}
	if cfg.Assignment.Hook() != nil {
		t.Fatal("routing off must build no hook")
	}
	if len(cfg.Assignment.Profiles) != 0 {
		t.Fatalf("routing off must configure no profiles, got %v", cfg.Assignment.Profiles)
	}
}

// TestAssignment_ParsesNamedNativeProfiles proves the configured surface: an
// executable, its fixed arguments, bounds, and named profiles that resolve to
// a concrete adapter, billing route, model and effort.
func TestAssignment_ParsesNamedNativeProfiles(t *testing.T) {
	cfg, err := loadAssignmentYAML(t, `agent: claude
assignment:
  hook_path: /opt/firstmate/bin/fm-route
  hook_args: ["--json"]
  hook_timeout: 15s
  profiles:
    - id: claude-opus
      agent: claude
      provider: anthropic-subscription
      model: opus
      effort: xhigh
    - id: codex-sol
      agent: codex
      provider: openai-subscription
      model: gpt-5.6-sol
      effort: high
    - id: pi-grok
      agent: pi
      provider: xai-subscription
      model: xai/grok-4.6
      effort: xhigh
      roles: ["review-fix", "review"]
    - id: pi-deepseek
      agent: pi
      provider: vercel-ai-gateway
      model: vercel-ai-gateway/deepseek/deepseek-v4.1-flash
      effort: xhigh
`)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	if !cfg.Assignment.Enabled() {
		t.Fatal("a hook plus profiles must enable routing")
	}
	if cfg.Assignment.HookPath != "/opt/firstmate/bin/fm-route" {
		t.Fatalf("hook path = %q", cfg.Assignment.HookPath)
	}
	if strings.Join(cfg.Assignment.HookArgs, ",") != "--json" {
		t.Fatalf("hook args = %v", cfg.Assignment.HookArgs)
	}
	if cfg.Assignment.HookTimeout != 15*time.Second {
		t.Fatalf("hook timeout = %v", cfg.Assignment.HookTimeout)
	}

	if len(cfg.Assignment.Profiles) != 4 {
		t.Fatalf("want 4 profiles, got %d", len(cfg.Assignment.Profiles))
	}
	grok := cfg.Assignment.Profiles[2]
	if grok.ID != "pi-grok" || grok.Agent != types.AgentPi || grok.Provider != "xai-subscription" {
		t.Fatalf("pi-grok = %+v", grok)
	}
	if grok.Tuning != (agentcfg.Profile{Model: "xai/grok-4.6", Effort: agentcfg.EffortXHigh}) {
		t.Fatalf("pi-grok tuning = %+v", grok.Tuning)
	}
	// Roles are normalized to a sorted, deduplicated set, so the same
	// restriction written in a different order is the same restriction.
	if strings.Join(grok.Roles, ",") != "review,review-fix" {
		t.Fatalf("pi-grok roles = %v", grok.Roles)
	}

	hook := cfg.Assignment.Hook()
	if hook == nil || hook.Path != cfg.Assignment.HookPath {
		t.Fatalf("hook = %+v", hook)
	}
	if hook.Timeout != 15*time.Second {
		t.Fatalf("hook timeout = %v", hook.Timeout)
	}
}

// TestAssignment_RefusesWhatCouldNotBeLaunched proves configuration fails
// closed. Every row here is something that would otherwise be discovered
// mid-run, when the only options left are to launch the wrong thing or to
// abort a run in flight.
func TestAssignment_RefusesWhatCouldNotBeLaunched(t *testing.T) {
	cases := []struct {
		name    string
		yaml    string
		wantMsg string
	}{
		{
			name: "non-native adapter",
			yaml: `assignment:
  hook_path: /bin/route
  profiles:
    - id: omp-deepseek
      agent: opencode
      provider: vercel-ai-gateway
      model: vercel-ai-gateway/deepseek/deepseek-v4.1-flash
`,
			wantMsg: "not a native no-mistakes adapter",
		},
		{
			name: "missing billing route",
			yaml: `assignment:
  hook_path: /bin/route
  profiles:
    - id: claude-opus
      agent: claude
      model: opus
`,
			wantMsg: "must name the billing route",
		},
		{
			name: "duplicate profile id",
			yaml: `assignment:
  hook_path: /bin/route
  profiles:
    - id: claude-opus
      agent: claude
      provider: anthropic-subscription
      model: opus
    - id: claude-opus
      agent: claude
      provider: anthropic-subscription
      model: sonnet
`,
			wantMsg: "duplicate profile id",
		},
		{
			name: "unknown effort",
			yaml: `assignment:
  hook_path: /bin/route
  profiles:
    - id: claude-opus
      agent: claude
      provider: anthropic-subscription
      model: opus
      effort: extreme
`,
			wantMsg: "invalid effort",
		},
		{
			name: "profiles with no hook to select among them",
			yaml: `assignment:
  profiles:
    - id: claude-opus
      agent: claude
      provider: anthropic-subscription
      model: opus
`,
			wantMsg: "nothing can select among them",
		},
		{
			name: "hook with nothing it may select",
			yaml: `assignment:
  hook_path: /bin/route
`,
			wantMsg: "nothing it may select",
		},
		{
			name: "unusable timeout",
			yaml: `assignment:
  hook_path: /bin/route
  hook_timeout: soon
  profiles:
    - id: claude-opus
      agent: claude
      provider: anthropic-subscription
      model: opus
`,
			wantMsg: "invalid assignment.hook_timeout",
		},
		{
			name: "negative timeout",
			yaml: `assignment:
  hook_path: /bin/route
  hook_timeout: -5m
  profiles:
    - id: claude-opus
      agent: claude
      provider: anthropic-subscription
      model: opus
`,
			wantMsg: "must not be negative",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := loadAssignmentYAML(t, tc.yaml)
			if err == nil {
				t.Fatal("want a load error")
			}
			if !strings.Contains(err.Error(), tc.wantMsg) {
				t.Fatalf("error %q must mention %q", err, tc.wantMsg)
			}
		})
	}
}

// TestAssignment_IsGlobalOnly proves a repository cannot configure routing.
// The merged per-run configuration takes its assignment from the global file
// and nowhere else, because routing decides which process runs with the
// maintainer's credentials against which subscription.
func TestAssignment_IsGlobalOnly(t *testing.T) {
	global, err := loadAssignmentYAML(t, `agent: claude
assignment:
  hook_path: /opt/firstmate/bin/fm-route
  profiles:
    - id: claude-opus
      agent: claude
      provider: anthropic-subscription
      model: opus
`)
	if err != nil {
		t.Fatalf("load global: %v", err)
	}

	// A repository config that tries to declare its own assignment block. The
	// repo schema has no such field, so the value is simply not read; what
	// this asserts is that the merged config still carries the operator's.
	repoYAML := []byte(`assignment:
  hook_path: /tmp/attacker-controlled
  profiles:
    - id: attacker
      agent: claude
      provider: anthropic-subscription
      model: opus
`)
	repo, err := LoadRepoFromBytes(repoYAML)
	if err != nil {
		t.Fatalf("parse repo config: %v", err)
	}

	merged := Merge(global, repo)
	if merged.Assignment.HookPath != "/opt/firstmate/bin/fm-route" {
		t.Fatalf("a repository must not redirect the assignment hook, got %q", merged.Assignment.HookPath)
	}
	if len(merged.Assignment.Profiles) != 1 || merged.Assignment.Profiles[0].ID != "claude-opus" {
		t.Fatalf("a repository must not add a profile, got %+v", merged.Assignment.Profiles)
	}
}

// TestAssignment_NormalizeRolesIsOrderInsensitive proves the same restriction
// written two ways is one restriction, and that blanks and duplicates are
// dropped rather than becoming a role nothing matches.
func TestAssignment_NormalizeRolesIsOrderInsensitive(t *testing.T) {
	first := normalizeRoles([]string{"review-fix", "review"})
	second := normalizeRoles([]string{"review", " review-fix ", "review"})
	if strings.Join(first, ",") != strings.Join(second, ",") {
		t.Fatalf("%v and %v must normalize to the same restriction", first, second)
	}
	if got := normalizeRoles([]string{"", "  "}); got != nil {
		t.Fatalf("a list of blanks must normalize to no restriction, got %v", got)
	}
}

// captureConfigWarnings redirects the default logger for one call and returns
// what was logged at warn level or above.
func captureConfigWarnings(t *testing.T, load func()) string {
	t.Helper()
	var logs bytes.Buffer
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelWarn})))
	t.Cleanup(func() { slog.SetDefault(previous) })
	load()
	return logs.String()
}

// TestAssignment_ModellessProfileLoadsButWarnsThatReuseIsOff proves the
// documented contract for a profile that leaves the model to the harness
// default: it stays legal, and the operator is told exactly once, by name,
// that it will never reuse a session.
//
// Without the warning this is silent. routing.ProfileKey returns an empty key
// for such a profile, an empty key never matches anything, so every turn it
// serves starts fresh and persists nothing for the whole life of every run.
// The only other signal is a per-turn log line an operator is unlikely to
// trace back to a missing `model`.
func TestAssignment_ModellessProfileLoadsButWarnsThatReuseIsOff(t *testing.T) {
	const body = `
assignment:
  hook_path: /usr/local/bin/fm-route.sh
  profiles:
    - id: claude-opus
      agent: claude
      provider: anthropic-subscription
      effort: xhigh
    - id: codex-sol
      agent: codex
      provider: openai-subscription
      model: gpt-5.6-sol
`
	var cfg *GlobalConfig
	var err error
	logs := captureConfigWarnings(t, func() {
		cfg, err = loadAssignmentYAML(t, body)
	})

	// Required change 1: this configuration must still load.
	if err != nil {
		t.Fatalf("a model-less profile must stay legal, got error: %v", err)
	}
	if got := len(cfg.Assignment.Profiles); got != 2 {
		t.Fatalf("parsed %d profiles, want 2", got)
	}

	// Required change 2: exactly one warning, naming the affected profile.
	if strings.Count(logs, "session reuse is disabled") != 1 {
		t.Fatalf("want exactly one reuse warning, got logs:\n%s", logs)
	}
	if !strings.Contains(logs, "claude-opus") {
		t.Fatalf("the warning must name the affected profile, got logs:\n%s", logs)
	}
	if strings.Contains(logs, "codex-sol") {
		t.Fatalf("the profile that sets a model must not be warned about, got logs:\n%s", logs)
	}

	// The warning describes a real consequence: that profile has no identity
	// to match, while its sibling does.
	if key := routing.ProfileKey(cfg.Assignment.Profiles[0]); key != "" {
		t.Fatalf("the model-less profile must have no identity, got %q", key)
	}
	if key := routing.ProfileKey(cfg.Assignment.Profiles[1]); key == "" {
		t.Fatal("the profile with a model must have an identity")
	}
}

// TestAssignment_ProfilesWithModelsLoadWithoutAReuseWarning proves the warning
// is scoped to the case it describes and does not fire on the documented
// configuration.
func TestAssignment_ProfilesWithModelsLoadWithoutAReuseWarning(t *testing.T) {
	const body = `
assignment:
  hook_path: /usr/local/bin/fm-route.sh
  profiles:
    - id: claude-opus
      agent: claude
      provider: anthropic-subscription
      model: opus
      effort: xhigh
    - id: pi-grok
      agent: pi
      provider: xai-subscription
      model: xai/grok-4.6
`
	var err error
	logs := captureConfigWarnings(t, func() {
		_, err = loadAssignmentYAML(t, body)
	})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if strings.Contains(logs, "session reuse is disabled") {
		t.Fatalf("profiles that set a model must not be warned about, got logs:\n%s", logs)
	}
}

// TestAssignment_EveryModellessProfileIsWarnedAboutByName proves the warning is
// per profile rather than one summary, so an operator fixing a multi-profile
// configuration is told about each one.
func TestAssignment_EveryModellessProfileIsWarnedAboutByName(t *testing.T) {
	const body = `
assignment:
  hook_path: /usr/local/bin/fm-route.sh
  profiles:
    - id: claude-default
      agent: claude
      provider: anthropic-subscription
    - id: codex-default
      agent: codex
      provider: openai-subscription
    - id: pi-grok
      agent: pi
      provider: xai-subscription
      model: xai/grok-4.6
`
	var err error
	logs := captureConfigWarnings(t, func() {
		_, err = loadAssignmentYAML(t, body)
	})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if got := strings.Count(logs, "session reuse is disabled"); got != 2 {
		t.Fatalf("want one warning per model-less profile (2), got %d; logs:\n%s", got, logs)
	}
	for _, id := range []string{"claude-default", "codex-default"} {
		if !strings.Contains(logs, id) {
			t.Fatalf("profile %q was not named in the warnings, got logs:\n%s", id, logs)
		}
	}
	if strings.Contains(logs, "pi-grok") {
		t.Fatalf("the profile that sets a model must not be warned about, got logs:\n%s", logs)
	}
}
