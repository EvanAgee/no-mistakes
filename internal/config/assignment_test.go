package config

import (
	"strings"
	"testing"
	"time"

	"github.com/kunchenguid/no-mistakes/internal/agentcfg"
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
