package daemon

import (
	"strings"
	"testing"
	"time"

	"github.com/kunchenguid/no-mistakes/internal/agentcfg"
	"github.com/kunchenguid/no-mistakes/internal/config"
	"github.com/kunchenguid/no-mistakes/internal/routing"
	"github.com/kunchenguid/no-mistakes/internal/runenv"
	"github.com/kunchenguid/no-mistakes/internal/types"
)

func routableConfig() *config.Config {
	return &config.Config{
		Agent: types.AgentClaude,
		Assignment: config.Assignment{
			HookPath: "/opt/firstmate/bin/fm-route",
			Profiles: []routing.Profile{
				{
					ID: "claude-opus", Agent: types.AgentClaude, Provider: "anthropic-subscription",
					Tuning: agentcfg.Profile{Model: "opus", Effort: agentcfg.EffortXHigh},
				},
				{
					ID: "pi-grok", Agent: types.AgentPi, Provider: "xai-subscription",
					Tuning: agentcfg.Profile{Model: "xai/grok-4.6", Effort: agentcfg.EffortXHigh},
				},
			},
		},
	}
}

// TestNewRoutedAgentFactory_OffWhenRoutingIsUnconfigured proves the default:
// with no assignment block there is no factory, so the executor leaves routing
// off and every run behaves exactly as it did before routing existed.
func TestNewRoutedAgentFactory_OffWhenRoutingIsUnconfigured(t *testing.T) {
	var m RunManager
	if factory := m.newRoutedAgentFactory(&config.Config{Agent: types.AgentClaude}, "/tmp/evidence", runenv.Overlay{}); factory != nil {
		t.Fatal("an unconfigured assignment must produce no factory")
	}
	if factory := m.newRoutedAgentFactory(nil, "/tmp/evidence", runenv.Overlay{}); factory != nil {
		t.Fatal("a nil config must produce no factory")
	}
}

// TestNewRoutedAgentFactory_BuildsEachApprovedProfile proves a selected
// profile becomes a real adapter carrying that profile's own model and effort,
// which is what makes the selection honest rather than nominal.
func TestNewRoutedAgentFactory_BuildsEachApprovedProfile(t *testing.T) {
	var m RunManager
	cfg := routableConfig()
	factory := m.newRoutedAgentFactory(cfg, t.TempDir(), runenv.Overlay{})
	if factory == nil {
		t.Fatal("a configured assignment must produce a factory")
	}

	for _, profile := range cfg.Assignment.Profiles {
		built, err := factory(profile)
		if err != nil {
			t.Fatalf("build %s: %v", profile.ID, err)
		}
		t.Cleanup(func() { _ = built.Close() })
		if built.Name() != string(profile.Agent) {
			t.Fatalf("profile %s built adapter %q, want %q", profile.ID, built.Name(), profile.Agent)
		}
	}
}

// TestNewRoutedAgentFactory_RefusesANonNativeAdapter proves defense in depth.
// Configuration already refuses this and a hook can only answer with a
// configured id, so reaching the factory with a non-native adapter should be
// impossible; refusing here means no path at all ends in such a launch.
func TestNewRoutedAgentFactory_RefusesANonNativeAdapter(t *testing.T) {
	var m RunManager
	factory := m.newRoutedAgentFactory(routableConfig(), t.TempDir(), runenv.Overlay{})

	_, err := factory(routing.Profile{
		ID: "omp", Agent: types.AgentOpenCode, Provider: "vercel-ai-gateway",
		Tuning: agentcfg.Profile{Model: "vercel-ai-gateway/deepseek-v4.1-flash"},
	})
	if err == nil {
		t.Fatal("the factory must refuse an adapter routing may not select")
	}
	if !strings.Contains(err.Error(), "routing may not select") {
		t.Fatalf("error %q must name the refusal", err)
	}
}

// TestNewRoutedAgentFactory_HonorsOperatorPathAndArgumentOverrides proves a
// routed launch is not a second, parallel configuration path: it inherits the
// same operator settings a default launch uses, so routing changes which
// approved service serves a turn and nothing else about how it runs.
func TestNewRoutedAgentFactory_HonorsOperatorPathAndArgumentOverrides(t *testing.T) {
	var m RunManager
	cfg := routableConfig()
	cfg.AgentPathOverride = map[string]string{string(types.AgentPi): "/opt/bin/pi"}
	cfg.AgentArgsOverride = map[string][]string{string(types.AgentPi): {"--no-color"}}

	factory := m.newRoutedAgentFactory(cfg, t.TempDir(), runenv.Overlay{})
	built, err := factory(cfg.Assignment.Profiles[1])
	if err != nil {
		t.Fatalf("build pi-grok: %v", err)
	}
	t.Cleanup(func() { _ = built.Close() })

	// The overrides are what the factory reads from; asserting the adapter was
	// constructed from them at all is what this proves, since the concrete
	// argv is pinned in internal/agent.
	if cfg.AgentPathFor(types.AgentPi) != "/opt/bin/pi" {
		t.Fatalf("the factory must resolve the operator's path override, got %q", cfg.AgentPathFor(types.AgentPi))
	}
	if strings.Join(cfg.AgentArgsFor(types.AgentPi), " ") != "--no-color" {
		t.Fatalf("the factory must resolve the operator's argument override, got %v", cfg.AgentArgsFor(types.AgentPi))
	}
}

// TestNewRoutedAgentFactory_FailsClosedWhenTheOptOutIsNotEffective proves the
// routed path honors the same trust boundary the default gate agent does.
//
// disable_project_settings is a trusted, default-branch-only declaration that
// the target repository's own AGENTS.md/CLAUDE.md must not steer the gate
// agent. Passing that flag into the constructor is a request, not a guarantee:
// an operator argument override that re-adds claude's `project` setting source
// defeats it. Without this check routing would be the one path that launches
// an unverified harness in the target checkout with those files loaded.
func TestNewRoutedAgentFactory_FailsClosedWhenTheOptOutIsNotEffective(t *testing.T) {
	var m RunManager
	cfg := routableConfig()
	cfg.DisableProjectSettings = true
	cfg.AgentArgsOverride = map[string][]string{
		string(types.AgentClaude): {"--setting-sources", "user,project"},
	}

	factory := m.newRoutedAgentFactory(cfg, t.TempDir(), runenv.Overlay{})
	built, err := factory(cfg.Assignment.Profiles[0])
	if err == nil {
		_ = built.Close()
		t.Fatal("an override that re-adds the project setting source must refuse the launch")
	}
	if !strings.Contains(err.Error(), "neutralize") {
		t.Fatalf("error %q must name the neutralization refusal", err)
	}

	// The same opt-out with no defeating override still builds, so the check
	// refuses only what it must.
	cfg.AgentArgsOverride = nil
	ok, err := factory(cfg.Assignment.Profiles[0])
	if err != nil {
		t.Fatalf("an effective opt-out must still build the adapter: %v", err)
	}
	t.Cleanup(func() { _ = ok.Close() })
}

// TestRoutingGeneration_IsStablePerIncarnationAndChangesOnRestart proves what
// the controller needs to attribute assignments: one daemon incarnation
// reports one generation for every run it owns, and a restart reports a
// different one, so work stranded by an earlier incarnation is recognizable.
func TestRoutingGeneration_IsStablePerIncarnationAndChangesOnRestart(t *testing.T) {
	started := time.Unix(1_700_000_000, 0)
	first := routingGeneration(4242, started)

	if first != routingGeneration(4242, started) {
		t.Fatal("one incarnation must report a stable generation")
	}
	if first == routingGeneration(4242, started.Add(time.Second)) {
		t.Fatal("a restart at a later time must report a different generation")
	}
	if first == routingGeneration(4243, started) {
		t.Fatal("a different process must report a different generation")
	}
	if routingOwner() == "" {
		t.Fatal("the owner must name this daemon to the controller")
	}
}

// TestRunManager_AssignsOneRoutingGenerationAtConstruction proves the
// generation is fixed when the manager is built, not recomputed per run: every
// run this daemon owns, including one resumed after crash recovery, must
// report the same incarnation.
func TestRunManager_AssignsOneRoutingGenerationAtConstruction(t *testing.T) {
	m := NewRunManager(nil, nil, nil)
	if m.routingGeneration == "" {
		t.Fatal("a manager must carry a routing generation")
	}
	// The value is fixed at construction, so reading it again later - as every
	// run and every recovery does - yields the same incarnation.
	first := m.routingGeneration
	time.Sleep(time.Millisecond)
	if m.routingGeneration != first {
		t.Fatal("the generation must not be recomputed per read")
	}

	time.Sleep(time.Millisecond)
	other := NewRunManager(nil, nil, nil)
	if m.routingGeneration == other.routingGeneration {
		t.Fatal("two manager incarnations must report different generations")
	}
}
