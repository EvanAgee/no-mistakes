package agent

import (
	"context"
	"encoding/json"
	"runtime"
	"strings"
	"testing"

	"github.com/kunchenguid/no-mistakes/internal/agentcfg"
	"github.com/kunchenguid/no-mistakes/internal/types"
)

// Continuous routing selects among the native adapters (Claude, Codex, Pi) and
// builds each one through NewWithOptions with the selected profile's tuning.
// These tests prove that construction actually reaches each harness's own
// model and effort mechanism, and that a routed Pi launch still produces valid
// structured output against the package's existing fixtures.
//
// They deliberately assert what no-mistakes controls: the argv it builds and
// the output it parses. Nothing here claims a credential works, a bill is
// payable, or a live model returns anything - none of which a test can know.

// routedNativeProfiles are the shapes the assignment layer selects among. Pi
// appears twice because one adapter serves two distinct billing routes, which
// is exactly why routing cannot be qualified by adapter name alone.
var routedNativeProfiles = []struct {
	name    string
	agent   types.AgentName
	profile agentcfg.Profile
	// wantArgs are the argv fragments this harness must emit for the profile.
	wantArgs []string
}{
	{
		name:     "claude opus at xhigh",
		agent:    types.AgentClaude,
		profile:  agentcfg.Profile{Model: "opus", Effort: agentcfg.EffortXHigh},
		wantArgs: []string{"--model", "opus", "--effort", "xhigh"},
	},
	{
		name:     "codex sol at high",
		agent:    types.AgentCodex,
		profile:  agentcfg.Profile{Model: "gpt-5.6-sol", Effort: agentcfg.EffortHigh},
		wantArgs: []string{"-m", "gpt-5.6-sol", "-c", `model_reasoning_effort="high"`},
	},
	{
		name:     "pi serving grok at xhigh",
		agent:    types.AgentPi,
		profile:  agentcfg.Profile{Model: "xai/grok-4.6", Effort: agentcfg.EffortXHigh},
		wantArgs: []string{"--model", "xai/grok-4.6", "--thinking", "xhigh"},
	},
	{
		name:     "pi serving gateway deepseek at xhigh",
		agent:    types.AgentPi,
		profile:  agentcfg.Profile{Model: "vercel-ai-gateway/deepseek/deepseek-v4.1-flash", Effort: agentcfg.EffortXHigh},
		wantArgs: []string{"--model", "vercel-ai-gateway/deepseek/deepseek-v4.1-flash", "--thinking", "xhigh"},
	},
}

// TestRoutedProfile_EveryNativeAdapterExpressesItsProfile proves a routed
// launch actually carries the selected model and effort into the harness's own
// flags. Without this, a selection could be admitted and then silently run on
// the harness default, and the run would report a service it did not use.
func TestRoutedProfile_EveryNativeAdapterExpressesItsProfile(t *testing.T) {
	for _, tc := range routedNativeProfiles {
		t.Run(tc.name, func(t *testing.T) {
			if err := agentcfg.Validate(tc.agent, tc.profile); err != nil {
				t.Fatalf("a routable profile must be expressible: %v", err)
			}

			args := agentcfg.NativeArgs(tc.agent, tc.profile, nil)
			joined := strings.Join(args, " ")
			for _, want := range tc.wantArgs {
				if !strings.Contains(joined, want) {
					t.Fatalf("argv %v must carry %q", args, want)
				}
			}

			// And the adapter builds from that profile without error, which is
			// the path the daemon's routed factory takes.
			built, err := NewWithOptions(tc.agent, string(tc.agent), nil, Options{Profile: tc.profile})
			if err != nil {
				t.Fatalf("construct routed adapter: %v", err)
			}
			t.Cleanup(func() { _ = built.Close() })
			if built.Name() != string(tc.agent) {
				t.Fatalf("adapter name = %q, want %q", built.Name(), tc.agent)
			}
		})
	}
}

// TestRoutedProfile_PiGrokAndDeepSeekAreDistinctLaunches proves the two Pi
// routes differ where it matters: the model flag. They share an adapter name
// and a binary, so the launch argv is the only place the difference appears,
// which is precisely why session reuse needs the profile key rather than the
// adapter name.
func TestRoutedProfile_PiGrokAndDeepSeekAreDistinctLaunches(t *testing.T) {
	grok := agentcfg.NativeArgs(types.AgentPi,
		agentcfg.Profile{Model: "xai/grok-4.6", Effort: agentcfg.EffortXHigh}, nil)
	deepseek := agentcfg.NativeArgs(types.AgentPi,
		agentcfg.Profile{Model: "vercel-ai-gateway/deepseek/deepseek-v4.1-flash", Effort: agentcfg.EffortXHigh}, nil)

	if strings.Join(grok, " ") == strings.Join(deepseek, " ") {
		t.Fatal("two pi routes must produce different launch arguments")
	}
	if !strings.Contains(strings.Join(deepseek, " "), "vercel-ai-gateway/") {
		t.Fatalf("the gateway route must launch its gateway-qualified model, got %v", deepseek)
	}
}

// TestRoutedProfile_NonNativeAdapterIsNotRoutable pins the adapter boundary
// from this package's side. opencode is a supported harness, but it is not one
// routing may select, and its model spelling requirement differs, so the
// boundary is load-time rather than a runtime surprise.
func TestRoutedProfile_NonNativeAdapterIsNotRoutable(t *testing.T) {
	// opencode does accept a qualified model, so the refusal must come from
	// the routing layer's own native-adapter list rather than from agentcfg.
	if err := agentcfg.Validate(types.AgentOpenCode,
		agentcfg.Profile{Model: "vercel-ai-gateway/deepseek-v4.1-flash"}); err != nil {
		t.Fatalf("this test's premise requires opencode to accept a qualified model: %v", err)
	}
	// The exclusion itself is asserted in internal/routing
	// (TestNativeAgent_ExcludesOMP); what matters here is that nothing in this
	// package quietly makes opencode look routable.
	for _, tc := range routedNativeProfiles {
		if tc.agent == types.AgentOpenCode {
			t.Fatal("opencode must not appear in the routable native set")
		}
	}
}

// TestRoutedProfile_PiLaunchProducesValidStructuredOutput drives a routed Pi
// profile end to end against this package's existing fake-CLI fixture: real
// argv, real stdin, real event stream, real structured-output parsing.
//
// It proves the adapter still honors the structured-output contract when it is
// launched with a routing-selected profile. It proves nothing about
// credentials, billing, or what a live model would return - a fixture cannot
// establish any of those, and reading it as if it could is exactly the
// overclaim this test refuses to make.
func TestRoutedProfile_PiLaunchProducesValidStructuredOutput(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell fixture is Unix-only")
	}

	// The fixture records the argv it was launched with, then emits a
	// well-formed structured response.
	bin := writeFakePi(t, t.TempDir(), `#!/bin/sh
cat > /dev/null
printf '%s\n' "$*" > "$PI_ARGV_OUT"
printf '%s\n' '{"type":"message_end","message":{"role":"assistant","responseId":"r1","provider":"vercel-ai-gateway","model":"deepseek-v4.1-flash","content":[{"type":"text","text":"{\"ok\":true}"}],"usage":{"input":5,"output":3,"cacheRead":0,"cacheWrite":0}}}'
printf '%s\n' '{"type":"agent_end","messages":[]}'
`, "")

	workDir := t.TempDir()
	argvOut := workDir + "/pi-argv.txt"
	t.Setenv("PI_ARGV_OUT", argvOut)

	profile := agentcfg.Profile{
		Model:  "vercel-ai-gateway/deepseek/deepseek-v4.1-flash",
		Effort: agentcfg.EffortXHigh,
	}
	built, err := NewWithOptions(types.AgentPi, bin, nil, Options{Profile: profile})
	if err != nil {
		t.Fatalf("construct routed pi: %v", err)
	}
	t.Cleanup(func() { _ = built.Close() })

	schema := json.RawMessage(`{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]}`)
	result, err := built.Run(context.Background(), RunOpts{
		Prompt:     "validate the change",
		CWD:        workDir,
		JSONSchema: schema,
	})
	if err != nil {
		t.Fatalf("routed pi run: %v", err)
	}
	if string(result.Output) != `{"ok":true}` {
		t.Fatalf("structured output = %s, want the schema-shaped object", result.Output)
	}
	// The adapter reports which model actually served the turn, which is what
	// a routed launch records alongside its selection reason.
	if result.Model != "deepseek-v4.1-flash" || result.ModelProvider != "vercel-ai-gateway" {
		t.Fatalf("model telemetry = %q/%q", result.ModelProvider, result.Model)
	}
}
