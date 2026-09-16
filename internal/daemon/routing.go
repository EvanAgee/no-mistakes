package daemon

import (
	"fmt"
	"time"

	"github.com/kunchenguid/no-mistakes/internal/agent"
	"github.com/kunchenguid/no-mistakes/internal/config"
	"github.com/kunchenguid/no-mistakes/internal/routing"
	"github.com/kunchenguid/no-mistakes/internal/runenv"
)

// newRoutedAgentFactory builds the adapter factory the pipeline uses when an
// assignment hook selects a profile. Adapter construction stays here, in the
// daemon, because it needs the evidence root, executable lookup and forge
// environment overlay the daemon owns. internal/routing therefore never
// constructs anything, and internal/pipeline never learns how.
//
// The factory takes a routing.Profile, not a name or a command. That profile
// came out of the operator's own global configuration and was matched to the
// hook's answer by id, so the only thing a hook can influence here is which of
// the operator's approved profiles is built. It can never supply a binary
// path, an argument, or a model string of its own.
//
// Every routed launch inherits the same operator-level settings a default
// launch would - path overrides, raw argument overrides, the project-settings
// opt-out, the forge environment - so routing changes which approved service
// serves a turn and nothing else about how that turn runs.
func (m *RunManager) newRoutedAgentFactory(cfg *config.Config, evidenceRoot string, environment runenv.Overlay) func(routing.Profile) (agent.Agent, error) {
	if cfg == nil || !cfg.Assignment.Enabled() {
		return nil
	}
	return func(profile routing.Profile) (agent.Agent, error) {
		if !routing.NativeAgent(profile.Agent) {
			// Defense in depth: config load already refused this, and the hook
			// can only answer with a configured id. Refusing again here means
			// no path at all reaches a launch of a non-native adapter.
			return nil, fmt.Errorf("profile %q names agent %q, which routing may not select", profile.ID, profile.Agent)
		}
		built, err := agent.NewWithOptions(profile.Agent, cfg.AgentPathFor(profile.Agent), cfg.AgentArgsFor(profile.Agent), agent.Options{
			ACPRegistryOverrides:   cfg.ACPRegistryOverrides,
			DisableProjectSettings: cfg.DisableProjectSettings,
			// The profile's own tuning wins for a routed launch: the whole
			// point of selecting this profile was its model and effort, so
			// substituting the agent's default agent_config entry here would
			// launch something other than what was admitted.
			Profile:     profile.Tuning,
			Environment: environment,
		})
		if err != nil {
			return nil, err
		}
		routed := agent.WithSteering(built, evidenceRoot)
		// Fail closed under the trusted opt-out, exactly as the default gate
		// agent does (see newGateAgent): passing DisableProjectSettings into
		// the constructor is a request, not a guarantee - an operator argument
		// override that re-adds `project` or `local` setting sources defeats
		// it. Without this check, routing would be the one path that launches
		// an unverified harness in the target checkout with the repository's
		// own project instructions loaded.
		if cfg.DisableProjectSettings {
			if err := agent.EnsureGateNeutralized(routed); err != nil {
				_ = routed.Close()
				return nil, err
			}
		}
		return routed, nil
	}
}

// routingOwner identifies this machine's no-mistakes daemon to the assignment
// hook, and routingGeneration identifies this incarnation of it.
//
// The split matters for recovery. A daemon that restarts and resumes parked
// runs is the SAME owner, so the hook can recognize the assignments it is
// reclaiming as its own rather than as an unrelated process reusing ids. The
// generation changes with every start, so the hook can also tell that the
// earlier incarnation is gone and that anything it left running will never be
// finished by that incarnation. Neither value is a secret and neither is a
// credential: they name a process, not an account.
func routingOwner() string { return "no-mistakes-daemon" }

func routingGeneration(pid int, startedAt time.Time) string {
	return fmt.Sprintf("%d-%d", pid, startedAt.UnixNano())
}
