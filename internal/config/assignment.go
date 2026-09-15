package config

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/kunchenguid/no-mistakes/internal/agentcfg"
	"github.com/kunchenguid/no-mistakes/internal/routing"
	"github.com/kunchenguid/no-mistakes/internal/types"
)

// Assignment is the operator's continuous-routing setting: an external hook
// that picks which approved native profile serves each agent invocation, and
// the named profiles it is allowed to pick from.
//
// It is global-only and never read from a repository, for the same reason
// agent_config and agent_args_override are: it decides which process launches
// with this machine's credentials, against which subscription, on every turn
// of a run. A pushed branch that could add a profile or point the hook
// somewhere else would be choosing what runs as the maintainer. The whole
// block is therefore ignored if it ever appears in repository configuration.
//
// Unset means routing is off, and a run behaves exactly as it did before this
// setting existed: the configured agent launches, once, for every invocation.
type Assignment struct {
	// HookPath is the executable consulted before each native invocation. It
	// is run by argv with a JSON request on stdin; there is no shell, and no
	// part of the request is ever spliced into a command line.
	HookPath string
	// HookArgs are fixed leading arguments for that executable. They come
	// from this file only - never from a hook response, a repository, or a
	// run - so a hook can never extend its own command line.
	HookArgs []string
	// HookTimeout bounds one hook call. Zero means routing.DefaultTimeout.
	HookTimeout time.Duration
	// MaxEvidenceAge bounds how old the hook's own observation may be before
	// a selection is logged as made from undatable evidence. Zero disables
	// the check. It never blocks a launch on its own: the hook owns
	// eligibility, and second-guessing its verdict here would mean two places
	// deciding admission.
	MaxEvidenceAge time.Duration
	// Profiles are the approved named execution profiles, in configured
	// order. A hook may select among these by id and nothing else.
	Profiles []routing.Profile
}

// Enabled reports whether routing is configured. Both halves are required: a
// hook with no profiles could select nothing, and profiles with no hook have
// nothing to select them.
func (a Assignment) Enabled() bool {
	return strings.TrimSpace(a.HookPath) != "" && len(a.Profiles) > 0
}

// Hook builds the routing hook for this setting.
func (a Assignment) Hook() *routing.Hook {
	if !a.Enabled() {
		return nil
	}
	return &routing.Hook{
		Path:           a.HookPath,
		Args:           append([]string(nil), a.HookArgs...),
		Timeout:        a.HookTimeout,
		MaxEvidenceAge: a.MaxEvidenceAge,
	}
}

// assignmentRaw is the on-disk YAML shape.
type assignmentRaw struct {
	HookPath       string                 `yaml:"hook_path"`
	HookArgs       []string               `yaml:"hook_args"`
	HookTimeout    string                 `yaml:"hook_timeout"`
	MaxEvidenceAge string                 `yaml:"max_evidence_age"`
	Profiles       []assignmentProfileRaw `yaml:"profiles"`
}

type assignmentProfileRaw struct {
	ID       string   `yaml:"id"`
	Agent    string   `yaml:"agent"`
	Provider string   `yaml:"provider"`
	Model    string   `yaml:"model"`
	Effort   string   `yaml:"effort"`
	Roles    []string `yaml:"roles"`
}

// parseAssignment validates the assignment block and resolves it to approved
// profiles. It fails closed on anything it cannot honestly launch later: an
// adapter routing does not natively support, a tuning knob that adapter cannot
// express, a duplicate or empty id, or a missing billing-route label. Catching
// those here means a selection can never be admitted and then silently
// downgraded to something the run would misreport.
func parseAssignment(raw assignmentRaw) (Assignment, error) {
	out := Assignment{
		HookPath: strings.TrimSpace(raw.HookPath),
		HookArgs: raw.HookArgs,
	}

	timeout, err := parseAssignmentDuration("hook_timeout", raw.HookTimeout)
	if err != nil {
		return Assignment{}, err
	}
	out.HookTimeout = timeout

	maxAge, err := parseAssignmentDuration("max_evidence_age", raw.MaxEvidenceAge)
	if err != nil {
		return Assignment{}, err
	}
	out.MaxEvidenceAge = maxAge

	seen := make(map[string]struct{}, len(raw.Profiles))
	for i, entry := range raw.Profiles {
		effort, err := agentcfg.ParseEffort(entry.Effort)
		if err != nil {
			return Assignment{}, fmt.Errorf("invalid assignment.profiles[%d]: %w", i, err)
		}
		profile := routing.Profile{
			ID:       strings.TrimSpace(entry.ID),
			Agent:    types.AgentName(strings.TrimSpace(entry.Agent)),
			Provider: strings.TrimSpace(entry.Provider),
			Tuning:   agentcfg.Profile{Model: strings.TrimSpace(entry.Model), Effort: effort},
			Roles:    normalizeRoles(entry.Roles),
		}
		if err := profile.Validate(); err != nil {
			return Assignment{}, fmt.Errorf("invalid assignment.profiles[%d]: %w", i, err)
		}
		if _, duplicate := seen[profile.ID]; duplicate {
			return Assignment{}, fmt.Errorf("invalid assignment.profiles[%d]: duplicate profile id %q", i, profile.ID)
		}
		seen[profile.ID] = struct{}{}
		out.Profiles = append(out.Profiles, profile)
	}

	if len(out.Profiles) > 0 && out.HookPath == "" {
		return Assignment{}, fmt.Errorf("invalid assignment: profiles are configured but assignment.hook_path is empty, so nothing can select among them")
	}
	if out.HookPath != "" && len(out.Profiles) == 0 {
		return Assignment{}, fmt.Errorf("invalid assignment: assignment.hook_path is set but no profiles are configured, so the hook has nothing it may select")
	}
	return out, nil
}

func parseAssignmentDuration(field, raw string) (time.Duration, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return 0, nil
	}
	value, err := time.ParseDuration(raw)
	if err != nil {
		return 0, fmt.Errorf("invalid assignment.%s: %w", field, err)
	}
	if value < 0 {
		return 0, fmt.Errorf("invalid assignment.%s: must not be negative", field)
	}
	return value, nil
}

// normalizeRoles trims, drops blanks, deduplicates and sorts, so the same set
// written in a different order is the same restriction.
func normalizeRoles(raw []string) []string {
	if len(raw) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(raw))
	out := make([]string, 0, len(raw))
	for _, role := range raw {
		role = strings.TrimSpace(role)
		if role == "" {
			continue
		}
		if _, duplicate := seen[role]; duplicate {
			continue
		}
		seen[role] = struct{}{}
		out = append(out, role)
	}
	if len(out) == 0 {
		return nil
	}
	sort.Strings(out)
	return out
}
