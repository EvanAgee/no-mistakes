package routing

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"sort"
	"strings"
	"time"
)

// The assignment hook is an external executable the operator configures once,
// globally. It is the only thing that decides which approved route serves the
// next invocation, and it is deliberately the only thing no-mistakes does not
// own here: balancing across subscriptions needs evidence (quota readings,
// outages, auth state, what every other local caller is already running) that
// belongs to whatever owns those accounts, not to one pipeline run.
//
// The protocol is bounded in both directions on purpose.
//
// Outbound, no-mistakes sends one JSON object on the hook's stdin and nothing
// else. The executable and its fixed arguments come from trusted global
// configuration and are executed by argv, never through a shell: no field of
// the request is ever concatenated into a command line, so no assignment id,
// branch name or route id can be read as a shell token.
//
// Inbound, the hook may say one of exactly two things: it selected one of the
// route ids no-mistakes offered it, or it deferred. It cannot name a command,
// a path, an argument, a model, a credential or a route that was not in the
// offered set. Everything else it returns is evidence for the operator's log.
// That asymmetry is the whole security property: the worst a wrong or hostile
// hook answer can do is pick a different approved route, or stall the turn.

// Verb names one hook call.
type Verb string

const (
	// VerbAcquire asks which approved route may serve one invocation.
	VerbAcquire Verb = "acquire"
	// VerbFinish reports how that invocation ended. It is idempotent by
	// contract: the hook must accept a repeat finish for an assignment it has
	// already closed without treating it as a new event.
	VerbFinish Verb = "finish"
)

// Outcome is the normalized result no-mistakes reports to finish. The
// vocabulary is closed so a hook can route on it without parsing prose, and
// deliberately distinguishes a failure to launch at all from a launch that ran
// and failed: only the former means the route was never actually exercised.
type Outcome string

const (
	// OutcomeSuccess means the invocation completed and returned a result.
	OutcomeSuccess Outcome = "success"
	// OutcomeFailed means the invocation launched and then failed, including
	// a timeout, a cancellation, or a rejected structured output. The route
	// served a turn; whether it served it well is not this field's business.
	OutcomeFailed Outcome = "failed"
	// OutcomeLaunchFailed means the process never started, so the route
	// consumed nothing. A hook that counts running work must release it.
	OutcomeLaunchFailed Outcome = "launch-failed"
)

// Request is the bounded JSON object sent on the hook's stdin.
type Request struct {
	Verb Verb `json:"verb"`
	// Assignment is the stable identity of one invocation attempt. A repeat
	// acquire with the same assignment and owner must return the same record,
	// so a retried or recovered attempt cannot be counted twice.
	Assignment string `json:"assignment"`
	// Owner identifies who holds the assignment, and Generation identifies
	// which incarnation of that owner. Together they let the hook tell a
	// recovered daemon reclaiming its own work from an unrelated process
	// reusing an id it does not own.
	Owner      string `json:"owner"`
	Generation string `json:"generation"`
	// Role is the pipeline duty this invocation serves. The hook may use it
	// as evidence, but it never widens what is allowed: Routes is already
	// filtered to the profiles this role may use.
	Role string `json:"role,omitempty"`
	// Routes are the approved profile ids this invocation may use, sorted.
	// A selection outside this set is refused by the caller.
	Routes []string `json:"routes,omitempty"`
	// Outcome is set for finish only.
	Outcome Outcome `json:"outcome,omitempty"`
	// Profile is the route id the invocation actually launched, set for
	// finish only, so the hook can record what it really spent.
	Profile string `json:"profile,omitempty"`
}

// Response is the bounded JSON object read from the hook's stdout.
type Response struct {
	// Status is "selected", "deferred" or "closed".
	Status string `json:"status"`
	// Route is the selected profile id. Required when status is "selected",
	// and meaningless otherwise.
	Route string `json:"route,omitempty"`
	// Reason is short, operator-facing evidence for the decision. It is
	// recorded verbatim in the run log and never parsed for control flow.
	Reason string `json:"reason,omitempty"`
	// Generation and ObservedAt are the freshness evidence behind the
	// decision: which eligibility snapshot it was made from, and when that
	// snapshot was taken (RFC 3339). A caller that requires fresh evidence
	// checks ObservedAt; see Decision.Fresh.
	Generation string `json:"generation,omitempty"`
	ObservedAt string `json:"observed_at,omitempty"`
}

const (
	statusSelected = "selected"
	statusDeferred = "deferred"
	statusClosed   = "closed"
)

// maxResponseBytes bounds what the hook may return. A hook that floods stdout
// is a malfunctioning hook, and reading it unboundedly would let one turn's
// routing decision exhaust the daemon's memory. The limit is far above any
// honest response and is enforced by reading one byte past it.
const maxResponseBytes = 64 << 10

// DefaultTimeout bounds one hook call. Routing sits in front of every agent
// invocation, so a hung hook would stall the whole pipeline; the call is cheap
// by design (it reads a snapshot another process refreshes on its own timer),
// so a short budget is generous.
const DefaultTimeout = 30 * time.Second

// Decision is the validated result of one acquire.
type Decision struct {
	// Profile is the approved profile the hook selected. Valid only when
	// Selected is true.
	Profile Profile
	// Selected reports whether a route was admitted. False means deferred:
	// the attempt may be retried later, and must not be remembered as a
	// permanent verdict, because the next refresh may admit it.
	Selected bool
	// Reason, Generation and ObservedAt are the hook's own evidence.
	Reason     string
	Generation string
	ObservedAt time.Time
	// Fresh reports whether ObservedAt was present and within the caller's
	// freshness window. A selection made from evidence the hook could not
	// date is recorded as not fresh rather than assumed current.
	Fresh bool
}

// Hook invokes the configured assignment executable.
type Hook struct {
	// Path and Args come from trusted global configuration. They are passed
	// to exec by argv; no part of a request is ever spliced into them.
	Path string
	Args []string
	// Timeout bounds one call. Zero means DefaultTimeout.
	Timeout time.Duration
	// MaxEvidenceAge bounds how old the hook's own observation may be for a
	// decision to count as fresh. Zero disables the check, which records
	// Fresh from presence alone.
	MaxEvidenceAge time.Duration
	// Now and run exist for tests. Nil means the real clock and exec.
	Now func() time.Time
	run func(ctx context.Context, path string, args []string, stdin []byte) ([]byte, error)
}

// ErrDeferred is returned by Acquire when the hook admitted no route. It is
// deliberately a distinct error: the attempt is not wrong and not permanently
// excluded, it is only not admissible right now, so a caller may retry it
// after recovery instead of caching the verdict.
var ErrDeferred = errors.New("assignment deferred by routing hook")

// Acquire asks the hook which of the allowed profiles may serve one
// invocation. allowed is already filtered to the profiles this role may use,
// so the hook is never offered a route the operator excluded for this duty.
//
// Every failure path is closed: an unknown or disallowed route id, malformed
// or oversized output, a nonzero exit, a timeout, or an empty answer all
// return an error and never launch anything. That is the point of the whole
// seam - a routing hook that misbehaves must be unable to start a process on
// a route the operator excluded.
func (h *Hook) Acquire(ctx context.Context, req Request, allowed []Profile) (Decision, error) {
	if err := validateAcquireRequest(req); err != nil {
		return Decision{}, err
	}
	if len(allowed) == 0 {
		return Decision{}, fmt.Errorf("routing: no approved profile is allowed for role %q", req.Role)
	}

	byID := make(map[string]Profile, len(allowed))
	ids := make([]string, 0, len(allowed))
	for _, profile := range allowed {
		byID[profile.ID] = profile
		ids = append(ids, profile.ID)
	}
	sort.Strings(ids)

	req.Verb = VerbAcquire
	req.Routes = ids
	resp, err := h.call(ctx, req)
	if err != nil {
		return Decision{}, err
	}

	switch resp.Status {
	case statusDeferred:
		return Decision{
			Selected:   false,
			Reason:     resp.Reason,
			Generation: resp.Generation,
		}, fmt.Errorf("%w: %s", ErrDeferred, evidence(resp.Reason))
	case statusSelected:
	default:
		return Decision{}, fmt.Errorf("routing: hook returned unusable status %q for acquire", truncate(resp.Status, 64))
	}

	// The selection is resolved against the offered set only. A route the
	// hook invented, or one that belongs to a different role, never reaches a
	// launch: an id no-mistakes did not offer has no approved profile behind
	// it, so there is nothing to run even if the hook insists.
	profile, ok := byID[resp.Route]
	if !ok {
		return Decision{}, fmt.Errorf("routing: hook selected route %q, which is not among the %d approved routes offered for role %q",
			truncate(resp.Route, 64), len(ids), req.Role)
	}

	decision := Decision{
		Profile:    profile,
		Selected:   true,
		Reason:     resp.Reason,
		Generation: resp.Generation,
	}
	observed, err := parseObservedAt(resp.ObservedAt)
	if err != nil {
		return Decision{}, fmt.Errorf("routing: hook reported unusable observation time: %w", err)
	}
	decision.ObservedAt = observed
	decision.Fresh = h.fresh(observed)
	return decision, nil
}

// Finish reports the normalized outcome of one invocation. It is called on
// success, on failure, and on a launch that never started, so a hook counting
// in-flight work is never left holding an assignment that has already ended.
//
// A finish error is returned but is never fatal to the caller's own work: the
// turn it describes has already happened, and refusing to report it would only
// lose the record. Callers log it and continue.
func (h *Hook) Finish(ctx context.Context, req Request) error {
	if strings.TrimSpace(req.Assignment) == "" {
		return errors.New("routing: finish requires an assignment id")
	}
	if strings.TrimSpace(req.Owner) == "" {
		return errors.New("routing: finish requires an owner")
	}
	switch req.Outcome {
	case OutcomeSuccess, OutcomeFailed, OutcomeLaunchFailed:
	default:
		return fmt.Errorf("routing: finish requires a normalized outcome (%s, %s or %s), got %q",
			OutcomeSuccess, OutcomeFailed, OutcomeLaunchFailed, truncate(string(req.Outcome), 64))
	}
	req.Verb = VerbFinish
	req.Routes = nil
	resp, err := h.call(ctx, req)
	if err != nil {
		return err
	}
	if resp.Status != statusClosed {
		return fmt.Errorf("routing: hook returned unusable status %q for finish", truncate(resp.Status, 64))
	}
	return nil
}

func validateAcquireRequest(req Request) error {
	if strings.TrimSpace(req.Assignment) == "" {
		return errors.New("routing: acquire requires an assignment id")
	}
	if strings.TrimSpace(req.Owner) == "" {
		return errors.New("routing: acquire requires an owner")
	}
	if strings.TrimSpace(req.Generation) == "" {
		return errors.New("routing: acquire requires an owner generation")
	}
	return nil
}

func (h *Hook) call(ctx context.Context, req Request) (Response, error) {
	if strings.TrimSpace(h.Path) == "" {
		return Response{}, errors.New("routing: no assignment hook is configured")
	}
	payload, err := json.Marshal(req)
	if err != nil {
		return Response{}, fmt.Errorf("routing: encode %s request: %w", req.Verb, err)
	}

	timeout := h.Timeout
	if timeout <= 0 {
		timeout = DefaultTimeout
	}
	callCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	run := h.run
	if run == nil {
		run = execHook
	}
	stdout, err := run(callCtx, h.Path, h.Args, payload)
	if err != nil {
		if errors.Is(callCtx.Err(), context.DeadlineExceeded) && ctx.Err() == nil {
			return Response{}, fmt.Errorf("routing: assignment hook %s timed out after %s", req.Verb, timeout)
		}
		return Response{}, fmt.Errorf("routing: assignment hook %s failed: %w", req.Verb, err)
	}
	if len(stdout) > maxResponseBytes {
		return Response{}, fmt.Errorf("routing: assignment hook %s returned more than %d bytes", req.Verb, maxResponseBytes)
	}

	var resp Response
	decoder := json.NewDecoder(bytes.NewReader(stdout))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&resp); err != nil {
		return Response{}, fmt.Errorf("routing: assignment hook %s returned unreadable output: %w", req.Verb, err)
	}
	// Exactly one JSON object, so a hook cannot append a second decision the
	// caller would never see.
	if err := decoder.Decode(new(json.RawMessage)); err != io.EOF {
		return Response{}, fmt.Errorf("routing: assignment hook %s returned more than one JSON object", req.Verb)
	}
	return resp, nil
}

// execHook runs the configured executable by argv with the request on stdin.
// There is no shell anywhere in this path: the operator's configured path and
// arguments are passed as separate argv entries, so nothing in the request can
// be interpreted as a command.
func execHook(ctx context.Context, path string, args []string, stdin []byte) ([]byte, error) {
	cmd := exec.CommandContext(ctx, path, args...)
	cmd.Stdin = bytes.NewReader(stdin)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		if text := strings.TrimSpace(stderr.String()); text != "" {
			return nil, fmt.Errorf("%w: %s", err, truncate(strings.Join(strings.Fields(text), " "), 400))
		}
		return nil, err
	}
	return stdout.Bytes(), nil
}

func (h *Hook) fresh(observed time.Time) bool {
	if observed.IsZero() {
		return false
	}
	if h.MaxEvidenceAge <= 0 {
		return true
	}
	now := time.Now
	if h.Now != nil {
		now = h.Now
	}
	age := now().Sub(observed)
	if age < 0 {
		// Evidence dated in the future is not proof of freshness; a skewed or
		// wrong clock on either side must not read as current.
		return false
	}
	return age <= h.MaxEvidenceAge
}

func parseObservedAt(raw string) (time.Time, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return time.Time{}, nil
	}
	observed, err := time.Parse(time.RFC3339, raw)
	if err != nil {
		return time.Time{}, fmt.Errorf("%q is not an RFC 3339 timestamp", truncate(raw, 64))
	}
	return observed, nil
}

func evidence(reason string) string {
	reason = strings.TrimSpace(reason)
	if reason == "" {
		return "the hook reported no reason"
	}
	return truncate(strings.Join(strings.Fields(reason), " "), 400)
}

func truncate(text string, max int) string {
	runes := []rune(text)
	if len(runes) <= max {
		return text
	}
	return string(runes[:max]) + "..."
}
