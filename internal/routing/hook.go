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
// The wire shape is the controller's published contract, not one invented
// here: the verb is a subcommand argument, the request is one JSON object on
// stdin, and the reply is one JSON object on stdout discriminated by `result`.
// There is deliberately no second protocol and no text-line variant.
//
// The protocol is bounded in both directions on purpose.
//
// Outbound, no-mistakes sends one JSON object on the hook's stdin and nothing
// else. The executable, its fixed arguments and the verb come from trusted
// global configuration and are executed by argv, never through a shell: no
// field of the request is ever concatenated into a command line, so no
// assignment id, branch name or route id can be read as a shell token.
//
// Inbound, only one reply authorizes a launch: `selected`, naming a route id
// no-mistakes itself offered. Every other reply - deferred, already-closed,
// error, or anything unrecognized - authorizes nothing. The hook cannot name a
// command, a path, an argument, a model, a credential or a route that was not
// in the offered set. That asymmetry is the whole security property: the worst
// a wrong or hostile hook answer can do is pick a different approved route, or
// stall the turn.

// Verb names one hook call. It is passed as the executable's subcommand
// argument, after the operator's own configured arguments.
type Verb string

const (
	// VerbAcquire asks which approved route may serve one invocation.
	VerbAcquire Verb = "acquire"
	// VerbFinish reports how that invocation ended. It is idempotent by
	// contract: the hook accepts a repeat finish for an assignment it has
	// already closed with no effect and no state change.
	VerbFinish Verb = "finish"
)

// Outcome is the normalized result no-mistakes reports to finish. The
// vocabulary is the controller's, closed so it can route on the value without
// parsing prose.
//
// The three failure values beyond a plain launch failure are verified evidence
// against the route itself: the controller excludes that route immediately
// rather than waiting for its next probe. no-mistakes therefore reports them
// only when the adapter actually established that condition, never as a guess
// about why a turn failed.
type Outcome string

const (
	// OutcomeSuccess means the invocation completed and returned a result.
	OutcomeSuccess Outcome = "success"
	// OutcomeLaunchFailed means the process never started, so the route
	// consumed nothing. A controller counting running work must release it.
	OutcomeLaunchFailed Outcome = "launch-failed"
	// OutcomeAuthFailed means the route's credentials were rejected.
	OutcomeAuthFailed Outcome = "auth-failed"
	// OutcomeExhausted means the route's quota is spent.
	OutcomeExhausted Outcome = "exhausted"
	// OutcomeOutage means the route's provider was unreachable or erroring.
	OutcomeOutage Outcome = "outage"
)

// Valid reports whether an outcome is one the controller accepts.
func (o Outcome) Valid() bool {
	switch o {
	case OutcomeSuccess, OutcomeLaunchFailed, OutcomeAuthFailed, OutcomeExhausted, OutcomeOutage:
		return true
	}
	return false
}

// outcomeNames lists the vocabulary for error text.
func outcomeNames() string {
	return strings.Join([]string{
		string(OutcomeSuccess), string(OutcomeLaunchFailed),
		string(OutcomeAuthFailed), string(OutcomeExhausted), string(OutcomeOutage),
	}, ", ")
}

// Owner identifies who holds an assignment and which incarnation of them.
//
// Identity is the idempotency key's other half: a repeat acquire with the same
// assignment id and the same identity returns the existing record. Generation
// is a freshness token that is NOT part of that key; the controller stores it
// and uses it to tell a live owner from one that was torn down, so a relaunched
// owner's stale assignments stop being counted against its route.
type Owner struct {
	Identity   string `json:"identity"`
	Generation string `json:"generation"`
}

// Request is the bounded JSON object sent on the hook's stdin. The verb is not
// a field: it is the subcommand argument.
type Request struct {
	// AssignmentID is the stable identity of one launch. A retry that may
	// land on a different route is its own launch and needs its own id.
	AssignmentID string `json:"assignment_id"`
	// Owner is who holds this assignment, and which incarnation.
	Owner Owner `json:"owner"`
	// Routes are the approved route ids this invocation may use, sorted. A
	// selection outside this set is refused by the caller.
	Routes []string `json:"routes,omitempty"`
	// Outcome is set for finish only.
	Outcome Outcome `json:"outcome,omitempty"`
	// Profile is the concrete, nonsecret identity that actually launched, set
	// for finish only, so the controller can record what was really spent.
	Profile *LaunchedProfile `json:"profile,omitempty"`
}

// LaunchedProfile is the resolved, nonsecret identity of what ran. It carries
// no credential and no account selection, only what the operator declared.
type LaunchedProfile struct {
	Adapter string `json:"adapter"`
	Model   string `json:"model,omitempty"`
	Effort  string `json:"effort,omitempty"`
}

// Response is the bounded JSON object read from the hook's stdout.
type Response struct {
	// Result discriminates the reply. Only resultSelected authorizes a launch.
	Result string `json:"result"`
	// RouteID is the selected route. Meaningful only for resultSelected.
	RouteID string `json:"route_id,omitempty"`
	// Reason is short, operator-facing evidence for the decision. It is
	// recorded verbatim in the run log and never parsed for control flow.
	Reason string `json:"reason,omitempty"`
	// Generation is the controller ledger's own monotonic decision generation,
	// distinct from the request's owner generation.
	Generation int64 `json:"generation,omitempty"`
	// AssignmentID echoes the closed assignment on a finish reply.
	AssignmentID string `json:"assignment_id,omitempty"`
	// Error carries the message for resultError.
	Error string `json:"error,omitempty"`
	// Note is optional free text the controller may attach. It is operator
	// evidence only and never changes what is authorized.
	Note string `json:"note,omitempty"`
}

const (
	// resultSelected is the ONLY reply that authorizes a launch.
	resultSelected = "selected"
	// resultDeferred means no approved route is admissible right now. It is
	// retriable: the controller re-evaluates the same id on the next call.
	resultDeferred = "deferred"
	// resultAlreadyClosed means this assignment has already run and been
	// closed. It carries no route and never authorizes another launch; a
	// relaunch is its own assignment id.
	resultAlreadyClosed = "already-closed"
	// resultClosed acknowledges a finish.
	resultClosed = "closed"
	// resultError is the controller's own refusal of a malformed request.
	resultError = "error"
)

// maxResponseBytes bounds what the hook may return. A hook that floods stdout
// is a malfunctioning hook, and reading it unboundedly would let one turn's
// routing decision exhaust the daemon's memory. The limit is far above any
// honest response.
const maxResponseBytes = 64 << 10

// DefaultTimeout bounds one hook call. Routing sits in front of every agent
// invocation, so a hung hook would stall the whole pipeline. The contract says
// acquire and finish never block on network I/O, so a short budget is generous.
const DefaultTimeout = 30 * time.Second

// Decision is the validated result of one acquire.
type Decision struct {
	// Profile is the approved profile the hook selected. Valid only when
	// Selected is true.
	Profile Profile
	// Selected reports whether a launch is authorized. It is true only for a
	// `selected` reply naming an offered route.
	Selected bool
	// Reason and Generation are the controller's own evidence for the
	// decision, recorded in the run log and never parsed for control flow.
	Reason     string
	Generation int64
}

// Hook invokes the configured assignment executable.
type Hook struct {
	// Path and Args come from trusted global configuration. They are passed
	// to exec by argv; no part of a request is ever spliced into them.
	Path string
	Args []string
	// Timeout bounds one call. Zero means DefaultTimeout.
	Timeout time.Duration
	// run exists for tests. Nil means the real exec.
	run func(ctx context.Context, path string, args []string, stdin []byte) ([]byte, error)
}

// ErrDeferred is returned by Acquire when the controller admitted no route. It
// is deliberately a distinct error: the attempt is not wrong and not
// permanently excluded, it is only not admissible right now, so a caller may
// retry it after the next refresh instead of caching the verdict.
var ErrDeferred = errors.New("assignment deferred by routing hook")

// ErrAlreadyClosed is returned when the controller reports that this
// assignment has already run and been closed.
//
// It is a distinct error because it means something specific: this exact
// launch already happened. Re-running it under the same id would double-count
// work the controller has already accounted for, and the reply carries no
// route precisely so it cannot authorize one. A genuine relaunch is a new
// launch and takes its own assignment id.
var ErrAlreadyClosed = errors.New("assignment is already closed; a relaunch needs its own assignment id")

// ErrHookRefused is returned when the controller itself refused the request as
// malformed. It is the controller's own error object, not a transport failure.
var ErrHookRefused = errors.New("routing hook refused the request")

// Acquire asks the hook which of the allowed profiles may serve one
// invocation. allowed is already filtered to the profiles this role may use,
// so the hook is never offered a route the operator excluded for this duty.
//
// Only a `selected` reply naming an offered route authorizes a launch. Every
// other path is closed: a deferral, an already-closed assignment, the
// controller's own error object, an unknown or disallowed route id, malformed
// or oversized output, a nonzero exit, or a timeout all return an error and
// launch nothing. That is the point of the whole seam - a routing hook that
// misbehaves must be unable to start a process on a route the operator
// excluded.
func (h *Hook) Acquire(ctx context.Context, req Request, allowed []Profile) (Decision, error) {
	if err := validateAcquireRequest(req); err != nil {
		return Decision{}, err
	}
	if len(allowed) == 0 {
		return Decision{}, fmt.Errorf("routing: no approved profile is allowed for this role")
	}

	byID := make(map[string]Profile, len(allowed))
	ids := make([]string, 0, len(allowed))
	for _, profile := range allowed {
		byID[profile.ID] = profile
		ids = append(ids, profile.ID)
	}
	sort.Strings(ids)

	req.Routes = ids
	req.Outcome = ""
	req.Profile = nil
	resp, err := h.call(ctx, VerbAcquire, req)
	if err != nil {
		return Decision{}, err
	}

	switch resp.Result {
	case resultSelected:
	case resultDeferred:
		return Decision{
			Selected:   false,
			Reason:     resp.Reason,
			Generation: resp.Generation,
		}, fmt.Errorf("%w: %s", ErrDeferred, evidence(resp.Reason))
	case resultAlreadyClosed:
		// Deliberately not a selection, even though the controller knows which
		// route once served it: this launch already happened, and re-running it
		// under the same id would double-count it.
		return Decision{
			Selected:   false,
			Reason:     resp.Reason,
			Generation: resp.Generation,
		}, fmt.Errorf("%w: %s", ErrAlreadyClosed, evidence(resp.Reason))
	case resultError:
		return Decision{}, fmt.Errorf("%w: %s", ErrHookRefused, evidence(resp.Error))
	default:
		return Decision{}, fmt.Errorf("routing: hook returned unusable result %q for acquire", truncate(resp.Result, 64))
	}

	// The selection is resolved against the offered set only. A route the hook
	// invented, or one that belongs to a different role, never reaches a
	// launch: an id no-mistakes did not offer has no approved profile behind
	// it, so there is nothing to run even if the hook insists.
	profile, ok := byID[resp.RouteID]
	if !ok {
		return Decision{}, fmt.Errorf("routing: hook selected route %q, which is not among the %d approved routes offered",
			truncate(resp.RouteID, 64), len(ids))
	}

	return Decision{
		Profile:    profile,
		Selected:   true,
		Reason:     resp.Reason,
		Generation: resp.Generation,
	}, nil
}

// Finish reports the normalized outcome of one invocation. It is called on
// success, on failure, and on a launch that never started, so a controller
// counting in-flight work is never left holding an assignment that has already
// ended.
//
// A finish error is returned but is never fatal to the caller's own work: the
// turn it describes has already happened, and refusing to report it would only
// lose the record. Callers log it and continue.
func (h *Hook) Finish(ctx context.Context, req Request) error {
	if strings.TrimSpace(req.AssignmentID) == "" {
		return errors.New("routing: finish requires an assignment id")
	}
	if !req.Outcome.Valid() {
		return fmt.Errorf("routing: finish requires a normalized outcome (%s), got %q",
			outcomeNames(), truncate(string(req.Outcome), 64))
	}
	req.Routes = nil
	resp, err := h.call(ctx, VerbFinish, req)
	if err != nil {
		return err
	}
	switch resp.Result {
	case resultClosed:
		return nil
	case resultError:
		return fmt.Errorf("%w: %s", ErrHookRefused, evidence(resp.Error))
	default:
		return fmt.Errorf("routing: hook returned unusable result %q for finish", truncate(resp.Result, 64))
	}
}

func validateAcquireRequest(req Request) error {
	if strings.TrimSpace(req.AssignmentID) == "" {
		return errors.New("routing: acquire requires an assignment id")
	}
	if strings.TrimSpace(req.Owner.Identity) == "" {
		return errors.New("routing: acquire requires an owner identity")
	}
	if strings.TrimSpace(req.Owner.Generation) == "" {
		return errors.New("routing: acquire requires an owner generation")
	}
	return nil
}

func (h *Hook) call(ctx context.Context, verb Verb, req Request) (Response, error) {
	if strings.TrimSpace(h.Path) == "" {
		return Response{}, errors.New("routing: no assignment hook is configured")
	}
	payload, err := json.Marshal(req)
	if err != nil {
		return Response{}, fmt.Errorf("routing: encode %s request: %w", verb, err)
	}

	timeout := h.Timeout
	if timeout <= 0 {
		timeout = DefaultTimeout
	}
	callCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	// The verb is the subcommand, after the operator's own fixed arguments.
	args := make([]string, 0, len(h.Args)+1)
	args = append(args, h.Args...)
	args = append(args, string(verb))

	run := h.run
	if run == nil {
		run = execHook
	}
	stdout, runErr := run(callCtx, h.Path, args, payload)
	// A refusal is reported on stdout as an error object AND with a nonzero
	// exit, so the JSON is parsed first: the controller's own message is far
	// more useful than "exit status 1".
	if resp, ok := decodeResponse(stdout); ok && resp.Result == resultError {
		return resp, nil
	}
	// The reader breaks a flooding hook off past the cap, which surfaces as a
	// run error. Report that as the size refusal it is rather than as whatever
	// the broken pipe made the process exit with.
	if len(stdout) > maxResponseBytes {
		return Response{}, fmt.Errorf("routing: assignment hook %s returned more than %d bytes", verb, maxResponseBytes)
	}
	if runErr != nil {
		if errors.Is(callCtx.Err(), context.DeadlineExceeded) && ctx.Err() == nil {
			return Response{}, fmt.Errorf("routing: assignment hook %s timed out after %s", verb, timeout)
		}
		return Response{}, fmt.Errorf("routing: assignment hook %s failed: %w", verb, runErr)
	}

	resp, err := strictDecodeResponse(stdout)
	if err != nil {
		return Response{}, fmt.Errorf("routing: assignment hook %s returned unreadable output: %w", verb, err)
	}
	return resp, nil
}

// strictDecodeResponse reads exactly one JSON object and refuses anything else,
// including a trailing second object that a caller would never see.
func strictDecodeResponse(stdout []byte) (Response, error) {
	var resp Response
	decoder := json.NewDecoder(bytes.NewReader(stdout))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&resp); err != nil {
		return Response{}, err
	}
	if err := decoder.Decode(new(json.RawMessage)); err != io.EOF {
		return Response{}, errors.New("expected exactly one JSON object")
	}
	return resp, nil
}

// decodeResponse is the lenient read used only to surface a refusal that
// accompanied a nonzero exit. It never authorizes anything on its own.
func decodeResponse(stdout []byte) (Response, bool) {
	if len(stdout) == 0 || len(stdout) > maxResponseBytes {
		return Response{}, false
	}
	resp, err := strictDecodeResponse(stdout)
	if err != nil {
		return Response{}, false
	}
	return resp, true
}

// execHook runs the configured executable by argv with the request on stdin.
// There is no shell anywhere in this path: the operator's configured path,
// arguments and the verb are passed as separate argv entries, so nothing in
// the request can be interpreted as a command.
func execHook(ctx context.Context, path string, args []string, stdin []byte) ([]byte, error) {
	cmd := exec.CommandContext(ctx, path, args...)
	cmd.Stdin = bytes.NewReader(stdin)
	// Both streams are capped where the reading actually happens, not after.
	// Checking the length of an already-buffered response would enforce
	// nothing: a hook that streams gigabytes is resident in the daemon's heap
	// long before any check runs, and the timeout does not help because such a
	// process exits normally. One byte past the limit is kept so the caller can
	// still tell "too long" from "exactly at the limit".
	stdout := &cappedBuffer{limit: maxResponseBytes + 1, refuseOverflow: true}
	// stderr is bounded too, but it discards the excess instead of refusing it:
	// it is only ever quoted as diagnostic text, and breaking a hook's pipe for
	// being chatty would turn a successful call into a failure.
	stderr := &cappedBuffer{limit: 4 << 10}
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	if err := cmd.Run(); err != nil {
		if text := strings.TrimSpace(stderr.String()); text != "" {
			return stdout.Bytes(), fmt.Errorf("%w: %s", err, truncate(strings.Join(strings.Fields(text), " "), 400))
		}
		return stdout.Bytes(), err
	}
	return stdout.Bytes(), nil
}

// cappedBuffer collects at most limit bytes.
//
// With refuseOverflow set, the first byte past the limit is answered with a
// write error. That is deliberate: os/exec's copier stops on a write error and
// closes its end of the pipe, so a flooding hook is broken off at the source
// rather than drained politely into the daemon's heap. What was kept is still
// returned, so an oversized reply is reported as oversized rather than as a
// lost stream.
//
// Without it the excess is discarded silently, which is what a stream that is
// only ever quoted as diagnostic text needs: bounding memory there must not
// turn a chatty but successful call into a failed one.
type cappedBuffer struct {
	limit          int
	refuseOverflow bool
	buf            bytes.Buffer
}

var errCappedBufferFull = errors.New("routing: assignment hook wrote more output than the limit allows")

func (b *cappedBuffer) Write(p []byte) (int, error) {
	room := b.limit - b.buf.Len()
	if room <= 0 {
		if b.refuseOverflow {
			return 0, errCappedBufferFull
		}
		return len(p), nil
	}
	if len(p) <= room {
		return b.buf.Write(p)
	}
	if _, err := b.buf.Write(p[:room]); err != nil {
		return 0, err
	}
	if b.refuseOverflow {
		return room, errCappedBufferFull
	}
	return len(p), nil
}

func (b *cappedBuffer) Bytes() []byte { return b.buf.Bytes() }

func (b *cappedBuffer) String() string { return b.buf.String() }

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
