package routing

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/kunchenguid/no-mistakes/internal/agentcfg"
	"github.com/kunchenguid/no-mistakes/internal/types"
)

// These tests drive the real transport: a compiled executable, launched by
// argv, reading a JSON request on stdin and writing a JSON response on stdout.
// The scripted tests elsewhere in this package pin the protocol's semantics;
// these pin that the wire actually behaves that way against a live process,
// including several at once.

// buildHookFixture compiles a small program that behaves like the assignment
// controller. It is compiled once per test binary, because compiling per test
// dominates the runtime of everything here.
var (
	hookFixtureOnce sync.Once
	hookFixturePath string
	hookFixtureErr  error
)

const hookFixtureSource = `package main

// A minimal stand-in for the assignment controller: it reads one bounded JSON
// request on stdin and writes one bounded JSON response on stdout, balancing
// acquires across the offered routes by round-robin through a shared counter
// file. It is deliberately independent of the package under test, so the test
// proves the wire contract rather than a shared struct definition.

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

type request struct {
	Verb       string   ` + "`json:\"verb\"`" + `
	Assignment string   ` + "`json:\"assignment\"`" + `
	Owner      string   ` + "`json:\"owner\"`" + `
	Generation string   ` + "`json:\"generation\"`" + `
	Role       string   ` + "`json:\"role\"`" + `
	Routes     []string ` + "`json:\"routes\"`" + `
	Outcome    string   ` + "`json:\"outcome\"`" + `
	Profile    string   ` + "`json:\"profile\"`" + `
}

func main() {
	var req request
	if err := json.NewDecoder(os.Stdin).Decode(&req); err != nil {
		fmt.Fprintln(os.Stderr, "unreadable request:", err)
		os.Exit(2)
	}

	mode := os.Getenv("FIXTURE_MODE")
	switch mode {
	case "garbage":
		fmt.Print("selected route=", strings.Join(req.Routes, ","))
		return
	case "invent-route":
		emit(map[string]any{"status": "selected", "route": "a-route-nobody-offered"})
		return
	case "hang":
		time.Sleep(time.Minute)
		return
	case "defer":
		emit(map[string]any{"status": "deferred", "reason": "no route proven eligible"})
		return
	}

	if req.Verb == "finish" {
		appendLine(os.Getenv("FIXTURE_LOG"), "finish "+req.Assignment+" "+req.Outcome+" "+req.Profile)
		emit(map[string]any{"status": "closed"})
		return
	}

	if req.Owner == "" || req.Generation == "" || req.Assignment == "" {
		fmt.Fprintln(os.Stderr, "incomplete assignment identity")
		os.Exit(3)
	}
	if len(req.Routes) == 0 {
		fmt.Fprintln(os.Stderr, "no routes offered")
		os.Exit(4)
	}

	// Round-robin across the offered routes through a counter file, so
	// concurrent callers genuinely contend for the same shared record.
	n := bumpCounter(os.Getenv("FIXTURE_COUNTER"))
	route := req.Routes[n%len(req.Routes)]
	appendLine(os.Getenv("FIXTURE_LOG"), "acquire "+req.Assignment+" "+req.Owner+" "+req.Generation+" "+route)
	emit(map[string]any{
		"status":      "selected",
		"route":       route,
		"reason":      "fewest-pending",
		"generation":  "obs-" + strconv.Itoa(n),
		"observed_at": time.Now().UTC().Format(time.RFC3339),
	})
}

func emit(payload map[string]any) {
	if err := json.NewEncoder(os.Stdout).Encode(payload); err != nil {
		os.Exit(5)
	}
}

// bumpCounter increments a shared counter under an O_EXCL lock directory, so
// two concurrent processes cannot read the same value.
func bumpCounter(path string) int {
	if path == "" {
		return 0
	}
	lock := path + ".lock"
	for {
		if err := os.Mkdir(lock, 0o700); err == nil {
			break
		}
		time.Sleep(time.Millisecond)
	}
	defer os.Remove(lock)

	raw, _ := os.ReadFile(path)
	n, _ := strconv.Atoi(strings.TrimSpace(string(raw)))
	_ = os.WriteFile(path, []byte(strconv.Itoa(n+1)), 0o600)
	return n
}

func appendLine(path, line string) {
	if path == "" {
		return
	}
	lock := path + ".lock"
	for {
		if err := os.Mkdir(lock, 0o700); err == nil {
			break
		}
		time.Sleep(time.Millisecond)
	}
	defer os.Remove(lock)

	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		return
	}
	defer f.Close()
	fmt.Fprintln(f, line)
}
`

func hookFixture(t *testing.T) string {
	t.Helper()
	hookFixtureOnce.Do(func() {
		dir, err := os.MkdirTemp("", "routing-hook-fixture")
		if err != nil {
			hookFixtureErr = err
			return
		}
		src := filepath.Join(dir, "main.go")
		if err := os.WriteFile(src, []byte(hookFixtureSource), 0o600); err != nil {
			hookFixtureErr = err
			return
		}
		if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module hookfixture\n\ngo 1.25\n"), 0o600); err != nil {
			hookFixtureErr = err
			return
		}
		bin := filepath.Join(dir, "hookfixture")
		if runtime.GOOS == "windows" {
			bin += ".exe"
		}
		// Built without -race deliberately: this is a separate process the
		// test only talks to over a pipe, and instrumenting it would multiply
		// each spawn cost for no added coverage.
		build := exec.Command("go", "build", "-o", bin, ".")
		build.Dir = dir
		if out, err := build.CombinedOutput(); err != nil {
			hookFixtureErr = errors.New("build hook fixture: " + err.Error() + ": " + string(out))
			return
		}
		hookFixturePath = bin
	})
	if hookFixtureErr != nil {
		t.Skipf("hook fixture unavailable: %v", hookFixtureErr)
	}
	return hookFixturePath
}

func processHook(t *testing.T, env map[string]string) *Hook {
	t.Helper()
	path := hookFixture(t)
	for key, value := range env {
		t.Setenv(key, value)
	}
	return &Hook{Path: path, Timeout: 20 * time.Second}
}

// TestHookProcess_RealExecutableSpeaksTheBoundedProtocol proves the wire works
// end to end against a live process: argv launch, JSON request on stdin, JSON
// response on stdout, and a selection that resolves to the operator's own
// configured profile.
func TestHookProcess_RealExecutableSpeaksTheBoundedProtocol(t *testing.T) {
	dir := t.TempDir()
	logPath := filepath.Join(dir, "log")
	hook := processHook(t, map[string]string{
		"FIXTURE_LOG":     logPath,
		"FIXTURE_COUNTER": filepath.Join(dir, "counter"),
	})

	allowed := []Profile{claudeProfile(), piGrokProfile()}
	decision, err := hook.Acquire(context.Background(), acquireRequest(), allowed)
	if err != nil {
		t.Fatalf("acquire against a real process: %v", err)
	}
	if !decision.Selected {
		t.Fatal("the fixture selects, so the decision must be selected")
	}
	if decision.Profile.ID != "claude-opus" && decision.Profile.ID != "pi-grok" {
		t.Fatalf("selection %q must be one of the offered routes", decision.Profile.ID)
	}
	if !decision.Fresh {
		t.Fatal("the fixture dates its observation, so the decision must read as fresh")
	}

	if err := hook.Finish(context.Background(), Request{
		Assignment: "run-1:1", Owner: "daemon", Generation: "gen-1",
		Outcome: OutcomeSuccess, Profile: decision.Profile.ID,
	}); err != nil {
		t.Fatalf("finish against a real process: %v", err)
	}

	logged := readFixtureLog(t, logPath)
	if !strings.Contains(logged, "acquire run-1:1 daemon gen-1 ") {
		t.Fatalf("the process must have received the full assignment identity, log:\n%s", logged)
	}
	if !strings.Contains(logged, "finish run-1:1 success "+decision.Profile.ID) {
		t.Fatalf("the process must have received the finish with its route, log:\n%s", logged)
	}
}

// TestHookProcess_ConcurrentAcquiresSpreadAcrossEligibleRoutes drives several
// real concurrent processes against one shared record and proves the
// assignments distribute rather than all landing on one route. It also proves
// the caller is safe under genuine concurrency: every returned selection is an
// approved profile, and no two assignments collide.
func TestHookProcess_ConcurrentAcquiresSpreadAcrossEligibleRoutes(t *testing.T) {
	dir := t.TempDir()
	hook := processHook(t, map[string]string{
		"FIXTURE_LOG":     filepath.Join(dir, "log"),
		"FIXTURE_COUNTER": filepath.Join(dir, "counter"),
	})

	allowed := []Profile{claudeProfile(), codexProfile(), piGrokProfile(), piDeepSeekProfile()}
	const attempts = 12

	type outcome struct {
		route string
		err   error
	}
	results := make([]outcome, attempts)
	var wg sync.WaitGroup
	for i := range attempts {
		wg.Add(1)
		go func() {
			defer wg.Done()
			req := acquireRequest()
			req.Assignment = "run-1:" + strconv.Itoa(i)
			decision, err := hook.Acquire(context.Background(), req, allowed)
			results[i] = outcome{route: decision.Profile.ID, err: err}
		}()
	}
	wg.Wait()

	seen := map[string]int{}
	approved := map[string]bool{}
	for _, p := range allowed {
		approved[p.ID] = true
	}
	for i, r := range results {
		if r.err != nil {
			t.Fatalf("attempt %d: %v", i, r.err)
		}
		if !approved[r.route] {
			t.Fatalf("attempt %d selected %q, which is not an approved route", i, r.route)
		}
		seen[r.route]++
	}

	if len(seen) < 2 {
		t.Fatalf("concurrent assignments must spread across eligible routes, all landed on %v", seen)
	}
	for _, p := range allowed {
		if seen[p.ID] == 0 {
			t.Fatalf("route %q received no assignment; distribution was %v", p.ID, seen)
		}
	}
}

// TestHookProcess_MisbehavingProcessNeverLaunchesAnything proves the refusals
// hold against a real process, not only against a scripted transport: text
// output, an invented route, and a hang each end in an error and no selection.
func TestHookProcess_MisbehavingProcessNeverLaunchesAnything(t *testing.T) {
	cases := []struct {
		name    string
		mode    string
		timeout time.Duration
		wantMsg string
	}{
		{name: "text output instead of JSON", mode: "garbage", wantMsg: "unreadable output"},
		{name: "route nobody offered", mode: "invent-route", wantMsg: "not among the"},
		{name: "process that never answers", mode: "hang", timeout: 200 * time.Millisecond, wantMsg: "timed out"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			hook := processHook(t, map[string]string{"FIXTURE_MODE": tc.mode})
			if tc.timeout > 0 {
				hook.Timeout = tc.timeout
			}
			decision, err := hook.Acquire(context.Background(), acquireRequest(), []Profile{claudeProfile(), piGrokProfile()})
			if err == nil {
				t.Fatalf("want a refusal, got %+v", decision)
			}
			if decision.Selected {
				t.Fatalf("a refused acquire must not report a selection, got %+v", decision)
			}
			if !strings.Contains(err.Error(), tc.wantMsg) {
				t.Fatalf("error %q must mention %q", err, tc.wantMsg)
			}
		})
	}
}

// TestHookProcess_DeferralFromARealProcessIsRetryable proves a deferred
// verdict crosses the wire as ErrDeferred, so the caller can distinguish it
// from a fault and retry the work after the next refresh.
func TestHookProcess_DeferralFromARealProcessIsRetryable(t *testing.T) {
	hook := processHook(t, map[string]string{"FIXTURE_MODE": "defer"})
	_, err := hook.Acquire(context.Background(), acquireRequest(), []Profile{piGrokProfile()})
	if !errors.Is(err, ErrDeferred) {
		t.Fatalf("a real process's deferral must arrive as ErrDeferred, got %v", err)
	}
}

// TestHookProcess_RequestIsNeverAShellToken proves the request travels on
// stdin and never through a shell. A route id shaped like a shell command
// substitution reaches the process as literal data, and nothing it names is
// executed.
func TestHookProcess_RequestIsNeverAShellToken(t *testing.T) {
	dir := t.TempDir()
	marker := filepath.Join(dir, "executed")
	logPath := filepath.Join(dir, "log")
	hook := processHook(t, map[string]string{
		"FIXTURE_LOG":     logPath,
		"FIXTURE_COUNTER": filepath.Join(dir, "counter"),
	})

	hostile := piGrokProfile()
	hostile.ID = "$(touch " + marker + ")"

	req := acquireRequest()
	req.Assignment = "run-1:`touch " + marker + "`"

	decision, err := hook.Acquire(context.Background(), req, []Profile{hostile})
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	if decision.Profile.ID != hostile.ID {
		t.Fatalf("the id must round-trip as literal data, got %q", decision.Profile.ID)
	}
	if _, err := os.Stat(marker); !os.IsNotExist(err) {
		t.Fatal("no part of the request may be evaluated by a shell")
	}

	logged := readFixtureLog(t, logPath)
	if !strings.Contains(logged, hostile.ID) {
		t.Fatalf("the process must have received the id verbatim, log:\n%s", logged)
	}
}

func readFixtureLog(t *testing.T, path string) string {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture log: %v", err)
	}
	return string(raw)
}

func codexProfile() Profile {
	return Profile{
		ID:       "codex-sol",
		Agent:    types.AgentCodex,
		Provider: "openai-subscription",
		Tuning:   agentcfg.Profile{Model: "gpt-5.6-sol", Effort: agentcfg.EffortHigh},
	}
}
