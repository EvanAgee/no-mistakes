package routing

import "context"

// NewTestHook builds a Hook whose transport is the given function instead of a
// subprocess, so another package can drive the assignment protocol without
// compiling and spawning an executable for every case.
//
// It exists because the transport field is deliberately unexported: nothing
// outside this package may substitute how the configured hook is executed, or
// the argv-only guarantee that keeps a request from ever being read as a shell
// token would be one assignment away from being bypassed. This constructor is
// the single, named exception, and it is only ever handed a function a test
// wrote.
//
// The protocol itself is unchanged: transport receives the verb exactly as it
// would arrive as the executable's subcommand argument, plus the same bounded
// JSON request the real executable would read on stdin, and returns the same
// bounded JSON response it would write on stdout. Every validation, refusal
// and bound in Acquire and Finish applies identically.
func NewTestHook(transport func(ctx context.Context, verb Verb, stdin []byte) ([]byte, error)) *Hook {
	return &Hook{
		Path: "test-hook",
		run: func(ctx context.Context, _ string, args []string, stdin []byte) ([]byte, error) {
			return transport(ctx, verbFromArgs(args), stdin)
		},
	}
}

// verbFromArgs reads the subcommand the caller appended, which is the last
// argument by construction (see Hook.call).
func verbFromArgs(args []string) Verb {
	if len(args) == 0 {
		return ""
	}
	return Verb(args[len(args)-1])
}
