package db

import (
	"path/filepath"
	"testing"
)

// TestRunAgentSessions_ProfileKeyRoundTrips proves the profile key is stored
// and read back beside the session id, so a later process can decide reuse
// from the same evidence the minting process recorded.
func TestRunAgentSessions_ProfileKeyRoundTrips(t *testing.T) {
	d, _, run := openSessionTestDB(t)

	const key = "v1:44e6f0f1c2a34b9d"
	if err := d.UpsertRunAgentSession(run.ID, "review-fixer", "pi", "sess-1", key); err != nil {
		t.Fatalf("upsert: %v", err)
	}

	sessions, err := d.GetRunAgentSessions(run.ID)
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if len(sessions) != 1 {
		t.Fatalf("want 1 session, got %d", len(sessions))
	}
	if sessions[0].SessionID != "sess-1" || sessions[0].ProfileKey != key {
		t.Fatalf("session = %+v, want sess-1 with its key", sessions[0])
	}
}

// TestRunAgentSessions_EmptyProfileKeyIsStoredAsNull proves "no key recorded"
// and "the empty key" cannot be confused. A legacy row and a row whose
// identity could not be established both read back as empty, which is what
// makes them unresumable by a routed turn.
func TestRunAgentSessions_EmptyProfileKeyIsStoredAsNull(t *testing.T) {
	d, _, run := openSessionTestDB(t)

	if err := d.UpsertRunAgentSession(run.ID, "review-fixer", "claude", "sess-1", ""); err != nil {
		t.Fatalf("upsert: %v", err)
	}

	var isNull bool
	row := d.sql.QueryRow(
		`SELECT profile_key IS NULL FROM run_agent_sessions WHERE run_id = ? AND role = ?`,
		run.ID, "review-fixer")
	if err := row.Scan(&isNull); err != nil {
		t.Fatalf("scan: %v", err)
	}
	if !isNull {
		t.Fatal("an empty profile key must be stored as SQL NULL")
	}

	sessions, err := d.GetRunAgentSessions(run.ID)
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if sessions[0].ProfileKey != "" {
		t.Fatalf("a NULL key must read back empty, got %q", sessions[0].ProfileKey)
	}
}

// TestRunAgentSessions_UpsertReplacesIDAndKeyTogether proves the two fields
// move as one. A row left holding a new session id beside a stale key would be
// resumed by a profile that never minted it, which is exactly the confusion
// the key exists to prevent.
func TestRunAgentSessions_UpsertReplacesIDAndKeyTogether(t *testing.T) {
	d, _, run := openSessionTestDB(t)

	if err := d.UpsertRunAgentSession(run.ID, "review-fixer", "pi", "grok-sess", "v1:grokkey"); err != nil {
		t.Fatalf("first upsert: %v", err)
	}
	// The same adapter, a different billing route: only the key can tell them
	// apart, so the key must be replaced along with the id.
	if err := d.UpsertRunAgentSession(run.ID, "review-fixer", "pi", "deepseek-sess", "v1:deepseekkey"); err != nil {
		t.Fatalf("second upsert: %v", err)
	}

	sessions, err := d.GetRunAgentSessions(run.ID)
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if len(sessions) != 1 {
		t.Fatalf("one role must keep one row, got %d", len(sessions))
	}
	if sessions[0].SessionID != "deepseek-sess" || sessions[0].ProfileKey != "v1:deepseekkey" {
		t.Fatalf("session = %+v, want the replacement id and key together", sessions[0])
	}
}

// TestRunAgentSessions_ClearingTheKeyIsPersisted proves a row can move from
// keyed back to key-less. It happens when a later turn's effective identity
// cannot be established: recording the new id with a stale key would be worse
// than recording it as unknown.
func TestRunAgentSessions_ClearingTheKeyIsPersisted(t *testing.T) {
	d, _, run := openSessionTestDB(t)

	if err := d.UpsertRunAgentSession(run.ID, "review-fixer", "pi", "sess-1", "v1:known"); err != nil {
		t.Fatalf("first upsert: %v", err)
	}
	if err := d.UpsertRunAgentSession(run.ID, "review-fixer", "pi", "sess-2", ""); err != nil {
		t.Fatalf("second upsert: %v", err)
	}

	sessions, err := d.GetRunAgentSessions(run.ID)
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if sessions[0].ProfileKey != "" {
		t.Fatalf("the key must be cleared, got %q", sessions[0].ProfileKey)
	}
}

// TestOpenMigratesRunAgentSessionProfileKey proves the additive migration
// reaches a database created before the column existed. A pre-existing row
// survives with a NULL key, so an upgrade never loses a parked run's session
// record and never invents an identity for it.
func TestOpenMigratesRunAgentSessionProfileKey(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.sqlite")

	d, err := Open(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	repo, err := d.InsertRepo("/tmp/repo", "https://github.com/test/repo", "main")
	if err != nil {
		t.Fatalf("insert repo: %v", err)
	}
	run, err := d.InsertRun(repo.ID, "feature/x", "head", "base")
	if err != nil {
		t.Fatalf("insert run: %v", err)
	}

	// Simulate the pre-migration shape: drop the column back off, leaving a
	// row exactly as an older build would have written it.
	if _, err := d.sql.Exec(`ALTER TABLE run_agent_sessions DROP COLUMN profile_key`); err != nil {
		t.Fatalf("drop column: %v", err)
	}
	if _, err := d.sql.Exec(
		`INSERT INTO run_agent_sessions (run_id, role, agent, session_id, created_at, updated_at)
		 VALUES (?, ?, ?, ?, ?, ?)`,
		run.ID, "review-fixer", "claude", "pre-upgrade-session", 1, 1); err != nil {
		t.Fatalf("insert legacy row: %v", err)
	}
	if err := d.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	// Reopening applies the migration.
	reopened, err := Open(path)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	t.Cleanup(func() { reopened.Close() })

	sessions, err := reopened.GetRunAgentSessions(run.ID)
	if err != nil {
		t.Fatalf("get after migration: %v", err)
	}
	if len(sessions) != 1 {
		t.Fatalf("the pre-existing row must survive, got %d rows", len(sessions))
	}
	if sessions[0].SessionID != "pre-upgrade-session" {
		t.Fatalf("session id = %q, want the pre-upgrade id", sessions[0].SessionID)
	}
	if sessions[0].ProfileKey != "" {
		t.Fatalf("a migrated row must carry no key, got %q", sessions[0].ProfileKey)
	}

	// And the column is now writable.
	if err := reopened.UpsertRunAgentSession(run.ID, "review-fixer", "claude", "sess-2", "v1:key"); err != nil {
		t.Fatalf("upsert after migration: %v", err)
	}
}
