package db

import (
	"database/sql"
	"fmt"
)

// RunAgentSession is the minimum session-resume metadata for one durable
// per-run, per-role agent session. Production resumes only the review-fixer
// role; legacy reviewer rows remain readable for crash recovery but are never
// resumed. Only the adapter-native session identity is stored - never prompts,
// transcripts, or any conversation content - so the review loop can resume
// its fixer session across parking and daemon process boundaries.
type RunAgentSession struct {
	RunID     string
	Role      string
	Agent     string
	SessionID string
	// ProfileKey is the execution profile the session was minted under (see
	// internal/pipeline/profilekey.go). It is empty for rows written before
	// routing existed and for rows whose effective identity could not be
	// established; such a row is readable but never resumed, because a native
	// session id belongs to the exact adapter, provider, model and billing
	// route that created it, and the adapter name alone does not prove those.
	ProfileKey string
	CreatedAt  int64
	UpdatedAt  int64
}

// UpsertRunAgentSession stores or replaces the session identity for a
// run+role. A run has at most one session per role. profileKey may be empty,
// which persists as SQL NULL and marks the row unresumable.
func (d *DB) UpsertRunAgentSession(runID, role, agent, sessionID, profileKey string) error {
	ts := now()
	_, err := d.sql.Exec(
		`INSERT INTO run_agent_sessions (run_id, role, agent, session_id, profile_key, created_at, updated_at)
		 VALUES (?, ?, ?, ?, ?, ?, ?)
		 ON CONFLICT(run_id, role) DO UPDATE SET agent = excluded.agent, session_id = excluded.session_id, profile_key = excluded.profile_key, updated_at = excluded.updated_at`,
		runID, role, agent, sessionID, nullableText(profileKey), ts, ts,
	)
	if err != nil {
		return fmt.Errorf("upsert run agent session: %w", err)
	}
	return nil
}

// GetRunAgentSessions returns all stored session identities for a run.
func (d *DB) GetRunAgentSessions(runID string) ([]RunAgentSession, error) {
	rows, err := d.sql.Query(
		`SELECT run_id, role, agent, session_id, profile_key, created_at, updated_at FROM run_agent_sessions WHERE run_id = ?`,
		runID,
	)
	if err != nil {
		return nil, fmt.Errorf("get run agent sessions: %w", err)
	}
	defer rows.Close()

	var sessions []RunAgentSession
	for rows.Next() {
		var s RunAgentSession
		var profileKey sql.NullString
		if err := rows.Scan(&s.RunID, &s.Role, &s.Agent, &s.SessionID, &profileKey, &s.CreatedAt, &s.UpdatedAt); err != nil {
			return nil, fmt.Errorf("scan run agent session: %w", err)
		}
		s.ProfileKey = profileKey.String
		sessions = append(sessions, s)
	}
	return sessions, rows.Err()
}

// DeleteRunAgentSession drops one role's session identity so the next turn
// starts a fresh same-role session (used after a failed resume).
func (d *DB) DeleteRunAgentSession(runID, role string) error {
	_, err := d.sql.Exec(`DELETE FROM run_agent_sessions WHERE run_id = ? AND role = ?`, runID, role)
	if err != nil {
		return fmt.Errorf("delete run agent session: %w", err)
	}
	return nil
}

// nullableText stores an empty string as SQL NULL, so "no key recorded" and
// "the empty key" cannot be confused by any later reader.
func nullableText(value string) any {
	if value == "" {
		return nil
	}
	return value
}
