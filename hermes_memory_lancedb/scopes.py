"""Multi-scope isolation for LanceDB memory.

Ported from memory-lancedb-pro TypeScript:
  - src/scopes.ts (538 LOC)
  - src/clawteam-scope.ts (63 LOC)

Scopes compose orthogonally: a single query can be filtered by any
combination of agent / user / project / team / workspace at once.
The legacy `user_id` filter is preserved when scope columns are absent.

Built-in scope patterns (string form, used as set members):
  - "global"
  - "agent:<agent_id>"
  - "user:<user_id>"
  - "project:<project_id>"
  - "team:<team_id>"
  - "workspace:<workspace_id>"
  - "custom:<name>"
  - "reflection:agent:<agent_id>"
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Reserved bypass identities
# ---------------------------------------------------------------------------

SYSTEM_BYPASS_IDS: Set[str] = {"system", "undefined"}


def is_system_bypass_id(agent_id: Optional[str]) -> bool:
    return isinstance(agent_id, str) and agent_id in SYSTEM_BYPASS_IDS


# ---------------------------------------------------------------------------
# Scope pattern helpers (mirror SCOPE_PATTERNS in scopes.ts)
# ---------------------------------------------------------------------------

GLOBAL_SCOPE = "global"


def agent_scope(agent_id: str) -> str:
    return f"agent:{agent_id}"


def user_scope(user_id: str) -> str:
    return f"user:{user_id}"


def project_scope(project_id: str) -> str:
    return f"project:{project_id}"


def team_scope(team_id: str) -> str:
    return f"team:{team_id}"


def workspace_scope(workspace_id: str) -> str:
    return f"workspace:{workspace_id}"


def custom_scope(name: str) -> str:
    return f"custom:{name}"


def reflection_scope(agent_id: str) -> str:
    return f"reflection:agent:{agent_id}"


_BUILTIN_PREFIXES = (
    "agent:",
    "user:",
    "project:",
    "team:",
    "workspace:",
    "custom:",
    "reflection:",
)


def _is_builtin_scope(scope: str) -> bool:
    if scope == GLOBAL_SCOPE:
        return True
    return any(scope.startswith(p) for p in _BUILTIN_PREFIXES)


_VALID_SCOPE_RE = re.compile(r"^[a-zA-Z0-9._:\-]+$")


def _is_valid_scope_format(scope: str) -> bool:
    if not scope or not isinstance(scope, str):
        return False
    trimmed = scope.strip()
    if not trimmed or len(trimmed) > 100:
        return False
    return bool(_VALID_SCOPE_RE.match(trimmed))


def parse_scope_id(scope: str) -> Optional[Tuple[str, str]]:
    """Return (type, id) for a scope, or None if malformed.

    Matches `parseScopeId` in scopes.ts. The "global" scope returns
    ("global", "").
    """
    if scope == GLOBAL_SCOPE:
        return ("global", "")
    idx = scope.find(":")
    if idx == -1:
        return None
    return (scope[:idx], scope[idx + 1 :])


def parse_agent_id_from_session_key(session_key: Optional[str]) -> Optional[str]:
    """Extract agentId from an OpenClaw session key.

    Supports formats:
      - "agent:main:discord:channel:123"
      - "agent:main"
    """
    if not session_key:
        return None
    sk = session_key.strip()
    if not sk.startswith("agent:"):
        return None
    rest = sk[len("agent:") :]
    colon = rest.find(":")
    candidate = (rest if colon == -1 else rest[:colon]).strip()
    if not candidate or is_system_bypass_id(candidate):
        return None
    return candidate


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ScopeDefinition:
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScopeConfig:
    default: str = GLOBAL_SCOPE
    definitions: Dict[str, ScopeDefinition] = field(default_factory=dict)
    agent_access: Dict[str, List[str]] = field(default_factory=dict)


def _default_scope_config() -> ScopeConfig:
    return ScopeConfig(
        default=GLOBAL_SCOPE,
        definitions={GLOBAL_SCOPE: ScopeDefinition(description="Shared knowledge across all agents")},
        agent_access={},
    )


def _normalize_agent_access(raw: Optional[Dict[str, List[str]]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    if not raw:
        return out
    for k, v in raw.items():
        if not isinstance(k, str):
            continue
        agent_id = k.strip()
        if not agent_id:
            continue
        if isinstance(v, list):
            out[agent_id] = list(v)
        else:
            out[agent_id] = []
    return out


# ---------------------------------------------------------------------------
# Scope manager
# ---------------------------------------------------------------------------


class ScopeManager:
    """Composable scope isolation for LanceDB rows.

    Two modes coexist:

    1. **Per-row scope membership** (TS-style): each row carries a `scope`
       string column (e.g. `agent:andrew`). The ScopeManager enumerates
       which scopes an agent can read.

    2. **Composable column filter** (Python-side, used by `_hybrid_search`):
       independent equality filters across `agent_id`, `user_id`,
       `project_id`, `team_id`, `workspace_id` columns. Empty (None)
       filters are skipped — the query still includes rows for which a
       given column is unset, preserving backwards compatibility with
       legacy data that has only `user_id` populated.
    """

    def __init__(self, config: Optional[ScopeConfig] = None):
        base = _default_scope_config()
        if config is not None:
            base.default = config.default or base.default
            base.definitions.update(config.definitions or {})
            base.agent_access.update(_normalize_agent_access(config.agent_access))
        self._config = base
        if GLOBAL_SCOPE not in self._config.definitions:
            self._config.definitions[GLOBAL_SCOPE] = ScopeDefinition(
                description="Shared knowledge across all agents"
            )
        self._validate_configuration()

    # ------------------------------------------------------------------
    # Configuration validation
    # ------------------------------------------------------------------

    def _validate_configuration(self) -> None:
        if self._config.default not in self._config.definitions:
            raise ValueError(
                f"Default scope '{self._config.default}' not found in definitions"
            )
        for agent_id, scopes in list(self._config.agent_access.items()):
            trimmed = agent_id.strip()
            if is_system_bypass_id(trimmed):
                raise ValueError(
                    f"Reserved bypass agent ID '{trimmed}' cannot have explicit access configured."
                )
            for scope in scopes:
                if scope not in self._config.definitions and not _is_builtin_scope(scope):
                    # Just a soft warning — TS does console.warn here. We swallow.
                    pass

    # ------------------------------------------------------------------
    # Public API — enumeration / accessibility
    # ------------------------------------------------------------------

    def get_all_scopes(self) -> List[str]:
        return list(self._config.definitions.keys())

    def get_scope_definition(self, scope: str) -> Optional[ScopeDefinition]:
        return self._config.definitions.get(scope)

    def validate_scope(self, scope: str) -> bool:
        if not isinstance(scope, str) or not scope.strip():
            return False
        s = scope.strip()
        return s in self._config.definitions or _is_builtin_scope(s)

    def get_default_scope(self, agent_id: Optional[str] = None) -> str:
        if not agent_id:
            return self._config.default
        if is_system_bypass_id(agent_id):
            raise ValueError(
                f"Reserved bypass agent ID '{agent_id}' must provide an explicit write scope."
            )
        agent = agent_scope(agent_id)
        accessible = self.get_accessible_scopes(agent_id)
        if agent in accessible:
            return agent
        return self._config.default

    def get_accessible_scopes(self, agent_id: Optional[str] = None) -> List[str]:
        if is_system_bypass_id(agent_id) or not agent_id:
            return self.get_all_scopes()
        normalized = agent_id.strip()
        explicit = self._config.agent_access.get(normalized)
        if explicit is not None:
            return _with_own_reflection(list(explicit), normalized)
        return _with_own_reflection([GLOBAL_SCOPE, agent_scope(normalized)], normalized)

    def get_scope_filter(self, agent_id: Optional[str] = None) -> Optional[List[str]]:
        """Return store-level scope filter (None == bypass)."""
        if not agent_id or is_system_bypass_id(agent_id):
            return None
        return self.get_accessible_scopes(agent_id)

    def is_accessible(self, scope: str, agent_id: Optional[str] = None) -> bool:
        if not agent_id or is_system_bypass_id(agent_id):
            return self.validate_scope(scope)
        return scope in self.get_accessible_scopes(agent_id)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_scope_definition(self, scope: str, definition: ScopeDefinition) -> None:
        if not _is_valid_scope_format(scope):
            raise ValueError(f"Invalid scope format: {scope}")
        self._config.definitions[scope] = definition

    def remove_scope_definition(self, scope: str) -> bool:
        if scope == GLOBAL_SCOPE:
            raise ValueError("Cannot remove global scope")
        if scope not in self._config.definitions:
            return False
        del self._config.definitions[scope]
        for agent_id, scopes in list(self._config.agent_access.items()):
            self._config.agent_access[agent_id] = [s for s in scopes if s != scope]
        return True

    def set_agent_access(self, agent_id: str, scopes: List[str]) -> None:
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ValueError("Invalid agent ID")
        normalized = agent_id.strip()
        if is_system_bypass_id(normalized):
            raise ValueError(
                f"Reserved bypass agent ID cannot have explicit access configured: {agent_id}"
            )
        for scope in scopes:
            if not self.validate_scope(scope):
                raise ValueError(f"Invalid scope: {scope}")
        self._config.agent_access[normalized] = list(scopes)

    def remove_agent_access(self, agent_id: str) -> bool:
        normalized = agent_id.strip()
        if normalized not in self._config.agent_access:
            return False
        del self._config.agent_access[normalized]
        return True

    # ------------------------------------------------------------------
    # ClawTeam shared scopes (port of clawteam-scope.ts)
    # ------------------------------------------------------------------

    def apply_clawteam_scopes(self, scopes: List[str]) -> None:
        """Register team scopes and grant them to every agent.

        Mirrors `applyClawteamScopes` in clawteam-scope.ts. Wraps
        `get_accessible_scopes` so all callers (including
        `get_scope_filter` and `is_accessible`) see the extra scopes.
        """
        cleaned = [s.strip() for s in scopes if isinstance(s, str) and s.strip()]
        if not cleaned:
            return
        for scope in cleaned:
            if scope not in self._config.definitions:
                self.add_scope_definition(
                    scope,
                    ScopeDefinition(description=f"ClawTeam shared scope: {scope}"),
                )
        original = self.get_accessible_scopes

        def _wrapped(agent_id: Optional[str] = None) -> List[str]:
            base = original(agent_id)
            for s in cleaned:
                if s not in base:
                    base.append(s)
            return base

        self.get_accessible_scopes = _wrapped  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Composable column filter — used by LanceDBMemoryProvider._hybrid_search
    # ------------------------------------------------------------------

    def build_where_clause(
        self,
        *,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        team_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        scope_columns_present: bool = True,
        legacy_user_id: Optional[str] = None,
    ) -> str:
        """Build a SQL `WHERE` predicate composing all non-empty scope filters.

        - When `scope_columns_present=False` (legacy table), only the
          `legacy_user_id` filter is emitted, preserving prior behaviour.
        - When the table HAS the new scope columns, each provided id
          becomes a clause `<col> = '<id>'`. Empty (None) filters are
          skipped — the query is permissive for those dimensions.
        - If `legacy_user_id` is provided alongside scope columns, it is
          OR-combined with `user_id` so existing rows with no `user_id`
          column-style migration still match (covers the migration grace
          period where rows may have an empty scope column).
        """
        parts: List[str] = []
        if not scope_columns_present:
            uid = legacy_user_id or user_id
            if uid:
                parts.append(f"user_id = '{_escape_sql(uid)}'")
            return " AND ".join(parts)

        # Map of column -> value for the composable filter.
        column_filters: List[Tuple[str, str]] = []
        if agent_id:
            column_filters.append(("agent_id", agent_id))
        if user_id:
            column_filters.append(("user_id", user_id))
        if project_id:
            column_filters.append(("project_id", project_id))
        if team_id:
            column_filters.append(("team_id", team_id))
        if workspace_id:
            column_filters.append(("workspace_id", workspace_id))

        for col, val in column_filters:
            parts.append(f"{col} = '{_escape_sql(val)}'")

        return " AND ".join(parts)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        scopes = self.get_all_scopes()
        by_type = {
            "global": 0,
            "agent": 0,
            "custom": 0,
            "project": 0,
            "user": 0,
            "team": 0,
            "workspace": 0,
            "reflection": 0,
            "other": 0,
        }
        for scope in scopes:
            if scope == GLOBAL_SCOPE:
                by_type["global"] += 1
            elif scope.startswith("agent:"):
                by_type["agent"] += 1
            elif scope.startswith("custom:"):
                by_type["custom"] += 1
            elif scope.startswith("project:"):
                by_type["project"] += 1
            elif scope.startswith("user:"):
                by_type["user"] += 1
            elif scope.startswith("team:"):
                by_type["team"] += 1
            elif scope.startswith("workspace:"):
                by_type["workspace"] += 1
            elif scope.startswith("reflection:"):
                by_type["reflection"] += 1
            else:
                by_type["other"] += 1
        return {
            "totalScopes": len(scopes),
            "agentsWithCustomAccess": len(self._config.agent_access),
            "scopesByType": by_type,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _with_own_reflection(scopes: List[str], agent_id: str) -> List[str]:
    refl = reflection_scope(agent_id)
    if refl in scopes:
        return list(scopes)
    return list(scopes) + [refl]


def _escape_sql(value: str) -> str:
    """Escape a string literal for use in a SQL `WHERE` predicate.

    LanceDB's SQL parser accepts standard single-quote escaping (double
    the quote). We also strip null bytes defensively.
    """
    return value.replace("\x00", "").replace("'", "''")


# ---------------------------------------------------------------------------
# ClawTeam env-var helpers
# ---------------------------------------------------------------------------


def parse_clawteam_scopes(env_value: Optional[str]) -> List[str]:
    if not env_value:
        return []
    return [s.strip() for s in env_value.split(",") if s.strip()]


def clawteam_scopes_from_env() -> List[str]:
    return parse_clawteam_scopes(os.environ.get("CLAWTEAM_MEMORY_SCOPE"))


# ---------------------------------------------------------------------------
# Schema migration helpers
# ---------------------------------------------------------------------------

# New scope columns (all nullable strings) added by P1.
SCOPE_COLUMNS: Tuple[str, ...] = (
    "agent_id",
    "project_id",
    "team_id",
    "workspace_id",
    "scope",  # canonical scope string e.g. "agent:andrew"
)

# Default-value SQL fragments used by LanceDB's `add_columns` API.
SCOPE_COLUMN_DEFAULTS: Dict[str, str] = {
    "agent_id": "''",
    "project_id": "''",
    "team_id": "''",
    "workspace_id": "''",
    "scope": "''",
}


__all__ = [
    "GLOBAL_SCOPE",
    "ScopeConfig",
    "ScopeDefinition",
    "ScopeManager",
    "SCOPE_COLUMNS",
    "SCOPE_COLUMN_DEFAULTS",
    "SYSTEM_BYPASS_IDS",
    "agent_scope",
    "clawteam_scopes_from_env",
    "custom_scope",
    "is_system_bypass_id",
    "parse_agent_id_from_session_key",
    "parse_clawteam_scopes",
    "parse_scope_id",
    "project_scope",
    "reflection_scope",
    "team_scope",
    "user_scope",
    "workspace_scope",
]
