from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

from llama_cpp import LlamaState


@dataclass(frozen=True)
class SessionStatus:
    initialized: bool
    turn_count: int
    used_tokens: int
    context_window: int


@dataclass
class SingleSessionManager:
    lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _base_state: LlamaState | None = field(default=None, init=False, repr=False)
    _current_state: LlamaState | None = field(default=None, init=False, repr=False)
    _turn_count: int = field(default=0, init=False, repr=False)

    @property
    def initialized(self) -> bool:
        return self._base_state is not None

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def active_state(self) -> LlamaState | None:
        return self._current_state or self._base_state

    def initialize(self, state: LlamaState) -> None:
        self._base_state = state
        self._current_state = state
        self._turn_count = 0

    def store_turn(self, state: LlamaState) -> None:
        if self._base_state is None:
            raise RuntimeError("Cannot store a turn before the session is initialized.")

        self._current_state = state
        self._turn_count += 1

    def reset(self) -> None:
        if self._base_state is None:
            self.clear()
            return

        self._current_state = self._base_state
        self._turn_count = 0

    def clear(self) -> None:
        self._base_state = None
        self._current_state = None
        self._turn_count = 0

    def status(self, context_window: int) -> SessionStatus:
        active_state = self.active_state
        used_tokens = active_state.n_tokens if active_state is not None else 0

        return SessionStatus(
            initialized=self.initialized,
            turn_count=self._turn_count,
            used_tokens=used_tokens,
            context_window=context_window,
        )
