# State Machine Design — Yamada Bee Farm Multi-Agent Voice System

This document covers the current state machine implementation and three proposed production-level alternatives. Each section includes a Mermaid diagram, data structure, and explanation of how it works within the LiveKit multi-agent workflow.

---

## Table of Contents

1. [Current Implementation — Flat Sequential Task List](#1-current-implementation--flat-sequential-task-list)
2. [Option 1 — Hierarchical Task Trees](#2-option-1--hierarchical-task-trees)
3. [Option 2 — Finite State Machine with Transitions (Recommended)](#3-option-2--finite-state-machine-with-transitions-recommended)
4. [Option 3 — Event-Driven Orchestrator](#4-option-3--event-driven-orchestrator)
5. [Comparison Matrix](#5-comparison-matrix)

---

## 1. Current Implementation — Flat Sequential Task List

### How It Works

Each agent has a flat ordered list of tasks stored in `session.userdata["state_machine"]`. Tasks are completed strictly in sequence — the `advance_task()` function only marks a task as completed if it is the **current task**. The next pending task becomes current automatically.

Two completion mechanisms exist:
- **Tool-based tasks**: Auto-completed when the associated tool is called (e.g., `get_session_context` auto-advances `load_context`).
- **Conversational tasks**: The LLM must explicitly call `complete_task(agent_name, task_id)` to mark them done.

The `format_state_machine()` function injects a checklist into every LLM call as a system message, showing `✓` (done), `→` (current), and `○` (pending).

### Data Structure

```python
"subscription": {
    "tasks": [
        {"id": "load_context",        "label": "Load session context",    "status": "pending"},
        {"id": "handle_request",      "label": "Handle request",          "status": "pending"},
        {"id": "record_outcome",      "label": "Record the final outcome","status": "pending"},
        {"id": "check_anything_else", "label": "Ask if anything else",    "status": "pending"},
        {"id": "transfer_out",        "label": "Transfer to finishing",   "status": "pending"},
    ],
    "current_task": "load_context",
}
```

### Mermaid Diagram — Current System (Subscription Agent Example)

```mermaid
stateDiagram-v2
    direction TB

    [*] --> load_context
    load_context --> handle_request : auto (get_session_context tool)
    handle_request --> record_outcome : LLM calls complete_task
    record_outcome --> check_anything_else : auto (log_agent_outcome tool)
    check_anything_else --> transfer_out : LLM calls complete_task
    transfer_out --> [*] : auto (transfer_to_* tool)

    note right of load_context
        Tool-based: auto-advances
        when get_session_context is called
    end note

    note right of handle_request
        Conversational: LLM must call
        complete_task("subscription", "handle_request")
    end note
```

### Mermaid Diagram — Current System (Full Multi-Agent Flow)

```mermaid
stateDiagram-v2
    direction TB

    state "Reception Agent" as reception {
        [*] --> r_greet
        r_greet --> r_verify : complete_task
        r_verify --> r_intent : auto (get_customer_info)
        r_intent --> r_fetch_order : complete_task
        r_fetch_order --> r_fetch_sub : auto (get_order_info)
        r_fetch_sub --> r_prepare : auto (get_regular_info)
        r_prepare --> r_transfer : auto (prepare_handoff_context)
        r_transfer --> [*] : auto (transfer_to_*)
    }

    state "Subscription Agent" as subscription {
        [*] --> s_load
        s_load --> s_handle : auto (get_session_context)
        s_handle --> s_record : complete_task
        s_record --> s_check : auto (log_agent_outcome)
        s_check --> s_transfer : complete_task
        s_transfer --> [*] : auto (transfer_to_*)
    }

    state "Sales Agent" as sales {
        [*] --> sa_load
        sa_load --> sa_present : auto (get_session_context)
        sa_present --> sa_close : complete_task
        sa_close --> sa_log : complete_task
        sa_log --> sa_transfer : auto (log_agent_outcome)
        sa_transfer --> [*] : auto (transfer_to_*)
    }

    state "Support Agent" as support {
        [*] --> su_load
        su_load --> su_diagnose : auto (get_session_context)
        su_diagnose --> su_resolve : complete_task
        su_resolve --> su_log : complete_task
        su_log --> su_transfer : auto (log_agent_outcome)
        su_transfer --> [*] : auto (transfer_to_*)
    }

    state "Decease Handling Agent" as decease {
        [*] --> d_load
        d_load --> d_condolences : auto (get_session_context)
        d_condolences --> d_caller : complete_task
        d_caller --> d_deceased : complete_task
        d_deceased --> d_explain : complete_task
        d_explain --> d_unpaid : complete_task
        d_unpaid --> d_register : complete_task
        d_register --> d_transfer : auto (collect_decease_information)
        d_transfer --> [*] : auto (transfer_to_*)
    }

    state "Finishing Agent" as finishing {
        [*] --> f_load
        f_load --> f_collate : auto (get_session_context)
        f_collate --> f_store : complete_task
        f_store --> f_confirm : auto (store_user_req)
        f_confirm --> f_followup : complete_task
        f_followup --> f_close : complete_task
        f_close --> [*] : complete_task
    }

    reception --> subscription : transfer_to_subscription
    reception --> sales : transfer_to_sales
    reception --> support : transfer_to_support
    reception --> decease : transfer_to_decease_handling
    subscription --> finishing : transfer_to_finishing
    sales --> finishing : transfer_to_finishing
    support --> finishing : transfer_to_finishing
    decease --> finishing : transfer_to_finishing
    finishing --> sales : transfer_to_sales (new request)
    finishing --> subscription : transfer_to_subscription (new request)
    finishing --> support : transfer_to_support (new request)
```

### Limitations

| Limitation | Impact |
|---|---|
| **No branching** | `handle_request` covers cancellation, changes, and escalation — all as one opaque step. The state machine can't enforce different flows for different intents. |
| **No loops** | If the customer says "actually I have another subscription question" after `check_anything_else`, the only option is `reset_agent_tasks()` which resets ALL tasks, losing progress. |
| **No nesting** | Retention (3 attempts) lives inside `handle_request` with no visibility to the state machine. The LLM manages retry counting on its own. |
| **LLM-dependent completion** | Conversational tasks rely on the LLM calling `complete_task`. If it forgets, the gate blocks forever. If it calls it prematurely, tasks get skipped. |
| **No conditional transitions** | Every task always leads to the next one in the list. There's no way to skip `record_outcome` if the customer was transferred to another agent mid-flow. |

---

## 2. Option 1 — Hierarchical Task Trees

### How It Works

Same dict-based approach, but tasks can have `children` (sub-tasks) and a `repeat` flag. The `advance_task()` function walks depth-first: it completes the current leaf node, moves to the next sibling, and when all children are done, the parent auto-completes.

`repeat: True` on a parent means that subtree can be reset independently without touching the rest of the agent's tasks. This handles "anything else?" loops cleanly.

Optional `condition` strings on tasks allow skipping (e.g., retention tasks only activate if intent is cancellation). The condition is evaluated against session context.

### Data Structure

```python
"subscription": {
    "current_path": ["handle_request", "understand_request"],  # depth-first pointer
    "tasks": [
        {
            "id": "load_context",
            "label": "Load session context",
            "status": "pending",
            "auto_tool": "get_session_context",
        },
        {
            "id": "handle_request",
            "label": "Handle subscription request",
            "status": "pending",
            "repeat": True,
            "children": [
                {
                    "id": "understand_request",
                    "label": "Clarify the request type",
                    "status": "pending",
                },
                {
                    "id": "retention",
                    "label": "Retention flow",
                    "status": "pending",
                    "condition": "intent == cancel",
                    "children": [
                        {"id": "retention_1", "label": "Attempt 1: solve problem",   "status": "pending"},
                        {"id": "retention_2", "label": "Attempt 2: make it personal","status": "pending"},
                        {"id": "retention_3", "label": "Attempt 3: show you care",   "status": "pending"},
                    ],
                },
                {
                    "id": "record_change",
                    "label": "Record the outcome",
                    "status": "pending",
                    "auto_tool": "log_agent_outcome",
                },
            ],
        },
        {
            "id": "check_anything_else",
            "label": "Ask if anything else",
            "status": "pending",
            "transitions": {
                "yes": "handle_request",  # re-enter the repeatable subtree
                "no": "transfer_out",
            },
        },
        {
            "id": "transfer_out",
            "label": "Transfer to finishing",
            "status": "pending",
            "auto_tool": "transfer_to_finishing",
        },
    ],
}
```

### Mermaid Diagram — Hierarchical Task Tree (Subscription Agent)

```mermaid
stateDiagram-v2
    direction TB

    [*] --> load_context
    load_context --> handle_request : auto (get_session_context)

    state "handle_request (repeatable)" as handle_request {
        [*] --> understand_request
        understand_request --> retention : intent == cancel
        understand_request --> record_change : intent != cancel

        state "retention (conditional)" as retention {
            [*] --> retention_1
            retention_1 --> retention_2 : customer declined
            retention_2 --> retention_3 : customer declined
            retention_3 --> [*] : accepted or exhausted
        }

        retention --> record_change : retention complete
        record_change --> [*] : auto (log_agent_outcome)
    }

    handle_request --> check_anything_else : subtree complete
    check_anything_else --> handle_request : "yes" (reset subtree)
    check_anything_else --> transfer_out : "no"
    transfer_out --> [*] : auto (transfer_to_finishing)
```

### Mermaid Diagram — Hierarchical Task Tree (Reception Agent)

```mermaid
stateDiagram-v2
    direction TB

    [*] --> greet
    greet --> verify_identity : complete_task

    state "verify_identity" as verify_identity {
        [*] --> ask_name_phone
        ask_name_phone --> call_get_customer_info : info provided
        call_get_customer_info --> verified : success
        call_get_customer_info --> retry_verify : failure
        retry_verify --> call_get_customer_info : retry (max 2)
        retry_verify --> escalate_human : max retries exceeded
    }

    verify_identity --> identify_intent : verified
    identify_intent --> fetch_data : complete_task

    state "fetch_data" as fetch_data {
        [*] --> fetch_order
        fetch_order --> fetch_subscription : auto
        fetch_subscription --> [*] : auto
    }

    fetch_data --> prepare_handoff : subtree complete
    prepare_handoff --> transfer : auto (prepare_handoff_context)
    transfer --> [*] : auto (transfer_to_*)
```

### How advance_task Works (Depth-First)

```
Current path: ["handle_request", "retention", "retention_1"]

1. Mark retention_1 as ✓
2. Next sibling of retention_1 → retention_2
3. New path: ["handle_request", "retention", "retention_2"]

When retention_3 completes:
1. Mark retention_3 as ✓
2. No more siblings → parent "retention" auto-completes ✓
3. Next sibling of retention → record_change
4. New path: ["handle_request", "record_change"]

When record_change completes:
1. Mark record_change as ✓
2. No more siblings → parent "handle_request" auto-completes ✓
3. Next top-level task → check_anything_else
4. New path: ["check_anything_else"]
```

### Pros & Cons

| Pros | Cons |
|---|---|
| Minimal refactor from current system | Condition strings (`"intent == cancel"`) are fragile — need an eval mechanism |
| LLM prompt format barely changes (still a checklist, just indented) | Tree traversal logic is more complex than flat list |
| Nesting gives visibility into retention attempts | Still not a full graph — can't express arbitrary transitions |
| `repeat` flag handles "anything else?" loops cleanly | Deep nesting can make the injected prompt verbose |

---

## 3. Option 2 — Finite State Machine with Transitions (Recommended)

### How It Works

Replace the task list with a proper state graph. Each agent defines a set of **states** and **transitions**. The current state is tracked, and movement between states happens via **events** — either emitted by tool calls automatically or by the LLM calling a `transition()` function.

Key concepts:
- **States** are named nodes (not ordered tasks). Each state can have `auto_tool` (auto-transition when a tool is called), `transitions` (event → next state mapping), or `on_complete` (default next state).
- **Events** are strings like `"cancel"`, `"retained"`, `"yes"`, `"no"`. They drive transitions.
- **Bounded retries** via `max_attempts` on a state. When attempts are exhausted, the `"exhausted"` event fires automatically.
- **Loops** are first-class: `check_anything_else → yes → understand_request` is a real cycle.
- **Terminal states** end the agent's flow.

The LLM prompt injection shows only the current state and available transitions — much simpler than a full checklist.

### Data Structure

```python
"subscription": {
    "current_state": "load_context",
    "history": [],  # audit trail of state transitions
    "states": {
        "load_context": {
            "label": "Load session context",
            "auto_tool": "get_session_context",
            "on_complete": "understand_request",
        },
        "understand_request": {
            "label": "Understand the customer's request",
            "transitions": {
                "cancel": "retention",
                "change": "record_outcome",
                "escalate": "escalation",
                "non_subscription": "transfer_other",
            },
        },
        "retention": {
            "label": "Retention conversation",
            "max_attempts": 3,
            "attempt": 0,
            "transitions": {
                "retained": "record_outcome",
                "still_cancelling": "retention",   # loops back
                "exhausted": "record_outcome",      # auto-fires at max
                "deceased": "transfer_other",
                "wants_other_product": "transfer_other",
            },
        },
        "record_outcome": {
            "label": "Record the outcome",
            "auto_tool": "log_agent_outcome",
            "on_complete": "check_anything_else",
        },
        "check_anything_else": {
            "label": "Ask if anything else",
            "transitions": {
                "yes": "understand_request",        # full loop back
                "no": "transfer_finishing",
            },
        },
        "escalation": {
            "label": "Escalate to human agent",
            "auto_tool": "log_agent_outcome",
            "on_complete": "transfer_finishing",
        },
        "transfer_finishing": {
            "label": "Transfer to finishing agent",
            "auto_tool": "transfer_to_finishing",
            "terminal": True,
        },
        "transfer_other": {
            "label": "Transfer to another agent",
            "terminal": True,
        },
    },
}
```

### Mermaid Diagram — FSM (Subscription Agent)

```mermaid
stateDiagram-v2
    direction TB

    [*] --> load_context

    load_context --> understand_request : auto (get_session_context)

    understand_request --> retention : event: cancel
    understand_request --> record_outcome : event: change
    understand_request --> escalation : event: escalate
    understand_request --> transfer_other : event: non_subscription

    retention --> retention : event: still_cancelling (attempt++)
    retention --> record_outcome : event: retained
    retention --> record_outcome : event: exhausted (auto at max_attempts)
    retention --> transfer_other : event: deceased
    retention --> transfer_other : event: wants_other_product

    record_outcome --> check_anything_else : auto (log_agent_outcome)

    check_anything_else --> understand_request : event: yes (LOOP)
    check_anything_else --> transfer_finishing : event: no

    escalation --> transfer_finishing : auto (log_agent_outcome)

    transfer_finishing --> [*] : auto (transfer_to_finishing)
    transfer_other --> [*] : auto (transfer_to_*)
```

### Mermaid Diagram — FSM (Reception Agent)

```mermaid
stateDiagram-v2
    direction TB

    [*] --> greet

    greet --> check_customer_type : event: greeted

    check_customer_type --> verify_identity : event: existing_customer
    check_customer_type --> identify_intent_new : event: new_customer

    verify_identity --> verify_identity : event: failed (retry, max 2)
    verify_identity --> identify_intent : event: verified
    verify_identity --> escalate_human : event: exhausted

    identify_intent --> fetch_data : event: intent_identified
    identify_intent_new --> prepare_handoff_new : event: intent_identified

    fetch_data --> prepare_handoff : auto (prepare_handoff_context)

    prepare_handoff --> route_subscription : event: subscription
    prepare_handoff --> route_sales : event: sales
    prepare_handoff --> route_support : event: support
    prepare_handoff --> route_decease : event: decease

    prepare_handoff_new --> route_sales_new : event: ready

    route_subscription --> [*] : auto (transfer_to_subscription)
    route_sales --> [*] : auto (transfer_to_sales)
    route_support --> [*] : auto (transfer_to_support)
    route_decease --> [*] : auto (transfer_to_decease_handling)
    route_sales_new --> [*] : auto (transfer_to_sales)
    escalate_human --> [*] : terminal
```

### Mermaid Diagram — FSM (Sales Agent)

```mermaid
stateDiagram-v2
    direction TB

    [*] --> load_context

    load_context --> check_customer : auto (get_session_context)

    check_customer --> collect_info : event: new_customer
    check_customer --> confirm_products : event: existing_customer

    collect_info --> confirm_products : event: info_collected (save_customer_info)

    confirm_products --> subscription_guidance : event: product_confirmed
    confirm_products --> confirm_products : event: product_not_found (retry)

    subscription_guidance --> delivery_date : event: no_subscription
    subscription_guidance --> transfer_subscription : event: wants_subscription

    delivery_date --> promotion_source : event: date_confirmed
    promotion_source --> calculate_total : event: source_recorded
    calculate_total --> payment_method : auto (calculate_order_total)
    payment_method --> check_gifts : event: payment_selected
    check_gifts --> readback : auto (check_eligible_gifts)
    readback --> place_order : event: confirmed
    readback --> confirm_products : event: correction_needed (LOOP)

    place_order --> log_outcome : auto (place_order tool)
    log_outcome --> transfer_finishing : auto (log_agent_outcome)
    transfer_finishing --> [*] : auto (transfer_to_finishing)
    transfer_subscription --> [*] : auto (transfer_to_subscription)
```

### Mermaid Diagram — FSM (Decease Handling Agent)

```mermaid
stateDiagram-v2
    direction TB

    [*] --> load_context

    load_context --> express_condolences : auto (get_session_context)
    express_condolences --> collect_caller_info : event: condolences_expressed

    collect_caller_info --> collect_deceased_info : event: caller_info_collected
    collect_deceased_info --> explain_procedures : event: deceased_confirmed

    explain_procedures --> check_unpaid : event: procedures_explained

    check_unpaid --> inform_unpaid : event: has_unpaid
    check_unpaid --> register_case : event: no_unpaid

    inform_unpaid --> collect_reissue_info : event: needs_reissue
    inform_unpaid --> register_case : event: no_reissue_needed

    collect_reissue_info --> register_case : event: reissue_info_collected

    register_case --> log_outcome : auto (collect_decease_information)
    log_outcome --> transfer_finishing : auto (log_agent_outcome)
    transfer_finishing --> [*] : auto (transfer_to_finishing)
```

### Mermaid Diagram — FSM (Finishing Agent)

```mermaid
stateDiagram-v2
    direction TB

    [*] --> load_context

    load_context --> collate_requirements : auto (get_session_context)
    collate_requirements --> store_requirements : event: collated

    store_requirements --> confirm_customer : auto (store_user_req)
    confirm_customer --> check_followup : event: confirmed

    check_followup --> log_and_route : event: new_request
    check_followup --> log_and_close : event: no_more

    log_and_route --> route_sales : event: order
    log_and_route --> route_subscription : event: subscription
    log_and_route --> route_support : event: support

    route_sales --> [*] : auto (transfer_to_sales)
    route_subscription --> [*] : auto (transfer_to_subscription)
    route_support --> [*] : auto (transfer_to_support)

    log_and_close --> close_call : auto (log_agent_outcome)
    close_call --> [*] : terminal
```

### Mermaid Diagram — FSM (Full Multi-Agent Orchestration)

```mermaid
stateDiagram-v2
    direction LR

    state "Reception" as R
    state "Subscription" as SUB
    state "Sales" as SAL
    state "Support" as SUP
    state "Decease" as DEC
    state "Finishing" as FIN

    [*] --> R : call starts

    R --> SUB : event: subscription
    R --> SAL : event: sales / new_customer
    R --> SUP : event: support
    R --> DEC : event: decease

    SUB --> FIN : transfer_finishing
    SUB --> SAL : wants_other_product
    SUB --> DEC : deceased
    SUB --> R : re_route

    SAL --> FIN : transfer_finishing
    SAL --> SUB : wants_subscription
    SAL --> R : existing_customer_detected

    SUP --> FIN : transfer_finishing
    SUP --> SUB : subscription_issue
    SUP --> R : re_route

    DEC --> FIN : transfer_finishing

    FIN --> SAL : new order request
    FIN --> SUB : new subscription request
    FIN --> SUP : new support request
    FIN --> [*] : call ends
```

### Core Functions

```python
def transition(session, agent_name: str, event: str) -> str | None:
    """Move to the next state based on current state + event.
    Returns new state name, or None if terminal/invalid.
    """
    sm = session.userdata["state_machine"][agent_name]
    current = sm["current_state"]
    state_def = sm["states"][current]

    if state_def.get("terminal"):
        return None

    # Handle bounded retries
    if "max_attempts" in state_def:
        state_def["attempt"] = state_def.get("attempt", 0) + 1
        if state_def["attempt"] >= state_def["max_attempts"]:
            event = "exhausted"

    # Resolve next state
    next_state = None
    if event and "transitions" in state_def:
        next_state = state_def["transitions"].get(event)
    if not next_state:
        next_state = state_def.get("on_complete")

    if next_state:
        sm["history"].append({"from": current, "to": next_state, "event": event})
        sm["current_state"] = next_state

    return next_state


def auto_transition_on_tool(session, agent_name: str, tool_name: str) -> str | None:
    """Called after any tool execution. If the current state has
    auto_tool matching this tool, fire on_complete automatically.
    """
    sm = session.userdata["state_machine"][agent_name]
    current = sm["current_state"]
    state_def = sm["states"][current]

    if state_def.get("auto_tool") == tool_name:
        return transition(session, agent_name, event="__auto__")
    return None
```

### LLM Prompt Injection (What the Model Sees)

Instead of a full checklist, the LLM sees only its current state and available actions:

```
[STATE: retention | attempt 2 of 3]
You are in the retention conversation. The customer wants to cancel.

Available actions (call emit_event with one of these):
  → "retained"            — customer agreed to stay     → next: record_outcome
  → "still_cancelling"    — customer declined, try again → next: retention (attempt 3)
  → "deceased"            — customer is deceased         → next: transfer_other
  → "wants_other_product" — customer wants a different product → next: transfer_other

If you reach attempt 3 and the customer still declines, "exhausted" fires automatically.
```

This is much cleaner than a 15-line checklist. The model knows exactly what it can do and what happens next.

### Pros & Cons

| Pros | Cons |
|---|---|
| Explicit transitions — no ambiguity about what comes next | More upfront design work per agent |
| Loops, retries, and branching are first-class | State graph can get complex for agents with many paths |
| LLM prompt is simpler and more actionable | Need to map every tool to its `auto_tool` state |
| Audit trail via `history` array | Migration from current system requires rewriting `build_initial_state_machine()` |
| `max_attempts` removes LLM counting responsibility | — |
| Race-condition safe (single `current_state` pointer) | — |

---

## 4. Option 3 — Event-Driven Orchestrator

### How It Works

The state machine is pulled out of `session.userdata` entirely and becomes a standalone orchestration layer that sits between LiveKit's `AgentSession` and your agents. Think of it as a lightweight workflow engine.

Key differences from Option 2:
- **Centralized orchestrator**: A single `WorkflowOrchestrator` class manages all agent state graphs. It's not a dict in userdata — it's a proper object with methods.
- **Event bus**: Tool calls, LLM responses, and agent handoffs all emit events to the orchestrator. The orchestrator decides transitions and can trigger agent swaps automatically.
- **Agent-agnostic routing**: The orchestrator knows the full multi-agent graph. When the subscription agent reaches `transfer_finishing`, the orchestrator doesn't need the LLM to call `transfer_to_finishing` — it does it directly.
- **Middleware hooks**: Pre/post transition hooks for logging, validation, metrics, etc.
- **Persistence**: State can be serialized to a database for crash recovery and analytics.

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                  LiveKit AgentSession                │
│                                                     │
│  ┌──────────┐   events    ┌──────────────────────┐  │
│  │  Agent    │ ─────────► │  WorkflowOrchestrator │  │
│  │ (LLM +   │            │                      │  │
│  │  Tools)   │ ◄───────── │  - state graphs      │  │
│  │          │  commands   │  - transition engine  │  │
│  └──────────┘            │  - event bus          │  │
│                          │  - audit log          │  │
│                          │  - agent router       │  │
│                          └──────────────────────┘  │
│                                   │                 │
│                                   ▼                 │
│                          ┌──────────────────┐       │
│                          │  Persistence     │       │
│                          │  (Redis / DB)    │       │
│                          └──────────────────┘       │
└─────────────────────────────────────────────────────┘
```

### Data Structure

```python
class WorkflowOrchestrator:
    def __init__(self, session: AgentSession):
        self.session = session
        self.graphs: dict[str, StateGraph] = {}
        self.event_log: list[Event] = []
        self.middleware: list[Callable] = []

    def register_agent(self, name: str, graph: StateGraph):
        """Register an agent's state graph."""
        self.graphs[name] = graph

    def emit(self, agent_name: str, event: str, payload: dict = None):
        """Process an event — transition state, trigger hooks, maybe swap agents."""
        graph = self.graphs[agent_name]
        old_state = graph.current
        new_state = graph.transition(event)

        # Run middleware (logging, metrics, validation)
        for mw in self.middleware:
            mw(agent_name, old_state, new_state, event, payload)

        # Log for audit
        self.event_log.append(Event(agent_name, old_state, new_state, event, payload))

        # Auto-swap agent if terminal state has a handoff target
        if graph.states[new_state].handoff_to:
            target = graph.states[new_state].handoff_to
            self._swap_agent(target)

    def _swap_agent(self, target_agent: str):
        """Programmatically swap the active agent — no LLM decision needed."""
        # Create the target agent and call session.update_agent()
        ...


@dataclass
class StateGraph:
    current: str
    states: dict[str, State]
    history: list[tuple[str, str, str]]  # (from, to, event)

    def transition(self, event: str) -> str:
        state = self.states[self.current]
        next_state = state.resolve(event)
        self.history.append((self.current, next_state, event))
        self.current = next_state
        return next_state


@dataclass
class State:
    name: str
    transitions: dict[str, str]
    auto_tool: str | None = None
    max_attempts: int | None = None
    attempt: int = 0
    terminal: bool = False
    handoff_to: str | None = None  # auto-swap to this agent on entry

    def resolve(self, event: str) -> str:
        if self.max_attempts and self.attempt >= self.max_attempts:
            event = "exhausted"
        return self.transitions.get(event, self.transitions.get("__default__"))
```

### Mermaid Diagram — Event-Driven Orchestrator Architecture

```mermaid
flowchart TB
    subgraph "LiveKit AgentSession"
        LLM["LLM (GPT-4o / Claude)"]
        Tools["@function_tools"]
        TTS["TTS Engine"]
    end

    subgraph "Workflow Orchestrator"
        EB["Event Bus"]
        TE["Transition Engine"]
        AR["Agent Router"]
        AL["Audit Log"]
        MW["Middleware Pipeline"]
    end

    subgraph "State Graphs (per agent)"
        SG_R["Reception Graph"]
        SG_SUB["Subscription Graph"]
        SG_SAL["Sales Graph"]
        SG_SUP["Support Graph"]
        SG_DEC["Decease Graph"]
        SG_FIN["Finishing Graph"]
    end

    subgraph "Persistence"
        Redis["Redis / PostgreSQL"]
    end

    LLM -->|"emit_event()"| EB
    Tools -->|"tool_completed"| EB
    EB --> MW
    MW --> TE
    TE --> SG_R
    TE --> SG_SUB
    TE --> SG_SAL
    TE --> SG_SUP
    TE --> SG_DEC
    TE --> SG_FIN
    TE -->|"handoff_to"| AR
    AR -->|"swap agent"| LLM
    TE --> AL
    AL --> Redis
```

### Mermaid Diagram — Event Flow (Subscription Cancellation Example)

```mermaid
sequenceDiagram
    participant C as Customer
    participant LLM as LLM Agent
    participant T as Tools
    participant O as Orchestrator
    participant G as State Graph

    Note over G: current_state = load_context

    LLM->>T: get_session_context()
    T->>O: event: tool_completed(get_session_context)
    O->>G: auto_transition → understand_request
    Note over G: current_state = understand_request

    C->>LLM: "I want to cancel my subscription"
    LLM->>O: emit_event("cancel")
    O->>G: transition → retention (attempt 0)
    Note over G: current_state = retention

    LLM->>C: "May I ask why you'd like to cancel?"
    C->>LLM: "Too much product accumulating"
    LLM->>C: "We could extend your delivery interval..."
    C->>LLM: "No thanks"
    LLM->>O: emit_event("still_cancelling")
    O->>G: transition → retention (attempt 1)

    LLM->>C: "You've been with us for 3 years..."
    C->>LLM: "I appreciate that but no"
    LLM->>O: emit_event("still_cancelling")
    O->>G: transition → retention (attempt 2)

    LLM->>C: "We truly value you as a customer..."
    C->>LLM: "Please just cancel it"
    LLM->>O: emit_event("still_cancelling")
    O->>G: attempt 3 >= max_attempts → auto "exhausted"
    O->>G: transition → record_outcome
    Note over G: current_state = record_outcome

    LLM->>T: log_agent_outcome()
    T->>O: event: tool_completed(log_agent_outcome)
    O->>G: auto_transition → check_anything_else
    Note over G: current_state = check_anything_else

    LLM->>C: "Is there anything else?"
    C->>LLM: "No, that's all"
    LLM->>O: emit_event("no")
    O->>G: transition → transfer_finishing (terminal, handoff_to: finishing)
    O->>O: _swap_agent("finishing") — automatic, no LLM call needed
```

### Pros & Cons

| Pros | Cons |
|---|---|
| Cleanest separation of concerns — LLM only does conversation, orchestrator handles flow | Largest refactor — requires new classes, event bus, middleware |
| Agent swaps are automatic — no reliance on LLM calling transfer functions | More moving parts to debug |
| Full audit trail with persistence | Needs Redis/DB for crash recovery |
| Middleware enables metrics, rate limiting, A/B testing | Over-engineered if you only have 6 agents |
| State graphs are testable independently of LLM | LiveKit's built-in handoff mechanism is partially bypassed |
| Crash recovery — orchestrator can resume from last persisted state | — |

---

## 5. Comparison Matrix

| Feature | Current (Flat List) | Option 1 (Task Trees) | Option 2 (FSM) | Option 3 (Orchestrator) |
|---|---|---|---|---|
| **Branching** | ❌ None | ⚠️ Via conditions | ✅ Event-based | ✅ Event-based |
| **Loops / Repeats** | ❌ Full reset only | ✅ Subtree reset | ✅ Transition cycles | ✅ Transition cycles |
| **Nesting** | ❌ None | ✅ Children array | ⚠️ Via sub-graphs | ✅ Composable graphs |
| **Bounded Retries** | ❌ LLM counts | ⚠️ Manual | ✅ `max_attempts` | ✅ `max_attempts` |
| **Auto-advance on tool** | ⚠️ Hardcoded | ✅ `auto_tool` | ✅ `auto_tool` | ✅ Event bus |
| **LLM prompt clarity** | ⚠️ Long checklist | ⚠️ Indented checklist | ✅ Current state + actions | ✅ Current state + actions |
| **Audit trail** | ❌ Logs only | ❌ Logs only | ✅ `history` array | ✅ Persistent event log |
| **Agent routing** | ⚠️ LLM decides | ⚠️ LLM decides | ⚠️ LLM decides | ✅ Automatic |
| **Crash recovery** | ❌ None | ❌ None | ⚠️ Userdata dump | ✅ DB persistence |
| **Refactor effort** | — | 🟢 Low | 🟡 Medium | 🔴 High |
| **Complexity** | 🟢 Simple | 🟡 Moderate | 🟡 Moderate | 🔴 High |
| **Production readiness** | ⚠️ MVP | 🟡 Good | ✅ Production | ✅ Enterprise |

### Recommendation

**Option 2 (FSM with Transitions)** is the sweet spot for your system. It gives you everything you need — branching, loops, retries, auto-advance, clean LLM prompts — without the overhead of a full orchestration layer. Your 6-agent system doesn't need Redis persistence or an event bus yet. If you scale to 15+ agents or need crash recovery, Option 3 becomes worth it.

The migration path is also clean: replace `build_initial_state_machine()` with FSM definitions, replace `advance_task()` with `transition()`, update `format_state_machine()` to show current state + transitions, and update `complete_task` to become `emit_event`. The agent code (prompts, tools, handoffs) stays mostly the same.
