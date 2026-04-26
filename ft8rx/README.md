# ft8rx

`ft8rx` is the receive, control, and web UI application for this repo. It combines:

- FT8 and FT4 audio capture and decode
- rig control
- a queue-based QSO scheduler
- an automated QSO FSM
- JSONL and text logging for post-mortem analysis

The app is designed around unattended or semi-attended operation from the web UI. It does not expose a direct `Start QSO` button anymore. Stations are queued, then the scheduler decides what to work next.

## Config

By default `ft8rx` loads [config/ft8rx.json](/home/bgelb/ft8x/config/ft8rx.json).

Current top-level sections:

- `station`
- `tx`
- `queue`
- `fsm`
- `logging`

### `station`

- `our_call`
- `our_grid`

Current defaults:

- `our_call = N1VF`
- `our_grid = CM97`

### `tx`

- `base_freq_hz`
- `drive_level`
- `playback_channels`
- `output_device`
- `power_w`
- `tx_freq_min_hz`
- `tx_freq_max_hz`

Current defaults:

- `power_w = 20.0`
- `base_freq_hz = 1000.0`
- `drive_level = 0.28`

The scheduler stores separate even/odd TX frequencies in memory. Those are UI/runtime controls, not separate config keys.

### `queue`

- `auto_add_direct_calls_default`
- `ignore_direct_calls_from_recently_worked_default`
- `cq_enabled_default`
- `cq_percent_default`
- `use_compound_rr73_handoff_default`
- `use_compound_73_once_handoff_default`
- `use_compound_for_direct_signal_callers_default`
- `no_message_retry_delay_seconds_default`
- `no_forward_retry_delay_seconds_default`

Current defaults:

- `auto_add_direct_calls_default = true`
- `ignore_direct_calls_from_recently_worked_default = true`
- `cq_enabled_default = false`
- `cq_percent_default = 80`
- `use_compound_rr73_handoff_default = true`
- `use_compound_73_once_handoff_default = false`
- `use_compound_for_direct_signal_callers_default = false`
- `no_message_retry_delay_seconds_default = 35`
- `no_forward_retry_delay_seconds_default = 300`

### `fsm`

- `rr73_enabled`
- `timeout_seconds`
- per-state thresholds

Current thresholds:

- `send_grid`: `no_msg = 3`, `no_fwd = 3`
- `send_sig`: `no_msg = 3`, `no_fwd = 3`
- `send_sig_ack`: `no_msg = 2`, `no_fwd = 5`
- `send_rr73`: `no_fwd = 3`
- `send_rrr`: `no_msg = 2`, `no_fwd = 5`
- `send_73`: `no_msg = 2`, `no_fwd = 3`

Current timeout:

- `timeout_seconds = 600`

### `logging`

- `fsm_log_path`
- `app_log_path`

Defaults:

- `logs/ft8rx-qso.jsonl`
- `logs/ft8rx.log`

## Web UI

The current UI layout is:

- Top row:
  - status / rig control / waterfall
  - station detail
- Second row:
  - stations to work queue
  - QSO pane
- Third row:
  - direct calls
  - QSO log

### Rig controls

- Band dropdown applies immediately.
- Power dropdown applies immediately.
- Power can be changed mid-QSO if TX is idle.
- Band changes are blocked while a QSO is active.
- `Tune 10s` sends a 1000 Hz tone for 10 seconds and returns the rig to RX.

When the band changes:

- bandmaps are cleared
- the queue is cleared
- worked-in-24h suppression continues to apply per band

### Bandmaps

There are separate even and odd bandmaps.

Each bandmap pane includes:

- calls currently heard on that parity
- per-bandmap `Add To Queue`
- per-call `Q` button to queue a single call

Calls are dimmed if they were already worked on the current band within the last 24 hours. Their `Q` button is hidden.

### Station detail pane

Clicking a callsign from the bandmap, queue, direct-calls pane, or QSO log populates the station detail pane.

From station detail:

- `Add To Queue` adds the selected call to the work queue

### Queue pane

The queue pane contains:

- queue count
- scheduler status
- queue controls
- the queue table itself

Queue controls currently include:

- `No Message Retry Delay`
- `No Fwd Retry Delay`
- `CQ %`
- `Flip Next CQ Parity`
- `Clear Queue`
- `Auto add direct calls`
- `Ignore direct calls from already worked stations`
- `Enable CQ`
- `Use compound RR73 handoff`
- `Use compound 73-once handoff`
- `Use compound for direct signal callers`

The queue table is dense and single-row-per-entry. Direct-priority entries that are currently eligible are floated to the top of the displayed list and highlighted.

### QSO pane

The QSO pane shows:

- current partner / idle status
- separate even and odd TX frequency controls
- separate auto-pick buttons for even and odd TX frequency
- `Auto QSO From Queue`
- `Stop`
- current FSM state
- counters
- timeout remaining
- RX classification
- scrollable transcript

`Stop` immediately aborts TX and returns the rig to RX.

### Direct calls pane

This pane shows messages to or from our call for the current process lifetime.

Important detail:

- RX rows come from the in-memory decode/tracker path
- TX rows come from live QSO/TX activity, not from decoder output
- old rows are not repopulated after restart

### QSO log pane

This pane is JSONL-backed and derived from the QSO log file, not from ad hoc in-memory UI state.

Current display policy:

- last 24 hours only
- only QSOs where `Rpl=Y` or `73=Y`

Columns:

- `Time`
- `Ago`
- `Call`
- `Bd`
- `R`
- `Rpl`
- `73`
- `Sent`
- `Recv`
- `Exit`

Meanings:

- `R = Y`: we received a directed roger/closing form from the other station
  - `R`
  - `RRR`
  - `RR73`
  - `73`
- `Rpl = Y`: we received any directed reply at all to our call, including plain grid/report messages
- `73 = Y`: the QSO reached one of our terminal send states:
  - `send_rr73`
  - `send_73`
  - `send_73_once`

## ADIF Export

Use the built-in exporter to turn the QSO FSM JSONL log into LoTW-friendly ADIF:

```bash
cargo run --bin ft8rx -- export-adif --config config/ft8rx.json --output logs/ft8rx-lotw.adi
```

Notes:

- input defaults to `logging.fsm_log_path` from the config
- output defaults to stdout if `--output` is omitted
- the command prints stderr status for ineligible QSOs and a count summary for eligible grids, countries, and DXCC entities
- only conservative award-valid QSOs are exported
- exported records use `MODE=MFSK` with `SUBMODE=FT8` or `SUBMODE=FT4`
- country/DXCC counting uses the local `~/.local/cty` database when available

## Queue model

The queue is in-memory only.

One queue entry exists per callsign.

Each entry tracks at least:

- `callsign`
- `queued_at`
- `ok_to_schedule_after`
- direct-call priority metadata
- direct-call count
- last direct-heard time
- last heard time
- last heard message class
- last heard parity

### Queue admission

A call may be added to the queue by:

- `Add To Queue` from station detail
- `Q` button from a bandmap entry
- bandmap `Add To Queue`
- auto-added direct calls to our station

Own callsign is never allowed into the queue.

Duplicate adds do not create multiple entries.

### Recent-worked suppression

The 24-hour suppression is band-specific.

That means:

- working `K1ABC` on `20m` does not block `K1ABC` on `40m`
- working `K1ABC` on `20m` does block `K1ABC` on `20m` for 24 hours

The recent-worked cache is rebuilt on startup by scanning the QSO JSONL log.

A station is considered worked on a band only if we actually launched a terminal transmit on that band:

- `send_rr73`
- `send_73`
- `send_73_once`

### Queue removal

Entries are removed when:

- manually removed
- queue is cleared
- band changes
- the station has not been heard for 10 minutes
- the station becomes recently worked on the current band
- the station is dispatched into a QSO

### Requeue behavior

Two retry-delay paths exist.

`No Message Retry Delay`:

- used when `SendGrid` exits via `send_grid_no_msg_limit`
- used when `SendSig` exits via `send_sig_no_msg_limit`
- default `35s`

`No Fwd Retry Delay`:

- used when `SendGrid` exits via `send_grid_no_fwd_limit`
- used when `SendSig` exits via `send_sig_no_fwd_limit`
- default `300s`

These exits mean “did not complete enough to count as worked, try again later.” The station is requeued to the back of the line with a fresh `queued_at` and a future `ok_to_schedule_after`.

## Direct calls

Direct calls are a first-class scheduler input.

If `Auto add direct calls` is enabled:

- any directed message to our call from a station we are not currently working is added to the queue
- if that station is already in the queue, its direct-call count is incremented
- a non-direct queue entry can be upgraded into a direct-priority entry by a later direct call

If `Ignore direct calls from already worked stations` is enabled:

- direct calls are still rejected by the same band-specific 24-hour worked rule

### Direct-call start-state mapping

When a direct call is chosen for dispatch, the opening FSM state is inferred from the message type:

- direct plain grid to us -> `SendSig`
- direct plain signal report to us -> `SendSigAck`
- direct `RRR` -> `Send73`
- direct `R`/ack -> `Send73`
- direct blank/no-ack -> `SendSig`
- direct `RR73` or `73` -> ignored, not queued
- compound DXpedition message whose next leg targets us -> `SendSigAck`

### Direct-priority scheduling

Before normal FRFCFS scheduling, the scheduler checks for pending direct-priority calls.

Priority logic:

1. Consider only direct calls heard within the last two RX cycles of the sender’s parity.
2. Prefer calls heard in the most recent eligible cycle.
3. If none were heard in that most recent cycle, consider the second-most-recent eligible cycle.
4. Tie-break by:
   - highest direct-call count
   - best SNR
   - oldest queue age
   - lexical callsign

If no eligible direct-priority call exists, the scheduler falls back to normal queue/CQ logic.

## CQ scheduling

If `Enable CQ` is on, the scheduler can choose CQ instead of a normal queued station some percentage of the time.

Current behavior:

- CQ is only considered when there is no eligible direct-priority call
- the `CQ %` setting controls how often CQ wins versus normal FRFCFS dispatch
- CQ can also be chosen while idle when no normal queued station is eligible

The scheduler also has a one-shot `Flip Next CQ Parity` control. It only affects the next scheduled CQ, then clears automatically.

## QSO start modes

A QSO can start in one of three modes:

- `normal`
- `direct`
- `cq`

`normal`:

- ordinary queued station
- initial state is `SendGrid`

`direct`:

- a direct call to us chosen from the queue
- initial state depends on the message that triggered the direct priority

`cq`:

- internal pseudo-partner `CQ`
- initial state is `SendCq`

## Decode feeding

The QSO FSM consumes RX in decode-stage order:

- `early41`
- `early47`
- `full`

Per RX slot:

- the first early stage that produces a relevant partner/direct event is used
- `full` is only used if neither early stage already produced a relevant event

This reduces missed slot opportunities relative to waiting for `full` only.

## FSM overview

The FSM lives in [ft8rx/src/qso.rs](/home/bgelb/ft8x/ft8rx/src/qso.rs).

General rules:

- TX state determines the transmitted message
- RX state transitions happen after the RX portion of the slot
- committed TX slots stay bound to their committed message/state
- late RX after TX commit is logged and queued for later application when applicable
- every QSO hard-times out after 10 minutes

Counters:

- `no_msg_count`: no qualifying partner message to our call
- `no_fwd_count`: partner did send to our call, but it did not advance the QSO

Counters reset on state entry.

### `Idle`

- no TX
- scheduler may dispatch a queue entry or CQ from here

### `SendCq`

- TX: `CQ <our_call> <our_grid>`
- direct replies are handled by the queue/direct-call machinery, not by binding a partner inside `SendCq`
- if no usable response arrives for 3 cycles, CQ exits
- a fresh direct-priority call can preempt CQ

### `SendGrid`

- TX: `<partner> <our_call> <our_grid>`
- directed plain grid/report to our call counts as forward progress
- directed `R`/ack to our call also counts as forward progress
- forward progress -> `SendSigAck`
- directed `RR73` -> `Send73Once`
- directed `RRR` or `73` -> `Send73`
- no partner message -> increment `no_msg`
- directed non-forward message -> increment `no_fwd`
- `no_msg` threshold -> exit with `send_grid_no_msg_limit`, requeue after no-message delay
- `no_fwd` threshold -> exit with `send_grid_no_fwd_limit`, requeue after no-forward delay
- if this is a normal cold-call QSO and no partner message has been received yet, a fresh priority direct call may preempt it

### `SendSig`

- TX: `<partner> <our_call> <report>`
- directed `R`/ack or `RRR`:
  - `SendRR73` if `rr73_enabled`
  - otherwise `SendRRR`
- directed `RR73` -> `Send73Once`
- directed `73` -> `Send73`
- no partner message -> increment `no_msg`
- directed non-forward message -> increment `no_fwd`
- `no_msg` threshold -> exit with `send_sig_no_msg_limit`, requeue after no-message delay
- `no_fwd` threshold -> exit with `send_sig_no_fwd_limit`, requeue after no-forward delay
- a fresh priority direct call may preempt `SendSig` only if:
  - the QSO started as `normal`
  - no partner RX has been received yet
  - we already transmitted at least once
  - we already completed at least one empty RX cycle

### `SendSigAck`

- TX: `<partner> <our_call> R<report>`
- directed `RR73` -> `Send73Once`
- directed `R`, `RRR`, or `73` -> `Send73`
- directed compound `RR73;` to us is treated as `RR73`
- no partner message -> increment `no_msg`
- directed non-forward message -> increment `no_fwd`
- either threshold -> `Send73Once`

### `SendRR73`

- TX: `<partner> <our_call> RR73`
- directed `73` -> exit
- partner CQ, directed-to-other, non-call-first-field, freeform, or no message -> exit
- other directed non-forward message -> increment `no_fwd`
- `no_fwd` threshold -> `Send73Once`

If enabled, `SendRR73` may instead be replaced by a compound handoff:

- `oldcall RR73; newcall <our_call> +nn`

This happens only if:

- there is an eligible pending direct-priority station
- compound RR73 handoff is enabled
- that next station is compound-eligible under the current settings

### `SendRRR`

- TX: `<partner> <our_call> RRR`
- directed `73` -> `Send73`
- directed `RR73` -> `Send73Once`
- directed-to-other or moved-on cases -> `Send73Once`
- no partner message -> increment `no_msg`
- directed but not `73` -> increment `no_fwd`
- either threshold -> `Send73Once`

### `Send73`

- TX: `<partner> <our_call> 73`
- directed `73` -> exit
- partner CQ, directed-to-other, non-call-first-field, or freeform -> exit
- no partner message -> increment `no_msg`
- directed but not `73` -> increment `no_fwd`
- either threshold -> exit

### `Send73Once`

- TX exactly one `<partner> <our_call> 73`
- exit after that exact TX completes

If enabled, `Send73Once` may instead be replaced by a compound handoff:

- `oldcall RR73; newcall <our_call> +nn`

This is WSJT-X-compatible because WSJT-X treats incoming `RR73` as a valid terminal close.

## Compound handoff behavior

`ft8rx` supports the WSJT-X DXpedition compound message:

- `oldcall RR73; newcall <our_call> +nn`

It is used only as a shared-slot handoff between a QSO being closed and a queued direct caller being started.

There are three toggles:

- `Use compound RR73 handoff`
- `Use compound 73-once handoff`
- `Use compound for direct signal callers`

Default behavior:

- RR73 handoff enabled
- 73-once handoff disabled
- direct-signal compound disabled

Why direct-signal compound is optional:

- the compound format can only send a plain report to the next station
- it cannot send `R+nn`
- WSJT-X does not infer `R`
- so if the next station already sent us a signal report, they will still answer with `R+nn` rather than jumping directly to closing

Once a compound handoff is armed:

- the next station is reserved
- it is no longer treated as a mutable queue candidate
- later decodes may refresh its context, but cannot replace it with a different station

## Logging

There are two logs:

- JSONL FSM/QSO log:
  - `logs/ft8rx-qso.jsonl`
- human-readable app log:
  - `logs/ft8rx.log`

The JSONL log is the source of truth for:

- QSO history pane
- recent-worked cache rebuild on startup
- sent/received exchange reconstruction

Each JSONL record includes, among other fields:

- session id
- partner call
- start mode
- rig frequency and band
- TX parity
- TX frequency
- state before / after
- counters
- timeout remaining
- RX stage
- RX classification
- RX rendered text
- serialized structured RX payload
- TX text
- compound handoff metadata when applicable

The text log includes:

- queue add/remove/requeue decisions
- scheduler decisions
- direct-call upgrades
- CQ decisions
- QSO lifecycle events
- rig control actions

Use the text log for quick debugging and the JSONL log when exact reconstruction matters.
