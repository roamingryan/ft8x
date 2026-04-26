mod config;
mod qso;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Html;
use axum::routing::{get, post};
use axum::{Json, Router};
use chrono::{DateTime, Local, Utc};
use clap::{Args, Parser, Subcommand};
use config::AppConfig;
use ft8_decoder::{
    AudioBuffer, CallModifier, DecodeOptions, DecodeProfile, DecodeStage, DecodedMessage,
    DecoderSession, DecoderState, Mode as DecoderMode, StageDecodeReport, StructuredCallField,
    StructuredCallValue, StructuredInfoField, StructuredInfoValue, StructuredMessage,
};
use hound::{SampleFormat, WavSpec, WavWriter};
use qso::{
    QsoCommand, QsoController, QsoOutcome, QsoStartMode, QsoState, RigTxBackend, StationStartInfo,
    WebQsoDefaults, WebQsoSnapshot,
};
use rigctl::audio::{AudioDevice, AudioStreamConfig, SampleStream, play_tone};
use rigctl::{
    Band, Mode as RigMode, Rig, RigConnectionConfig, RigKind, RigPowerRequest, RigPowerState,
    RigSnapshot as CommonRigSnapshot, RigTelemetry, TxMeterMode, detect_audio_device_for_rig,
    detect_audio_output_device_for_rig, resolve_rig_kind,
};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex32;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt::Write as _;
use std::fs::OpenOptions;
use std::io::{IsTerminal, Write as IoWrite};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{error, info, warn};
use tracing_subscriber::Registry;
use tracing_subscriber::fmt::writer::MakeWriter;
use tracing_subscriber::layer::SubscriberExt;

const DECODER_SAMPLE_RATE_HZ: u32 = 12_000;
const WATERFALL_MAX_HZ: f32 = 4_000.0;
const WATERFALL_BUCKETS: usize = 400;
const WATERFALL_HISTORY_ROWS: usize = 180;
const WATERFALL_SAMPLES: usize = 4096;
const WATERFALL_UPDATE_MS: u64 = 200;
const WEB_BIND_DEFAULT: &str = "127.0.0.1:8000";
const BANDMAP_CELL_HZ: f32 = 400.0;
const BANDMAP_COLUMNS: usize = 10;
const BANDMAP_ROWS: usize = 1;
const BANDMAP_MAX_AGE_SLOTS: u64 = 10;
const DT_HISTORY_FRAMES: usize = 40;
const STATION_RETENTION: Duration = Duration::from_secs(60 * 60);
const QUEUE_HEARD_RETENTION: Duration = Duration::from_secs(10 * 60);
const CQ_ACTIVITY_WINDOW: Duration = Duration::from_secs(5 * 60);
const DIRECT_CALL_PANE_RETENTION: Duration = Duration::from_secs(60 * 60);
const RECENT_WORKED_RETENTION: Duration = Duration::from_secs(24 * 60 * 60);
const QSO_JSONL_REFRESH_INTERVAL: Duration = Duration::from_secs(2);
const TX_TELEMETRY_REFRESH_INTERVAL: Duration = Duration::from_millis(500);
const CONFIG_DEFAULT: &str = "config/ft8rx.json";

#[derive(Debug, Parser)]
#[command(name = "ft8rx")]
struct Cli {
    #[command(subcommand)]
    command: Option<CliCommand>,
    #[arg(long)]
    oneshot: bool,
    #[arg(long, default_value = WEB_BIND_DEFAULT)]
    web_bind: String,
    #[arg(long, default_value = CONFIG_DEFAULT)]
    config: PathBuf,
    #[arg(long)]
    save_wav: Option<PathBuf>,
    #[arg(long)]
    save_raw_wav: Option<PathBuf>,
    #[arg(long)]
    device: Option<String>,
    #[arg(long, default_value = "medium", value_name = "quick|medium|deepest")]
    decode_profile: String,
}

#[derive(Debug, Subcommand)]
enum CliCommand {
    ExportAdif(ExportAdifArgs),
}

#[derive(Debug, Args)]
struct ExportAdifArgs {
    #[arg(long, default_value = CONFIG_DEFAULT)]
    config: PathBuf,
    #[arg(long)]
    input: Option<PathBuf>,
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, thiserror::Error)]
enum AppError {
    #[error("rig error: {0}")]
    Rig(#[from] rigctl::Error),
    #[error("audio error: {0}")]
    Audio(#[from] rigctl::audio::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("wav error: {0}")]
    Wav(#[from] hound::Error),
    #[error("decoder error: {0}")]
    Decoder(String),
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("system clock error")]
    Clock,
}

#[derive(Debug, Clone)]
struct DisplayState {
    rig: Option<RigSnapshot>,
    app_mode: DecoderMode,
    audio: AudioDevice,
    capture_rms_dbfs: f32,
    capture_latest_sample_time: Option<SystemTime>,
    capture_channel_rms_dbfs: Vec<f32>,
    capture_channel: usize,
    capture_recoveries: u64,
    decode_status: String,
    early41_wall_ms: Option<u128>,
    early47_wall_ms: Option<u128>,
    tx_margin_ms: Option<i128>,
    full_wall_ms: Option<u128>,
    last_decode_wall_ms: Option<u128>,
    dropped_slots: u64,
    last_slot_start: Option<SystemTime>,
    early41_decodes: Vec<DecodedMessage>,
    early47_decodes: Vec<DecodedMessage>,
    full_decodes: Vec<DecodedMessage>,
}

#[derive(Debug, Clone)]
struct RigSnapshot {
    kind: RigKind,
    frequency_hz: u64,
    mode: RigMode,
    band: Band,
    power: RigPowerState,
    bar_graph: Option<u8>,
    telemetry_rx_s_meter: Option<f32>,
    telemetry_tx_forward_power_w: Option<f32>,
    telemetry_tx_swr: Option<f32>,
    transmitting: Option<bool>,
}

#[derive(Debug, Clone)]
struct CompositeDecodeRow {
    display: DecodedMessage,
    seen: &'static str,
}

type SharedWebSnapshot = Arc<Mutex<WebSnapshot>>;
type SharedRig = Arc<Mutex<Option<Rig>>>;

#[derive(Clone)]
struct WebAppState {
    snapshot: SharedWebSnapshot,
    qso_control: SharedQsoControl,
    rig_control: SharedRigControl,
    queue_control: SharedQueueControl,
    tune_available: bool,
}

type SharedQsoControl = Arc<QsoControlPlane>;
type SharedRigControl = Arc<RigControlPlane>;
type SharedQueueControl = Arc<QueueControlPlane>;

#[derive(Debug, Default)]
struct QsoControlPlane {
    commands: Mutex<VecDeque<QsoCommand>>,
}

impl QsoControlPlane {
    fn enqueue(&self, command: QsoCommand) {
        self.commands
            .lock()
            .expect("qso control poisoned")
            .push_back(command);
    }

    fn drain(&self) -> Vec<QsoCommand> {
        let mut guard = self.commands.lock().expect("qso control poisoned");
        guard.drain(..).collect()
    }
}

#[derive(Debug, Default)]
struct RigControlPlane {
    commands: Mutex<VecDeque<RigCommand>>,
}

impl RigControlPlane {
    fn enqueue(&self, command: RigCommand) {
        self.commands
            .lock()
            .expect("rig control poisoned")
            .push_back(command);
    }

    fn drain(&self) -> Vec<RigCommand> {
        let mut guard = self.commands.lock().expect("rig control poisoned");
        guard.drain(..).collect()
    }
}

#[derive(Debug, Default)]
struct QueueControlPlane {
    commands: Mutex<VecDeque<QueueCommand>>,
}

impl QueueControlPlane {
    fn enqueue(&self, command: QueueCommand) {
        self.commands
            .lock()
            .expect("queue control poisoned")
            .push_back(command);
    }

    fn drain(&self) -> Vec<QueueCommand> {
        let mut guard = self.commands.lock().expect("queue control poisoned");
        guard.drain(..).collect()
    }
}

#[derive(Debug, Clone)]
enum RigCommand {
    Configure {
        band: Band,
        power: Option<RigPowerRequest>,
        app_mode: DecoderMode,
    },
    Tune10s,
}

#[derive(Debug, Clone)]
enum QueueCommand {
    Add {
        callsign: String,
    },
    Remove {
        callsign: String,
    },
    Clear,
    SetAuto {
        enabled: bool,
    },
    SetTxFreq {
        slot_family: qso::SlotFamily,
        tx_freq_hz: f32,
    },
    SetRetryDelay {
        kind: QueueRetryDelayKind,
        retry_delay_seconds: u64,
    },
    SetAutoAddAllDecodedCalls {
        enabled: bool,
    },
    SetAutoAddDecodedMinCount5m {
        count: u32,
    },
    SetAutoAddDirect {
        enabled: bool,
    },
    SetIgnoreDirectWorked {
        enabled: bool,
    },
    SetCqEnabled {
        enabled: bool,
    },
    SetCqPercent {
        percent: u8,
    },
    SetPauseCqWhenFewUniqueCalls {
        enabled: bool,
    },
    SetCqPauseMinUniqueCalls5m {
        count: u32,
    },
    SetCompoundRr73Handoff {
        enabled: bool,
    },
    SetCompound73OnceHandoff {
        enabled: bool,
    },
    SetCompoundForDirectSignalCallers {
        enabled: bool,
    },
    ToggleNextCqParity,
}

#[derive(Debug, Clone, Default)]
struct BandMapStore {
    even: BTreeMap<String, BandMapEntry>,
    odd: BTreeMap<String, BandMapEntry>,
    even_last_updated_at: Option<SystemTime>,
    odd_last_updated_at: Option<SystemTime>,
}

#[derive(Debug, Clone)]
struct BandMapEntry {
    callsign: String,
    detail: Option<String>,
    freq_hz: f32,
    last_seen_slot_index: u64,
}

#[derive(Debug, Clone, Default, Serialize)]
struct WebSnapshot {
    time_utc: String,
    our_call: String,
    rig_kind: Option<String>,
    rig_frequency_hz: Option<u64>,
    rig_mode: String,
    rig_band: String,
    app_mode: String,
    rig_power_w: Option<f32>,
    rig_power_label: Option<String>,
    rig_power_current_id: Option<String>,
    rig_power_settings: Vec<WebRigPowerSetting>,
    rig_power_settable: bool,
    rig_power_is_discrete: bool,
    rig_bargraph: Option<u8>,
    rig_rx_s_meter: Option<f32>,
    rig_tx_forward_power_w: Option<f32>,
    rig_tx_swr: Option<f32>,
    rig_is_tx: Option<bool>,
    rig_tune_active: bool,
    decode_status: String,
    audio_stats: WebAudioStats,
    decode_times: WebDecodeTimes,
    dt_stats: WebDtStats,
    current_slot: String,
    last_done_slot: Option<String>,
    decodes: Vec<WebDecodeRow>,
    waterfall: Vec<Vec<u8>>,
    bandmaps: WebBandMaps,
    stations: Vec<WebStationSummary>,
    station_logs: Vec<WebStationLog>,
    direct_calls: Vec<WebDirectCallLog>,
    qso: WebQsoSnapshot,
    qso_defaults: WebQsoDefaults,
    queue: WebQueueSnapshot,
    qso_history: Vec<WebQsoHistoryEntry>,
}

#[derive(Debug, Clone, Default, Serialize)]
struct WebAudioStats {
    latest_sample: Option<String>,
    selected_channel: usize,
    overall_dbfs: f32,
    left_dbfs: f32,
    right_dbfs: f32,
    recoveries: u64,
}

#[derive(Debug, Clone, Default, Serialize)]
struct WebDecodeTimes {
    early_seconds: Option<f32>,
    mid_seconds: Option<f32>,
    late_seconds: Option<f32>,
    tx_margin_seconds: Option<f32>,
}

#[derive(Debug, Clone, Default, Serialize)]
struct WebDtStats {
    current_mean_seconds: Option<f32>,
    current_median_seconds: Option<f32>,
    current_stddev_seconds: Option<f32>,
    current_count: usize,
    ten_minute_mean_seconds: Option<f32>,
    ten_minute_median_seconds: Option<f32>,
    ten_minute_count: usize,
}

#[derive(Debug, Clone, Serialize)]
struct WebDecodeRow {
    seen: String,
    utc: String,
    snr_db: i32,
    dt_seconds: f32,
    freq_hz: f32,
    kind: String,
    field1: String,
    field1_select_call: Option<String>,
    field2: String,
    field2_select_call: Option<String>,
    info: String,
    text: String,
}

#[derive(Debug, Clone, Default, Serialize)]
struct WebBandMaps {
    even: Vec<Vec<Vec<WebBandMapCall>>>,
    odd: Vec<Vec<Vec<WebBandMapCall>>>,
    even_age_seconds: Option<u64>,
    odd_age_seconds: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
struct WebBandMapCall {
    callsign: String,
    detail: Option<String>,
    age_slots: u64,
    worked_recently: bool,
}

#[derive(Debug, Clone, Serialize)]
struct WebStationSummary {
    callsign: String,
    last_heard_at: String,
    last_heard_freq_hz: f32,
    last_heard_snr_db: i32,
    last_heard_slot_family: String,
    is_in_qso: bool,
    in_qso_since: Option<String>,
    qso_with: Option<String>,
    last_qso_ended_at: Option<String>,
    qso_history: Vec<WebCompletedQso>,
}

#[derive(Debug, Clone, Serialize)]
struct WebCompletedQso {
    started_at: String,
    ended_at: String,
    peer: String,
}

#[derive(Debug, Clone, Serialize)]
struct WebStationLog {
    timestamp: String,
    sender_call: String,
    peer: Option<String>,
    peer_before: Option<String>,
    peer_after: Option<String>,
    related_calls: Vec<String>,
    snr_db: i32,
    dt_seconds: f32,
    freq_hz: f32,
    kind: String,
    field1: String,
    field2: String,
    info: String,
    text: String,
}

#[derive(Debug, Clone, Serialize)]
struct WebDirectCallLog {
    sort_epoch_ms: u64,
    timestamp: String,
    from_call: String,
    to_call: String,
    snr_db: Option<i32>,
    dt_seconds: Option<f32>,
    freq_hz: Option<f32>,
    text: String,
    is_ours: bool,
}

#[derive(Debug, Deserialize)]
struct RigConfigRequest {
    band: String,
    power_w: Option<f32>,
    power_setting_id: Option<String>,
    app_mode: String,
}

#[derive(Debug, Clone, Serialize)]
struct WebRigPowerSetting {
    id: String,
    label: String,
    nominal_watts: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct QueueCallRequest {
    callsign: String,
}

#[derive(Debug, Deserialize)]
struct QueueAutoRequest {
    enabled: bool,
}

#[derive(Debug, Deserialize)]
struct QueueFlagRequest {
    enabled: bool,
}

#[derive(Debug, Deserialize)]
struct QueueCountRequest {
    count: u32,
}

#[derive(Debug, Deserialize)]
struct QueueTxFreqRequest {
    slot_family: String,
    tx_freq_hz: f32,
}

#[derive(Debug, Deserialize)]
struct QueueRetryDelayRequest {
    kind: String,
    retry_delay_seconds: u64,
}

#[derive(Debug, Clone, Copy)]
enum QueueRetryDelayKind {
    NoMessage,
    NoForward,
}

impl QueueRetryDelayKind {
    fn as_str(self) -> &'static str {
        match self {
            QueueRetryDelayKind::NoMessage => "no_message",
            QueueRetryDelayKind::NoForward => "no_forward",
        }
    }
}

#[derive(Debug, Deserialize)]
struct QueueCqPercentRequest {
    percent: u8,
}

#[derive(Debug, Serialize)]
struct ApiStatus {
    ok: bool,
    message: String,
}

#[derive(Debug, Clone, Default, Serialize)]
struct WebQueueSnapshot {
    auto_enabled: bool,
    auto_add_all_decoded_calls: bool,
    auto_add_decoded_min_count_5m: u32,
    auto_add_direct_calls: bool,
    ignore_direct_calls_from_recently_worked: bool,
    cq_enabled: bool,
    cq_percent: u8,
    pause_cq_when_few_unique_calls: bool,
    cq_pause_min_unique_calls_5m: u32,
    unique_calls_last_5m: u32,
    use_compound_rr73_handoff: bool,
    use_compound_73_once_handoff: bool,
    use_compound_for_direct_signal_callers: bool,
    next_cq_parity_flipped: bool,
    even_tx_freq_hz: f32,
    odd_tx_freq_hz: f32,
    no_message_retry_delay_seconds: u64,
    no_forward_retry_delay_seconds: u64,
    scheduler_status: String,
    entries: Vec<WebQueueEntry>,
}

#[derive(Debug, Clone, Serialize)]
struct WebQueueEntry {
    callsign: String,
    queued_at: String,
    ok_to_schedule_after: String,
    direct_pending: bool,
    priority_direct: bool,
    direct_count: u32,
    direct_last_heard_at: Option<String>,
    last_heard_at: Option<String>,
    last_heard_message: String,
    last_heard_slot_family: Option<String>,
    ready: bool,
    status: String,
}

#[derive(Debug, Clone, Serialize)]
struct WebQsoHistoryEntry {
    time: String,
    age: String,
    age_seconds: u64,
    callsign: String,
    band: String,
    mode: String,
    sent_info: String,
    received_info: String,
    got_roger: bool,
    got_reply: bool,
    reached_73: bool,
    exit_reason: String,
}

#[derive(Debug, Clone)]
struct StationTracker {
    mode: DecoderMode,
    stations: BTreeMap<String, StationState>,
    logs: VecDeque<LoggedDecode>,
    hash12_resolutions: BTreeMap<u16, String>,
    hash22_resolutions: BTreeMap<u32, String>,
}

#[derive(Debug, Clone)]
struct StationState {
    last_heard_at: SystemTime,
    last_heard_slot_index: u64,
    last_heard_decode_stage: DecodeStage,
    last_heard_freq_hz: f32,
    last_heard_snr_db: i32,
    last_heard_slot_family: qso::SlotFamily,
    last_message_kind: StationLastMessageKind,
    last_text: String,
    last_structured_json: String,
    active_qso: Option<ActiveQso>,
    last_qso_ended_at: Option<SystemTime>,
    qso_history: Vec<CompletedQso>,
}

#[derive(Debug, Clone)]
struct ActiveQso {
    since: SystemTime,
    peer: PeerRef,
}

#[derive(Debug, Clone)]
struct CompletedQso {
    started_at: SystemTime,
    ended_at: SystemTime,
    peer: PeerRef,
}

#[derive(Debug, Clone)]
enum PeerRef {
    Callsign(String),
    Hash12(u16),
    Hash22(u32),
}

#[derive(Debug, Clone)]
struct LoggedDecode {
    received_at: SystemTime,
    slot_index: u64,
    decode_stage: DecodeStage,
    sender_call: String,
    peer: Option<PeerRef>,
    peer_before: Option<PeerRef>,
    peer_after: Option<PeerRef>,
    related_calls: Vec<String>,
    snr_db: i32,
    dt_seconds: f32,
    freq_hz: f32,
    kind: String,
    field1: String,
    field2: String,
    info: String,
    text: String,
}

#[derive(Debug, Clone)]
struct WorkQueueState {
    auto_enabled: bool,
    our_call: String,
    current_band: Option<String>,
    current_mode: DecoderMode,
    auto_add_all_decoded_calls: bool,
    auto_add_decoded_min_count_5m: u32,
    auto_add_direct_calls: bool,
    ignore_direct_calls_from_recently_worked: bool,
    cq_enabled: bool,
    cq_percent: u8,
    pause_cq_when_few_unique_calls: bool,
    cq_pause_min_unique_calls_5m: u32,
    use_compound_rr73_handoff: bool,
    use_compound_73_once_handoff: bool,
    use_compound_for_direct_signal_callers: bool,
    next_cq_parity_flipped: bool,
    even_tx_freq_hz: f32,
    odd_tx_freq_hz: f32,
    no_message_retry_delay: Duration,
    no_forward_retry_delay: Duration,
    entries: VecDeque<WorkQueueEntry>,
    recent_worked: BTreeMap<WorkedBandKey, SystemTime>,
    scheduler_status: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct WorkedBandKey {
    callsign: String,
    band: String,
}

#[derive(Debug, Clone)]
struct WorkQueueEntry {
    callsign: String,
    queued_at: SystemTime,
    ok_to_schedule_after: SystemTime,
    last_observed_at: SystemTime,
    direct_pending: bool,
    direct_count: u32,
    last_direct_at: Option<SystemTime>,
    last_direct_slot_index: Option<u64>,
    last_direct_slot_family: Option<qso::SlotFamily>,
    last_direct_snr_db: Option<i32>,
    direct_start_state: Option<QsoState>,
    direct_compound_eligible: bool,
    last_direct_text: Option<String>,
    last_direct_structured_json: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StationLastMessageKind {
    Cq,
    Rr73,
    SeventyThree,
    Other,
}

impl StationLastMessageKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Cq => "cq",
            Self::Rr73 => "rr73",
            Self::SeventyThree => "73",
            Self::Other => "other",
        }
    }

    fn is_ready_last_message(self) -> bool {
        matches!(self, Self::Cq | Self::Rr73 | Self::SeventyThree)
    }
}

#[derive(Debug, Clone)]
struct QueueEntryStatus {
    last_heard_at: Option<SystemTime>,
    last_heard_message: StationLastMessageKind,
    last_heard_slot_family: Option<qso::SlotFamily>,
    ready: bool,
    status: String,
}

#[derive(Debug, Clone)]
struct DirectCallObservation {
    callsign: String,
    observed_at: SystemTime,
    slot_index: u64,
    slot_family: qso::SlotFamily,
    snr_db: i32,
    start_state: QsoState,
    compound_eligible: bool,
    text: String,
    structured_json: String,
}

#[derive(Debug, Clone)]
enum QueueDispatchKind {
    Station {
        callsign: String,
        initial_state: QsoState,
        start_mode: QsoStartMode,
        context_last_heard_at: Option<SystemTime>,
        context_last_heard_slot_family: Option<qso::SlotFamily>,
        context_text: Option<String>,
        context_structured_json: Option<String>,
        context_snr_db: Option<i32>,
    },
    Cq {
        tx_slot_family_override: Option<qso::SlotFamily>,
    },
}

#[derive(Debug, Clone)]
struct QueueDispatch {
    kind: QueueDispatchKind,
    callsign: String,
    tx_slot_family: qso::SlotFamily,
    tx_freq_hz: f32,
}

#[derive(Debug, Clone)]
struct QsoJsonlCache {
    path: PathBuf,
    direct_calls_since: SystemTime,
    last_modified: Option<SystemTime>,
    last_len: u64,
    last_refresh_at: Option<SystemTime>,
    history: Vec<WebQsoHistoryEntry>,
    direct_calls: Vec<WebDirectCallLog>,
    recent_worked: BTreeMap<WorkedBandKey, SystemTime>,
}

#[derive(Debug, Default)]
struct QsoJsonlSessionSummary {
    partner_call: String,
    rig_band: Option<String>,
    app_mode: Option<String>,
    started_at: Option<SystemTime>,
    ended_at: Option<SystemTime>,
    last_seen_at: Option<SystemTime>,
    got_roger: bool,
    got_reply: bool,
    reached_73: bool,
    sent_infos: Vec<String>,
    received_infos: Vec<String>,
    exit_reason: Option<String>,
}

#[derive(Debug, Default)]
struct AdifSessionSummary {
    session_id: u64,
    partner_call: String,
    started_at: Option<SystemTime>,
    ended_at: Option<SystemTime>,
    last_seen_at: Option<SystemTime>,
    rig_band: Option<String>,
    rig_frequency_hz: Option<u64>,
    app_mode: Option<String>,
    got_reply: bool,
    got_roger: bool,
    reached_73: bool,
    sent_report: Option<String>,
    received_report: Option<String>,
    remote_grid: Option<String>,
    exit_reason: Option<String>,
    observed_station_calls: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AdifRecord {
    session_id: u64,
    call: String,
    qso_date: String,
    time_on: String,
    qso_date_off: Option<String>,
    time_off: Option<String>,
    band: String,
    freq: Option<String>,
    mode: String,
    submode: Option<String>,
    rst_sent: String,
    rst_rcvd: String,
    station_callsign: String,
    my_gridsquare: String,
    gridsquare: Option<String>,
}

#[derive(Debug, Default)]
struct AdifExportReport {
    eligible_records: Vec<AdifRecord>,
    ineligible_sessions: Vec<IneligibleQso>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct IneligibleQso {
    session_id: u64,
    call: String,
    started_at: Option<SystemTime>,
    ended_at: Option<SystemTime>,
    reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CtyLookup {
    country: String,
    dxcc_adif: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct QsoHistoryKey {
    session_id: u64,
    ordinal: u64,
}

impl Default for StationTracker {
    fn default() -> Self {
        Self {
            mode: DecoderMode::Ft8,
            stations: BTreeMap::new(),
            logs: VecDeque::new(),
            hash12_resolutions: BTreeMap::new(),
            hash22_resolutions: BTreeMap::new(),
        }
    }
}

impl WorkQueueState {
    fn set_current_band(&mut self, band: Option<String>) {
        self.current_band = band;
    }

    fn set_current_mode(&mut self, mode: DecoderMode) {
        self.current_mode = mode;
    }

    fn worked_band_key(callsign: &str, band: &str) -> WorkedBandKey {
        WorkedBandKey {
            callsign: callsign.to_string(),
            band: band.to_string(),
        }
    }

    fn new(
        config: &AppConfig,
        tx_freq_hz: f32,
        recent_worked: BTreeMap<WorkedBandKey, SystemTime>,
    ) -> Self {
        Self {
            auto_enabled: false,
            our_call: config.station.our_call.clone(),
            current_band: None,
            current_mode: DecoderMode::Ft8,
            auto_add_all_decoded_calls: config.queue.auto_add_all_decoded_calls_default,
            auto_add_decoded_min_count_5m: config
                .queue
                .auto_add_decoded_min_count_5m_default
                .max(1),
            auto_add_direct_calls: config.queue.auto_add_direct_calls_default,
            ignore_direct_calls_from_recently_worked: config
                .queue
                .ignore_direct_calls_from_recently_worked_default,
            cq_enabled: config.queue.cq_enabled_default,
            cq_percent: config.queue.cq_percent_default.min(100),
            pause_cq_when_few_unique_calls: config.queue.pause_cq_when_few_unique_calls_default,
            cq_pause_min_unique_calls_5m: config.queue.cq_pause_min_unique_calls_5m_default,
            use_compound_rr73_handoff: config.queue.use_compound_rr73_handoff_default,
            use_compound_73_once_handoff: config.queue.use_compound_73_once_handoff_default,
            use_compound_for_direct_signal_callers: config
                .queue
                .use_compound_for_direct_signal_callers_default,
            next_cq_parity_flipped: false,
            even_tx_freq_hz: tx_freq_hz,
            odd_tx_freq_hz: tx_freq_hz,
            no_message_retry_delay: Duration::from_secs(
                config.queue.no_message_retry_delay_seconds_default,
            ),
            no_forward_retry_delay: Duration::from_secs(
                config.queue.no_forward_retry_delay_seconds_default,
            ),
            entries: VecDeque::new(),
            recent_worked,
            scheduler_status: "auto disabled".to_string(),
        }
    }

    fn add_station(
        &mut self,
        callsign: &str,
        last_observed_at: SystemTime,
        now: SystemTime,
    ) -> Result<(), String> {
        self.prune_recent_worked(now);
        if callsign.eq_ignore_ascii_case(&self.our_call) {
            info!(callsign, "queue_add_rejected_own_call");
            return Err("cannot queue our own call".to_string());
        }
        if self.was_worked_recently_on_current_band(callsign, now) {
            info!(callsign, "queue_add_rejected_recently_worked");
            return Err("station worked on this band in last 24h".to_string());
        }
        if self.entries.iter().any(|entry| entry.callsign == callsign) {
            info!(callsign, "queue_add_ignored_duplicate");
            return Ok(());
        }
        self.entries.push_back(WorkQueueEntry {
            callsign: callsign.to_string(),
            queued_at: now,
            ok_to_schedule_after: now,
            last_observed_at,
            direct_pending: false,
            direct_count: 0,
            last_direct_at: None,
            last_direct_slot_index: None,
            last_direct_slot_family: None,
            last_direct_snr_db: None,
            direct_start_state: None,
            direct_compound_eligible: false,
            last_direct_text: None,
            last_direct_structured_json: None,
        });
        info!(callsign, queued_at = %format_time(now), "queue_add_accepted");
        Ok(())
    }

    fn add_direct_observation(
        &mut self,
        observation: DirectCallObservation,
        now: SystemTime,
    ) -> Result<(), String> {
        self.prune_recent_worked(now);
        if observation.callsign.eq_ignore_ascii_case(&self.our_call) {
            info!(
                callsign = observation.callsign,
                "queue_direct_rejected_own_call"
            );
            return Err("cannot queue our own call".to_string());
        }
        if self.ignore_direct_calls_from_recently_worked
            && self.was_worked_recently_on_current_band(&observation.callsign, now)
        {
            info!(
                callsign = observation.callsign,
                "queue_direct_rejected_recently_worked"
            );
            return Err("direct caller worked on this band in last 24h".to_string());
        }
        if let Some(entry) = self
            .entries
            .iter_mut()
            .find(|entry| entry.callsign == observation.callsign)
        {
            let duplicate_slot = entry.last_direct_slot_index == Some(observation.slot_index)
                && entry.last_direct_slot_family == Some(observation.slot_family);
            entry.last_observed_at = entry.last_observed_at.max(observation.observed_at);
            entry.direct_pending = true;
            entry.last_direct_at = Some(observation.observed_at);
            entry.last_direct_slot_index = Some(observation.slot_index);
            entry.last_direct_slot_family = Some(observation.slot_family);
            entry.last_direct_snr_db = Some(
                entry
                    .last_direct_snr_db
                    .map(|existing| existing.max(observation.snr_db))
                    .unwrap_or(observation.snr_db),
            );
            entry.direct_start_state = Some(observation.start_state);
            entry.direct_compound_eligible = observation.compound_eligible;
            entry.last_direct_text = Some(observation.text.clone());
            entry.last_direct_structured_json = Some(observation.structured_json.clone());
            entry.ok_to_schedule_after = now;
            if !duplicate_slot {
                entry.direct_count = entry.direct_count.saturating_add(1);
            }
            info!(
                callsign = entry.callsign,
                direct_count = entry.direct_count,
                start_state = entry.direct_start_state.map(QsoState::as_str).unwrap_or(""),
                duplicate_slot,
                "queue_direct_updated"
            );
            return Ok(());
        }
        self.entries.push_back(WorkQueueEntry {
            callsign: observation.callsign.clone(),
            queued_at: now,
            ok_to_schedule_after: now,
            last_observed_at: observation.observed_at,
            direct_pending: true,
            direct_count: 1,
            last_direct_at: Some(observation.observed_at),
            last_direct_slot_index: Some(observation.slot_index),
            last_direct_slot_family: Some(observation.slot_family),
            last_direct_snr_db: Some(observation.snr_db),
            direct_start_state: Some(observation.start_state),
            direct_compound_eligible: observation.compound_eligible,
            last_direct_text: Some(observation.text),
            last_direct_structured_json: Some(observation.structured_json),
        });
        info!(
            callsign = observation.callsign,
            queued_at = %format_time(now),
            "queue_direct_added"
        );
        Ok(())
    }

    fn best_priority_direct_index_excluding(
        &self,
        now: SystemTime,
        excluded_callsign: Option<&str>,
    ) -> Option<usize> {
        let most_recent_slot = self
            .entries
            .iter()
            .filter(|entry| {
                excluded_callsign
                    .is_none_or(|excluded| !entry.callsign.eq_ignore_ascii_case(excluded))
            })
            .filter_map(|entry| self.direct_priority_meta(entry, now))
            .map(|meta| meta.slot_index)
            .max()?;
        let target_slot = if self
            .entries
            .iter()
            .filter(|entry| {
                excluded_callsign
                    .is_none_or(|excluded| !entry.callsign.eq_ignore_ascii_case(excluded))
            })
            .filter_map(|entry| self.direct_priority_meta(entry, now))
            .any(|meta| meta.slot_index == most_recent_slot)
        {
            most_recent_slot
        } else {
            most_recent_slot.saturating_sub(2)
        };
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                excluded_callsign
                    .is_none_or(|excluded| !entry.callsign.eq_ignore_ascii_case(excluded))
            })
            .filter_map(|(index, entry)| {
                let meta = self.direct_priority_meta(entry, now)?;
                if meta.slot_index != target_slot {
                    return None;
                }
                Some((index, meta))
            })
            .max_by(|(_, left), (_, right)| {
                left.count
                    .cmp(&right.count)
                    .then_with(|| left.snr_db.cmp(&right.snr_db))
                    .then_with(|| right.queued_at.cmp(&left.queued_at))
                    .then_with(|| right.callsign.cmp(&left.callsign))
            })
            .map(|(index, _)| index)
    }

    fn remove_station(&mut self, callsign: &str, reason: &str) -> bool {
        if let Some(index) = self
            .entries
            .iter()
            .position(|entry| entry.callsign == callsign)
        {
            self.entries.remove(index);
            info!(callsign, reason, "queue_remove");
            true
        } else {
            false
        }
    }

    fn clear(&mut self, reason: &str) {
        let removed = self.entries.len();
        self.entries.clear();
        info!(removed, reason, "queue_cleared");
        self.scheduler_status = if self.auto_enabled {
            "queue empty".to_string()
        } else {
            "auto disabled".to_string()
        };
    }

    fn set_auto_enabled(&mut self, enabled: bool) {
        self.auto_enabled = enabled;
        self.scheduler_status = if enabled {
            "auto enabled".to_string()
        } else {
            "auto disabled".to_string()
        };
        info!(enabled, "queue_auto_changed");
    }

    fn set_auto_add_all_decoded_calls(&mut self, enabled: bool) {
        self.auto_add_all_decoded_calls = enabled;
        info!(enabled, "queue_auto_add_all_decoded_changed");
    }

    fn set_auto_add_decoded_min_count_5m(&mut self, count: u32) {
        self.auto_add_decoded_min_count_5m = count.max(1);
        info!(
            min_count_5m = self.auto_add_decoded_min_count_5m,
            "queue_auto_add_decoded_min_count_changed"
        );
    }

    fn set_auto_add_direct_calls(&mut self, enabled: bool) {
        self.auto_add_direct_calls = enabled;
        info!(enabled, "queue_auto_add_direct_changed");
    }

    fn set_ignore_direct_calls_from_recently_worked(&mut self, enabled: bool) {
        self.ignore_direct_calls_from_recently_worked = enabled;
        info!(enabled, "queue_ignore_direct_recently_worked_changed");
    }

    fn set_cq_enabled(&mut self, enabled: bool) {
        self.cq_enabled = enabled;
        info!(enabled, "queue_cq_enabled_changed");
    }

    fn set_cq_percent(&mut self, percent: u8) {
        self.cq_percent = percent.min(100);
        info!(cq_percent = self.cq_percent, "queue_cq_percent_changed");
    }

    fn set_pause_cq_when_few_unique_calls(&mut self, enabled: bool) {
        self.pause_cq_when_few_unique_calls = enabled;
        info!(enabled, "queue_pause_cq_low_activity_changed");
    }

    fn set_cq_pause_min_unique_calls_5m(&mut self, count: u32) {
        self.cq_pause_min_unique_calls_5m = count;
        info!(
            min_unique_calls_5m = self.cq_pause_min_unique_calls_5m,
            "queue_cq_low_activity_threshold_changed"
        );
    }

    fn set_use_compound_rr73_handoff(&mut self, enabled: bool) {
        self.use_compound_rr73_handoff = enabled;
        info!(enabled, "queue_compound_rr73_handoff_changed");
    }

    fn set_use_compound_73_once_handoff(&mut self, enabled: bool) {
        self.use_compound_73_once_handoff = enabled;
        info!(enabled, "queue_compound_73_once_handoff_changed");
    }

    fn set_use_compound_for_direct_signal_callers(&mut self, enabled: bool) {
        self.use_compound_for_direct_signal_callers = enabled;
        info!(enabled, "queue_compound_direct_signal_changed");
    }

    fn use_compound_73_once_handoff(&self) -> bool {
        self.use_compound_73_once_handoff
    }

    fn toggle_next_cq_parity(&mut self) {
        self.next_cq_parity_flipped = !self.next_cq_parity_flipped;
        info!(
            next_cq_parity_flipped = self.next_cq_parity_flipped,
            "queue_next_cq_parity_toggled"
        );
    }

    fn set_tx_freq_hz(&mut self, slot_family: qso::SlotFamily, tx_freq_hz: f32) {
        match slot_family {
            qso::SlotFamily::Even => self.even_tx_freq_hz = tx_freq_hz,
            qso::SlotFamily::Odd => self.odd_tx_freq_hz = tx_freq_hz,
        }
        info!(
            slot_family = slot_family.as_str(),
            tx_freq_hz, "queue_tx_freq_changed"
        );
    }

    fn tx_freq_hz_for(&self, slot_family: qso::SlotFamily) -> f32 {
        match slot_family {
            qso::SlotFamily::Even => self.even_tx_freq_hz,
            qso::SlotFamily::Odd => self.odd_tx_freq_hz,
        }
    }

    fn set_retry_delay(&mut self, kind: QueueRetryDelayKind, retry_delay: Duration) {
        match kind {
            QueueRetryDelayKind::NoMessage => self.no_message_retry_delay = retry_delay,
            QueueRetryDelayKind::NoForward => self.no_forward_retry_delay = retry_delay,
        }
        info!(
            kind = kind.as_str(),
            retry_delay_seconds = retry_delay.as_secs(),
            "queue_retry_delay_changed"
        );
    }

    fn sync_recent_worked(&mut self, recent_worked: BTreeMap<WorkedBandKey, SystemTime>) {
        for (key, worked_at) in recent_worked {
            match self.recent_worked.get(&key).copied() {
                Some(existing) if existing >= worked_at => {}
                _ => {
                    self.recent_worked.insert(key, worked_at);
                }
            }
        }
    }

    fn prune_recent_worked(&mut self, now: SystemTime) {
        self.recent_worked.retain(|key, worked_at| {
            let keep =
                now.duration_since(*worked_at).unwrap_or_default() <= RECENT_WORKED_RETENTION;
            if !keep {
                info!(
                    callsign = key.callsign,
                    band = key.band,
                    worked_at = %format_time(*worked_at),
                    "recent_worked_expired"
                );
            }
            keep
        });
    }

    fn mark_worked(&mut self, callsign: &str, band: &str, worked_at: SystemTime) {
        self.recent_worked
            .insert(Self::worked_band_key(callsign, band), worked_at);
        info!(
            callsign,
            band,
            worked_at = %format_time(worked_at),
            "recent_worked_updated"
        );
    }

    fn was_worked_recently(&self, callsign: &str, band: Option<&str>, now: SystemTime) -> bool {
        let Some(band) = band else {
            return false;
        };
        self.recent_worked
            .get(&Self::worked_band_key(callsign, band))
            .is_some_and(|worked_at| {
                now.duration_since(*worked_at).unwrap_or_default() <= RECENT_WORKED_RETENTION
            })
    }

    fn was_worked_recently_on_current_band(&self, callsign: &str, now: SystemTime) -> bool {
        self.was_worked_recently(callsign, self.current_band.as_deref(), now)
    }

    fn handle_qso_outcome(&mut self, outcome: &QsoOutcome, tracker: &StationTracker) {
        if outcome.sent_terminal_73 {
            if let Some(band) = outcome.rig_band.as_deref() {
                self.mark_worked(&outcome.partner_call, band, outcome.finished_at);
            }
            self.remove_station(&outcome.partner_call, "already_worked");
        }
        match outcome.exit_reason.as_str() {
            "send_grid_no_msg_limit" | "send_sig_no_msg_limit" => {
                self.remove_station(&outcome.partner_call, "requeue_replace");
                let last_observed_at = tracker
                    .start_info(&outcome.partner_call)
                    .map(|info| info.last_heard_at)
                    .unwrap_or(outcome.finished_at);
                self.entries.push_back(WorkQueueEntry {
                    callsign: outcome.partner_call.clone(),
                    queued_at: outcome.finished_at,
                    ok_to_schedule_after: outcome.finished_at + self.no_message_retry_delay,
                    last_observed_at,
                    direct_pending: false,
                    direct_count: 0,
                    last_direct_at: None,
                    last_direct_slot_index: None,
                    last_direct_slot_family: None,
                    last_direct_snr_db: None,
                    direct_start_state: None,
                    direct_compound_eligible: false,
                    last_direct_text: None,
                    last_direct_structured_json: None,
                });
                info!(
                    callsign = outcome.partner_call,
                    ok_to_schedule_after = %format_time(outcome.finished_at + self.no_message_retry_delay),
                    exit_reason = outcome.exit_reason,
                    "queue_requeue_after_no_msg"
                );
            }
            "send_grid_no_fwd_limit" | "send_sig_no_fwd_limit" => {
                self.remove_station(&outcome.partner_call, "requeue_replace");
                let last_observed_at = tracker
                    .start_info(&outcome.partner_call)
                    .map(|info| info.last_heard_at)
                    .unwrap_or(outcome.finished_at);
                self.entries.push_back(WorkQueueEntry {
                    callsign: outcome.partner_call.clone(),
                    queued_at: outcome.finished_at,
                    ok_to_schedule_after: outcome.finished_at + self.no_forward_retry_delay,
                    last_observed_at,
                    direct_pending: false,
                    direct_count: 0,
                    last_direct_at: None,
                    last_direct_slot_index: None,
                    last_direct_slot_family: None,
                    last_direct_snr_db: None,
                    direct_start_state: None,
                    direct_compound_eligible: false,
                    last_direct_text: None,
                    last_direct_structured_json: None,
                });
                info!(
                    callsign = outcome.partner_call,
                    ok_to_schedule_after = %format_time(outcome.finished_at + self.no_forward_retry_delay),
                    exit_reason = outcome.exit_reason,
                    "queue_requeue_after_no_fwd"
                );
            }
            _ => {
                info!(
                    callsign = outcome.partner_call,
                    exit_reason = outcome.exit_reason,
                    "queue_drop_after_qso_exit"
                );
            }
        }
    }

    fn scheduler_pick(
        &mut self,
        now: SystemTime,
        tracker: &StationTracker,
        qso_busy: bool,
        tune_active: bool,
    ) -> Option<QueueDispatch> {
        let unique_calls_last_5m = tracker.unique_sender_count_since_excluding(
            now.checked_sub(CQ_ACTIVITY_WINDOW).unwrap_or(now),
            &self.our_call,
        );
        self.prune_recent_worked(now);
        self.refresh_entry_observed_times(tracker);
        self.prune_queue(now, tracker);
        if !self.auto_enabled {
            self.scheduler_status = "auto disabled".to_string();
            return None;
        }
        if qso_busy {
            self.scheduler_status = "waiting: qso active".to_string();
            return None;
        }
        if tune_active {
            self.scheduler_status = "waiting: rig tune active".to_string();
            return None;
        }
        if let Some(dispatch) = self.pick_priority_direct(now) {
            self.scheduler_status = format!("dispatching direct {}", dispatch.callsign);
            info!(
                callsign = dispatch.callsign,
                tx_slot_family = dispatch.tx_slot_family.as_str(),
                tx_freq_hz = dispatch.tx_freq_hz,
                "queue_dispatch_direct"
            );
            return Some(dispatch);
        }
        let normal_pick = self.pick_oldest_ready_normal(now, tracker);
        let cq_paused_for_low_activity = self.cq_paused_for_low_activity(unique_calls_last_5m);
        if self.cq_enabled && !cq_paused_for_low_activity && self.should_choose_cq(now) {
            let tx_slot_family = if self.next_cq_parity_flipped {
                Some(
                    qso::slot_family_for_mode(
                        self.current_mode,
                        next_slot_boundary_for_mode(self.current_mode, now),
                    )
                    .opposite(),
                )
            } else {
                None
            }
            .unwrap_or_else(|| {
                qso::slot_family_for_mode(
                    self.current_mode,
                    next_slot_boundary_for_mode(self.current_mode, now),
                )
            });
            self.next_cq_parity_flipped = false;
            self.scheduler_status = if normal_pick.is_some() {
                format!("dispatching cq at {}%", self.cq_percent)
            } else {
                format!("cq idle roll at {}%", self.cq_percent)
            };
            info!(
                cq_percent = self.cq_percent,
                tx_slot_family = tx_slot_family.as_str(),
                tx_freq_hz = self.tx_freq_hz_for(tx_slot_family),
                "queue_dispatch_cq"
            );
            return Some(QueueDispatch {
                kind: QueueDispatchKind::Cq {
                    tx_slot_family_override: Some(tx_slot_family),
                },
                callsign: "CQ".to_string(),
                tx_slot_family,
                tx_freq_hz: self.tx_freq_hz_for(tx_slot_family),
            });
        }
        if let Some(dispatch) = normal_pick {
            self.scheduler_status = format!("dispatching {}", dispatch.callsign);
            info!(
                callsign = dispatch.callsign,
                tx_slot_family = dispatch.tx_slot_family.as_str(),
                tx_freq_hz = dispatch.tx_freq_hz,
                "queue_dispatch"
            );
            return Some(dispatch);
        }
        self.scheduler_status = if self.entries.is_empty() {
            if self.cq_enabled && cq_paused_for_low_activity {
                format!(
                    "idle; cq paused at {}/{} unique calls in 5m",
                    unique_calls_last_5m, self.cq_pause_min_unique_calls_5m
                )
            } else if self.cq_enabled {
                format!("idle; cq skipped at {}%", self.cq_percent)
            } else {
                "queue empty".to_string()
            }
        } else if cq_paused_for_low_activity {
            format!(
                "no ready queued stations; cq paused at {}/{} unique calls in 5m",
                unique_calls_last_5m, self.cq_pause_min_unique_calls_5m
            )
        } else {
            "no ready queued stations".to_string()
        };
        None
    }

    fn web_snapshot(&self, tracker: &StationTracker, now: SystemTime) -> WebQueueSnapshot {
        let unique_calls_last_5m = tracker.unique_sender_count_since_excluding(
            now.checked_sub(CQ_ACTIVITY_WINDOW).unwrap_or(now),
            &self.our_call,
        ) as u32;
        let mut entries = self
            .entries
            .iter()
            .map(|entry| {
                let status = self.entry_status(entry, tracker, now);
                let priority_direct = self.direct_priority_meta(entry, now).is_some();
                WebQueueEntry {
                    callsign: entry.callsign.clone(),
                    queued_at: format_time(entry.queued_at),
                    ok_to_schedule_after: format_time(entry.ok_to_schedule_after),
                    direct_pending: entry.direct_pending,
                    priority_direct,
                    direct_count: entry.direct_count,
                    direct_last_heard_at: entry.last_direct_at.map(format_time),
                    last_heard_at: status.last_heard_at.map(format_time),
                    last_heard_message: status.last_heard_message.as_str().to_string(),
                    last_heard_slot_family: status
                        .last_heard_slot_family
                        .map(|family| family.as_str().to_string()),
                    ready: status.ready,
                    status: status.status,
                }
            })
            .collect::<Vec<_>>();
        entries.sort_by_key(|entry| !entry.priority_direct);
        WebQueueSnapshot {
            auto_enabled: self.auto_enabled,
            auto_add_all_decoded_calls: self.auto_add_all_decoded_calls,
            auto_add_decoded_min_count_5m: self.auto_add_decoded_min_count_5m,
            auto_add_direct_calls: self.auto_add_direct_calls,
            ignore_direct_calls_from_recently_worked: self.ignore_direct_calls_from_recently_worked,
            cq_enabled: self.cq_enabled,
            cq_percent: self.cq_percent,
            pause_cq_when_few_unique_calls: self.pause_cq_when_few_unique_calls,
            cq_pause_min_unique_calls_5m: self.cq_pause_min_unique_calls_5m,
            unique_calls_last_5m,
            use_compound_rr73_handoff: self.use_compound_rr73_handoff,
            use_compound_73_once_handoff: self.use_compound_73_once_handoff,
            use_compound_for_direct_signal_callers: self.use_compound_for_direct_signal_callers,
            next_cq_parity_flipped: self.next_cq_parity_flipped,
            even_tx_freq_hz: self.even_tx_freq_hz,
            odd_tx_freq_hz: self.odd_tx_freq_hz,
            no_message_retry_delay_seconds: self.no_message_retry_delay.as_secs(),
            no_forward_retry_delay_seconds: self.no_forward_retry_delay.as_secs(),
            scheduler_status: self.scheduler_status.clone(),
            entries,
        }
    }

    fn refresh_entry_observed_times(&mut self, tracker: &StationTracker) {
        for entry in &mut self.entries {
            if let Some(info) = tracker.start_info(&entry.callsign) {
                if info.last_heard_at > entry.last_observed_at {
                    entry.last_observed_at = info.last_heard_at;
                }
            }
        }
    }

    fn prune_queue(&mut self, now: SystemTime, _tracker: &StationTracker) {
        let recent_worked = self.recent_worked.clone();
        let current_band = self.current_band.clone();
        let ignore_direct = self.ignore_direct_calls_from_recently_worked;
        self.entries.retain(|entry| {
            let worked_recently = current_band.as_deref().is_some_and(|band| {
                recent_worked
                    .get(&WorkQueueState::worked_band_key(&entry.callsign, band))
                    .is_some_and(|worked_at| {
                        now.duration_since(*worked_at).unwrap_or_default()
                            <= RECENT_WORKED_RETENTION
                    })
            });
            if worked_recently && (ignore_direct || !entry.direct_pending) {
                info!(
                    callsign = entry.callsign,
                    band = current_band.clone().unwrap_or_default(),
                    "queue_remove_already_worked"
                );
                return false;
            }
            if now
                .duration_since(entry.last_observed_at)
                .unwrap_or_default()
                > QUEUE_HEARD_RETENTION
            {
                info!(
                    callsign = entry.callsign,
                    last_observed_at = %format_time(entry.last_observed_at),
                    "queue_remove_stale_unheard"
                );
                return false;
            }
            true
        });
    }

    fn entry_status(
        &self,
        entry: &WorkQueueEntry,
        tracker: &StationTracker,
        now: SystemTime,
    ) -> QueueEntryStatus {
        if self.was_worked_recently_on_current_band(&entry.callsign, now)
            && (self.ignore_direct_calls_from_recently_worked || !entry.direct_pending)
        {
            return QueueEntryStatus {
                last_heard_at: Some(entry.last_observed_at),
                last_heard_message: StationLastMessageKind::Other,
                last_heard_slot_family: None,
                ready: false,
                status: "already worked on this band in last 24h".to_string(),
            };
        }
        if now < entry.ok_to_schedule_after {
            let wait = entry
                .ok_to_schedule_after
                .duration_since(now)
                .unwrap_or_default()
                .as_secs();
            return QueueEntryStatus {
                last_heard_at: Some(entry.last_observed_at),
                last_heard_message: StationLastMessageKind::Other,
                last_heard_slot_family: None,
                ready: false,
                status: format!("retry blocked for {wait}s"),
            };
        }
        let Some(station) = tracker.stations.get(&entry.callsign) else {
            return QueueEntryStatus {
                last_heard_at: Some(entry.last_observed_at),
                last_heard_message: StationLastMessageKind::Other,
                last_heard_slot_family: None,
                ready: false,
                status: "not currently visible".to_string(),
            };
        };
        let last_heard_at = Some(station.last_heard_at);
        let last_heard_message = station.last_message_kind;
        let last_heard_slot_family = Some(station.last_heard_slot_family);
        let latest_same_family_slot =
            latest_slot_index_for_family(self.current_mode, now, station.last_heard_slot_family);
        if latest_same_family_slot.saturating_sub(station.last_heard_slot_index) > 2 {
            return QueueEntryStatus {
                last_heard_at,
                last_heard_message,
                last_heard_slot_family,
                ready: false,
                status: "not heard in last two parity slots".to_string(),
            };
        }
        if !station.last_message_kind.is_ready_last_message() {
            return QueueEntryStatus {
                last_heard_at,
                last_heard_message,
                last_heard_slot_family,
                ready: false,
                status: format!(
                    "last heard message {} is not schedulable",
                    station.last_message_kind.as_str()
                ),
            };
        }
        QueueEntryStatus {
            last_heard_at,
            last_heard_message,
            last_heard_slot_family,
            ready: true,
            status: if entry.direct_pending {
                format!("direct x{}", entry.direct_count)
            } else {
                "ready".to_string()
            },
        }
    }

    fn has_recent_priority_direct_for_slot(&self, slot_start: SystemTime, now: SystemTime) -> bool {
        let slot_index = slot_index_for_mode(self.current_mode, slot_start);
        self.entries.iter().any(|entry| {
            entry.direct_pending
                && now >= entry.ok_to_schedule_after
                && entry.last_direct_slot_index == Some(slot_index)
                && !(self.ignore_direct_calls_from_recently_worked
                    && self.was_worked_recently_on_current_band(&entry.callsign, now))
        })
    }

    fn best_priority_direct_index(&self, now: SystemTime) -> Option<usize> {
        self.best_priority_direct_index_excluding(now, None)
    }

    fn direct_dispatch_from_entry(&self, entry: &WorkQueueEntry, now: SystemTime) -> QueueDispatch {
        let tx_slot_family = entry
            .last_direct_slot_family
            .map(|family| family.opposite())
            .unwrap_or(qso::slot_family_for_mode(
                self.current_mode,
                next_slot_boundary_for_mode(self.current_mode, now),
            ));
        QueueDispatch {
            kind: QueueDispatchKind::Station {
                callsign: entry.callsign.clone(),
                initial_state: entry.direct_start_state.unwrap_or(QsoState::SendSig),
                start_mode: QsoStartMode::Direct,
                context_last_heard_at: entry.last_direct_at,
                context_last_heard_slot_family: entry.last_direct_slot_family,
                context_text: entry.last_direct_text.clone(),
                context_structured_json: entry.last_direct_structured_json.clone(),
                context_snr_db: entry.last_direct_snr_db,
            },
            callsign: entry.callsign.clone(),
            tx_slot_family,
            tx_freq_hz: self.tx_freq_hz_for(tx_slot_family),
        }
    }

    fn peek_compound_handoff_candidate(
        &self,
        now: SystemTime,
        excluded_callsign: Option<&str>,
    ) -> Option<QueueDispatch> {
        if !self.use_compound_rr73_handoff && !self.use_compound_73_once_handoff {
            return None;
        }
        let index = self.best_priority_direct_index_excluding(now, excluded_callsign)?;
        let entry = self.entries.get(index)?;
        let eligible = entry.direct_compound_eligible
            || (self.use_compound_for_direct_signal_callers
                && entry.direct_start_state == Some(QsoState::SendSigAck));
        if !eligible {
            return None;
        }
        Some(self.direct_dispatch_from_entry(entry, now))
    }

    fn pick_priority_direct(&mut self, now: SystemTime) -> Option<QueueDispatch> {
        let best_index = self.best_priority_direct_index(now)?;
        let entry = self.entries.remove(best_index).expect("queue index valid");
        Some(self.direct_dispatch_from_entry(&entry, now))
    }

    fn pick_oldest_ready_normal(
        &mut self,
        now: SystemTime,
        tracker: &StationTracker,
    ) -> Option<QueueDispatch> {
        let index = self
            .entries
            .iter()
            .position(|entry| self.entry_status(entry, tracker, now).ready)?;
        let entry = self.entries.remove(index).expect("queue index valid");
        let tx_slot_family = tracker
            .start_info(&entry.callsign)
            .map(|info| info.last_heard_slot_family.opposite())
            .unwrap_or(qso::slot_family_for_mode(
                self.current_mode,
                next_slot_boundary_for_mode(self.current_mode, now),
            ));
        Some(QueueDispatch {
            kind: QueueDispatchKind::Station {
                callsign: entry.callsign.clone(),
                initial_state: QsoState::SendGrid,
                start_mode: QsoStartMode::Normal,
                context_last_heard_at: None,
                context_last_heard_slot_family: None,
                context_text: None,
                context_structured_json: None,
                context_snr_db: None,
            },
            callsign: entry.callsign,
            tx_slot_family,
            tx_freq_hz: self.tx_freq_hz_for(tx_slot_family),
        })
    }

    fn direct_priority_meta(
        &self,
        entry: &WorkQueueEntry,
        now: SystemTime,
    ) -> Option<DirectPriorityMeta> {
        if !entry.direct_pending || now < entry.ok_to_schedule_after {
            return None;
        }
        if self.ignore_direct_calls_from_recently_worked
            && self.was_worked_recently_on_current_band(&entry.callsign, now)
        {
            return None;
        }
        let slot_family = entry.last_direct_slot_family?;
        let slot_index = entry.last_direct_slot_index?;
        let latest_same_family_slot =
            latest_slot_index_for_family(self.current_mode, now, slot_family);
        if latest_same_family_slot.saturating_sub(slot_index) > 2 {
            return None;
        }
        Some(DirectPriorityMeta {
            callsign: entry.callsign.clone(),
            queued_at: entry.queued_at,
            slot_index,
            count: entry.direct_count,
            snr_db: entry.last_direct_snr_db.unwrap_or(i32::MIN),
        })
    }

    fn should_choose_cq(&self, now: SystemTime) -> bool {
        if !self.cq_enabled || self.cq_percent == 0 {
            return false;
        }
        let roll = ((slot_index_for_mode(self.current_mode, now)
            .wrapping_mul(1103515245)
            .wrapping_add(12345))
            % 100) as u8;
        roll < self.cq_percent
    }

    fn cq_paused_for_low_activity(&self, unique_calls_last_5m: usize) -> bool {
        self.pause_cq_when_few_unique_calls
            && unique_calls_last_5m < self.cq_pause_min_unique_calls_5m as usize
    }
}

#[derive(Debug, Clone)]
struct DirectPriorityMeta {
    callsign: String,
    queued_at: SystemTime,
    slot_index: u64,
    count: u32,
    snr_db: i32,
}

impl QsoJsonlCache {
    fn new(path: PathBuf, direct_calls_since: SystemTime) -> Self {
        Self {
            path,
            direct_calls_since,
            last_modified: None,
            last_len: 0,
            last_refresh_at: None,
            history: Vec::new(),
            direct_calls: Vec::new(),
            recent_worked: BTreeMap::new(),
        }
    }

    fn refresh(&mut self, now: SystemTime) {
        if let Some(last_refresh_at) = self.last_refresh_at
            && now.duration_since(last_refresh_at).unwrap_or_default() < QSO_JSONL_REFRESH_INTERVAL
        {
            return;
        }
        self.last_refresh_at = Some(now);
        let Ok(metadata) = std::fs::metadata(&self.path) else {
            self.history.clear();
            self.direct_calls.clear();
            self.recent_worked.clear();
            self.last_modified = None;
            self.last_len = 0;
            return;
        };
        let modified = metadata.modified().ok();
        let len = metadata.len();
        if self.last_modified == modified && self.last_len == len {
            return;
        }
        let Ok(contents) = std::fs::read_to_string(&self.path) else {
            return;
        };
        let scan = scan_qso_jsonl(&contents, now, self.direct_calls_since);
        self.history = scan.history;
        self.direct_calls = scan.direct_calls;
        self.recent_worked = scan.recent_worked;
        self.last_modified = modified;
        self.last_len = len;
    }

    fn apply_refresh(&mut self, refresh: QsoJsonlRefresh) {
        self.history = refresh.scan.history;
        self.direct_calls = refresh.scan.direct_calls;
        self.recent_worked = refresh.scan.recent_worked;
        self.last_modified = refresh.modified;
        self.last_len = refresh.len;
        self.last_refresh_at = Some(refresh.scanned_at);
    }
}

#[derive(Debug, Default)]
struct QsoJsonlScan {
    history: Vec<WebQsoHistoryEntry>,
    direct_calls: Vec<WebDirectCallLog>,
    recent_worked: BTreeMap<WorkedBandKey, SystemTime>,
}

#[derive(Debug)]
struct QsoJsonlRefresh {
    scanned_at: SystemTime,
    modified: Option<SystemTime>,
    len: u64,
    scan: QsoJsonlScan,
}

fn spawn_qso_jsonl_refresh_worker(
    path: PathBuf,
    direct_calls_since: SystemTime,
    mut last_modified: Option<SystemTime>,
    mut last_len: u64,
) -> mpsc::Receiver<QsoJsonlRefresh> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        loop {
            let now = SystemTime::now();
            if let Ok(metadata) = std::fs::metadata(&path) {
                let modified = metadata.modified().ok();
                let len = metadata.len();
                if (modified != last_modified || len != last_len)
                    && let Ok(contents) = std::fs::read_to_string(&path)
                {
                    let scan = scan_qso_jsonl(&contents, now, direct_calls_since);
                    if tx
                        .send(QsoJsonlRefresh {
                            scanned_at: now,
                            modified,
                            len,
                            scan,
                        })
                        .is_err()
                    {
                        break;
                    }
                    last_modified = modified;
                    last_len = len;
                }
            }
            thread::sleep(QSO_JSONL_REFRESH_INTERVAL);
        }
    });
    rx
}

#[derive(Debug)]
struct DecodeJob {
    slot_start: SystemTime,
    mode: DecoderMode,
    stage: DecodeStage,
    capture_end: SystemTime,
    samples: Vec<i16>,
    sample_rate_hz: u32,
    raw_path: Option<PathBuf>,
}

#[derive(Debug)]
enum DecodeEvent {
    Finished {
        slot_start: SystemTime,
        mode: DecoderMode,
        stage: DecodeStage,
        wall_ms: u128,
        worker_wait_ms: u128,
        decode_run_ms: u128,
        result: Result<StageDecodeReport, AppError>,
    },
}

#[derive(Debug)]
struct DecodeSummary {
    final_decodes: Vec<DecodedMessage>,
}

#[derive(Debug, Clone, Copy, Default)]
struct SlotStageState {
    early41: bool,
    early47: bool,
    full: bool,
}

impl SlotStageState {
    fn is_handled(self, stage: DecodeStage) -> bool {
        match stage {
            DecodeStage::Early41 => self.early41,
            DecodeStage::Early47 => self.early47,
            DecodeStage::Full => self.full,
        }
    }

    fn mark_handled(&mut self, stage: DecodeStage) {
        match stage {
            DecodeStage::Early41 => self.early41 = true,
            DecodeStage::Early47 => self.early47 = true,
            DecodeStage::Full => self.full = true,
        }
    }

    fn next_due_stage(
        self,
        mode: DecoderMode,
        profile: DecodeProfile,
        slot_start: SystemTime,
        latest_sample_time: Option<SystemTime>,
    ) -> Option<DecodeStage> {
        let latest_sample_time = latest_sample_time?;
        for stage in DecodeStage::ordered() {
            if !decode_stage_enabled_for_profile(mode, profile, stage) {
                continue;
            }
            if self.is_handled(stage) {
                continue;
            }
            let ready_at = stage_capture_end(mode, slot_start, stage).ok()?;
            if latest_sample_time >= ready_at {
                return Some(stage);
            }
            return None;
        }
        None
    }
}

#[derive(Debug, Default)]
struct DecodeCycleLogState {
    slot_start: Option<SystemTime>,
    mode: Option<DecoderMode>,
    early41_wall_ms: Option<u128>,
    early47_wall_ms: Option<u128>,
    full_wall_ms: Option<u128>,
    early41_worker_wait_ms: Option<u128>,
    early47_worker_wait_ms: Option<u128>,
    full_worker_wait_ms: Option<u128>,
    early41_decode_run_ms: Option<u128>,
    early47_decode_run_ms: Option<u128>,
    full_decode_run_ms: Option<u128>,
    early41_decode_count: Option<usize>,
    early47_decode_count: Option<usize>,
    full_decode_count: Option<usize>,
    status: Option<&'static str>,
    error_stage: Option<DecodeStage>,
    error_message: Option<String>,
}

impl DecodeCycleLogState {
    fn reset(&mut self) {
        *self = Self::default();
    }

    fn begin(&mut self, slot_start: SystemTime, mode: DecoderMode) {
        if self.slot_start != Some(slot_start) || self.mode != Some(mode) {
            self.reset();
            self.slot_start = Some(slot_start);
            self.mode = Some(mode);
        }
    }

    fn record_ok(
        &mut self,
        stage: DecodeStage,
        wall_ms: u128,
        worker_wait_ms: u128,
        decode_run_ms: u128,
        decode_count: usize,
    ) {
        self.record_timing(stage, wall_ms, worker_wait_ms, decode_run_ms);
        self.record_decode_count(stage, decode_count);
    }

    fn record_decode_failed(
        &mut self,
        stage: DecodeStage,
        wall_ms: u128,
        worker_wait_ms: u128,
        decode_run_ms: u128,
        error: &AppError,
    ) {
        self.record_timing(stage, wall_ms, worker_wait_ms, decode_run_ms);
        self.record_error("decode_failed", stage, error.to_string());
    }

    fn record_capture_failed(&mut self, stage: DecodeStage, error: &AppError) {
        self.record_error("capture_failed", stage, error.to_string());
    }

    fn record_dropped(&mut self, stage: DecodeStage, reason: &'static str) {
        self.record_error("dropped", stage, reason.to_string());
    }

    fn record_timing(
        &mut self,
        stage: DecodeStage,
        wall_ms: u128,
        worker_wait_ms: u128,
        decode_run_ms: u128,
    ) {
        match stage {
            DecodeStage::Early41 => {
                self.early41_wall_ms = Some(wall_ms);
                self.early41_worker_wait_ms = Some(worker_wait_ms);
                self.early41_decode_run_ms = Some(decode_run_ms);
            }
            DecodeStage::Early47 => {
                self.early47_wall_ms = Some(wall_ms);
                self.early47_worker_wait_ms = Some(worker_wait_ms);
                self.early47_decode_run_ms = Some(decode_run_ms);
            }
            DecodeStage::Full => {
                self.full_wall_ms = Some(wall_ms);
                self.full_worker_wait_ms = Some(worker_wait_ms);
                self.full_decode_run_ms = Some(decode_run_ms);
            }
        }
    }

    fn record_decode_count(&mut self, stage: DecodeStage, decode_count: usize) {
        match stage {
            DecodeStage::Early41 => self.early41_decode_count = Some(decode_count),
            DecodeStage::Early47 => self.early47_decode_count = Some(decode_count),
            DecodeStage::Full => self.full_decode_count = Some(decode_count),
        }
    }

    fn record_error(&mut self, status: &'static str, stage: DecodeStage, message: String) {
        self.status = Some(status);
        self.error_stage = Some(stage);
        self.error_message = Some(message);
    }

    fn emit(
        &self,
        slot_start: SystemTime,
        mode: DecoderMode,
        decode_profile: DecodeProfile,
        tx_margin_ms: Option<i128>,
    ) {
        info!(
            slot = %format_slot_time_for_mode(mode, slot_start),
            mode = %mode.as_str(),
            decode_profile = %decode_profile.as_str(),
            status = self.status.unwrap_or("ok"),
            early41_wall_ms = option_u128_i128(self.early41_wall_ms),
            early47_wall_ms = option_u128_i128(self.early47_wall_ms),
            full_wall_ms = option_u128_i128(self.full_wall_ms),
            early41_worker_wait_ms = option_u128_i128(self.early41_worker_wait_ms),
            early47_worker_wait_ms = option_u128_i128(self.early47_worker_wait_ms),
            full_worker_wait_ms = option_u128_i128(self.full_worker_wait_ms),
            early41_decode_run_ms = option_u128_i128(self.early41_decode_run_ms),
            early47_decode_run_ms = option_u128_i128(self.early47_decode_run_ms),
            full_decode_run_ms = option_u128_i128(self.full_decode_run_ms),
            tx_margin_ms = tx_margin_ms.unwrap_or(-1),
            early41_decode_count = option_usize_i64(self.early41_decode_count),
            early47_decode_count = option_usize_i64(self.early47_decode_count),
            full_decode_count = option_usize_i64(self.full_decode_count),
            error_stage = self.error_stage.map(DecodeStage::as_str).unwrap_or(""),
            error_message = self.error_message.as_deref().unwrap_or(""),
            "decode_cycle"
        );
    }
}

#[derive(Debug, Clone, Copy)]
struct ActiveDecodeJob {
    slot_start: SystemTime,
    stage: DecodeStage,
}

const INDEX_HTML: &str = r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ft8rx</title>
  <style>
    :root {
      --bg: #06111a;
      --panel: #0d1d29;
      --panel-2: #132838;
      --ink: #e7f2f7;
      --muted: #8fb0c0;
      --grid: #20394b;
      --accent: #6ad3ff;
      --good: #97f0a9;
      --warn: #ffd26a;
      --focus-station: #8fe9ff;
      --partner-station: #ffd26a;
      --font: "Iosevka Term", "SF Mono", "Menlo", monospace;
      --waterfall-width: 800px;
      --waterfall-shell-width: 840px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(74, 144, 226, 0.16), transparent 30%),
        linear-gradient(180deg, #071019 0%, #06111a 100%);
      color: var(--ink);
      font-family: var(--font);
    }
    .page {
      width: 100%;
      max-width: 1800px;
      margin: 0 auto;
      padding: 16px;
      display: grid;
      gap: 16px;
    }
    .top-layout {
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(320px, 0.85fr);
      gap: 16px;
      align-items: start;
    }
    .second-row {
      display: grid;
      grid-template-columns: repeat(2, minmax(320px, 1fr));
      gap: 16px;
      align-items: start;
    }
    .third-row {
      display: grid;
      grid-template-columns: minmax(320px, 0.85fr) minmax(0, 1.15fr);
      gap: 16px;
      align-items: start;
    }
    .top-main {
      display: grid;
      gap: 16px;
      width: 100%;
      min-width: 0;
    }
    .panel {
      background: rgba(13, 29, 41, 0.95);
      border: 1px solid rgba(143, 176, 192, 0.16);
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 14px 40px rgba(0, 0, 0, 0.24);
      margin: 0;
      min-width: 0;
      min-height: 0;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .status-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px 18px;
    }
    .label {
      font-size: 11px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 4px;
    }
    .value {
      font-size: 20px;
      line-height: 1.2;
    }
    .value.small {
      font-size: 15px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      min-width: 0;
    }
    #rig {
      white-space: pre-line;
    }
    .status-grid > div {
      min-width: 0;
    }
    #waterfall {
      width: 100%;
      max-width: var(--waterfall-width);
      height: 110px;
      display: block;
      margin: 0 auto;
      image-rendering: pixelated;
      background: #02070c;
      border-radius: 10px;
      border: 1px solid rgba(143, 176, 192, 0.12);
    }
    .status-panel,
    .waterfall-panel {
      width: 100%;
      max-width: none;
    }
    .maps {
      display: grid;
      grid-template-columns: 1fr;
      gap: 16px;
    }
    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .panel-head .label {
      margin-bottom: 0;
    }
    .map-grid {
      display: grid;
      grid-template-columns: repeat(10, minmax(0, 1fr));
      gap: 6px;
      margin-top: 10px;
    }
    .cell {
      min-height: 314px;
      background: linear-gradient(180deg, rgba(19, 40, 56, 0.95), rgba(10, 23, 34, 0.95));
      border: 1px solid rgba(143, 176, 192, 0.12);
      border-radius: 8px;
      padding: 6px;
      overflow: hidden;
    }
    .cell-title {
      font-size: 10px;
      color: var(--muted);
      margin-bottom: 4px;
    }
    .call {
      display: flex;
      align-items: center;
      gap: 6px;
      min-width: 0;
      font-size: 12px;
      line-height: 1.3;
      color: var(--good);
    }
    .call-text {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      min-width: 0;
    }
    .queue-tag {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      border: 1px solid rgba(255, 255, 255, 0.28);
      color: #ffffff;
      background: rgba(255, 255, 255, 0.08);
      font-size: 11px;
      font-weight: 700;
      cursor: pointer;
      flex: 0 0 auto;
    }
    .queue-tag.placeholder {
      visibility: hidden;
      pointer-events: none;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }
    th, td {
      padding: 8px 10px;
      border-bottom: 1px solid rgba(143, 176, 192, 0.09);
      text-align: left;
    }
    th {
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    td.num { text-align: right; }
    .seen-early { color: var(--good); }
    .seen-mid { color: var(--warn); }
    .seen-late { color: #ff9e80; }
    .meta-line {
      display: flex;
      justify-content: center;
      gap: 18px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 13px;
      margin-top: 8px;
    }
    .pickable {
      color: inherit;
      cursor: pointer;
      text-decoration: none;
    }
    .detail-panel {
      height: 740px;
    }
    .queue-panel,
    .qso-panel {
      height: 760px;
    }
    .queue-panel {
      display: grid;
      grid-template-rows: auto auto auto auto minmax(0, 1fr);
      align-content: stretch;
    }
    .direct-panel,
    .log-panel {
      height: 380px;
    }
    .detail-block {
      margin-top: 0;
      display: grid;
      gap: 8px;
      min-height: 0;
    }
    .detail-panel .detail-block:last-child {
      flex: 1 1 auto;
      display: flex;
      flex-direction: column;
      min-height: 0;
    }
    .queue-panel .detail-block:last-child,
    .qso-panel .detail-block:last-child,
    .direct-panel .detail-block:last-child,
    .log-panel .detail-block:last-child {
      flex: 1 1 auto;
      display: flex;
      flex-direction: column;
      min-height: 0;
    }
    .queue-panel .detail-block:last-child {
      overflow: hidden;
    }
    .detail-lines {
      color: var(--ink);
      font-size: 13px;
      line-height: 1.45;
      white-space: pre-wrap;
    }
    #detail-state {
      min-height: 3.2em;
    }
    #queue-status {
      min-height: 1.5em;
    }
    .history-list {
      max-height: none;
      overflow-y: auto;
      overflow-x: hidden;
      padding: 8px;
      height: 108px;
    }
    .detail-empty {
      color: var(--muted);
      font-size: 13px;
    }
    .scroll-surface {
      min-height: 0;
      border: 1px solid rgba(143, 176, 192, 0.1);
      border-radius: 10px;
      background: rgba(6, 17, 26, 0.55);
      padding: 6px 6px 18px;
      scrollbar-gutter: stable;
      scroll-padding-bottom: 18px;
    }
    .queue-panel .scroll-surface,
    .qso-panel .scroll-surface,
    .direct-panel .scroll-surface,
    .log-panel .scroll-surface {
      flex: 1 1 auto;
      display: flex;
      min-height: 0;
    }
    .activity-list {
      margin-top: 0;
      display: flex;
      flex-direction: column;
      align-items: stretch;
      gap: 4px;
      max-height: none;
      overflow-y: auto;
      overflow-x: auto;
      min-height: 0;
      padding-bottom: 0;
    }
    .activity-list::after {
      content: "";
      display: block;
      flex: 0 0 18px;
      min-height: 18px;
    }
    #detail-logs {
      height: auto;
      flex: 1 1 auto;
      min-height: 0;
    }
    #direct-list {
      height: auto;
      flex: 1 1 auto;
      min-height: 0;
    }
    .activity-item {
      border: 1px solid rgba(143, 176, 192, 0.1);
      border-radius: 8px;
      padding: 4px 6px;
      background: rgba(19, 40, 56, 0.45);
      font-size: 12px;
      line-height: 1.15;
      display: grid;
      grid-template-columns: 64px 110px 78px 34px 46px 50px minmax(0, 1fr);
      gap: 6px;
      align-items: baseline;
      min-width: 520px;
    }
    .activity-head {
      position: sticky;
      top: 0;
      z-index: 1;
      border: 0;
      padding: 0 6px 2px;
      background: linear-gradient(180deg, rgba(13, 29, 41, 0.98), rgba(13, 29, 41, 0.92));
      font-size: 10px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .activity-col {
      color: var(--muted);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .activity-station {
      font-weight: 600;
      color: var(--focus-station);
    }
    .activity-partner {
      color: var(--partner-station);
    }
    .activity-msg {
      color: var(--ink);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .activity-item.direct-sent {
      background: rgba(38, 61, 84, 0.75);
      border-color: rgba(186, 225, 255, 0.18);
    }
    .activity-item.direct-sent .activity-col,
    .activity-item.direct-sent .activity-msg {
      color: #eef7ff;
    }
    .control-row {
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
      margin-top: 8px;
    }
    .control-inline {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 8px;
      align-items: end;
    }
    .control-inline.dual {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .input-wrap {
      display: grid;
      gap: 4px;
    }
    .control-input {
      width: 100%;
      border: 1px solid rgba(143, 176, 192, 0.18);
      border-radius: 9px;
      background: rgba(6, 17, 26, 0.9);
      color: var(--ink);
      padding: 10px 11px;
      font: inherit;
    }
    .button {
      border: 1px solid rgba(143, 176, 192, 0.18);
      border-radius: 10px;
      background: linear-gradient(180deg, rgba(31, 111, 145, 0.95), rgba(18, 69, 91, 0.95));
      color: var(--ink);
      padding: 10px 12px;
      font: inherit;
      cursor: pointer;
    }
    .button.secondary {
      background: linear-gradient(180deg, rgba(43, 55, 65, 0.95), rgba(22, 31, 38, 0.95));
      color: var(--muted);
    }
    .button.compact {
      padding: 2px 6px;
      min-height: 22px;
      border-radius: 7px;
      font-size: 11px;
      line-height: 1.1;
    }
    .button.warn {
      background: linear-gradient(180deg, rgba(153, 84, 33, 0.95), rgba(110, 56, 18, 0.95));
    }
    .button:disabled {
      cursor: not-allowed;
      opacity: 0.5;
    }
    .qso-summary {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 6px 12px;
      margin-top: 8px;
    }
    .qso-kv {
      display: flex;
      align-items: baseline;
      gap: 8px;
      min-width: 0;
      font-size: 12px;
      line-height: 1.3;
    }
    .qso-kv .label {
      margin-bottom: 0;
      flex: 0 0 auto;
    }
    .qso-status-line {
      color: var(--ink);
      font-size: 12px;
      line-height: 1.3;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      min-width: 0;
    }
    .qso-transcript {
      margin-top: 0;
      display: flex;
      flex-direction: column;
      align-items: stretch;
      gap: 4px;
      max-height: none;
      overflow-y: auto;
      overflow-x: hidden;
      min-height: 0;
      padding-bottom: 0;
    }
    .qso-transcript::after {
      content: "";
      display: block;
      flex: 0 0 18px;
      min-height: 18px;
    }
    #qso-transcript {
      height: auto;
      flex: 1 1 auto;
      min-height: 0;
    }
    .qso-entry {
      border: 1px solid rgba(143, 176, 192, 0.1);
      border-radius: 8px;
      padding: 4px 6px;
      background: rgba(19, 40, 56, 0.45);
      display: grid;
      grid-template-columns: 62px 42px 82px minmax(0, 1fr);
      gap: 6px;
      font-size: 11px;
      line-height: 1.15;
    }
    .qso-entry .stamp,
    .qso-entry .dir,
    .qso-entry .state {
      color: var(--muted);
      white-space: nowrap;
    }
    .qso-entry .msg {
      color: var(--ink);
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .hint {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
      min-height: 2.8em;
    }
    .toggle-row {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--ink);
      font-size: 13px;
    }
    .queue-list {
      margin-top: 0;
      display: flex;
      flex-direction: column;
      align-items: stretch;
      gap: 2px;
      max-height: none;
      overflow-y: auto;
      overflow-x: auto;
      min-height: 0;
      font-size: 11px;
      line-height: 1.15;
      padding-bottom: 0;
    }
    .queue-list::after {
      content: "";
      display: block;
      flex: 0 0 18px;
      min-height: 18px;
    }
    #queue-list {
      height: auto;
      flex: 1 1 auto;
      min-height: 0;
    }
    .queue-item {
      display: grid;
      grid-template-columns: 78px 30px 36px 88px 88px 88px 58px 44px minmax(0, 1fr) 56px;
      gap: 6px;
      align-items: baseline;
      padding: 3px 6px;
      border: 1px solid rgba(143, 176, 192, 0.08);
      border-radius: 6px;
      background: rgba(19, 40, 56, 0.45);
      min-width: 760px;
    }
    .queue-head {
      position: sticky;
      top: 0;
      z-index: 1;
      background: linear-gradient(180deg, rgba(13, 29, 41, 0.98), rgba(13, 29, 41, 0.92));
      border: 0;
    }
    .queue-item-direct {
      border-color: rgba(244, 190, 92, 0.38);
      background: rgba(76, 61, 20, 0.42);
    }
    .queue-call {
      font-weight: 600;
      color: var(--focus-station);
    }
    .queue-meta {
      color: var(--muted);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .history-grid {
      margin-top: 0;
      display: flex;
      flex-direction: column;
      align-items: stretch;
      gap: 2px;
      max-height: none;
      overflow-y: auto;
      overflow-x: auto;
      min-height: 0;
      font-size: 11px;
      line-height: 1.15;
      padding-bottom: 0;
    }
    .history-grid::after {
      content: "";
      display: block;
      flex: 0 0 18px;
      min-height: 18px;
    }
    #qso-history-list {
      height: auto;
      flex: 1 1 auto;
      min-height: 0;
    }
    .history-row {
      display: grid;
      grid-template-columns: 126px 52px 64px 24px 28px 24px 18px 18px 64px 64px minmax(96px, 1fr);
      gap: 3px;
      align-items: baseline;
      padding: 2px 3px;
      border: 1px solid rgba(143, 176, 192, 0.08);
      border-radius: 6px;
      background: rgba(19, 40, 56, 0.45);
      min-width: 600px;
    }
    .history-head {
      position: sticky;
      top: 0;
      z-index: 1;
      background: linear-gradient(180deg, rgba(13, 29, 41, 0.98), rgba(13, 29, 41, 0.92));
      border: 0;
      padding: 0 6px 2px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      font-size: 10px;
    }
    .history-cell {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      color: var(--ink);
      min-width: 0;
    }
    .history-cell.muted {
      color: var(--muted);
    }
    .history-cell-exit {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      min-width: 0;
    }
    .table-scroll {
      min-height: 0;
      max-height: 360px;
      overflow: auto;
      border: 1px solid rgba(143, 176, 192, 0.1);
      border-radius: 10px;
      background: rgba(6, 17, 26, 0.55);
      scrollbar-gutter: stable;
    }
    .table-scroll table {
      min-width: 980px;
    }
    @media (max-width: 1100px) {
      .status-grid { grid-template-columns: 1fr; }
      .top-layout { grid-template-columns: 1fr; }
      .second-row { grid-template-columns: 1fr; }
      .third-row { grid-template-columns: 1fr; }
      .maps { grid-template-columns: 1fr; }
      .detail-panel,
      .queue-panel,
      .direct-panel,
      .qso-panel,
      .log-panel {
        height: auto;
      }
      .history-list,
      #detail-logs,
      #direct-list,
      #qso-transcript,
      #queue-list,
      #qso-history-list {
        height: auto;
      }
    }
  </style>
</head>
<body>
  <div class="page">
    <div class="top-layout">
      <div class="top-main">
        <section class="panel status-panel">
          <div class="status-grid">
            <div><div class="label">UTC</div><div class="value" id="time"></div></div>
            <div><div class="label">Rig</div><div class="value small" id="rig"></div></div>
            <div><div class="label">Audio</div><div class="value small" id="audio"></div></div>
            <div><div class="label">Status</div><div class="value small" id="status"></div></div>
            <div><div class="label">Slot</div><div class="value small" id="slot"></div></div>
            <div><div class="label">Decode</div><div class="value small" id="times"></div></div>
            <div><div class="label">dT</div><div class="value small" id="dtstats"></div></div>
            <div><div class="label">Decodes</div><div class="value small" id="count"></div></div>
          </div>
          <div class="detail-block">
            <div class="label">Rig Control</div>
            <div class="control-row">
              <div class="control-inline dual">
                <div class="input-wrap">
                  <div class="label">Band</div>
                  <select id="rig-band" class="control-input"></select>
                </div>
                <div class="input-wrap">
                  <div class="label">Mode</div>
                  <select id="rig-app-mode" class="control-input"></select>
                </div>
                <div class="input-wrap">
                  <div class="label">Power</div>
                  <select id="rig-power" class="control-input"></select>
                </div>
              </div>
              <div class="control-inline">
                <button id="rig-tune" class="button warn" type="button">Tune 10s</button>
              </div>
              <div class="hint" id="rig-hint">Band and power apply immediately on selection.</div>
            </div>
          </div>
        </section>
        <section class="panel waterfall-panel">
          <canvas id="waterfall" width="300" height="180"></canvas>
          <div class="meta-line">
            <span>Waterfall: 0-4000 Hz</span>
            <span>Rows: latest at top</span>
          </div>
        </section>
      </div>
      <section class="panel detail-panel">
        <div class="label">Station Detail</div>
        <div class="value" id="detail-call">No station selected</div>
        <div class="control-row">
          <button id="detail-add-queue" class="button" type="button">Add To Queue</button>
          <div class="hint" id="detail-queue-hint">Select a station in the monitor pane to add it to the work queue.</div>
        </div>
        <div class="detail-block">
          <div class="label">Current State</div>
          <div class="detail-lines" id="detail-state">Click a callsign in the bandmap or decode table.</div>
        </div>
        <div class="detail-block">
          <div class="label">Recent QSOs</div>
          <div class="detail-lines history-list scroll-surface" id="detail-history"></div>
        </div>
        <div class="detail-block">
          <div class="label">Live Activity</div>
          <div class="activity-list scroll-surface" id="detail-logs">
            <div class="activity-item activity-head">
              <div class="activity-col">Time</div>
              <div class="activity-col">To</div>
              <div class="activity-col">De</div>
              <div class="activity-col">SNR</div>
              <div class="activity-col">dT</div>
              <div class="activity-col">Freq</div>
              <div class="activity-col">Msg</div>
            </div>
          </div>
        </div>
      </section>
    </div>
    <div class="second-row">
      <section class="panel queue-panel">
        <div class="label">Stations To Work</div>
        <div class="value" id="queue-count">0 queued</div>
        <div class="detail-block">
          <div class="label">Scheduler</div>
          <div class="detail-lines" id="queue-status">auto disabled</div>
        </div>
        <div class="detail-block">
          <div class="label">Queue Controls</div>
          <div class="control-inline">
            <div class="input-wrap">
              <div class="label">No Message Retry Delay</div>
              <input id="queue-no-message-retry-delay" class="control-input" type="number" min="1" max="3600" step="1">
            </div>
            <div class="input-wrap">
              <div class="label">No Fwd Retry Delay</div>
              <input id="queue-no-forward-retry-delay" class="control-input" type="number" min="1" max="3600" step="1">
            </div>
            <div class="input-wrap">
              <div class="label">CQ %</div>
              <select id="queue-cq-percent" class="control-input">
                <option value="0">0%</option>
                <option value="10">10%</option>
                <option value="20">20%</option>
                <option value="30">30%</option>
                <option value="40">40%</option>
                <option value="50">50%</option>
                <option value="60">60%</option>
                <option value="70">70%</option>
                <option value="80">80%</option>
                <option value="90">90%</option>
                <option value="100">100%</option>
              </select>
            </div>
            <div class="input-wrap">
              <div class="label">Auto-Add Min Decodes / 5m</div>
              <input id="queue-auto-add-decoded-min-count-5m" class="control-input" type="number" min="1" max="1000" step="1">
            </div>
            <div class="input-wrap">
              <div class="label">CQ 5m Unique Min</div>
              <input id="queue-cq-min-unique-calls-5m" class="control-input" type="number" min="0" max="1000" step="1">
            </div>
            <div class="input-wrap">
              <div class="label">5m Unique Now</div>
              <div class="value small" id="queue-unique-calls-last-5m">0</div>
            </div>
            <button id="queue-next-cq-parity" class="button secondary" type="button">Flip Next CQ Parity</button>
            <button id="queue-clear" class="button secondary" type="button">Clear Queue</button>
          </div>
        <div class="control-row">
          <label class="toggle-row"><input id="queue-auto-add-decoded" type="checkbox"> Auto add eligible decodes</label>
          <label class="toggle-row"><input id="queue-auto-add-direct" type="checkbox"> Auto add direct calls</label>
          <label class="toggle-row"><input id="queue-ignore-direct-worked" type="checkbox"> Ignore direct calls from already worked stations</label>
          <label class="toggle-row"><input id="queue-cq-enabled" type="checkbox"> Enable CQ</label>
          <label class="toggle-row"><input id="queue-pause-cq-low-activity" type="checkbox"> Pause CQ on low 5m activity</label>
          <label class="toggle-row"><input id="queue-compound-rr73" type="checkbox"> Use compound RR73 handoff</label>
          <label class="toggle-row"><input id="queue-compound-73-once" type="checkbox"> Use compound 73-once handoff</label>
          <label class="toggle-row"><input id="queue-compound-direct-signal" type="checkbox"> Use compound for direct signal callers</label>
        </div>
        </div>
        <div class="detail-block">
          <div class="label">Queue</div>
          <div class="queue-list scroll-surface" id="queue-list"></div>
        </div>
      </section>
      <section class="panel qso-panel">
        <div class="label">QSO</div>
        <div class="value" id="qso-call">Idle</div>
        <div class="control-row">
          <div class="control-inline">
            <div class="input-wrap">
              <div class="label">Even TX Hz</div>
              <input id="qso-freq-even" class="control-input" type="number" min="200" max="3500" step="1">
            </div>
            <button id="qso-autopick-even" class="button secondary" type="button">Auto-Pick Even</button>
            <div class="input-wrap">
              <div class="label">Odd TX Hz</div>
              <input id="qso-freq-odd" class="control-input" type="number" min="200" max="3500" step="1">
            </div>
            <button id="qso-autopick-odd" class="button secondary" type="button">Auto-Pick Odd</button>
          </div>
          <div class="control-inline dual">
            <label class="toggle-row"><input id="qso-auto" type="checkbox"> Auto QSO From Queue</label>
            <button id="qso-stop" class="button warn" type="button">Stop</button>
          </div>
          <div class="hint" id="qso-hint">Auto QSO starts the oldest ready station from the work queue.</div>
        </div>
        <div class="qso-summary">
          <div class="qso-kv"><div class="label">State</div><div class="qso-status-line" id="qso-state">idle</div></div>
          <div class="qso-kv"><div class="label">Timeout</div><div class="qso-status-line" id="qso-timeout">-</div></div>
          <div class="qso-kv"><div class="label">Counters</div><div class="qso-status-line" id="qso-counters">no_msg=0 no_fwd=0</div></div>
          <div class="qso-kv"><div class="label">Latest RX</div><div class="qso-status-line" id="qso-last-rx">-</div></div>
          <div class="qso-kv"><div class="label">TX Parity</div><div class="qso-status-line" id="qso-parity">-</div></div>
          <div class="qso-kv"><div class="label">Signal</div><div class="qso-status-line" id="qso-snr">-</div></div>
        </div>
        <div class="detail-block">
          <div class="label">Transcript</div>
          <div class="qso-transcript scroll-surface" id="qso-transcript"></div>
        </div>
      </section>
    </div>
    <div class="third-row">
      <section class="panel direct-panel">
          <div class="label">Direct Calls (1h)</div>
        <div class="value" id="direct-count">0 heard</div>
        <div class="detail-block">
          <div class="label">Messages To Or From Our Call</div>
          <div class="activity-list scroll-surface" id="direct-list"></div>
        </div>
      </section>
      <section class="panel log-panel">
        <div class="label">QSO Log (24h)</div>
        <div class="value" id="qso-history-count">0 QSOs</div>
        <div class="detail-block">
          <div class="label">QSOs From Last 24 Hours</div>
          <div class="label">Rpl = any reply to our call. 73 = reached RR73/73 sending state.</div>
          <div class="history-grid scroll-surface" id="qso-history-list"></div>
        </div>
      </section>
    </div>
    <div class="maps">
      <section class="panel">
        <div class="panel-head">
          <div class="label">Even Slots</div>
          <button id="queue-even-map" class="button secondary" type="button">Add To Queue</button>
        </div>
        <div id="even-map" class="map-grid"></div>
      </section>
      <section class="panel">
        <div class="panel-head">
          <div class="label">Odd Slots</div>
          <button id="queue-odd-map" class="button secondary" type="button">Add To Queue</button>
        </div>
        <div id="odd-map" class="map-grid"></div>
      </section>
    </div>
    <section class="panel">
      <div class="label">Recent Decodes</div>
      <div class="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Seen</th>
              <th>UTC</th>
              <th>SNR</th>
              <th>dT</th>
              <th>Freq</th>
              <th>Kind</th>
              <th>Field 1</th>
              <th>Field 2</th>
              <th>Info</th>
              <th>Message</th>
            </tr>
          </thead>
          <tbody id="decodes"></tbody>
        </table>
      </div>
    </section>
  </div>
  <script>
    const canvas = document.getElementById('waterfall');
    const ctx = canvas.getContext('2d');
    const BAND_OPTIONS = ['160m', '80m', '60m', '40m', '30m', '20m', '17m', '15m', '12m', '10m', '6m'];
    const APP_MODE_OPTIONS = ['FT8', 'FT4'];
    const POWER_OPTIONS = ['5', '10', '20', '50', '100'];
    let selectedCall = null;
    let lastSnapshot = null;
    let autoFollowLogs = true;
    let autoFollowQso = true;
    let autoFollowDirect = true;
    let pendingRigConfig = null;
    function initRigBandOptions() {
      const select = document.getElementById('rig-band');
      if (select.options.length) return;
      for (const band of BAND_OPTIONS) {
        const option = document.createElement('option');
        option.value = band;
        option.textContent = band;
        select.appendChild(option);
      }
    }
    function initRigPowerOptions(data) {
      const select = document.getElementById('rig-power');
      const isDiscrete = !!data?.rig_power_is_discrete;
      const desired = isDiscrete
        ? (data?.rig_power_settings || []).map((setting) => ({ value: setting.id, label: setting.label }))
        : POWER_OPTIONS.map((power) => ({ value: power, label: `${power} W` }));
      const existing = [...select.options].map((option) => ({ value: option.value, label: option.textContent }));
      if (JSON.stringify(existing) === JSON.stringify(desired)) {
        return;
      }
      select.replaceChildren();
      for (const entry of desired) {
        const option = document.createElement('option');
        option.value = entry.value;
        option.textContent = entry.label;
        select.appendChild(option);
      }
    }
    function initRigModeOptions() {
      const select = document.getElementById('rig-app-mode');
      if (select.options.length) return;
      for (const mode of APP_MODE_OPTIONS) {
        const option = document.createElement('option');
        option.value = mode;
        option.textContent = mode;
        select.appendChild(option);
      }
    }
    function fmtSec(value) {
      return value == null ? '-' : `${value.toFixed(2)}s`;
    }
    function pickCall(call) {
      if (!call) return;
      selectedCall = call;
      if (lastSnapshot) {
        renderDetail(lastSnapshot);
        renderQso(lastSnapshot);
      }
    }
    function renderCallValue(value, call) {
      if (!call) return value;
      return `<span class="pickable" data-call="${call}">${value}</span>`;
    }
    function escapeHtml(value) {
      return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }
    function renderParty(value, selected) {
      const text = value || '-';
      if (!selected) {
        return escapeHtml(text);
      }
      const cls = text === selected ? 'activity-station' : 'activity-partner';
      return `<span class="${cls}">${escapeHtml(text)}</span>`;
    }
    function stationMap(data) {
      return new Map((data.stations || []).map((entry) => [entry.callsign, entry]));
    }
    function selectedStation(data) {
      if (!selectedCall) return null;
      return stationMap(data).get(selectedCall) || null;
    }
    function inferredTxParity(station) {
      if (!station || !station.last_heard_slot_family) return null;
      return station.last_heard_slot_family === 'even' ? 'odd' : 'even';
    }
    function currentTxFreqValue(data, txParity) {
      const input = document.getElementById(txParity === 'even' ? 'qso-freq-even' : 'qso-freq-odd');
      const current = Number(input.value);
      if (Number.isFinite(current) && input.value !== '') {
        return current;
      }
      if (txParity === 'even' && data?.queue?.even_tx_freq_hz != null) {
        return data.queue.even_tx_freq_hz;
      }
      if (txParity === 'odd' && data?.queue?.odd_tx_freq_hz != null) {
        return data.queue.odd_tx_freq_hz;
      }
      return data?.qso_defaults?.tx_freq_default_hz ?? 1000;
    }
    function txFreqValid(data, value) {
      const min = data?.qso_defaults?.tx_freq_min_hz ?? 200;
      const max = data?.qso_defaults?.tx_freq_max_hz ?? 3500;
      return Number.isFinite(value) && value >= min && value <= max;
    }
    function quietSpotFrequency(data, txParity) {
      const defaults = data.qso_defaults || {};
      const min = Math.max(defaults.tx_freq_min_hz ?? 200, 350);
      const max = Math.min(defaults.tx_freq_max_hz ?? 3500, 2600);
      const fallback = defaults.tx_freq_default_hz ?? 1000;
      const grid = txParity === 'even' ? data.bandmaps?.even : data.bandmaps?.odd;
      if (!grid || !grid.length) return fallback;
      let best = null;
      for (let row = 0; row < grid.length; row++) {
        for (let col = 0; col < grid[row].length; col++) {
          const startHz = col * 400 + row * 400;
          const centerHz = startHz + 200;
          if (centerHz < min || centerHz > max) continue;
          const entries = grid[row][col] || [];
          const occupancy = entries.length;
          const ageScore = entries.reduce((sum, entry) => sum + (entry.age_slots || 0), 0);
          const candidate = { occupancy, ageScore, centerHz };
          if (
            !best ||
            candidate.occupancy < best.occupancy ||
            (candidate.occupancy === best.occupancy && candidate.ageScore > best.ageScore) ||
            (candidate.occupancy === best.occupancy &&
              candidate.ageScore === best.ageScore &&
              candidate.centerHz < best.centerHz)
          ) {
            best = candidate;
          }
        }
      }
      return best ? best.centerHz : fallback;
    }
    function bandmapAgeSeconds(data, txParity) {
      return txParity === 'even'
        ? data?.bandmaps?.even_age_seconds
        : data?.bandmaps?.odd_age_seconds;
    }
    function autoPickAllowed(data, txParity) {
      const ageSeconds = bandmapAgeSeconds(data, txParity);
      if (ageSeconds == null || ageSeconds >= 120) return false;
      if (data?.rig_tune_active) return false;
      if (data?.qso?.tx_active && data?.qso?.tx_slot_family === txParity) return false;
      return true;
    }
    function eligibleBandmapCallsigns(data, grid) {
      const queuedCalls = new Set((data.queue?.entries || []).map((entry) => entry.callsign));
      const calls = [];
      for (const row of grid || []) {
        for (const cell of row || []) {
          for (const entry of cell || []) {
            if (queuedCalls.has(entry.callsign) || entry.worked_recently) continue;
            calls.push(entry.callsign);
          }
        }
      }
      return [...new Set(calls)];
    }
    async function postJson(url, body) {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body || {}),
      });
      if (!response.ok) {
        let detail = response.statusText;
        try {
          const payload = await response.json();
          detail = payload.message || detail;
        } catch (_) {}
        throw new Error(detail || `request failed: ${response.status}`);
      }
      return response.json();
    }
    async function addSelectedToQueue() {
      if (!selectedCall) return;
      await postJson('/api/queue/add', { callsign: selectedCall });
      scheduleRefresh(10);
    }
    async function stopQso() {
      await postJson('/api/qso/stop', {});
      scheduleRefresh(10);
    }
    async function applyRigSettings() {
      const band = document.getElementById('rig-band').value;
      const appMode = document.getElementById('rig-app-mode').value;
      const powerInput = document.getElementById('rig-power');
      if (lastSnapshot?.rig_power_is_discrete) {
        const powerSettingId = powerInput.value || null;
        const powerLabel = powerInput.selectedOptions[0]?.textContent || powerSettingId || '-';
        const body = { band, app_mode: appMode };
        pendingRigConfig = { band, app_mode: appMode, power_label: powerLabel };
        if (lastSnapshot?.rig_power_settable) {
          body.power_setting_id = powerSettingId;
          pendingRigConfig.power_setting_id = powerSettingId;
        }
        await postJson('/api/rig/config', body);
      } else {
        const power = Number(powerInput.value);
        pendingRigConfig = { band, app_mode: appMode, power_w: power, power_label: `${power.toFixed(0)} W` };
        await postJson('/api/rig/config', {
          band,
          app_mode: appMode,
          power_w: power,
        });
      }
      scheduleRefresh(10);
    }
    async function tuneRigForTenSeconds() {
      await postJson('/api/rig/tune', {});
      scheduleRefresh(10);
    }
    async function updateQueueAuto(enabled) {
      await postJson('/api/queue/auto', { enabled });
      scheduleRefresh(10);
    }
    async function updateQueueAutoAddDecoded(enabled) {
      await postJson('/api/queue/auto-add-decoded', { enabled });
      scheduleRefresh(10);
    }
    async function updateQueueAutoAddDecodedMinCount5m(count) {
      await postJson('/api/queue/auto-add-decoded-min-count-5m', { count });
      scheduleRefresh(10);
    }
    async function updateQueueAutoAddDirect(enabled) {
      await postJson('/api/queue/auto-add-direct', { enabled });
      scheduleRefresh(10);
    }
    async function updateQueueIgnoreDirectWorked(enabled) {
      await postJson('/api/queue/ignore-direct-worked', { enabled });
      scheduleRefresh(10);
    }
    async function updateQueueCqEnabled(enabled) {
      await postJson('/api/queue/cq-enabled', { enabled });
      scheduleRefresh(10);
    }
    async function updateQueueCompoundRr73Handoff(enabled) {
      await postJson('/api/queue/compound-rr73-handoff', { enabled });
      scheduleRefresh(10);
    }
    async function updateQueueCompound73OnceHandoff(enabled) {
      await postJson('/api/queue/compound-73-once-handoff', { enabled });
      scheduleRefresh(10);
    }
    async function updateQueueCompoundForDirectSignal(enabled) {
      await postJson('/api/queue/compound-direct-signal', { enabled });
      scheduleRefresh(10);
    }
    async function updateQueueCqPercent(percent) {
      await postJson('/api/queue/cq-percent', { percent });
      scheduleRefresh(10);
    }
    async function updateQueuePauseCqLowActivity(enabled) {
      await postJson('/api/queue/pause-cq-low-activity', { enabled });
      scheduleRefresh(10);
    }
    async function updateQueueCqMinUniqueCalls5m(count) {
      await postJson('/api/queue/cq-min-unique-calls-5m', { count });
      scheduleRefresh(10);
    }
    async function updateQueueTxFreq(txParity, txFreqHz) {
      await postJson('/api/queue/tx-freq', { slot_family: txParity, tx_freq_hz: txFreqHz });
      scheduleRefresh(10);
    }
    async function updateQueueRetryDelay(kind, seconds) {
      await postJson('/api/queue/retry-delay', { kind, retry_delay_seconds: seconds });
      scheduleRefresh(10);
    }
    async function removeQueuedCall(callsign) {
      await postJson('/api/queue/remove', { callsign });
      scheduleRefresh(10);
    }
    async function clearQueue() {
      await postJson('/api/queue/clear', {});
      scheduleRefresh(10);
    }
    async function addBandmapCallsToQueue(slotFamily) {
      if (!lastSnapshot) return;
      const grid = slotFamily === 'even' ? lastSnapshot.bandmaps?.even : lastSnapshot.bandmaps?.odd;
      const callsigns = eligibleBandmapCallsigns(lastSnapshot, grid);
      if (!callsigns.length) return;
      const results = await Promise.allSettled(
        callsigns.map((callsign) => postJson('/api/queue/add', { callsign }))
      );
      const failures = results.filter((result) => result.status === 'rejected');
      if (failures.length) {
        console.error(`bulk queue add had ${failures.length} failures`, failures);
      }
      scheduleRefresh(10);
    }
    async function autoPickQuietSpot(txParity) {
      if (!lastSnapshot) return;
      if (!autoPickAllowed(lastSnapshot, txParity)) return;
      const picked = quietSpotFrequency(lastSnapshot, txParity).toFixed(0);
      document.getElementById(txParity === 'even' ? 'qso-freq-even' : 'qso-freq-odd').value = picked;
      renderQso(lastSnapshot);
      await updateQueueTxFreq(txParity, Number(picked));
    }
    function renderRigControls(data) {
      initRigBandOptions();
      initRigModeOptions();
      initRigPowerOptions(data);
      const bandInput = document.getElementById('rig-band');
      const modeInput = document.getElementById('rig-app-mode');
      const powerInput = document.getElementById('rig-power');
      const tune = document.getElementById('rig-tune');
      const hint = document.getElementById('rig-hint');
      const rigAvailable = data.rig_mode && data.rig_mode !== 'unavailable';
      const tuneActive = !!data.rig_tune_active;
      const qsoActive = !!data.qso?.active;
      const txBusy = tuneActive || !!data.qso?.tx_active || !!data.rig_is_tx;
      if (
        pendingRigConfig &&
        data.rig_band === pendingRigConfig.band &&
        data.app_mode === pendingRigConfig.app_mode &&
        (
          (data.rig_power_is_discrete &&
            (pendingRigConfig.power_setting_id == null ||
              data.rig_power_current_id === pendingRigConfig.power_setting_id)) ||
          (!data.rig_power_is_discrete &&
            data.rig_power_w != null &&
            pendingRigConfig.power_w != null &&
            Math.abs(data.rig_power_w - pendingRigConfig.power_w) < 0.05)
        )
      ) {
        pendingRigConfig = null;
      }
      const effectiveBand = pendingRigConfig?.band ?? data.rig_band;
      const effectiveMode = pendingRigConfig?.app_mode ?? data.app_mode;
      const effectivePower = data.rig_power_is_discrete
        ? (pendingRigConfig?.power_setting_id ?? data.rig_power_current_id ?? null)
        : (pendingRigConfig?.power_w ?? (data.rig_power_w == null ? null : Math.round(data.rig_power_w).toString()));
      if (rigAvailable && effectiveBand) {
        bandInput.value = effectiveBand;
      }
      if (rigAvailable && effectiveMode) {
        modeInput.value = effectiveMode;
      }
      if (rigAvailable && effectivePower != null) {
        powerInput.value = String(effectivePower);
      }
      bandInput.disabled = !rigAvailable || txBusy || qsoActive;
      modeInput.disabled = !rigAvailable || txBusy || qsoActive;
      powerInput.disabled = !rigAvailable || txBusy || (data.rig_power_is_discrete && !data.rig_power_settable);
      tune.disabled = !rigAvailable || tuneActive || qsoActive || txBusy;
      if (!rigAvailable) {
        hint.textContent = 'Rig control unavailable.';
      } else if (tuneActive) {
        hint.textContent = 'Tuning: 1000 Hz tone for 10 seconds. TX will return to RX automatically.';
      } else if (txBusy) {
        hint.textContent = 'Rig control disabled while transmit is active.';
      } else if (qsoActive) {
        hint.textContent = 'Power can be changed mid-QSO while not transmitting. Band and mode changes wait until the QSO is idle.';
      } else if (pendingRigConfig) {
        hint.textContent = `Applying ${pendingRigConfig.band} ${pendingRigConfig.app_mode}${pendingRigConfig.power_label ? ` at ${pendingRigConfig.power_label}` : ''}...`;
      } else if (data.rig_power_is_discrete && !data.rig_power_settable) {
        hint.textContent = `Rig is ${bandInput.value || data.rig_band} ${modeInput.value || data.app_mode}; power is ${data.rig_power_label || '-'} and is not CAT-settable. Tune sends a 1000 Hz tone for 10 seconds.`;
      } else {
        hint.textContent = `Rig is ${bandInput.value || data.rig_band} ${modeInput.value || data.app_mode} at ${data.rig_power_label || powerInput.value || '-'}. Tune sends a 1000 Hz tone for 10 seconds.`;
      }
    }
    function renderWaterfall(rows) {
      if (!rows || rows.length === 0) {
        ctx.fillStyle = '#02070c';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        return;
      }
      const width = rows[0].length;
      const height = rows.length;
      canvas.width = width;
      canvas.height = height;
      const image = ctx.createImageData(width, height);
      function gradient(v) {
        const t = Math.max(0, Math.min(1, v / 255));
        if (t < 0.18) {
          const u = t / 0.18;
          return [4 + u * 10, 8 + u * 18, 18 + u * 42];
        }
        if (t < 0.42) {
          const u = (t - 0.18) / 0.24;
          return [14 + u * 14, 26 + u * 92, 60 + u * 120];
        }
        if (t < 0.68) {
          const u = (t - 0.42) / 0.26;
          return [28 + u * 152, 118 + u * 74, 180 - u * 76];
        }
        if (t < 0.86) {
          const u = (t - 0.68) / 0.18;
          return [180 + u * 50, 192 + u * 32, 104 - u * 48];
        }
        const u = (t - 0.86) / 0.14;
        return [230 + u * 25, 224 + u * 28, 56 + u * 120];
      }
      for (let y = 0; y < height; y++) {
        const row = rows[y];
        for (let x = 0; x < width; x++) {
          const value = row[x] || 0;
          const [r, g, b] = gradient(value);
          const i = (y * width + x) * 4;
          image.data[i + 0] = Math.round(r);
          image.data[i + 1] = Math.round(g);
          image.data[i + 2] = Math.round(b);
          image.data[i + 3] = 255;
        }
      }
      ctx.putImageData(image, 0, 0);
    }
    function renderBandMap(rootId, grid) {
      const root = document.getElementById(rootId);
      const queuedCalls = new Set((lastSnapshot?.queue?.entries || []).map((entry) => entry.callsign));
      root.innerHTML = '';
      for (let row = 0; row < grid.length; row++) {
        for (let col = 0; col < grid[row].length; col++) {
          const startHz = col * 400 + row * 400;
          const cell = document.createElement('div');
          cell.className = 'cell';
          const title = document.createElement('div');
          title.className = 'cell-title';
          title.textContent = `${startHz}-${startHz + 399} Hz`;
          cell.appendChild(title);
          const entries = grid[row][col] || [];
          for (const entry of entries) {
            const line = document.createElement('div');
            line.className = 'call';
            const fade = Math.min(1, (entry.age_slots || 0) / 4);
            if (entry.worked_recently) {
              const lightness = 82 - fade * 14;
              line.style.color = `hsl(0 0% ${lightness}%)`;
            } else {
              const lightness = 72 - fade * 34;
              const saturation = 88 - fade * 58;
              line.style.color = `hsl(135 ${saturation}% ${lightness}%)`;
            }
            const queueButton = queuedCalls.has(entry.callsign) || entry.worked_recently
              ? '<span class="queue-tag placeholder">Q</span>'
              : `<span class="queue-tag" data-queue-call="${escapeHtml(entry.callsign)}">Q</span>`;
            line.innerHTML = `
              ${queueButton}
              <span class="call-text">${renderCallValue(
                entry.detail ? `${entry.callsign} ${entry.detail}` : entry.callsign,
                entry.callsign
              )}</span>`;
            cell.appendChild(line);
          }
          root.appendChild(cell);
        }
      }
    }
    function renderDecodes(rows) {
      const body = document.getElementById('decodes');
      body.innerHTML = '';
      if (!rows.length) {
        const tr = document.createElement('tr');
        const td = document.createElement('td');
        td.colSpan = 10;
        td.textContent = 'No decodes yet';
        tr.appendChild(td);
        body.appendChild(tr);
        return;
      }
      for (const row of rows) {
        const tr = document.createElement('tr');
        const seen = row.seen;
        tr.innerHTML = `
          <td class="seen-${seen}">${seen}</td>
          <td>${row.utc}</td>
          <td class="num">${row.snr_db}</td>
          <td class="num">${row.dt_seconds.toFixed(2)}</td>
          <td class="num">${Math.round(row.freq_hz)}</td>
          <td>${row.kind}</td>
          <td>${renderCallValue(row.field1, row.field1_select_call)}</td>
          <td>${renderCallValue(row.field2, row.field2_select_call)}</td>
          <td>${row.info}</td>
          <td>${row.text}</td>`;
        body.appendChild(tr);
      }
    }
    function renderDetail(data) {
      const title = document.getElementById('detail-call');
      const state = document.getElementById('detail-state');
      const history = document.getElementById('detail-history');
      const logs = document.getElementById('detail-logs');
      const addButton = document.getElementById('detail-add-queue');
      const addHint = document.getElementById('detail-queue-hint');
      const queuedCalls = new Set((data.queue?.entries || []).map((entry) => entry.callsign));
      const stations = new Map((data.stations || []).map((entry) => [entry.callsign, entry]));
      if (!selectedCall || !stations.has(selectedCall)) {
        title.textContent = selectedCall ? `${selectedCall} not active in last 60m` : 'No station selected';
        state.textContent = selectedCall ? '' : 'Click a callsign in the bandmap or decode table.';
        history.textContent = '';
        logs.innerHTML = '';
        addButton.disabled = true;
        addHint.textContent = 'Select a station in the monitor pane to add it to the work queue.';
        return;
      }
      const station = stations.get(selectedCall);
      addButton.disabled = queuedCalls.has(station.callsign);
      addHint.textContent = queuedCalls.has(station.callsign)
        ? `${station.callsign} is already in the queue.`
        : `Add ${station.callsign} to the work queue.`;
      title.textContent = station.callsign;
      const current = station.is_in_qso
        ? `In QSO since ${station.in_qso_since ?? '-'} with ${station.qso_with ?? '?'}`
        : `Idle${station.last_qso_ended_at ? `, last ended ${station.last_qso_ended_at}` : ''}`;
      state.textContent = `Last heard: ${station.last_heard_at}\n${current}`;
      history.textContent = station.qso_history.length
        ? [...station.qso_history]
            .reverse()
            .map((item) => `${item.started_at} -> ${item.ended_at}  ${item.peer}`)
            .join('\n')
        : 'No QSO history in last 60m.';
      const related = (data.station_logs || []).filter((item) => (item.related_calls || []).includes(selectedCall));
      logs.innerHTML = `
        <div class="activity-item activity-head">
          <div class="activity-col">Time</div>
          <div class="activity-col">To</div>
          <div class="activity-col">De</div>
          <div class="activity-col">SNR</div>
          <div class="activity-col">dT</div>
          <div class="activity-col">Freq</div>
          <div class="activity-col">Msg</div>
        </div>`;
      if (!related.length) {
        logs.innerHTML += '<div class="detail-empty">No related decodes in last 60m.</div>';
        return;
      }
      for (const item of related) {
        const div = document.createElement('div');
        div.className = 'activity-item';
        const peer = item.peer ?? '-';
        div.innerHTML = `
          <div class="activity-col">${item.timestamp}</div>
          <div class="activity-col">${renderParty(peer, selectedCall)}</div>
          <div class="activity-col">${renderParty(item.sender_call, selectedCall)}</div>
          <div class="activity-col">${item.snr_db}</div>
          <div class="activity-col">${item.dt_seconds.toFixed(2)}</div>
          <div class="activity-col">${Math.round(item.freq_hz)}</div>
          <div class="activity-msg">${escapeHtml(item.text)}</div>`;
        logs.appendChild(div);
      }
      if (autoFollowLogs) {
        logs.scrollTop = logs.scrollHeight;
      }
    }
    function renderQso(data) {
      const call = document.getElementById('qso-call');
      const hint = document.getElementById('qso-hint');
      const state = document.getElementById('qso-state');
      const timeout = document.getElementById('qso-timeout');
      const counters = document.getElementById('qso-counters');
      const lastRx = document.getElementById('qso-last-rx');
      const parity = document.getElementById('qso-parity');
      const snr = document.getElementById('qso-snr');
      const transcript = document.getElementById('qso-transcript');
      const autoToggle = document.getElementById('qso-auto');
      const stop = document.getElementById('qso-stop');
      const evenAutoPick = document.getElementById('qso-autopick-even');
      const oddAutoPick = document.getElementById('qso-autopick-odd');
      const evenFreqInput = document.getElementById('qso-freq-even');
      const oddFreqInput = document.getElementById('qso-freq-odd');
      const queue = data.queue || {};
      const station = selectedStation(data);
      const qso = data.qso || {};
      const readyQueueEntry = (queue.entries || []).find((entry) => entry.ready);
      const candidateStation = station || (readyQueueEntry ? stationMap(data).get(readyQueueEntry.callsign) : null);
      const txParity = qso.active ? qso.tx_slot_family : inferredTxParity(candidateStation);
      const evenFreq = currentTxFreqValue(data, 'even');
      const oddFreq = currentTxFreqValue(data, 'odd');
      evenFreqInput.min = String(data.qso_defaults?.tx_freq_min_hz ?? 200);
      evenFreqInput.max = String(data.qso_defaults?.tx_freq_max_hz ?? 3500);
      oddFreqInput.min = String(data.qso_defaults?.tx_freq_min_hz ?? 200);
      oddFreqInput.max = String(data.qso_defaults?.tx_freq_max_hz ?? 3500);
      if (evenFreqInput.value === '' || Number(evenFreqInput.value) !== Number(evenFreq)) {
        evenFreqInput.value = Number(evenFreq).toFixed(0);
      }
      if (oddFreqInput.value === '' || Number(oddFreqInput.value) !== Number(oddFreq)) {
        oddFreqInput.value = Number(oddFreq).toFixed(0);
      }
      autoToggle.checked = !!queue.auto_enabled;
      call.textContent = qso.active
        ? `Active with ${qso.partner_call}`
        : (readyQueueEntry ? `Idle, next ready ${readyQueueEntry.callsign}` : 'Idle');
      state.textContent = qso.state || 'idle';
      timeout.textContent = qso.timeout_remaining_seconds == null ? '-' : `${qso.timeout_remaining_seconds}s`;
      counters.textContent = `no_msg=${qso.no_msg_count ?? 0}  no_fwd=${qso.no_fwd_count ?? 0}`;
      lastRx.textContent = qso.last_rx_event || '-';
      parity.textContent = txParity || '-';
      snr.textContent = qso.latest_partner_snr_db == null ? '-' : `${qso.latest_partner_snr_db} dB`;
      if (qso.active) {
        hint.textContent = `TX ${qso.selected_tx_freq_hz?.toFixed(0) ?? '-'} Hz, ${qso.tx_slot_family ?? '?'} slots. Escape stops immediately.`;
      } else if (data.rig_tune_active) {
        hint.textContent = 'Tune tone active. Queue dispatch is paused until TX returns to RX.';
      } else if (candidateStation) {
        hint.textContent =
          `Latest ${candidateStation.callsign}: ${candidateStation.last_heard_at}, ${Math.round(candidateStation.last_heard_freq_hz)} Hz, ${candidateStation.last_heard_snr_db} dB, TX parity ${txParity ?? '?'} .`;
      } else {
        hint.textContent = queue.scheduler_status || 'Auto QSO waits for a ready station in the queue.';
      }
      transcript.innerHTML = '';
      const rows = qso.transcript || [];
      if (!rows.length) {
        transcript.innerHTML = '<div class="detail-empty">No QSO transcript yet.</div>';
      } else {
        for (const entry of rows) {
          const row = document.createElement('div');
          row.className = 'qso-entry';
          row.innerHTML = `
            <div class="stamp">${escapeHtml(entry.timestamp)}</div>
            <div class="dir">${escapeHtml(entry.direction)}</div>
            <div class="state">${escapeHtml(entry.state)}</div>
            <div class="msg">${escapeHtml(entry.text)}</div>`;
          transcript.appendChild(row);
        }
      }
      if (autoFollowQso) {
        transcript.scrollTop = transcript.scrollHeight;
      }
      stop.disabled = !qso.active && !qso.tx_active;
      autoToggle.disabled = !!data.rig_tune_active;
      evenFreqInput.disabled = !!qso.tx_active || !!data.rig_tune_active;
      oddFreqInput.disabled = !!qso.tx_active || !!data.rig_tune_active;
      evenAutoPick.disabled = !autoPickAllowed(data, 'even');
      oddAutoPick.disabled = !autoPickAllowed(data, 'odd');
      evenAutoPick.textContent = autoPickAllowed(data, 'even') ? 'Auto-Pick Even' : 'Auto-Pick Even';
      oddAutoPick.textContent = autoPickAllowed(data, 'odd') ? 'Auto-Pick Odd' : 'Auto-Pick Odd';
    }
    function renderQueue(data) {
      const queue = data.queue || {};
      const count = document.getElementById('queue-count');
      const status = document.getElementById('queue-status');
      const noMessageRetryDelay = document.getElementById('queue-no-message-retry-delay');
      const noForwardRetryDelay = document.getElementById('queue-no-forward-retry-delay');
      const autoAddDecoded = document.getElementById('queue-auto-add-decoded');
      const autoAddDecodedMinCount5m = document.getElementById('queue-auto-add-decoded-min-count-5m');
      const autoAddDirect = document.getElementById('queue-auto-add-direct');
      const ignoreDirectWorked = document.getElementById('queue-ignore-direct-worked');
      const cqEnabled = document.getElementById('queue-cq-enabled');
      const pauseCqLowActivity = document.getElementById('queue-pause-cq-low-activity');
      const cqMinUniqueCalls5m = document.getElementById('queue-cq-min-unique-calls-5m');
      const uniqueCallsLast5m = document.getElementById('queue-unique-calls-last-5m');
      const compoundRr73 = document.getElementById('queue-compound-rr73');
      const compound73Once = document.getElementById('queue-compound-73-once');
      const compoundDirectSignal = document.getElementById('queue-compound-direct-signal');
      const cqPercent = document.getElementById('queue-cq-percent');
      const nextCqParity = document.getElementById('queue-next-cq-parity');
      const clearButton = document.getElementById('queue-clear');
      const list = document.getElementById('queue-list');
      count.textContent = `${(queue.entries || []).length} queued`;
      status.textContent = queue.scheduler_status || 'auto disabled';
      clearButton.disabled = !(queue.entries || []).length;
      autoAddDecoded.checked = !!queue.auto_add_all_decoded_calls;
      if (
        autoAddDecodedMinCount5m.value === ''
        || Number(autoAddDecodedMinCount5m.value) !== Number(queue.auto_add_decoded_min_count_5m)
      ) {
        autoAddDecodedMinCount5m.value = String(queue.auto_add_decoded_min_count_5m ?? 2);
      }
      autoAddDirect.checked = !!queue.auto_add_direct_calls;
      ignoreDirectWorked.checked = !!queue.ignore_direct_calls_from_recently_worked;
      cqEnabled.checked = !!queue.cq_enabled;
      pauseCqLowActivity.checked = !!queue.pause_cq_when_few_unique_calls;
      compoundRr73.checked = !!queue.use_compound_rr73_handoff;
      compound73Once.checked = !!queue.use_compound_73_once_handoff;
      compoundDirectSignal.checked = !!queue.use_compound_for_direct_signal_callers;
      cqPercent.value = String(queue.cq_percent ?? 80);
      uniqueCallsLast5m.textContent = String(queue.unique_calls_last_5m ?? 0);
      nextCqParity.textContent = queue.next_cq_parity_flipped ? 'Next CQ Parity Flipped' : 'Flip Next CQ Parity';
      if (
        noMessageRetryDelay.value === ''
        || Number(noMessageRetryDelay.value) !== Number(queue.no_message_retry_delay_seconds)
      ) {
        noMessageRetryDelay.value = String(queue.no_message_retry_delay_seconds ?? 35);
      }
      if (
        noForwardRetryDelay.value === ''
        || Number(noForwardRetryDelay.value) !== Number(queue.no_forward_retry_delay_seconds)
      ) {
        noForwardRetryDelay.value = String(queue.no_forward_retry_delay_seconds ?? 300);
      }
      if (
        cqMinUniqueCalls5m.value === ''
        || Number(cqMinUniqueCalls5m.value) !== Number(queue.cq_pause_min_unique_calls_5m)
      ) {
        cqMinUniqueCalls5m.value = String(queue.cq_pause_min_unique_calls_5m ?? 3);
      }
      list.innerHTML = `
        <div class="queue-item queue-head">
          <div>Call</div>
          <div>Dir</div>
          <div>Cnt</div>
          <div>Queued</div>
          <div>Ready</div>
          <div>Heard</div>
          <div>Msg</div>
          <div>Par</div>
          <div>Status</div>
          <div>Act</div>
        </div>`;
      if (!(queue.entries || []).length) {
        list.innerHTML += '<div class="detail-empty">No queued stations.</div>';
        return;
      }
      for (const entry of queue.entries) {
        const row = document.createElement('div');
        row.className = `queue-item${entry.priority_direct ? ' queue-item-direct' : ''}`;
        row.innerHTML = `
          <div class="queue-call">${renderCallValue(entry.callsign, entry.callsign)}</div>
          <div class="queue-meta">${entry.direct_pending ? 'Y' : '-'}</div>
          <div class="queue-meta">${escapeHtml(entry.direct_count ?? 0)}</div>
          <div class="queue-meta">${escapeHtml(entry.queued_at)}</div>
          <div class="queue-meta">${escapeHtml(entry.ok_to_schedule_after)}</div>
          <div class="queue-meta">${escapeHtml(entry.direct_last_heard_at ?? entry.last_heard_at ?? '-')}</div>
          <div class="queue-meta">${escapeHtml(entry.last_heard_message)}</div>
          <div class="queue-meta">${escapeHtml(entry.last_heard_slot_family ?? '-')}</div>
          <div class="queue-meta">${escapeHtml(entry.ready ? 'ready' : entry.status)}</div>
          <div><button class="button secondary compact" type="button" data-queue-remove="${escapeHtml(entry.callsign)}">X</button></div>`;
        list.appendChild(row);
      }
    }
    function renderDirectCalls(data) {
      const direct = data.direct_calls || [];
      const count = document.getElementById('direct-count');
      const list = document.getElementById('direct-list');
      const previousScrollTop = list.scrollTop;
      count.textContent = `${direct.length} msgs`;
      list.innerHTML = `
        <div class="activity-item activity-head">
          <div class="activity-col">Time</div>
          <div class="activity-col">To</div>
          <div class="activity-col">De</div>
          <div class="activity-col">SNR</div>
          <div class="activity-col">dT</div>
          <div class="activity-col">Freq</div>
          <div class="activity-col">Msg</div>
        </div>`;
      if (!direct.length) {
        list.innerHTML += '<div class="detail-empty">No recent direct traffic.</div>';
        return;
      }
      for (const item of direct) {
        const row = document.createElement('div');
        row.className = item.is_ours ? 'activity-item direct-sent' : 'activity-item';
        row.innerHTML = `
          <div class="activity-col">${escapeHtml(item.timestamp)}</div>
          <div class="activity-col">${renderCallValue(escapeHtml(item.to_call), item.to_call)}</div>
          <div class="activity-col">${renderCallValue(escapeHtml(item.from_call), item.from_call)}</div>
          <div class="activity-col">${item.snr_db == null ? '-' : item.snr_db}</div>
          <div class="activity-col">${item.dt_seconds == null ? '-' : item.dt_seconds.toFixed(2)}</div>
          <div class="activity-col">${item.freq_hz == null ? '-' : Math.round(item.freq_hz)}</div>
          <div class="activity-msg">${escapeHtml(item.text)}</div>`;
        list.appendChild(row);
      }
      if (autoFollowDirect) {
        list.scrollTop = list.scrollHeight;
      } else {
        list.scrollTop = previousScrollTop;
      }
    }
    function renderQsoHistory(data) {
      const history = (data.qso_history || []).filter((entry) => entry.got_reply || entry.reached_73);
      const count = document.getElementById('qso-history-count');
      const list = document.getElementById('qso-history-list');
      const last10m = history.filter((entry) => (entry.age_seconds ?? Number.MAX_SAFE_INTEGER) <= 10 * 60).length;
      const last1h = history.filter((entry) => (entry.age_seconds ?? Number.MAX_SAFE_INTEGER) <= 60 * 60).length;
      count.textContent = `${history.length} QSOs  10m=${last10m}  1h=${last1h}`;
      list.innerHTML = `
        <div class="history-row history-head">
          <div>Time</div>
          <div>Ago</div>
          <div>Call</div>
          <div>Bd</div>
          <div>Md</div>
          <div>Rpl</div>
          <div>R</div>
          <div>73</div>
          <div>Sent</div>
          <div>Recv</div>
          <div>Exit</div>
        </div>`;
      if (!history.length) {
        list.innerHTML += '<div class="detail-empty">No QSO history from the last 24 hours in the JSONL log.</div>';
        return;
      }
      for (const entry of history) {
        const row = document.createElement('div');
        row.className = 'history-row';
        row.innerHTML = `
          <div class="history-cell muted">${escapeHtml(entry.time)}</div>
          <div class="history-cell muted">${escapeHtml(entry.age)}</div>
          <div class="history-cell">${renderCallValue(escapeHtml(entry.callsign), entry.callsign)}</div>
          <div class="history-cell muted">${escapeHtml(entry.band ?? '-')}</div>
          <div class="history-cell muted">${escapeHtml(entry.mode ?? '-')}</div>
          <div class="history-cell muted">${entry.got_reply ? 'Y' : '-'}</div>
          <div class="history-cell muted">${entry.got_roger ? 'Y' : '-'}</div>
          <div class="history-cell muted">${entry.reached_73 ? 'Y' : '-'}</div>
          <div class="history-cell">${escapeHtml(entry.sent_info)}</div>
          <div class="history-cell">${escapeHtml(entry.received_info)}</div>
          <div class="history-cell history-cell-exit muted">${escapeHtml(entry.exit_reason)}</div>`;
        list.appendChild(row);
      }
    }
    let refreshInFlight = { value: false };
    let refreshTimer = { value: null };
    function scheduleRefresh(delayMs) {
      if (refreshTimer.value != null) {
        clearTimeout(refreshTimer.value);
      }
      refreshTimer.value = setTimeout(() => refresh().catch(console.error), delayMs);
    }
    async function refresh() {
      if (refreshInFlight.value) {
        return;
      }
      refreshInFlight.value = true;
      try {
        const response = await fetch('/api/state', { cache: 'no-store' });
        const data = await response.json();
        lastSnapshot = data;
        document.getElementById('time').textContent = data.time_utc || '-';
      const freq = data.rig_frequency_hz == null ? 'unavailable' : `${(data.rig_frequency_hz / 1e6).toFixed(4)} MHz`;
      const rigDir = data.rig_is_tx == null ? '?' : (data.rig_is_tx ? 'TX' : 'RX');
      const rigPower = data.rig_power_label || (data.rig_power_w == null ? '-' : `${data.rig_power_w.toFixed(1)}W`);
      const rigBg = data.rig_bargraph == null ? '-' : data.rig_bargraph;
      const rigSmeter = data.rig_rx_s_meter == null ? '-' : data.rig_rx_s_meter.toFixed(1);
      const rigFwd = data.rig_tx_forward_power_w == null ? '-' : `${data.rig_tx_forward_power_w.toFixed(1)}W`;
      const rigSwr = data.rig_tx_swr == null ? '-' : data.rig_tx_swr.toFixed(1);
      const rigKind = data.rig_kind || '?';
      const rigTelemetry = [];
      if (rigKind !== 'mchf') rigTelemetry.push(`BG=${rigBg}`);
      if (data.rig_rx_s_meter != null) rigTelemetry.push(`S=${rigSmeter}`);
      if (data.rig_tx_forward_power_w != null) rigTelemetry.push(`FWD=${rigFwd}`);
      if (data.rig_tx_swr != null) rigTelemetry.push(`SWR=${rigSwr}`);
      const rigCore = `${freq} ${rigKind} ${data.rig_mode}/${data.app_mode} ${data.rig_band} ${rigDir} P=${rigPower}`;
      document.getElementById('rig').textContent = rigTelemetry.length
        ? `${rigCore}\n${rigTelemetry.join('  ')}`
        : rigCore;
      document.getElementById('audio').textContent =
        `t=${data.audio_stats.latest_sample ?? '-'} ch=${data.audio_stats.selected_channel} L=${data.audio_stats.left_dbfs.toFixed(1)} R=${data.audio_stats.right_dbfs.toFixed(1)} all=${data.audio_stats.overall_dbfs.toFixed(1)} rec=${data.audio_stats.recoveries}`;
      document.getElementById('status').textContent = data.decode_status || '-';
      document.getElementById('slot').textContent =
        `${data.current_slot}${data.last_done_slot ? ` last=${data.last_done_slot}` : ''}`;
      document.getElementById('times').textContent =
        data.app_mode === 'FT8'
          ? `e=${fmtSec(data.decode_times.early_seconds)} m=${fmtSec(data.decode_times.mid_seconds)} l=${fmtSec(data.decode_times.late_seconds)} tx=${fmtSec(data.decode_times.tx_margin_seconds)}`
          : `m=${fmtSec(data.decode_times.mid_seconds)} tx=${fmtSec(data.decode_times.tx_margin_seconds)}`;
      document.getElementById('dtstats').textContent =
        `cur=${fmtSec(data.dt_stats.current_mean_seconds)}/${fmtSec(data.dt_stats.current_median_seconds)} sd=${fmtSec(data.dt_stats.current_stddev_seconds)} n=${data.dt_stats.current_count} 10m=${fmtSec(data.dt_stats.ten_minute_mean_seconds)}/${fmtSec(data.dt_stats.ten_minute_median_seconds)} n=${data.dt_stats.ten_minute_count}`;
      document.getElementById('count').textContent = `${data.decodes.length} visible`;
      renderWaterfall(data.waterfall);
      renderBandMap('even-map', data.bandmaps.even);
      renderBandMap('odd-map', data.bandmaps.odd);
      document.getElementById('queue-even-map').disabled = eligibleBandmapCallsigns(data, data.bandmaps?.even).length === 0;
      document.getElementById('queue-odd-map').disabled = eligibleBandmapCallsigns(data, data.bandmaps?.odd).length === 0;
      renderDecodes(data.decodes);
      renderRigControls(data);
      renderDetail(data);
      renderQso(data);
      renderQueue(data);
      renderDirectCalls(data);
      renderQsoHistory(data);
      document.querySelectorAll('[data-call]').forEach((node) => {
        node.addEventListener('click', (event) => {
          event.preventDefault();
          pickCall(node.dataset.call);
        });
      });
      document.querySelectorAll('[data-queue-call]').forEach((node) => {
        node.addEventListener('click', (event) => {
          event.preventDefault();
          event.stopPropagation();
          postJson('/api/queue/add', { callsign: node.dataset.queueCall }).then(() => scheduleRefresh(10)).catch((error) => console.error(error));
        });
      });
      document.querySelectorAll('[data-queue-remove]').forEach((node) => {
        node.addEventListener('click', (event) => {
          event.preventDefault();
          removeQueuedCall(node.dataset.queueRemove).catch((error) => console.error(error));
        });
      });
      } finally {
        refreshInFlight.value = false;
        scheduleRefresh(250);
      }
    }
    refresh().catch(console.error);
    initRigBandOptions();
    initRigModeOptions();
    initRigPowerOptions();
    document.getElementById('detail-logs').addEventListener('scroll', (event) => {
      const node = event.currentTarget;
      const remaining = node.scrollHeight - node.clientHeight - node.scrollTop;
      autoFollowLogs = remaining < 12;
    });
    document.getElementById('qso-transcript').addEventListener('scroll', (event) => {
      const node = event.currentTarget;
      const remaining = node.scrollHeight - node.clientHeight - node.scrollTop;
      autoFollowQso = remaining < 12;
    });
    document.getElementById('direct-list').addEventListener('scroll', (event) => {
      const node = event.currentTarget;
      const remaining = node.scrollHeight - node.clientHeight - node.scrollTop;
      autoFollowDirect = remaining < 12;
    });
    document.getElementById('qso-stop').addEventListener('click', () => {
      stopQso().catch((error) => console.error(error));
    });
    document.getElementById('detail-add-queue').addEventListener('click', () => {
      addSelectedToQueue().catch((error) => console.error(error));
    });
    document.getElementById('qso-auto').addEventListener('change', (event) => {
      updateQueueAuto(event.currentTarget.checked).catch((error) => console.error(error));
    });
    document.getElementById('rig-tune').addEventListener('click', () => {
      tuneRigForTenSeconds().catch((error) => console.error(error));
    });
    document.getElementById('rig-band').addEventListener('change', () => {
      applyRigSettings().catch((error) => {
        pendingRigConfig = null;
        console.error(error);
      });
    });
    document.getElementById('rig-app-mode').addEventListener('change', () => {
      applyRigSettings().catch((error) => {
        pendingRigConfig = null;
        console.error(error);
      });
    });
    document.getElementById('rig-power').addEventListener('change', () => {
      applyRigSettings().catch((error) => {
        pendingRigConfig = null;
        console.error(error);
      });
    });
    document.getElementById('qso-autopick-even').addEventListener('click', () => {
      autoPickQuietSpot('even').catch((error) => console.error(error));
    });
    document.getElementById('qso-autopick-odd').addEventListener('click', () => {
      autoPickQuietSpot('odd').catch((error) => console.error(error));
    });
    document.getElementById('qso-freq-even').addEventListener('input', () => {
      if (lastSnapshot) renderQso(lastSnapshot);
    });
    document.getElementById('qso-freq-even').addEventListener('change', (event) => {
      const value = Number(event.currentTarget.value);
      if (!lastSnapshot || !txFreqValid(lastSnapshot, value)) {
        if (lastSnapshot) renderQso(lastSnapshot);
        return;
      }
      updateQueueTxFreq('even', value).catch((error) => console.error(error));
    });
    document.getElementById('qso-freq-odd').addEventListener('input', () => {
      if (lastSnapshot) renderQso(lastSnapshot);
    });
    document.getElementById('qso-freq-odd').addEventListener('change', (event) => {
      const value = Number(event.currentTarget.value);
      if (!lastSnapshot || !txFreqValid(lastSnapshot, value)) {
        if (lastSnapshot) renderQso(lastSnapshot);
        return;
      }
      updateQueueTxFreq('odd', value).catch((error) => console.error(error));
    });
    document.getElementById('queue-no-message-retry-delay').addEventListener('change', (event) => {
      const value = Number(event.currentTarget.value);
      if (!Number.isFinite(value) || value < 1 || value > 3600) {
        if (lastSnapshot) renderQueue(lastSnapshot);
        return;
      }
      updateQueueRetryDelay('no_message', Math.round(value)).catch((error) => console.error(error));
    });
    document.getElementById('queue-no-forward-retry-delay').addEventListener('change', (event) => {
      const value = Number(event.currentTarget.value);
      if (!Number.isFinite(value) || value < 1 || value > 3600) {
        if (lastSnapshot) renderQueue(lastSnapshot);
        return;
      }
      updateQueueRetryDelay('no_forward', Math.round(value)).catch((error) => console.error(error));
    });
    document.getElementById('queue-clear').addEventListener('click', () => {
      clearQueue().catch((error) => console.error(error));
    });
    document.getElementById('queue-auto-add-decoded').addEventListener('change', (event) => {
      updateQueueAutoAddDecoded(event.currentTarget.checked).catch((error) => console.error(error));
    });
    document.getElementById('queue-auto-add-decoded-min-count-5m').addEventListener('change', (event) => {
      const value = Number(event.currentTarget.value);
      if (!Number.isFinite(value) || value < 1 || value > 1000) {
        if (lastSnapshot) renderQueue(lastSnapshot);
        return;
      }
      updateQueueAutoAddDecodedMinCount5m(Math.round(value)).catch((error) => console.error(error));
    });
    document.getElementById('queue-auto-add-direct').addEventListener('change', (event) => {
      updateQueueAutoAddDirect(event.currentTarget.checked).catch((error) => console.error(error));
    });
    document.getElementById('queue-ignore-direct-worked').addEventListener('change', (event) => {
      updateQueueIgnoreDirectWorked(event.currentTarget.checked).catch((error) => console.error(error));
    });
    document.getElementById('queue-cq-enabled').addEventListener('change', (event) => {
      updateQueueCqEnabled(event.currentTarget.checked).catch((error) => console.error(error));
    });
    document.getElementById('queue-pause-cq-low-activity').addEventListener('change', (event) => {
      updateQueuePauseCqLowActivity(event.currentTarget.checked).catch((error) => console.error(error));
    });
    document.getElementById('queue-compound-rr73').addEventListener('change', (event) => {
      updateQueueCompoundRr73Handoff(event.currentTarget.checked).catch((error) => console.error(error));
    });
    document.getElementById('queue-compound-73-once').addEventListener('change', (event) => {
      updateQueueCompound73OnceHandoff(event.currentTarget.checked).catch((error) => console.error(error));
    });
    document.getElementById('queue-compound-direct-signal').addEventListener('change', (event) => {
      updateQueueCompoundForDirectSignal(event.currentTarget.checked).catch((error) => console.error(error));
    });
    document.getElementById('queue-cq-percent').addEventListener('change', (event) => {
      updateQueueCqPercent(Number(event.currentTarget.value)).catch((error) => console.error(error));
    });
    document.getElementById('queue-cq-min-unique-calls-5m').addEventListener('change', (event) => {
      const value = Number(event.currentTarget.value);
      if (!Number.isFinite(value) || value < 0 || value > 1000) {
        if (lastSnapshot) renderQueue(lastSnapshot);
        return;
      }
      updateQueueCqMinUniqueCalls5m(Math.round(value)).catch((error) => console.error(error));
    });
    document.getElementById('queue-next-cq-parity').addEventListener('click', () => {
      fetchJson('/api/queue/next-cq-parity', { method: 'POST' }).then(refresh).catch((error) => console.error(error));
    });
    document.getElementById('queue-even-map').addEventListener('click', () => {
      addBandmapCallsToQueue('even').catch((error) => console.error(error));
    });
    document.getElementById('queue-odd-map').addEventListener('click', () => {
      addBandmapCallsToQueue('odd').catch((error) => console.error(error));
    });
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && lastSnapshot?.qso && (lastSnapshot.qso.active || lastSnapshot.qso.tx_active)) {
        event.preventDefault();
        stopQso().catch((error) => console.error(error));
      }
    });
  </script>
</body>
</html>
"#;

fn main() -> Result<(), AppError> {
    let cli = Cli::parse();
    if let Some(command) = cli.command {
        return match command {
            CliCommand::ExportAdif(args) => run_export_adif(args),
        };
    }
    if cli.oneshot {
        run_oneshot(cli)
    } else {
        run_continuous(cli)
    }
}

fn run_export_adif(args: ExportAdifArgs) -> Result<(), AppError> {
    let config = AppConfig::load(&args.config)?;
    let input_path = args
        .input
        .clone()
        .unwrap_or_else(|| PathBuf::from(&config.logging.fsm_log_path));
    let contents = std::fs::read_to_string(&input_path)?;
    let report = build_adif_export_report(
        &contents,
        &config.station.our_call,
        &config.station.our_grid,
    );
    print_adif_export_status(&report);
    let rendered = render_adif(&report.eligible_records);
    if let Some(output_path) = args.output {
        if let Some(parent) = output_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(&output_path, rendered)?;
        eprintln!(
            "exported {} award-valid QSOs from {} to {}",
            report.eligible_records.len(),
            input_path.display(),
            output_path.display()
        );
    } else {
        print!("{rendered}");
        eprintln!(
            "exported {} award-valid QSOs from {}",
            report.eligible_records.len(),
            input_path.display()
        );
    }
    Ok(())
}

fn start_web_server(bind: &str, state: WebAppState) -> Result<(), AppError> {
    let addr: SocketAddr = bind.parse().map_err(|error| {
        AppError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("invalid web bind address '{bind}': {error}"),
        ))
    })?;
    thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");
        runtime.block_on(async move {
            let app = Router::new()
                .route("/", get(index_handler))
                .route("/api/state", get(api_state_handler))
                .route("/api/qso/stop", post(api_qso_stop_handler))
                .route("/api/rig/config", post(api_rig_config_handler))
                .route("/api/rig/tune", post(api_rig_tune_handler))
                .route("/api/queue/add", post(api_queue_add_handler))
                .route("/api/queue/remove", post(api_queue_remove_handler))
                .route("/api/queue/clear", post(api_queue_clear_handler))
                .route("/api/queue/auto", post(api_queue_auto_handler))
                .route(
                    "/api/queue/auto-add-decoded",
                    post(api_queue_auto_add_decoded_handler),
                )
                .route(
                    "/api/queue/auto-add-decoded-min-count-5m",
                    post(api_queue_auto_add_decoded_min_count_5m_handler),
                )
                .route(
                    "/api/queue/auto-add-direct",
                    post(api_queue_auto_add_direct_handler),
                )
                .route(
                    "/api/queue/ignore-direct-worked",
                    post(api_queue_ignore_direct_worked_handler),
                )
                .route("/api/queue/cq-enabled", post(api_queue_cq_enabled_handler))
                .route("/api/queue/cq-percent", post(api_queue_cq_percent_handler))
                .route(
                    "/api/queue/pause-cq-low-activity",
                    post(api_queue_pause_cq_low_activity_handler),
                )
                .route(
                    "/api/queue/cq-min-unique-calls-5m",
                    post(api_queue_cq_min_unique_calls_5m_handler),
                )
                .route(
                    "/api/queue/compound-rr73-handoff",
                    post(api_queue_compound_rr73_handoff_handler),
                )
                .route(
                    "/api/queue/compound-73-once-handoff",
                    post(api_queue_compound_73_once_handoff_handler),
                )
                .route(
                    "/api/queue/compound-direct-signal",
                    post(api_queue_compound_direct_signal_handler),
                )
                .route(
                    "/api/queue/next-cq-parity",
                    post(api_queue_next_cq_parity_handler),
                )
                .route("/api/queue/tx-freq", post(api_queue_tx_freq_handler))
                .route(
                    "/api/queue/retry-delay",
                    post(api_queue_retry_delay_handler),
                )
                .with_state(state);
            match tokio::net::TcpListener::bind(addr).await {
                Ok(listener) => {
                    if let Err(error) = axum::serve(listener, app).await {
                        eprintln!("web server failed: {error}");
                    }
                }
                Err(error) => eprintln!("web bind failed on {addr}: {error}"),
            }
        });
    });
    Ok(())
}

async fn index_handler() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn api_state_handler(State(state): State<WebAppState>) -> Json<WebSnapshot> {
    Json(
        state
            .snapshot
            .lock()
            .expect("web snapshot poisoned")
            .clone(),
    )
}

async fn api_qso_stop_handler(State(state): State<WebAppState>) -> (StatusCode, Json<ApiStatus>) {
    state.qso_control.enqueue(QsoCommand::Stop {
        reason: "web_stop".to_string(),
    });
    (
        StatusCode::ACCEPTED,
        Json(ApiStatus {
            ok: true,
            message: "qso stop queued".to_string(),
        }),
    )
}

async fn api_queue_add_handler(
    State(state): State<WebAppState>,
    Json(request): Json<QueueCallRequest>,
) -> (StatusCode, Json<ApiStatus>) {
    let callsign = request.callsign.trim().to_uppercase();
    if callsign.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ApiStatus {
                ok: false,
                message: "callsign required".to_string(),
            }),
        );
    }
    state.queue_control.enqueue(QueueCommand::Add { callsign });
    (
        StatusCode::ACCEPTED,
        Json(ApiStatus {
            ok: true,
            message: "queue add queued".to_string(),
        }),
    )
}

async fn api_queue_remove_handler(
    State(state): State<WebAppState>,
    Json(request): Json<QueueCallRequest>,
) -> (StatusCode, Json<ApiStatus>) {
    let callsign = request.callsign.trim().to_uppercase();
    if callsign.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ApiStatus {
                ok: false,
                message: "callsign required".to_string(),
            }),
        );
    }
    state
        .queue_control
        .enqueue(QueueCommand::Remove { callsign });
    (
        StatusCode::ACCEPTED,
        Json(ApiStatus {
            ok: true,
            message: "queue remove queued".to_string(),
        }),
    )
}

async fn api_queue_clear_handler(
    State(state): State<WebAppState>,
) -> (StatusCode, Json<ApiStatus>) {
    state.queue_control.enqueue(QueueCommand::Clear);
    (
        StatusCode::ACCEPTED,
        Json(ApiStatus {
            ok: true,
            message: "queue clear queued".to_string(),
        }),
    )
}

async fn api_queue_auto_handler(
    State(state): State<WebAppState>,
    Json(request): Json<QueueAutoRequest>,
) -> (StatusCode, Json<ApiStatus>) {
    state.queue_control.enqueue(QueueCommand::SetAuto {
        enabled: request.enabled,
    });
    (
        StatusCode::ACCEPTED,
        Json(ApiStatus {
            ok: true,
            message: "queue auto updated".to_string(),
        }),
    )
}

async fn api_queue_auto_add_decoded_handler(
    State(state): State<WebAppState>,
    Json(request): Json<QueueFlagRequest>,
) -> (StatusCode, Json<ApiStatus>) {
    state
        .queue_control
        .enqueue(QueueCommand::SetAutoAddAllDecodedCalls {
            enabled: request.enabled,
        });
    (
        StatusCode::ACCEPTED,
        Json(ApiStatus {
            ok: true,
            message: "queue decoded auto-add updated".to_string(),
        }),
    )
}

async fn api_queue_auto_add_decoded_min_count_5m_handler(
    State(state): State<WebAppState>,
    Json(request): Json<QueueCountRequest>,
) -> (StatusCode, Json<ApiStatus>) {
    if !(1..=1000).contains(&request.count) {
        return (
            StatusCode::BAD_REQUEST,
            Json(ApiStatus {
                ok: false,
                message: "count must be between 1 and 1000".to_string(),
            }),
        );
    }
    state
        .queue_control
        .enqueue(QueueCommand::SetAutoAddDecodedMinCount5m {
            count: request.count,
        });
    (
        StatusCode::ACCEPTED,
        Json(ApiStatus {
            ok: true,
            message: "queue decoded auto-add threshold updated".to_string(),
        }),
    )
}

async fn api_queue_auto_add_direct_handler(
    State(state): State<WebAppState>,
    Json(request): Json<QueueFlagRequest>,
) -> (StatusCode, Json<ApiStatus>) {
    state.queue_control.enqueue(QueueCommand::SetAutoAddDirect {
        enabled: request.enabled,
    });
    (
        StatusCode::ACCEPTED,
        Json(ApiStatus {
            ok: true,
            message: "queue direct auto-add updated".to_string(),
        }),
    )
}

async fn api_queue_ignore_direct_worked_handler(
    State(state): State<WebAppState>,
    Json(request): Json<QueueFlagRequest>,
) -> (StatusCode, Json<ApiStatus>) {
    state
        .queue_control
        .enqueue(QueueCommand::SetIgnoreDirectWorked {
            enabled: request.enabled,
        });
    (
        StatusCode::ACCEPTED,
        Json(ApiStatus {
            ok: true,
            message: "queue direct worked filter updated".to_string(),
        }),
    )
}

async fn api_queue_cq_enabled_handler(
    State(state): State<WebAppState>,
    Json(request): Json<QueueFlagRequest>,
) -> (StatusCode, Json<ApiStatus>) {
    state.queue_control.enqueue(QueueCommand::SetCqEnabled {
        enabled: request.enabled,
    });
    (
        StatusCode::ACCEPTED,
        Json(ApiStatus {
            ok: true,
            message: "queue cq enabled updated".to_string(),
        }),
    )
}

async fn api_queue_cq_percent_handler(
    State(state): State<WebAppState>,
    Json(request): Json<QueueCqPercentRequest>,
) -> (StatusCode, Json<ApiStatus>) {
    if request.percent > 100 || request.percent % 10 != 0 {
        return (
            StatusCode::BAD_REQUEST,
            Json(ApiStatus {
                ok: false,
                message: "cq percent must be one of 0,10,...,100".to_string(),
            }),
        );
    }
    state.queue_control.enqueue(QueueCommand::SetCqPercent {
        percent: request.percent,
    });
    (
        StatusCode::ACCEPTED,
        Json(ApiStatus {
            ok: true,
            message: "queue cq percent updated".to_string(),
        }),
    )
}

async fn api_queue_pause_cq_low_activity_handler(
    State(state): State<WebAppState>,
    Json(request): Json<QueueFlagRequest>,
) -> (StatusCode, Json<ApiStatus>) {
    state
        .queue_control
        .enqueue(QueueCommand::SetPauseCqWhenFewUniqueCalls {
            enabled: request.enabled,
        });
    (
        StatusCode::ACCEPTED,
        Json(ApiStatus {
            ok: true,
            message: "queue low-activity cq pause updated".to_string(),
        }),
    )
}

async fn api_queue_cq_min_unique_calls_5m_handler(
    State(state): State<WebAppState>,
    Json(request): Json<QueueCountRequest>,
) -> (StatusCode, Json<ApiStatus>) {
    if request.count > 1000 {
        return (
            StatusCode::BAD_REQUEST,
            Json(ApiStatus {
                ok: false,
                message: "count must be between 0 and 1000".to_string(),
            }),
        );
    }
    state
        .queue_control
        .enqueue(QueueCommand::SetCqPauseMinUniqueCalls5m {
            count: request.count,
        });
    (
        StatusCode::ACCEPTED,
        Json(ApiStatus {
            ok: true,
            message: "queue cq 5m unique threshold updated".to_string(),
        }),
    )
}

async fn api_queue_compound_rr73_handoff_handler(
    State(state): State<WebAppState>,
    Json(request): Json<QueueFlagRequest>,
) -> (StatusCode, Json<ApiStatus>) {
    state
        .queue_control
        .enqueue(QueueCommand::SetCompoundRr73Handoff {
            enabled: request.enabled,
        });
    (
        StatusCode::ACCEPTED,
        Json(ApiStatus {
            ok: true,
            message: "compound rr73 handoff updated".to_string(),
        }),
    )
}

async fn api_queue_compound_73_once_handoff_handler(
    State(state): State<WebAppState>,
    Json(request): Json<QueueFlagRequest>,
) -> (StatusCode, Json<ApiStatus>) {
    state
        .queue_control
        .enqueue(QueueCommand::SetCompound73OnceHandoff {
            enabled: request.enabled,
        });
    (
        StatusCode::ACCEPTED,
        Json(ApiStatus {
            ok: true,
            message: "compound 73 once handoff updated".to_string(),
        }),
    )
}

async fn api_queue_compound_direct_signal_handler(
    State(state): State<WebAppState>,
    Json(request): Json<QueueFlagRequest>,
) -> (StatusCode, Json<ApiStatus>) {
    state
        .queue_control
        .enqueue(QueueCommand::SetCompoundForDirectSignalCallers {
            enabled: request.enabled,
        });
    (
        StatusCode::ACCEPTED,
        Json(ApiStatus {
            ok: true,
            message: "compound direct signal callers updated".to_string(),
        }),
    )
}

async fn api_queue_next_cq_parity_handler(
    State(state): State<WebAppState>,
) -> (StatusCode, Json<ApiStatus>) {
    state
        .queue_control
        .enqueue(QueueCommand::ToggleNextCqParity);
    (
        StatusCode::ACCEPTED,
        Json(ApiStatus {
            ok: true,
            message: "next cq parity toggled".to_string(),
        }),
    )
}

async fn api_queue_tx_freq_handler(
    State(state): State<WebAppState>,
    Json(request): Json<QueueTxFreqRequest>,
) -> (StatusCode, Json<ApiStatus>) {
    let snapshot = state
        .snapshot
        .lock()
        .expect("web snapshot poisoned")
        .clone();
    let min = snapshot.qso_defaults.tx_freq_min_hz;
    let max = snapshot.qso_defaults.tx_freq_max_hz;
    if !request.tx_freq_hz.is_finite() || request.tx_freq_hz < min || request.tx_freq_hz > max {
        return (
            StatusCode::BAD_REQUEST,
            Json(ApiStatus {
                ok: false,
                message: format!("tx freq must be between {min:.0} and {max:.0} Hz"),
            }),
        );
    }
    let Some(slot_family) = parse_slot_family_name(&request.slot_family) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(ApiStatus {
                ok: false,
                message: "slot_family must be 'even' or 'odd'".to_string(),
            }),
        );
    };
    state.queue_control.enqueue(QueueCommand::SetTxFreq {
        slot_family,
        tx_freq_hz: request.tx_freq_hz,
    });
    (
        StatusCode::ACCEPTED,
        Json(ApiStatus {
            ok: true,
            message: format!("queue {} tx freq updated", slot_family.as_str()),
        }),
    )
}

async fn api_queue_retry_delay_handler(
    State(state): State<WebAppState>,
    Json(request): Json<QueueRetryDelayRequest>,
) -> (StatusCode, Json<ApiStatus>) {
    if request.retry_delay_seconds == 0 || request.retry_delay_seconds > 3600 {
        return (
            StatusCode::BAD_REQUEST,
            Json(ApiStatus {
                ok: false,
                message: "retry delay must be between 1 and 3600 seconds".to_string(),
            }),
        );
    }
    let kind = match request.kind.as_str() {
        "no_message" => QueueRetryDelayKind::NoMessage,
        "no_forward" => QueueRetryDelayKind::NoForward,
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ApiStatus {
                    ok: false,
                    message: "kind must be 'no_message' or 'no_forward'".to_string(),
                }),
            );
        }
    };
    state.queue_control.enqueue(QueueCommand::SetRetryDelay {
        kind,
        retry_delay_seconds: request.retry_delay_seconds,
    });
    (
        StatusCode::ACCEPTED,
        Json(ApiStatus {
            ok: true,
            message: format!("queue {} retry delay updated", kind.as_str()),
        }),
    )
}

async fn api_rig_config_handler(
    State(state): State<WebAppState>,
    Json(request): Json<RigConfigRequest>,
) -> (StatusCode, Json<ApiStatus>) {
    info!(
        requested_band = %request.band,
        requested_app_mode = %request.app_mode,
        requested_power_w = ?request.power_w,
        requested_power_setting_id = ?request.power_setting_id,
        "api_rig_config_received"
    );
    let snapshot = state
        .snapshot
        .lock()
        .expect("web snapshot poisoned")
        .clone();
    if snapshot.rig_mode == "unavailable" {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ApiStatus {
                ok: false,
                message: "rig unavailable".to_string(),
            }),
        );
    }
    let tx_active =
        snapshot.qso.tx_active || snapshot.rig_tune_active || snapshot.rig_is_tx == Some(true);
    if tx_active {
        return (
            StatusCode::CONFLICT,
            Json(ApiStatus {
                ok: false,
                message: "transmit active".to_string(),
            }),
        );
    }
    let band = match request.band.parse::<Band>() {
        Ok(band) => band,
        Err(error) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ApiStatus {
                    ok: false,
                    message: error,
                }),
            );
        }
    };
    let app_mode = match parse_supported_app_mode(&request.app_mode) {
        Ok(mode) => mode,
        Err(error) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ApiStatus {
                    ok: false,
                    message: error,
                }),
            );
        }
    };
    let power = if snapshot.rig_power_is_discrete {
        if let Some(power_w) = request.power_w {
            return (
                StatusCode::BAD_REQUEST,
                Json(ApiStatus {
                    ok: false,
                    message: format!(
                        "continuous watts are not valid for this rig ({power_w:.1} requested)"
                    ),
                }),
            );
        }
        if let Some(setting_id) = request.power_setting_id.clone() {
            if !snapshot.rig_power_settable {
                if snapshot.rig_power_current_id.as_deref() == Some(setting_id.as_str()) {
                    None
                } else {
                    warn!(
                        requested_power_setting_id = %setting_id,
                        current_power_setting_id = ?snapshot.rig_power_current_id,
                        "rig_config_rejected_power_not_settable"
                    );
                    return (
                        StatusCode::CONFLICT,
                        Json(ApiStatus {
                            ok: false,
                            message: "power setting is not CAT-settable on this rig".to_string(),
                        }),
                    );
                }
            } else {
                if !snapshot
                    .rig_power_settings
                    .iter()
                    .any(|setting| setting.id == setting_id)
                {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(ApiStatus {
                            ok: false,
                            message: "unknown power setting".to_string(),
                        }),
                    );
                }
                Some(RigPowerRequest::SettingId(setting_id))
            }
        } else {
            None
        }
    } else if let Some(power_w) = request.power_w {
        if !(0.1..=110.0).contains(&power_w) {
            return (
                StatusCode::BAD_REQUEST,
                Json(ApiStatus {
                    ok: false,
                    message: "power must be between 0.1 and 110.0 W".to_string(),
                }),
            );
        }
        Some(RigPowerRequest::ContinuousWatts(power_w))
    } else {
        None
    };
    if snapshot.qso.active && snapshot.rig_band != band.to_string() {
        return (
            StatusCode::CONFLICT,
            Json(ApiStatus {
                ok: false,
                message: "cannot change band during active qso".to_string(),
            }),
        );
    }
    if snapshot.qso.active && snapshot.app_mode != app_mode.as_str().to_uppercase() {
        return (
            StatusCode::CONFLICT,
            Json(ApiStatus {
                ok: false,
                message: "cannot change mode during active qso".to_string(),
            }),
        );
    }
    info!(
        requested_band = %band,
        requested_app_mode = %app_mode.as_str().to_uppercase(),
        requested_power = ?power,
        current_rig_kind = ?snapshot.rig_kind,
        current_band = %snapshot.rig_band,
        current_frequency_hz = ?snapshot.rig_frequency_hz,
        "rig_config_request_queued"
    );
    state.rig_control.enqueue(RigCommand::Configure {
        band,
        power,
        app_mode,
    });
    (
        StatusCode::ACCEPTED,
        Json(ApiStatus {
            ok: true,
            message: "rig config queued".to_string(),
        }),
    )
}

fn parse_supported_app_mode(value: &str) -> Result<DecoderMode, String> {
    let mode = value.parse::<DecoderMode>()?;
    match mode {
        DecoderMode::Ft8 | DecoderMode::Ft4 => Ok(mode),
        DecoderMode::Ft2 => Err("FT2 is not supported in ft8rx; use FT8 or FT4".to_string()),
    }
}

async fn api_rig_tune_handler(State(state): State<WebAppState>) -> (StatusCode, Json<ApiStatus>) {
    if !state.tune_available {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ApiStatus {
                ok: false,
                message: "audio output unavailable".to_string(),
            }),
        );
    }
    let snapshot = state
        .snapshot
        .lock()
        .expect("web snapshot poisoned")
        .clone();
    if snapshot.rig_mode == "unavailable" {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ApiStatus {
                ok: false,
                message: "rig unavailable".to_string(),
            }),
        );
    }
    if snapshot.qso.active
        || snapshot.qso.tx_active
        || snapshot.rig_tune_active
        || snapshot.rig_is_tx == Some(true)
    {
        return (
            StatusCode::CONFLICT,
            Json(ApiStatus {
                ok: false,
                message: "rig busy".to_string(),
            }),
        );
    }
    state.rig_control.enqueue(RigCommand::Tune10s);
    (
        StatusCode::ACCEPTED,
        Json(ApiStatus {
            ok: true,
            message: "tune queued".to_string(),
        }),
    )
}

#[derive(Clone)]
struct SharedFileWriter(Arc<Mutex<std::fs::File>>);

impl<'a> MakeWriter<'a> for SharedFileWriter {
    type Writer = SharedFileGuard<'a>;

    fn make_writer(&'a self) -> Self::Writer {
        SharedFileGuard(self.0.lock().expect("log file mutex poisoned"))
    }
}

struct SharedFileGuard<'a>(std::sync::MutexGuard<'a, std::fs::File>);

impl std::io::Write for SharedFileGuard<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.0.flush()
    }
}

fn init_tracing(config: &AppConfig) -> Result<(), AppError> {
    let json_path = PathBuf::from(&config.logging.fsm_log_path);
    if let Some(parent) = json_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let text_path = PathBuf::from(&config.logging.app_log_path);
    if let Some(parent) = text_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(json_path)?;
    let text_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(text_path)?;
    let json_writer = SharedFileWriter(Arc::new(Mutex::new(json_file)));
    let text_writer = SharedFileWriter(Arc::new(Mutex::new(text_file)));
    let json_layer = tracing_subscriber::fmt::layer()
        .with_writer(json_writer)
        .json()
        .with_current_span(false)
        .with_target(false);
    let text_layer = tracing_subscriber::fmt::layer()
        .with_writer(text_writer)
        .with_ansi(false)
        .with_target(false);
    let subscriber = Registry::default().with(json_layer).with(text_layer);
    let _ = tracing::subscriber::set_global_default(subscriber);
    Ok(())
}

fn scan_qso_jsonl(contents: &str, now: SystemTime, direct_calls_since: SystemTime) -> QsoJsonlScan {
    let mut sessions = BTreeMap::<QsoHistoryKey, QsoJsonlSessionSummary>::new();
    let mut active_keys = BTreeMap::<u64, QsoHistoryKey>::new();
    let mut next_ordinal = 1_u64;
    let mut direct_calls = Vec::<WebDirectCallLog>::new();
    let mut recent_worked = BTreeMap::<WorkedBandKey, SystemTime>::new();
    for line in contents.lines() {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let Some(fields) = value.get("fields").and_then(|fields| fields.as_object()) else {
            continue;
        };
        if fields.get("message").and_then(|value| value.as_str()) != Some("qso_fsm") {
            continue;
        }
        let Some(session_id) = fields.get("session_id").and_then(|value| value.as_u64()) else {
            continue;
        };
        let Some(partner_call) = fields.get("partner_call").and_then(|value| value.as_str()) else {
            continue;
        };
        let Some(timestamp_str) = value.get("timestamp").and_then(|value| value.as_str()) else {
            continue;
        };
        let Ok(parsed) = DateTime::parse_from_rfc3339(timestamp_str) else {
            continue;
        };
        let timestamp: SystemTime = parsed.with_timezone(&Utc).into();
        let event = fields
            .get("event")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        let state_before = fields
            .get("state_before")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        let state_after = fields
            .get("state_after")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        let last_rx_event = fields
            .get("last_rx_event")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        let rx_text = fields
            .get("rx_text")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        let tx_text = fields
            .get("tx_text")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        let compound_next_call = fields
            .get("compound_next_call")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        let rig_band = fields
            .get("rig_band")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        let app_mode = fields
            .get("app_mode")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        let key = if event == "start" {
            let key = QsoHistoryKey {
                session_id,
                ordinal: next_ordinal,
            };
            next_ordinal += 1;
            active_keys.insert(session_id, key);
            key
        } else if let Some(existing) = active_keys.get(&session_id).copied() {
            existing
        } else {
            let key = QsoHistoryKey {
                session_id,
                ordinal: next_ordinal,
            };
            next_ordinal += 1;
            active_keys.insert(session_id, key);
            key
        };
        let session = sessions
            .entry(key)
            .or_insert_with(|| QsoJsonlSessionSummary {
                partner_call: partner_call.to_string(),
                ..QsoJsonlSessionSummary::default()
            });
        if session.partner_call.is_empty() {
            session.partner_call = partner_call.to_string();
        }
        if !rig_band.is_empty() && session.rig_band.is_none() {
            session.rig_band = Some(rig_band.to_string());
        }
        if !app_mode.is_empty() && session.app_mode.is_none() {
            session.app_mode = Some(app_mode.to_uppercase());
        }
        session.last_seen_at = Some(timestamp);
        if event == "start" && session.started_at.is_none() {
            session.started_at = Some(timestamp);
        }
        if event == "exit" {
            session.ended_at = Some(timestamp);
            session.exit_reason = Some(last_rx_event.to_string());
            active_keys.remove(&session_id);
        }
        if last_rx_event.starts_with("to_us_") {
            session.got_reply = true;
        }
        if matches!(
            last_rx_event,
            "to_us_ack" | "to_us_reply_rrr" | "to_us_reply_rr73" | "to_us_reply_73"
        ) {
            session.got_roger = true;
        }
        if matches!(state_before, "send_rr73" | "send_73" | "send_73_once")
            || matches!(state_after, "send_rr73" | "send_73" | "send_73_once")
        {
            session.reached_73 = true;
        }
        if event == "tx_launch" && matches!(state_before, "send_rr73" | "send_73" | "send_73_once")
        {
            session.reached_73 = true;
            if now.duration_since(timestamp).unwrap_or_default() <= RECENT_WORKED_RETENTION
                && !rig_band.is_empty()
            {
                let key = WorkedBandKey {
                    callsign: partner_call.to_string(),
                    band: rig_band.to_string(),
                };
                match recent_worked.get(&key).copied() {
                    Some(existing) if existing >= timestamp => {}
                    _ => {
                        recent_worked.insert(key, timestamp);
                    }
                }
            }
        }
        if event == "tx_launch" && !tx_text.is_empty() && timestamp >= direct_calls_since {
            let (to_call, from_call) = parse_direct_message_calls(tx_text)
                .unwrap_or_else(|| (partner_call.to_string(), "TX".to_string()));
            direct_calls.push(WebDirectCallLog {
                sort_epoch_ms: system_time_to_epoch_ms(timestamp),
                timestamp: format_time(timestamp),
                from_call,
                to_call,
                snr_db: None,
                dt_seconds: None,
                freq_hz: None,
                text: tx_text.to_string(),
                is_ours: true,
            });
        }
        if let Some(info) =
            extract_tx_exchange_info(tx_text, event, partner_call, compound_next_call)
        {
            push_unique_info(&mut session.sent_infos, info);
        }
        if let Some(info) = extract_exchange_info(rx_text) {
            push_unique_info(&mut session.received_infos, info);
        }
    }

    let mut session_rows = sessions
        .into_values()
        .filter_map(|session| {
            let time = session
                .ended_at
                .or(session.last_seen_at)
                .or(session.started_at)?;
            let age = now.duration_since(time).unwrap_or_default();
            Some((
                time,
                WebQsoHistoryEntry {
                    time: format_datetime(time),
                    age: format_relative_age(age),
                    age_seconds: age.as_secs(),
                    callsign: session.partner_call,
                    band: session.rig_band.unwrap_or_else(|| "-".to_string()),
                    mode: session.app_mode.unwrap_or_else(|| "-".to_string()),
                    sent_info: if session.sent_infos.is_empty() {
                        "-".to_string()
                    } else {
                        session.sent_infos.join(", ")
                    },
                    received_info: if session.received_infos.is_empty() {
                        "-".to_string()
                    } else {
                        session.received_infos.join(", ")
                    },
                    got_roger: session.got_roger,
                    got_reply: session.got_reply,
                    reached_73: session.reached_73,
                    exit_reason: session
                        .exit_reason
                        .unwrap_or_else(|| "in_progress".to_string()),
                },
            ))
        })
        .collect::<Vec<_>>();
    session_rows.retain(|(_, entry)| entry.age_seconds <= RECENT_WORKED_RETENTION.as_secs());
    session_rows.sort_by_key(|row| std::cmp::Reverse(row.0));
    let history = session_rows
        .into_iter()
        .map(|(_, entry)| entry)
        .collect::<Vec<_>>();
    direct_calls.sort_by_key(|entry| entry.sort_epoch_ms);
    QsoJsonlScan {
        history,
        direct_calls,
        recent_worked,
    }
}

fn extract_exchange_info(text: &str) -> Option<String> {
    let mut parts = text.split_whitespace();
    let _first = parts.next()?;
    let _second = parts.next()?;
    let info = parts.next()?.trim().to_uppercase();
    if info.is_empty() {
        return None;
    }
    if matches!(info.as_str(), "73" | "RR73" | "RRR") {
        return None;
    }
    Some(info)
}

fn extract_tx_exchange_info(
    text: &str,
    event: &str,
    partner_call: &str,
    compound_next_call: &str,
) -> Option<String> {
    if text.contains(" RR73; ") {
        if event == "compound_start"
            || (!compound_next_call.is_empty()
                && compound_next_call.eq_ignore_ascii_case(partner_call))
        {
            return text
                .split_whitespace()
                .last()
                .map(|report| report.trim().to_uppercase())
                .filter(|report| !report.is_empty());
        }
        return None;
    }
    extract_exchange_info(text)
}

fn parse_direct_message_calls(text: &str) -> Option<(String, String)> {
    if let Some((_, rest)) = text.split_once(" RR73; ") {
        let mut parts = rest.split_whitespace();
        let to_call = parts.next()?.trim();
        let from_call = parts
            .next()?
            .trim()
            .trim_start_matches('<')
            .trim_end_matches('>');
        if !to_call.is_empty() && !from_call.is_empty() {
            return Some((to_call.to_string(), from_call.to_string()));
        }
    }
    let mut parts = text.split_whitespace();
    let to_call = parts.next()?.trim();
    let from_call = parts.next()?.trim();
    if to_call.is_empty() || from_call.is_empty() {
        return None;
    }
    Some((to_call.to_string(), from_call.to_string()))
}

fn push_unique_info(values: &mut Vec<String>, value: String) {
    if !values.iter().any(|existing| existing == &value) {
        values.push(value);
    }
}

fn build_adif_export_report(
    contents: &str,
    station_callsign: &str,
    my_gridsquare: &str,
) -> AdifExportReport {
    let mut sessions = BTreeMap::<QsoHistoryKey, AdifSessionSummary>::new();
    let mut active_keys = BTreeMap::<u64, QsoHistoryKey>::new();
    let mut next_ordinal = 1_u64;
    for line in contents.lines() {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let Some(fields) = value.get("fields").and_then(|fields| fields.as_object()) else {
            continue;
        };
        if fields.get("message").and_then(|value| value.as_str()) != Some("qso_fsm") {
            continue;
        }
        let Some(session_id) = fields.get("session_id").and_then(|value| value.as_u64()) else {
            continue;
        };
        let Some(partner_call) = fields.get("partner_call").and_then(|value| value.as_str()) else {
            continue;
        };
        if partner_call.eq_ignore_ascii_case("CQ") {
            continue;
        }
        let Some(timestamp_str) = value.get("timestamp").and_then(|value| value.as_str()) else {
            continue;
        };
        let Ok(parsed) = DateTime::parse_from_rfc3339(timestamp_str) else {
            continue;
        };
        let timestamp: SystemTime = parsed.with_timezone(&Utc).into();
        let event = fields
            .get("event")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        let state_before = fields
            .get("state_before")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        let state_after = fields
            .get("state_after")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        let last_rx_event = fields
            .get("last_rx_event")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        let rx_text = fields
            .get("rx_text")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        let tx_text = fields
            .get("tx_text")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        let compound_next_call = fields
            .get("compound_next_call")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        let rig_band = fields
            .get("rig_band")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        let rig_frequency_hz = fields
            .get("rig_frequency_hz")
            .and_then(|value| value.as_u64())
            .filter(|value| *value > 0);
        let app_mode = fields
            .get("app_mode")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        let key = if event == "start" {
            let key = QsoHistoryKey {
                session_id,
                ordinal: next_ordinal,
            };
            next_ordinal += 1;
            active_keys.insert(session_id, key);
            key
        } else if let Some(existing) = active_keys.get(&session_id).copied() {
            existing
        } else {
            let key = QsoHistoryKey {
                session_id,
                ordinal: next_ordinal,
            };
            next_ordinal += 1;
            active_keys.insert(session_id, key);
            key
        };
        let session = sessions.entry(key).or_insert_with(|| AdifSessionSummary {
            session_id,
            partner_call: partner_call.to_string(),
            ..AdifSessionSummary::default()
        });
        if session.partner_call.is_empty() {
            session.partner_call = partner_call.to_string();
        }
        session.last_seen_at = Some(timestamp);
        if event == "start" && session.started_at.is_none() {
            session.started_at = Some(timestamp);
        }
        if event == "exit" {
            session.ended_at = Some(timestamp);
            session.exit_reason = Some(last_rx_event.to_string());
            active_keys.remove(&session_id);
        }
        if !rig_band.is_empty() && session.rig_band.is_none() {
            session.rig_band = Some(rig_band.to_string());
        }
        if session.rig_frequency_hz.is_none() {
            session.rig_frequency_hz = rig_frequency_hz;
        }
        if !app_mode.is_empty() && session.app_mode.is_none() {
            session.app_mode = Some(app_mode.to_uppercase());
        }
        if let Some(observed) = extract_station_call_from_tx_text(tx_text) {
            session.observed_station_calls.insert(observed);
        }
        if let Some(observed) = extract_station_call_from_rx_text(rx_text) {
            session.observed_station_calls.insert(observed);
        }
        if last_rx_event.starts_with("to_us_") {
            session.got_reply = true;
        }
        if matches!(
            last_rx_event,
            "to_us_ack" | "to_us_reply_rrr" | "to_us_reply_rr73" | "to_us_reply_73"
        ) {
            session.got_roger = true;
        }
        if matches!(state_before, "send_rr73" | "send_73" | "send_73_once")
            || matches!(state_after, "send_rr73" | "send_73" | "send_73_once")
        {
            session.reached_73 = true;
        }
        if let Some(exchange) = classify_exchange_info(rx_text) {
            match exchange {
                ExchangeInfo::Grid(grid) => {
                    if session.remote_grid.is_none() {
                        session.remote_grid = Some(grid);
                    }
                }
                ExchangeInfo::Report(report) => {
                    if session.received_report.is_none() {
                        session.received_report = Some(report);
                    }
                }
            }
        }
        if let Some(tx_info) =
            extract_tx_exchange_info(tx_text, event, partner_call, compound_next_call)
        {
            if let Some(exchange) = classify_exchange_token(&tx_info) {
                match exchange {
                    ExchangeInfo::Grid(_) => {}
                    ExchangeInfo::Report(report) => {
                        if session.sent_report.is_none() {
                            session.sent_report = Some(report);
                        }
                    }
                }
            }
        }
    }

    let mut eligible_records = Vec::new();
    let mut ineligible_sessions = Vec::new();
    for session in sessions.into_values() {
        match adif_record_from_session(session, station_callsign, my_gridsquare) {
            Ok(record) => eligible_records.push(record),
            Err(ineligible) => ineligible_sessions.push(ineligible),
        }
    }
    eligible_records.sort_by(|left, right| {
        left.qso_date
            .cmp(&right.qso_date)
            .then(left.time_on.cmp(&right.time_on))
            .then(left.call.cmp(&right.call))
    });
    ineligible_sessions.sort_by(|left, right| {
        left.started_at
            .unwrap_or(UNIX_EPOCH)
            .cmp(&right.started_at.unwrap_or(UNIX_EPOCH))
            .then(left.call.cmp(&right.call))
    });
    AdifExportReport {
        eligible_records,
        ineligible_sessions,
    }
}

#[cfg(test)]
fn collect_adif_records(
    contents: &str,
    station_callsign: &str,
    my_gridsquare: &str,
) -> Vec<AdifRecord> {
    build_adif_export_report(contents, station_callsign, my_gridsquare).eligible_records
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ExchangeInfo {
    Grid(String),
    Report(String),
}

fn classify_exchange_info(text: &str) -> Option<ExchangeInfo> {
    let mut parts = text.split_whitespace();
    let _first = parts.next()?;
    let _second = parts.next()?;
    let token = parts.next()?;
    classify_exchange_token(token)
}

fn classify_exchange_token(token: &str) -> Option<ExchangeInfo> {
    let token = token.trim().to_uppercase();
    if token.is_empty() || matches!(token.as_str(), "73" | "RR73" | "RRR") {
        return None;
    }
    if let Some(report) = normalize_signal_report(&token) {
        return Some(ExchangeInfo::Report(report));
    }
    if is_gridsquare(&token) {
        return Some(ExchangeInfo::Grid(token));
    }
    None
}

fn normalize_signal_report(token: &str) -> Option<String> {
    let raw = token.strip_prefix('R').unwrap_or(token);
    let bytes = raw.as_bytes();
    if bytes.len() != 3 {
        return None;
    }
    if bytes[0] != b'+' && bytes[0] != b'-' {
        return None;
    }
    if !bytes[1].is_ascii_digit() || !bytes[2].is_ascii_digit() {
        return None;
    }
    Some(raw.to_string())
}

fn extract_station_call_from_tx_text(text: &str) -> Option<String> {
    if text.is_empty() {
        return None;
    }
    if let Some((_, rest)) = text.split_once(" RR73; ") {
        let mut parts = rest.split_whitespace();
        let _partner = parts.next()?;
        return normalize_call_token(parts.next()?);
    }
    let mut parts = text.split_whitespace();
    let first = parts.next()?;
    let second = parts.next()?;
    if first.eq_ignore_ascii_case("CQ") {
        return normalize_call_token(second);
    }
    normalize_call_token(second)
}

fn extract_station_call_from_rx_text(text: &str) -> Option<String> {
    let mut parts = text.split_whitespace();
    let first = parts.next()?;
    if first.eq_ignore_ascii_case("CQ") {
        return None;
    }
    normalize_call_token(first)
}

fn normalize_call_token(token: &str) -> Option<String> {
    let normalized = token
        .trim()
        .trim_start_matches('<')
        .trim_end_matches('>')
        .trim();
    if normalized.is_empty() {
        return None;
    }
    Some(normalized.to_uppercase())
}

fn is_gridsquare(token: &str) -> bool {
    let bytes = token.as_bytes();
    if !(bytes.len() == 4 || bytes.len() == 6) {
        return false;
    }
    bytes[0].is_ascii_uppercase()
        && bytes[1].is_ascii_uppercase()
        && bytes[0] <= b'R'
        && bytes[1] <= b'R'
        && bytes[2].is_ascii_digit()
        && bytes[3].is_ascii_digit()
        && (bytes.len() == 4
            || (bytes[4].is_ascii_uppercase()
                && bytes[5].is_ascii_uppercase()
                && bytes[4] <= b'X'
                && bytes[5] <= b'X'))
}

fn adif_record_from_session(
    session: AdifSessionSummary,
    station_callsign: &str,
    my_gridsquare: &str,
) -> Result<AdifRecord, IneligibleQso> {
    let started_at = session.started_at.or(session.last_seen_at);
    let ended_at = session.ended_at.or(session.last_seen_at);
    let configured_station = station_callsign.trim().to_uppercase();
    let mismatched_station_calls = session
        .observed_station_calls
        .iter()
        .filter(|call| *call != &configured_station)
        .cloned()
        .collect::<Vec<_>>();
    let primary_reason = if started_at.is_none() {
        Some("missing QSO start time".to_string())
    } else if !session.observed_station_calls.is_empty()
        && !session.observed_station_calls.contains(&configured_station)
    {
        Some(format!(
            "session logged under different station callsign(s): {}",
            mismatched_station_calls.join(", ")
        ))
    } else if !session.got_reply {
        Some("no directed reply from peer".to_string())
    } else if session.received_report.is_none() {
        Some("missing received report".to_string())
    } else if session.sent_report.is_none() {
        Some("missing sent report".to_string())
    } else if !session.got_roger {
        Some("no roger/ack from peer".to_string())
    } else if !session.reached_73 {
        Some("we never sent terminal 73/RR73".to_string())
    } else if session
        .rig_band
        .as_deref()
        .and_then(normalize_adif_band)
        .is_none()
    {
        Some("missing rig band".to_string())
    } else if session
        .app_mode
        .as_deref()
        .and_then(normalize_adif_mode)
        .is_none()
    {
        Some("missing or unsupported app mode".to_string())
    } else {
        None
    };
    if let Some(reason) = primary_reason {
        return Err(IneligibleQso {
            session_id: session.session_id,
            call: session.partner_call,
            started_at,
            ended_at,
            reason,
        });
    }
    let band = session
        .rig_band
        .as_deref()
        .and_then(normalize_adif_band)
        .expect("validated band");
    let mode_parts = session
        .app_mode
        .as_deref()
        .and_then(normalize_adif_mode)
        .expect("validated mode");
    let rst_sent = session.sent_report.clone().expect("validated sent report");
    let rst_rcvd = session
        .received_report
        .clone()
        .expect("validated received report");
    let (mode, submode) = mode_parts;
    Ok(AdifRecord {
        session_id: session.session_id,
        call: session.partner_call,
        qso_date: format_adif_date(started_at.expect("validated start")),
        time_on: format_adif_time(started_at.expect("validated start")),
        qso_date_off: ended_at.map(format_adif_date),
        time_off: ended_at.map(format_adif_time),
        band,
        freq: session.rig_frequency_hz.map(format_adif_freq_mhz),
        mode: mode.to_string(),
        submode: submode.map(str::to_string),
        rst_sent,
        rst_rcvd,
        station_callsign: station_callsign.to_string(),
        my_gridsquare: my_gridsquare.to_string(),
        gridsquare: session.remote_grid,
    })
}

fn normalize_adif_band(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(trimmed.to_uppercase())
}

fn normalize_adif_mode(value: &str) -> Option<(&'static str, Option<&'static str>)> {
    match value.trim().to_uppercase().as_str() {
        "FT8" => Some(("MFSK", Some("FT8"))),
        "FT4" => Some(("MFSK", Some("FT4"))),
        _ => None,
    }
}

fn format_adif_date(time: SystemTime) -> String {
    let timestamp: DateTime<Utc> = time.into();
    timestamp.format("%Y%m%d").to_string()
}

fn format_adif_time(time: SystemTime) -> String {
    let timestamp: DateTime<Utc> = time.into();
    timestamp.format("%H%M%S").to_string()
}

fn format_adif_freq_mhz(frequency_hz: u64) -> String {
    format!("{:.6}", frequency_hz as f64 / 1_000_000.0)
}

fn print_adif_export_status(report: &AdifExportReport) {
    eprintln!(
        "ADIF export status: {} eligible, {} ineligible",
        report.eligible_records.len(),
        report.ineligible_sessions.len()
    );
    if !report.ineligible_sessions.is_empty() {
        let mut by_reason = BTreeMap::<String, usize>::new();
        for session in &report.ineligible_sessions {
            *by_reason.entry(session.reason.clone()).or_default() += 1;
        }
        eprintln!("Ineligible summary:");
        for (reason, count) in by_reason {
            eprintln!("  {count:>3}  {reason}");
        }
        eprintln!("Ineligible QSOs:");
        for session in &report.ineligible_sessions {
            let when = session
                .started_at
                .or(session.ended_at)
                .map(format_datetime)
                .unwrap_or_else(|| "-".to_string());
            eprintln!(
                "  {}  #{}  {}  {}",
                when, session.session_id, session.call, session.reason
            );
        }
    }

    let unique_grids = report
        .eligible_records
        .iter()
        .filter_map(|record| record.gridsquare.clone())
        .collect::<BTreeSet<_>>();
    let calls = report
        .eligible_records
        .iter()
        .map(|record| record.call.clone())
        .collect::<Vec<_>>();
    match lookup_cty_for_calls(&calls) {
        Ok(lookups) => {
            let unique_countries = lookups
                .values()
                .map(|entry| entry.country.clone())
                .collect::<BTreeSet<_>>();
            let unique_dxcc = lookups
                .values()
                .map(|entry| entry.dxcc_adif)
                .collect::<BTreeSet<_>>();
            eprintln!(
                "Eligible summary: {} unique grids, {} unique countries, {} unique DXCC entities",
                unique_grids.len(),
                unique_countries.len(),
                unique_dxcc.len()
            );
        }
        Err(error) => {
            eprintln!(
                "Eligible summary: {} unique grids; country/DXCC summary unavailable ({error})",
                unique_grids.len()
            );
        }
    }
}

fn lookup_cty_for_calls(calls: &[String]) -> Result<BTreeMap<String, CtyLookup>, String> {
    if calls.is_empty() {
        return Ok(BTreeMap::new());
    }
    let mut child = Command::new("python3")
        .arg("-c")
        .arg(
            r#"
import dbm.gnu, json, marshal, os, sys

def variants(call):
    call = call.strip().upper()
    values = []
    if call:
        values.append(call)
        for part in call.split('/'):
            part = part.strip()
            if part and part not in values:
                values.append(part)
    return values

def resolve(db, call):
    best = None
    for candidate in variants(call):
        for width in range(len(candidate), 0, -1):
            key = candidate[:width].encode()
            if key not in db:
                continue
            value = marshal.loads(db[key])
            if value.get('ExactCallsign') and width != len(candidate):
                continue
            best = value
            break
        if best is not None:
            break
    if best is None:
        return None
    adif = best.get('ADIF')
    country = best.get('Country')
    if adif is None or not country:
        return None
    return {'country': country, 'dxcc_adif': int(adif)}

path = os.path.expanduser(os.environ.get('CTY_DB_PATH', '~/.local/cty'))
calls = json.load(sys.stdin)
db = dbm.gnu.open(path, 'r')
result = {}
for call in calls:
    resolved = resolve(db, call)
    if resolved is not None:
        result[call] = resolved
print(json.dumps(result))
"#,
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("failed to start python3 lookup helper: {error}"))?;
    let input =
        serde_json::to_vec(calls).map_err(|error| format!("json encode failed: {error}"))?;
    child
        .stdin
        .as_mut()
        .ok_or_else(|| "python3 lookup helper stdin unavailable".to_string())?
        .write_all(&input)
        .map_err(|error| format!("failed to write calls to lookup helper: {error}"))?;
    let output = child
        .wait_with_output()
        .map_err(|error| format!("python3 lookup helper failed: {error}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(if stderr.is_empty() {
            format!("python3 lookup helper exited with {}", output.status)
        } else {
            stderr
        });
    }
    let raw = serde_json::from_slice::<BTreeMap<String, serde_json::Value>>(&output.stdout)
        .map_err(|error| format!("lookup helper returned invalid json: {error}"))?;
    let mut resolved = BTreeMap::new();
    for (call, value) in raw {
        let Some(country) = value.get("country").and_then(|value| value.as_str()) else {
            continue;
        };
        let Some(dxcc_adif) = value
            .get("dxcc_adif")
            .and_then(|value| value.as_u64())
            .and_then(|value| u32::try_from(value).ok())
        else {
            continue;
        };
        resolved.insert(
            call,
            CtyLookup {
                country: country.to_string(),
                dxcc_adif,
            },
        );
    }
    Ok(resolved)
}

fn render_adif(records: &[AdifRecord]) -> String {
    let mut output = String::new();
    append_adif_field(&mut output, "ADIF_VER", "3.1.4");
    append_adif_field(&mut output, "PROGRAMID", "ft8rx");
    append_adif_field(&mut output, "PROGRAMVERSION", env!("CARGO_PKG_VERSION"));
    output.push_str("<EOH>\n");
    for record in records {
        append_adif_field(&mut output, "CALL", &record.call);
        append_adif_field(&mut output, "QSO_DATE", &record.qso_date);
        append_adif_field(&mut output, "TIME_ON", &record.time_on);
        if let Some(value) = record.qso_date_off.as_deref() {
            append_adif_field(&mut output, "QSO_DATE_OFF", value);
        }
        if let Some(value) = record.time_off.as_deref() {
            append_adif_field(&mut output, "TIME_OFF", value);
        }
        append_adif_field(&mut output, "BAND", &record.band);
        if let Some(value) = record.freq.as_deref() {
            append_adif_field(&mut output, "FREQ", value);
        }
        append_adif_field(&mut output, "MODE", &record.mode);
        if let Some(value) = record.submode.as_deref() {
            append_adif_field(&mut output, "SUBMODE", value);
        }
        append_adif_field(&mut output, "RST_SENT", &record.rst_sent);
        append_adif_field(&mut output, "RST_RCVD", &record.rst_rcvd);
        append_adif_field(&mut output, "STATION_CALLSIGN", &record.station_callsign);
        append_adif_field(&mut output, "MY_GRIDSQUARE", &record.my_gridsquare);
        if let Some(value) = record.gridsquare.as_deref() {
            append_adif_field(&mut output, "GRIDSQUARE", value);
        }
        output.push_str("<EOR>\n");
    }
    output
}

fn append_adif_field(output: &mut String, name: &str, value: &str) {
    let _ = write!(output, "<{}:{}>{}", name, value.len(), value);
}

fn resolve_app_rig_kind(config: &AppConfig) -> Result<RigKind, AppError> {
    resolve_rig_kind(
        config
            .rig
            .as_ref()
            .and_then(|rig| rig.kind.as_deref())
            .map(str::parse)
            .transpose()
            .map_err(|error: String| {
                AppError::Rig(rigctl::Error::Unsupported { operation: error })
            })?,
    )
    .map_err(AppError::Rig)
}

fn connect_app_rig(config: &AppConfig, kind: RigKind) -> Result<Rig, AppError> {
    Rig::connect(RigConnectionConfig {
        kind,
        port_path: config.rig.as_ref().and_then(|rig| rig.port_path.clone()),
        timeout: rigctl::DEFAULT_TIMEOUT,
    })
    .map_err(AppError::Rig)
}

fn configured_input_device_override(config: &AppConfig) -> Option<&str> {
    config
        .rig
        .as_ref()
        .and_then(|rig| rig.input_device.as_deref())
}

fn configured_output_device_override(config: &AppConfig) -> Option<&str> {
    config
        .rig
        .as_ref()
        .and_then(|rig| rig.output_device.as_deref())
        .or(config.tx.output_device.as_deref())
}

fn configured_power_request(config: &AppConfig, kind: RigKind) -> Option<RigPowerRequest> {
    match kind {
        RigKind::K3s => config
            .rig
            .as_ref()
            .and_then(|rig| rig.power_w)
            .or(config.tx.power_w)
            .map(RigPowerRequest::ContinuousWatts),
        RigKind::Mchf => config
            .rig
            .as_ref()
            .and_then(|rig| rig.power_setting.clone())
            .map(RigPowerRequest::SettingId),
    }
}

fn parse_decode_profile(value: &str) -> Result<DecodeProfile, AppError> {
    value
        .parse::<DecodeProfile>()
        .map_err(AppError::InvalidArgument)
}

fn run_continuous(cli: Cli) -> Result<(), AppError> {
    let config = AppConfig::load(&cli.config)?;
    let decode_profile = parse_decode_profile(&cli.decode_profile)?;
    init_tracing(&config)?;
    let rig_kind = resolve_app_rig_kind(&config)?;
    let audio = detect_audio_device_for_rig(
        rig_kind,
        cli.device
            .as_deref()
            .or(configured_input_device_override(&config)),
    )?;
    let capture = SampleStream::start(audio.clone(), AudioStreamConfig::default())?;
    let rig: SharedRig = Arc::new(Mutex::new(Some(connect_app_rig(&config, rig_kind)?)));
    if let Some(power) = configured_power_request(&config, rig_kind) {
        if let Some(rig_inner) = rig.lock().expect("rig mutex poisoned").as_mut() {
            let _ = rig_inner.apply_power_request(&power);
        }
    }
    let tx_busy = Arc::new(AtomicBool::new(false));
    let tune_active = Arc::new(AtomicBool::new(false));
    let output_device =
        detect_audio_output_device_for_rig(rig_kind, configured_output_device_override(&config));
    let tx_backend: Box<dyn qso::TxBackend> = match &output_device {
        Ok(device) => Box::new(RigTxBackend::new(
            Arc::clone(&rig),
            device.clone(),
            Arc::clone(&tx_busy),
        )),
        Err(error) => Box::new(qso::UnavailableTxBackend::new(error.to_string())),
    };
    let mut qso_controller = QsoController::new(config.clone(), tx_backend);
    let process_started_at = SystemTime::now();
    let mut qso_jsonl_cache = QsoJsonlCache::new(
        PathBuf::from(&config.logging.fsm_log_path),
        process_started_at,
    );
    qso_jsonl_cache.refresh(process_started_at);
    let qso_jsonl_refresh_rx = spawn_qso_jsonl_refresh_worker(
        qso_jsonl_cache.path.clone(),
        qso_jsonl_cache.direct_calls_since,
        qso_jsonl_cache.last_modified,
        qso_jsonl_cache.last_len,
    );
    let mut work_queue = WorkQueueState::new(
        &config,
        qso_controller.defaults().tx_freq_default_hz,
        qso_jsonl_cache.recent_worked.clone(),
    );
    let web_snapshot = Arc::new(Mutex::new(WebSnapshot {
        qso_defaults: qso_controller.defaults(),
        qso: qso_controller.snapshot(SystemTime::now()),
        queue: work_queue.web_snapshot(&StationTracker::default(), SystemTime::now()),
        qso_history: qso_jsonl_cache.history.clone(),
        ..WebSnapshot::default()
    }));
    let qso_control = Arc::new(QsoControlPlane::default());
    let rig_control = Arc::new(RigControlPlane::default());
    let queue_control = Arc::new(QueueControlPlane::default());
    start_web_server(
        &cli.web_bind,
        WebAppState {
            snapshot: Arc::clone(&web_snapshot),
            qso_control: Arc::clone(&qso_control),
            rig_control: Arc::clone(&rig_control),
            queue_control: Arc::clone(&queue_control),
            tune_available: output_device.is_ok(),
        },
    )?;
    let (job_tx, job_rx) = mpsc::sync_channel::<DecodeJob>(1);
    let (event_tx, event_rx) = mpsc::channel::<DecodeEvent>();
    thread::spawn(move || {
        let mut session_slot: Option<SystemTime> = None;
        let mut session_mode: DecoderMode = DecoderMode::Ft8;
        let mut session = DecoderSession::new();
        let mut state = DecoderState::new();
        while let Ok(job) = job_rx.recv() {
            let full_authoritative_reset = job.mode == DecoderMode::Ft8
                && job.stage == DecodeStage::Full
                && !session.emitted_stages().contains(&DecodeStage::Early47);
            if full_authoritative_reset
                || session_slot != Some(job.slot_start)
                || session_mode != job.mode
            {
                session.reset();
                session_slot = Some(job.slot_start);
                session_mode = job.mode;
            }
            let worker_started_at = SystemTime::now();
            let decode_started_at = Instant::now();
            let result = decode_stage_from_samples(
                &mut session,
                &mut state,
                &job.samples,
                job.sample_rate_hz,
                job.mode,
                job.stage,
                job.slot_start,
                decode_profile,
                job.raw_path.as_deref(),
            );
            let decode_run_ms = decode_started_at.elapsed().as_millis();
            let finished_at = SystemTime::now();
            let wall_ms = finished_at
                .duration_since(job.capture_end)
                .unwrap_or_default()
                .as_millis();
            let worker_wait_ms = worker_started_at
                .duration_since(job.capture_end)
                .unwrap_or_default()
                .as_millis();
            let _ = event_tx.send(DecodeEvent::Finished {
                slot_start: job.slot_start,
                mode: job.mode,
                stage: job.stage,
                wall_ms,
                worker_wait_ms,
                decode_run_ms,
                result,
            });
            if job.stage == DecodeStage::Full {
                session.reset();
                session_slot = None;
            }
        }
    });
    let stop = Arc::new(AtomicBool::new(false));
    let signal = Arc::clone(&stop);
    ctrlc::set_handler(move || {
        signal.store(true, Ordering::Relaxed);
    })
    .map_err(std::io::Error::other)?;

    let mut display = DisplayState {
        rig: read_rig_snapshot_shared(&rig),
        app_mode: DecoderMode::Ft8,
        audio,
        capture_rms_dbfs: -120.0,
        capture_latest_sample_time: None,
        capture_channel_rms_dbfs: vec![-120.0; capture.config().channels],
        capture_channel: 0,
        capture_recoveries: 0,
        decode_status: "Idle".to_string(),
        early41_wall_ms: None,
        early47_wall_ms: None,
        tx_margin_ms: None,
        full_wall_ms: None,
        last_decode_wall_ms: None,
        dropped_slots: 0,
        last_slot_start: None,
        early41_decodes: Vec::new(),
        early47_decodes: Vec::new(),
        full_decodes: Vec::new(),
    };

    let mut next_slot = next_slot_boundary_for_mode(display.app_mode, SystemTime::now());
    let mut next_slot_stages = SlotStageState::default();
    let mut decode_cycle_log = DecodeCycleLogState::default();
    let mut slot_direct_skip_start: Option<SystemTime> = None;
    let mut slot_direct_skip_calls = BTreeSet::<String>::new();
    let mut last_rig_poll = UNIX_EPOCH;
    let mut last_tx_telemetry_poll = UNIX_EPOCH;
    let mut active_decode: Option<ActiveDecodeJob> = None;
    let mut waterfall_rows = seeded_waterfall_rows();
    let mut last_waterfall_update = UNIX_EPOCH;
    let mut last_waterfall_sample_time = UNIX_EPOCH;
    let render_terminal = std::io::stdout().is_terminal();
    let mut last_terminal_render = UNIX_EPOCH;
    let mut bandmaps = BandMapStore::default();
    let mut dt_frame_history = VecDeque::<Vec<DecodedMessage>>::with_capacity(DT_HISTORY_FRAMES);
    let mut station_tracker = StationTracker::default();
    qso_controller.update_rig_context(
        display.rig.as_ref().map(|rig| rig.frequency_hz),
        display.rig.as_ref().map(|rig| rig.band.to_string()),
        display.app_mode,
    );
    work_queue.set_current_band(display.rig.as_ref().map(|rig| rig.band.to_string()));
    work_queue.set_current_mode(display.app_mode);

    if render_terminal {
        print!("\x1b[?25l");
    }
    while !stop.load(Ordering::Relaxed) {
        let stats = capture.stats();
        display.capture_rms_dbfs = stats.last_chunk_rms_dbfs;
        display.capture_latest_sample_time = stats.latest_sample_time;
        display.capture_channel_rms_dbfs = stats.channel_rms_dbfs;
        display.capture_channel = stats.selected_channel;
        display.capture_recoveries = stats.recoveries;

        let now = SystemTime::now();
        qso_controller.tick(now);
        for outcome in qso_controller.drain_outcomes() {
            work_queue.handle_qso_outcome(&outcome, &station_tracker);
        }
        while let Ok(refresh) = qso_jsonl_refresh_rx.try_recv() {
            qso_jsonl_cache.apply_refresh(refresh);
            work_queue.sync_recent_worked(qso_jsonl_cache.recent_worked.clone());
        }

        for command in qso_control.drain() {
            let station_info = match &command {
                QsoCommand::Start { partner_call, .. } => station_tracker.start_info(partner_call),
                QsoCommand::Stop { .. } => None,
            };
            qso_controller.handle_command(command, station_info, now);
        }
        for command in queue_control.drain() {
            match command {
                QueueCommand::Add { callsign } => {
                    if let Some(info) = station_tracker.start_info(&callsign) {
                        if let Err(message) =
                            work_queue.add_station(&callsign, info.last_heard_at, now)
                        {
                            warn!(callsign, message, "queue_add_failed");
                        }
                    } else {
                        warn!(callsign, "queue_add_rejected_station_unavailable");
                    }
                }
                QueueCommand::Remove { callsign } => {
                    work_queue.remove_station(&callsign, "manual_remove");
                }
                QueueCommand::Clear => work_queue.clear("manual_clear"),
                QueueCommand::SetAuto { enabled } => work_queue.set_auto_enabled(enabled),
                QueueCommand::SetTxFreq {
                    slot_family,
                    tx_freq_hz,
                } => {
                    if config.validate_tx_freq_hz(tx_freq_hz) {
                        work_queue.set_tx_freq_hz(slot_family, tx_freq_hz);
                    } else {
                        warn!(
                            tx_freq_hz,
                            slot_family = slot_family.as_str(),
                            "queue_tx_freq_invalid"
                        );
                    }
                }
                QueueCommand::SetRetryDelay {
                    kind,
                    retry_delay_seconds,
                } => work_queue.set_retry_delay(kind, Duration::from_secs(retry_delay_seconds)),
                QueueCommand::SetAutoAddAllDecodedCalls { enabled } => {
                    work_queue.set_auto_add_all_decoded_calls(enabled)
                }
                QueueCommand::SetAutoAddDecodedMinCount5m { count } => {
                    work_queue.set_auto_add_decoded_min_count_5m(count)
                }
                QueueCommand::SetAutoAddDirect { enabled } => {
                    work_queue.set_auto_add_direct_calls(enabled)
                }
                QueueCommand::SetIgnoreDirectWorked { enabled } => {
                    work_queue.set_ignore_direct_calls_from_recently_worked(enabled)
                }
                QueueCommand::SetCqEnabled { enabled } => work_queue.set_cq_enabled(enabled),
                QueueCommand::SetCqPercent { percent } => work_queue.set_cq_percent(percent),
                QueueCommand::SetPauseCqWhenFewUniqueCalls { enabled } => {
                    work_queue.set_pause_cq_when_few_unique_calls(enabled)
                }
                QueueCommand::SetCqPauseMinUniqueCalls5m { count } => {
                    work_queue.set_cq_pause_min_unique_calls_5m(count)
                }
                QueueCommand::SetCompoundRr73Handoff { enabled } => {
                    work_queue.set_use_compound_rr73_handoff(enabled)
                }
                QueueCommand::SetCompound73OnceHandoff { enabled } => {
                    work_queue.set_use_compound_73_once_handoff(enabled)
                }
                QueueCommand::SetCompoundForDirectSignalCallers { enabled } => {
                    work_queue.set_use_compound_for_direct_signal_callers(enabled)
                }
                QueueCommand::ToggleNextCqParity => work_queue.toggle_next_cq_parity(),
            }
        }
        for command in rig_control.drain() {
            match command {
                RigCommand::Configure {
                    band,
                    power,
                    app_mode,
                } => {
                    if let Err(error) = apply_rig_config(&rig, band, power, app_mode) {
                        display.decode_status = format!("Rig config failed: {error}");
                    } else {
                        let previous_band = display.rig.as_ref().map(|rig| rig.band.to_string());
                        let previous_mode = display.app_mode;
                        display.rig = read_rig_snapshot_shared(&rig);
                        let current_band = display.rig.as_ref().map(|rig| rig.band.to_string());
                        display.app_mode = app_mode;
                        if previous_band != current_band || previous_mode != app_mode {
                            clear_bandmaps(&mut bandmaps);
                            work_queue.clear(if previous_mode != app_mode {
                                "mode_change"
                            } else {
                                "band_change"
                            });
                            station_tracker.reset_for_mode(app_mode);
                            dt_frame_history.clear();
                            display.early41_wall_ms = None;
                            display.early47_wall_ms = None;
                            display.tx_margin_ms = None;
                            display.full_wall_ms = None;
                            display.last_decode_wall_ms = None;
                            display.last_slot_start = None;
                            display.early41_decodes.clear();
                            display.early47_decodes.clear();
                            display.full_decodes.clear();
                            next_slot = next_slot_boundary_for_mode(app_mode, now);
                            next_slot_stages = SlotStageState::default();
                            decode_cycle_log.reset();
                            active_decode = None;
                        }
                        qso_controller.update_rig_context(
                            display.rig.as_ref().map(|rig| rig.frequency_hz),
                            current_band.clone(),
                            display.app_mode,
                        );
                        work_queue.set_current_band(current_band);
                        work_queue.set_current_mode(display.app_mode);
                        last_rig_poll = now;
                        last_tx_telemetry_poll = now;
                    }
                }
                RigCommand::Tune10s => {
                    if let Ok(device) = &output_device {
                        if let Err(error) = start_tune_thread(
                            Arc::clone(&rig),
                            device.clone(),
                            config.tx.playback_channels,
                            config.tx.drive_level,
                            Arc::clone(&tx_busy),
                            Arc::clone(&tune_active),
                        ) {
                            display.decode_status = format!("Tune failed: {error}");
                        } else {
                            display.decode_status = "Tune 10s started".to_string();
                        }
                    } else if let Err(error) = &output_device {
                        display.decode_status = format!("Tune unavailable: {error}");
                    }
                }
            }
        }

        while let Ok(event) = event_rx.try_recv() {
            match event {
                DecodeEvent::Finished {
                    slot_start,
                    mode,
                    stage,
                    wall_ms,
                    worker_wait_ms,
                    decode_run_ms,
                    result,
                } => {
                    if mode != display.app_mode {
                        continue;
                    }
                    active_decode = None;
                    if slot_direct_skip_start != Some(slot_start) {
                        slot_direct_skip_start = Some(slot_start);
                        slot_direct_skip_calls.clear();
                    }
                    if let Some(active_partner) = qso_controller.active_partner_call() {
                        slot_direct_skip_calls.insert(active_partner);
                    }
                    match result {
                        Ok(update) => {
                            decode_cycle_log.record_ok(
                                stage,
                                wall_ms,
                                worker_wait_ms,
                                decode_run_ms,
                                update.report.decodes.len(),
                            );
                            match stage {
                                DecodeStage::Early41 => {
                                    display.early41_wall_ms = Some(wall_ms);
                                    display.early41_decodes = update.report.decodes.clone();
                                    station_tracker.ingest_stage(
                                        slot_start,
                                        stage,
                                        &display.early41_decodes,
                                    );
                                    maybe_track_priority_directs(
                                        &mut work_queue,
                                        &mut qso_controller,
                                        &display.early41_decodes,
                                        &config.station.our_call,
                                        slot_start,
                                        &slot_direct_skip_calls,
                                    );
                                    qso_controller.on_decode_stage(
                                        slot_start,
                                        stage,
                                        &display.early41_decodes,
                                        SystemTime::now(),
                                    );
                                    maybe_arm_compound_handoff_from_queue(
                                        &mut work_queue,
                                        &mut qso_controller,
                                        &station_tracker,
                                        slot_start,
                                        SystemTime::now(),
                                    );
                                    if work_queue.has_recent_priority_direct_for_slot(
                                        slot_start,
                                        SystemTime::now(),
                                    ) && qso_controller
                                        .preempt_for_priority_direct(SystemTime::now())
                                    {
                                        info!(
                                            slot = %format_slot_time(slot_start),
                                            "qso_preempted_for_priority_direct"
                                        );
                                    }
                                }
                                DecodeStage::Early47 => {
                                    display.early47_wall_ms = Some(wall_ms);
                                    display.early47_decodes = update.report.decodes.clone();
                                    station_tracker.ingest_stage(
                                        slot_start,
                                        stage,
                                        &display.early47_decodes,
                                    );
                                    maybe_track_priority_directs(
                                        &mut work_queue,
                                        &mut qso_controller,
                                        &display.early47_decodes,
                                        &config.station.our_call,
                                        slot_start,
                                        &slot_direct_skip_calls,
                                    );
                                    qso_controller.on_decode_stage(
                                        slot_start,
                                        stage,
                                        &display.early47_decodes,
                                        SystemTime::now(),
                                    );
                                    maybe_arm_compound_handoff_from_queue(
                                        &mut work_queue,
                                        &mut qso_controller,
                                        &station_tracker,
                                        slot_start,
                                        SystemTime::now(),
                                    );
                                    if work_queue.has_recent_priority_direct_for_slot(
                                        slot_start,
                                        SystemTime::now(),
                                    ) && qso_controller
                                        .preempt_for_priority_direct(SystemTime::now())
                                    {
                                        info!(
                                            slot = %format_slot_time(slot_start),
                                            "qso_preempted_for_priority_direct"
                                        );
                                    }
                                }
                                DecodeStage::Full => {
                                    display.last_slot_start = Some(slot_start);
                                    display.full_wall_ms = Some(wall_ms);
                                    display.last_decode_wall_ms = Some(wall_ms);
                                    let tx_margin_ms = tx_margin_after_stage_decode_ms(
                                        display.app_mode,
                                        slot_start,
                                        stage,
                                        wall_ms,
                                    )?;
                                    display.tx_margin_ms = Some(tx_margin_ms);
                                    display.full_decodes = update.report.decodes.clone();
                                    if dt_frame_history.len() == DT_HISTORY_FRAMES {
                                        dt_frame_history.pop_front();
                                    }
                                    dt_frame_history.push_back(display.full_decodes.clone());
                                    station_tracker.ingest_frame(slot_start, &display.full_decodes);
                                    update_bandmaps(
                                        &mut bandmaps,
                                        display.app_mode,
                                        slot_start,
                                        &display.full_decodes,
                                    );
                                    maybe_auto_add_decoded_calls(
                                        &mut work_queue,
                                        &station_tracker,
                                        &qso_controller,
                                        &display.full_decodes,
                                        slot_start,
                                    );
                                    maybe_track_priority_directs(
                                        &mut work_queue,
                                        &mut qso_controller,
                                        &display.full_decodes,
                                        &config.station.our_call,
                                        slot_start,
                                        &slot_direct_skip_calls,
                                    );
                                    qso_controller.on_decode_stage(
                                        slot_start,
                                        stage,
                                        &display.full_decodes,
                                        SystemTime::now(),
                                    );
                                    maybe_arm_compound_handoff_from_queue(
                                        &mut work_queue,
                                        &mut qso_controller,
                                        &station_tracker,
                                        slot_start,
                                        SystemTime::now(),
                                    );
                                    if work_queue.has_recent_priority_direct_for_slot(
                                        slot_start,
                                        SystemTime::now(),
                                    ) && qso_controller
                                        .preempt_for_priority_direct(SystemTime::now())
                                    {
                                        info!(
                                            slot = %format_slot_time(slot_start),
                                            "qso_preempted_for_priority_direct"
                                        );
                                    }
                                    decode_cycle_log.emit(
                                        slot_start,
                                        mode,
                                        decode_profile,
                                        Some(tx_margin_ms),
                                    );
                                    decode_cycle_log.reset();
                                }
                            }
                        }
                        Err(error) => {
                            decode_cycle_log.record_decode_failed(
                                stage,
                                wall_ms,
                                worker_wait_ms,
                                decode_run_ms,
                                &error,
                            );
                            display.decode_status = format!(
                                "Last {} {} failed: {}",
                                stage_display_label(display.app_mode, stage),
                                format_slot_time_for_mode(display.app_mode, slot_start),
                                error
                            );
                            if stage == DecodeStage::Full {
                                display.last_slot_start = Some(slot_start);
                                display.full_wall_ms = Some(wall_ms);
                                display.last_decode_wall_ms = Some(wall_ms);
                                display.early41_decodes.clear();
                                display.early47_decodes.clear();
                                display.full_decodes.clear();
                                let tx_margin_ms = tx_margin_after_stage_decode_ms(
                                    display.app_mode,
                                    slot_start,
                                    stage,
                                    wall_ms,
                                )
                                .ok();
                                display.tx_margin_ms = tx_margin_ms;
                                decode_cycle_log.emit(
                                    slot_start,
                                    mode,
                                    decode_profile,
                                    tx_margin_ms,
                                );
                                decode_cycle_log.reset();
                            }
                        }
                    }
                }
            }
        }

        while let Some(stage) = next_slot_stages.next_due_stage(
            display.app_mode,
            decode_profile,
            next_slot,
            display.capture_latest_sample_time,
        ) {
            let slot_start = next_slot;
            let app_mode = display.app_mode;
            decode_cycle_log.begin(slot_start, app_mode);
            let capture_end = stage_capture_end(display.app_mode, slot_start, stage)?;
            let samples = match extract_stage_capture(&capture, slot_start, display.app_mode, stage)
            {
                Ok(raw) => raw,
                Err(AppError::Audio(rigctl::audio::Error::WindowNotReady)) => {
                    break;
                }
                Err(error) => {
                    display.decode_status = format!(
                        "Capture error for {} {}: {}",
                        stage_display_label(app_mode, stage),
                        format_slot_time_for_mode(app_mode, slot_start),
                        error
                    );
                    decode_cycle_log.record_capture_failed(stage, &error);
                    next_slot_stages.mark_handled(stage);
                    if stage == DecodeStage::Full {
                        display.last_slot_start = Some(slot_start);
                        display.early41_decodes.clear();
                        display.early47_decodes.clear();
                        display.full_decodes.clear();
                        display.tx_margin_ms = None;
                        decode_cycle_log.emit(slot_start, app_mode, decode_profile, None);
                        decode_cycle_log.reset();
                        next_slot += slot_duration_for_mode(display.app_mode);
                        next_slot_stages = SlotStageState::default();
                    }
                    continue;
                }
            };

            let raw_path = if stage == DecodeStage::Full {
                Some(
                    cli.save_raw_wav
                        .clone()
                        .unwrap_or_else(|| temp_path("ft8rx-raw.wav")),
                )
            } else {
                None
            };

            let send_result = job_tx.try_send(DecodeJob {
                slot_start,
                stage,
                capture_end,
                samples,
                sample_rate_hz: capture.config().sample_rate_hz,
                raw_path,
                mode: display.app_mode,
            });
            next_slot_stages.mark_handled(stage);
            match send_result {
                Ok(()) => {
                    if stage == DecodeStage::Early41 {
                        display.early41_wall_ms = None;
                        display.early47_wall_ms = None;
                        display.tx_margin_ms = None;
                        display.full_wall_ms = None;
                        display.last_decode_wall_ms = None;
                        display.early41_decodes.clear();
                        display.early47_decodes.clear();
                        display.full_decodes.clear();
                    }
                    active_decode = Some(ActiveDecodeJob { slot_start, stage });
                }
                Err(mpsc::TrySendError::Full(_)) => {
                    decode_cycle_log.record_dropped(stage, "decode_worker_busy");
                    if stage == DecodeStage::Full {
                        display.dropped_slots += 1;
                        display.decode_status = format!(
                            "capture=active decode=busy dropping={} drops={} next={}",
                            format_slot_time_for_mode(display.app_mode, slot_start),
                            display.dropped_slots,
                            format_slot_time_for_mode(
                                display.app_mode,
                                next_slot + slot_duration_for_mode(display.app_mode)
                            )
                        );
                        display.last_slot_start = Some(slot_start);
                        display.early41_decodes.clear();
                        display.early47_decodes.clear();
                        display.full_decodes.clear();
                        display.tx_margin_ms = None;
                        decode_cycle_log.emit(slot_start, app_mode, decode_profile, None);
                        decode_cycle_log.reset();
                    }
                }
                Err(mpsc::TrySendError::Disconnected(_)) => {
                    return Err(AppError::Io(std::io::Error::new(
                        std::io::ErrorKind::BrokenPipe,
                        "decode worker disconnected",
                    )));
                }
            }

            if stage == DecodeStage::Full {
                next_slot += slot_duration_for_mode(display.app_mode);
                next_slot_stages = SlotStageState::default();
            }
        }

        let qso_runtime_snapshot = qso_controller.snapshot(SystemTime::now());
        if let Some(dispatch) = work_queue.scheduler_pick(
            SystemTime::now(),
            &station_tracker,
            qso_runtime_snapshot.active || qso_runtime_snapshot.tx_active,
            tune_active.load(Ordering::Relaxed) || tx_busy.load(Ordering::Relaxed),
        ) {
            let dispatch_now = SystemTime::now();
            let station_info = station_info_from_dispatch(&station_tracker, &dispatch.kind);
            let command = qso_start_command_from_dispatch(&dispatch);
            qso_controller.handle_command(command, station_info, dispatch_now);
        }
        qso_controller.tick(SystemTime::now());
        for outcome in qso_controller.drain_outcomes() {
            work_queue.handle_qso_outcome(&outcome, &station_tracker);
        }

        let qso_runtime_snapshot = qso_controller.snapshot(SystemTime::now());
        let tx_path_active = qso_runtime_snapshot.tx_active
            || tune_active.load(Ordering::Relaxed)
            || tx_busy.load(Ordering::Relaxed)
            || display
                .rig
                .as_ref()
                .and_then(|state| state.transmitting)
                .unwrap_or(false);
        if should_poll_rig_on_main_loop(
            now,
            last_rig_poll,
            next_slot,
            next_slot_stages,
            display.app_mode,
            decode_profile,
            active_decode,
            &qso_runtime_snapshot,
            tune_active.load(Ordering::Relaxed) || tx_busy.load(Ordering::Relaxed),
        ) {
            let previous_band = display.rig.as_ref().map(|rig| rig.band.to_string());
            display.rig = read_rig_snapshot_shared(&rig);
            let current_band = display.rig.as_ref().map(|rig| rig.band.to_string());
            if previous_band != current_band {
                clear_bandmaps(&mut bandmaps);
                work_queue.clear("band_change");
            }
            qso_controller.update_rig_context(
                display.rig.as_ref().map(|rig| rig.frequency_hz),
                current_band.clone(),
                display.app_mode,
            );
            work_queue.set_current_band(current_band);
            work_queue.set_current_mode(display.app_mode);
            last_rig_poll = now;
            last_tx_telemetry_poll = now;
        } else if should_poll_rig_telemetry_during_tx(
            now,
            last_tx_telemetry_poll,
            active_decode,
            tx_path_active,
        ) {
            refresh_rig_telemetry_shared(&rig, &mut display.rig);
            last_tx_telemetry_poll = now;
        }

        if should_refresh_waterfall(
            now,
            last_waterfall_update,
            display.capture_latest_sample_time,
            last_waterfall_sample_time,
        ) {
            if let Some(latest_sample_time) = display.capture_latest_sample_time {
                if let Ok(row) = compute_latest_waterfall_row(&capture, latest_sample_time) {
                    push_waterfall_row(&mut waterfall_rows, row);
                    last_waterfall_update = now;
                    last_waterfall_sample_time = latest_sample_time;
                }
            }
        }

        display.decode_status = format_status(
            active_decode,
            next_slot,
            display.dropped_slots,
            capture.config().sample_rate_hz,
            display.app_mode,
        );
        refresh_web_snapshot(
            &web_snapshot,
            &display,
            &waterfall_rows,
            &bandmaps,
            &dt_frame_history,
            &station_tracker,
            &work_queue,
            &qso_controller.snapshot(SystemTime::now()),
            &qso_controller.defaults(),
            &work_queue.web_snapshot(&station_tracker, SystemTime::now()),
            &qso_jsonl_cache.history,
            &qso_jsonl_cache.direct_calls,
            &config.station.our_call,
            tune_active.load(Ordering::Relaxed),
        );
        if render_terminal
            && now.duration_since(last_terminal_render).unwrap_or_default()
                >= Duration::from_millis(250)
        {
            render(&display);
            last_terminal_render = now;
        }
        thread::sleep(Duration::from_millis(50));
    }

    qso_controller.shutdown(SystemTime::now());
    if render_terminal {
        print!("\x1b[?25h");
    }
    Ok(())
}

fn refresh_web_snapshot(
    snapshot: &SharedWebSnapshot,
    display: &DisplayState,
    waterfall_rows: &VecDeque<Vec<u8>>,
    bandmaps: &BandMapStore,
    dt_frame_history: &VecDeque<Vec<DecodedMessage>>,
    station_tracker: &StationTracker,
    work_queue: &WorkQueueState,
    qso_snapshot: &WebQsoSnapshot,
    qso_defaults: &WebQsoDefaults,
    queue_snapshot: &WebQueueSnapshot,
    qso_history: &[WebQsoHistoryEntry],
    qso_direct_calls: &[WebDirectCallLog],
    our_call: &str,
    tune_active: bool,
) {
    let now = SystemTime::now();
    let current_slot = current_slot_boundary_for_mode(display.app_mode, now);
    let composite = composite_rows(display);
    let preferred_decodes = preferred_stage_decodes(display);
    let current_dt_stats = summarize_dt(preferred_decodes);
    let ten_minute_dt_stats = summarize_dt_history(dt_frame_history);
    let decodes = composite
        .into_iter()
        .map(|row| {
            let (kind, field1, field2, info) = decode_columns(&row.display);
            WebDecodeRow {
                seen: row.seen.to_string(),
                utc: row.display.utc,
                snr_db: row.display.snr_db,
                dt_seconds: row.display.dt_seconds,
                freq_hz: row.display.freq_hz,
                kind,
                field1,
                field1_select_call: semantic_first_call_display_call(&row.display.message),
                field2,
                field2_select_call: semantic_sender_call(&row.display.message),
                info,
                text: row.display.text,
            }
        })
        .collect::<Vec<_>>();
    let current_slot_index = slot_index_for_mode(display.app_mode, current_slot);
    let mut guard = snapshot.lock().expect("web snapshot poisoned");
    guard.time_utc = {
        let now_utc: DateTime<Utc> = now.into();
        now_utc.format("%Y-%m-%d %H:%M:%S UTC").to_string()
    };
    guard.our_call = our_call.to_string();
    guard.rig_frequency_hz = display.rig.as_ref().map(|state| state.frequency_hz);
    guard.rig_kind = display.rig.as_ref().map(|state| state.kind.to_string());
    guard.rig_mode = display
        .rig
        .as_ref()
        .map(|state| state.mode.to_string())
        .unwrap_or_else(|| "unavailable".to_string());
    guard.rig_band = display
        .rig
        .as_ref()
        .map(|state| state.band.to_string())
        .unwrap_or_else(|| "unavailable".to_string());
    guard.app_mode = display.app_mode.as_str().to_uppercase();
    let (
        rig_power_w,
        rig_power_label,
        rig_power_current_id,
        rig_power_settings,
        rig_power_settable,
        rig_power_is_discrete,
    ) = display.rig.as_ref().map(rig_power_web_fields).unwrap_or((
        None,
        None,
        None,
        Vec::new(),
        false,
        false,
    ));
    guard.rig_power_w = rig_power_w;
    guard.rig_power_label = rig_power_label;
    guard.rig_power_current_id = rig_power_current_id;
    guard.rig_power_settings = rig_power_settings;
    guard.rig_power_settable = rig_power_settable;
    guard.rig_power_is_discrete = rig_power_is_discrete;
    guard.rig_bargraph = display.rig.as_ref().and_then(|state| state.bar_graph);
    guard.rig_rx_s_meter = display
        .rig
        .as_ref()
        .and_then(|state| state.telemetry_rx_s_meter);
    guard.rig_tx_forward_power_w = display
        .rig
        .as_ref()
        .and_then(|state| state.telemetry_tx_forward_power_w);
    guard.rig_tx_swr = display
        .rig
        .as_ref()
        .and_then(|state| state.telemetry_tx_swr);
    guard.rig_is_tx = display.rig.as_ref().and_then(|state| state.transmitting);
    guard.rig_tune_active = tune_active;
    guard.decode_status = display.decode_status.clone();
    guard.audio_stats = WebAudioStats {
        latest_sample: display
            .capture_latest_sample_time
            .map(|time| format_slot_time_for_mode(display.app_mode, time)),
        selected_channel: display.capture_channel,
        overall_dbfs: display.capture_rms_dbfs,
        left_dbfs: display
            .capture_channel_rms_dbfs
            .first()
            .copied()
            .unwrap_or(-120.0),
        right_dbfs: display
            .capture_channel_rms_dbfs
            .get(1)
            .copied()
            .unwrap_or(-120.0),
        recoveries: display.capture_recoveries,
    };
    guard.decode_times = if display.app_mode == DecoderMode::Ft8 {
        WebDecodeTimes {
            early_seconds: display.early41_wall_ms.map(ms_to_seconds),
            mid_seconds: display.early47_wall_ms.map(ms_to_seconds),
            late_seconds: display.full_wall_ms.map(ms_to_seconds),
            tx_margin_seconds: display.tx_margin_ms.map(ms_to_signed_seconds),
        }
    } else {
        WebDecodeTimes {
            early_seconds: None,
            mid_seconds: display.full_wall_ms.map(ms_to_seconds),
            late_seconds: None,
            tx_margin_seconds: display.tx_margin_ms.map(ms_to_signed_seconds),
        }
    };
    guard.dt_stats = WebDtStats {
        current_mean_seconds: current_dt_stats.mean,
        current_median_seconds: current_dt_stats.median,
        current_stddev_seconds: current_dt_stats.stddev,
        current_count: current_dt_stats.count,
        ten_minute_mean_seconds: ten_minute_dt_stats.mean,
        ten_minute_median_seconds: ten_minute_dt_stats.median,
        ten_minute_count: ten_minute_dt_stats.count,
    };
    guard.current_slot = format_slot_time_for_mode(display.app_mode, current_slot);
    guard.last_done_slot = display
        .last_slot_start
        .map(|time| format_slot_time_for_mode(display.app_mode, time));
    guard.decodes = decodes;
    guard.waterfall = waterfall_rows.iter().cloned().collect();
    guard.bandmaps = WebBandMaps {
        even: build_bandmap_grid(&bandmaps.even, current_slot_index, work_queue, now),
        odd: build_bandmap_grid(&bandmaps.odd, current_slot_index, work_queue, now),
        even_age_seconds: bandmap_age_seconds(bandmaps.even_last_updated_at, now),
        odd_age_seconds: bandmap_age_seconds(bandmaps.odd_last_updated_at, now),
    };
    guard.stations = station_tracker.web_station_summaries();
    guard.station_logs = station_tracker.web_logs();
    let direct_calls_since = now
        .checked_sub(DIRECT_CALL_PANE_RETENTION)
        .unwrap_or(UNIX_EPOCH);
    let direct_calls_since_ms = system_time_to_epoch_ms(direct_calls_since);
    let mut direct_calls = station_tracker.web_direct_calls(our_call, direct_calls_since);
    direct_calls.extend(
        qso_direct_calls
            .iter()
            .filter(|entry| entry.sort_epoch_ms >= direct_calls_since_ms)
            .cloned(),
    );
    direct_calls.sort_by_key(|entry| entry.sort_epoch_ms);
    guard.direct_calls = direct_calls;
    guard.qso = qso_snapshot.clone();
    guard.qso_defaults = qso_defaults.clone();
    guard.queue = queue_snapshot.clone();
    guard.qso_history = qso_history.to_vec();
}

fn run_oneshot(cli: Cli) -> Result<(), AppError> {
    let config = AppConfig::load(&cli.config)?;
    let decode_profile = parse_decode_profile(&cli.decode_profile)?;
    let app_mode = DecoderMode::Ft8;
    let rig_kind = resolve_app_rig_kind(&config)?;
    let audio = detect_audio_device_for_rig(
        rig_kind,
        cli.device
            .as_deref()
            .or(configured_input_device_override(&config)),
    )?;
    let capture = SampleStream::start(audio.clone(), AudioStreamConfig::default())?;
    let target_slot = next_slot_boundary_for_mode(app_mode, SystemTime::now());

    println!("audio=\"{}\" spec={}", audio.name, audio.spec);
    println!("target_slot={}", format_slot_time(target_slot));

    let ready_at = slot_capture_end(target_slot, capture.config().sample_rate_hz, app_mode)?;
    while SystemTime::now() < ready_at {
        let stats = capture.stats();
        let latest = stats
            .latest_sample_time
            .map(format_slot_time)
            .unwrap_or_else(|| "------".to_string());
        let left = stats.channel_rms_dbfs.first().copied().unwrap_or(-120.0);
        let right = stats.channel_rms_dbfs.get(1).copied().unwrap_or(-120.0);
        println!(
            "waiting latest_sample={} ch={} rms={:.1}dBFS left={:.1} right={:.1} recoveries={}",
            latest,
            stats.selected_channel,
            stats.last_chunk_rms_dbfs,
            left,
            right,
            stats.recoveries
        );
        thread::sleep(Duration::from_secs(1));
    }

    let summary = decode_slot_from_capture(
        &capture,
        target_slot,
        app_mode,
        cli.save_raw_wav.as_deref().or(cli.save_wav.as_deref()),
        decode_profile,
    )?;
    println!("decodes={}", summary.final_decodes.len());
    for decode in summary.final_decodes {
        println!(
            "{} {:>4} {:+5.2} {:>6.0} {}",
            decode.utc, decode.snr_db, decode.dt_seconds, decode.freq_hz, decode.text
        );
    }
    Ok(())
}

fn read_rig_snapshot(rig: &mut Option<Rig>) -> Option<RigSnapshot> {
    let rig = rig.as_mut()?;
    let CommonRigSnapshot {
        kind,
        frequency_hz,
        mode,
        band,
        transmitting,
        telemetry,
        power,
    } = rig.read_snapshot().ok()?;
    Some(RigSnapshot {
        kind,
        frequency_hz,
        mode,
        band,
        power,
        bar_graph: telemetry.bar_graph,
        telemetry_rx_s_meter: telemetry.rx_s_meter,
        telemetry_tx_forward_power_w: telemetry.tx_forward_power_w,
        telemetry_tx_swr: telemetry.tx_swr,
        transmitting: Some(transmitting),
    })
}

fn read_rig_snapshot_shared(rig: &SharedRig) -> Option<RigSnapshot> {
    let mut guard = rig.lock().expect("rig mutex poisoned");
    read_rig_snapshot(&mut guard)
}

fn apply_rig_telemetry(
    snapshot: &mut Option<RigSnapshot>,
    telemetry: RigTelemetry,
    transmitting: Option<bool>,
) {
    let Some(snapshot) = snapshot.as_mut() else {
        return;
    };
    snapshot.bar_graph = telemetry.bar_graph;
    snapshot.telemetry_rx_s_meter = telemetry.rx_s_meter;
    snapshot.telemetry_tx_forward_power_w = telemetry.tx_forward_power_w;
    snapshot.telemetry_tx_swr = telemetry.tx_swr;
    if transmitting.is_some() {
        snapshot.transmitting = transmitting;
    }
}

fn refresh_rig_telemetry_shared(rig: &SharedRig, snapshot: &mut Option<RigSnapshot>) {
    let mut guard = rig.lock().expect("rig mutex poisoned");
    let Some(rig) = guard.as_mut() else {
        return;
    };
    let transmitting = rig.is_transmitting().ok();
    if let Ok(telemetry) = rig.read_telemetry() {
        apply_rig_telemetry(snapshot, telemetry, transmitting);
    }
}

fn rig_power_web_fields(
    state: &RigSnapshot,
) -> (
    Option<f32>,
    Option<String>,
    Option<String>,
    Vec<WebRigPowerSetting>,
    bool,
    bool,
) {
    match &state.power {
        RigPowerState::Continuous { current_watts, .. } => (
            *current_watts,
            current_watts.map(|watts| format!("{watts:.1} W")),
            None,
            Vec::new(),
            true,
            false,
        ),
        RigPowerState::Discrete {
            current_id,
            current_label,
            settings,
            can_set,
        } => (
            settings
                .iter()
                .find(|setting| current_id.as_deref() == Some(setting.id.as_str()))
                .and_then(|setting| setting.nominal_watts),
            current_label.clone(),
            current_id.clone(),
            settings
                .iter()
                .cloned()
                .map(|setting| WebRigPowerSetting {
                    id: setting.id,
                    label: setting.label,
                    nominal_watts: setting.nominal_watts,
                })
                .collect(),
            *can_set,
            true,
        ),
    }
}

fn calling_frequency_hz(band: Band, app_mode: DecoderMode) -> u64 {
    match (app_mode, band) {
        (DecoderMode::Ft8, Band::M160) => 1_840_000,
        (DecoderMode::Ft8, Band::M80) => 3_573_000,
        (DecoderMode::Ft8, Band::M60) => 5_357_000,
        (DecoderMode::Ft8, Band::M40) => 7_074_000,
        (DecoderMode::Ft8, Band::M30) => 10_136_000,
        (DecoderMode::Ft8, Band::M20) => 14_074_000,
        (DecoderMode::Ft8, Band::M17) => 18_100_000,
        (DecoderMode::Ft8, Band::M15) => 21_074_000,
        (DecoderMode::Ft8, Band::M12) => 24_915_000,
        (DecoderMode::Ft8, Band::M10) => 28_074_000,
        (DecoderMode::Ft8, Band::M6) => 50_313_000,
        (DecoderMode::Ft4, Band::M80) => 3_575_000,
        (DecoderMode::Ft4, Band::M40) => 7_047_500,
        (DecoderMode::Ft4, Band::M30) => 10_140_000,
        (DecoderMode::Ft4, Band::M20) => 14_080_000,
        (DecoderMode::Ft4, Band::M17) => 18_104_000,
        (DecoderMode::Ft4, Band::M15) => 21_140_000,
        (DecoderMode::Ft4, Band::M12) => 24_919_000,
        (DecoderMode::Ft4, Band::M10) => 28_180_000,
        (DecoderMode::Ft4, Band::M6) => 50_318_000,
        (DecoderMode::Ft4, Band::M160) | (DecoderMode::Ft4, Band::M60) => {
            calling_frequency_hz(band, DecoderMode::Ft8)
        }
        (DecoderMode::Ft2, _) => unreachable!("FT2 is not supported in ft8rx"),
        (_, Band::Xvtr(_)) => 0,
    }
}

fn apply_rig_config(
    rig: &SharedRig,
    band: Band,
    power: Option<RigPowerRequest>,
    app_mode: DecoderMode,
) -> Result<(), AppError> {
    let mut guard = rig.lock().expect("rig mutex poisoned");
    let rig = guard.as_mut().ok_or_else(|| {
        AppError::Io(std::io::Error::new(
            std::io::ErrorKind::NotConnected,
            "rig unavailable",
        ))
    })?;
    let before_snapshot = rig.read_snapshot().ok();
    let frequency_hz = calling_frequency_hz(band, app_mode);
    info!(
        rig_kind = %rig.kind(),
        target_band = %band,
        target_frequency_hz = frequency_hz,
        target_app_mode = %app_mode.as_str().to_uppercase(),
        requested_power = ?power,
        before_frequency_hz = before_snapshot.as_ref().map(|snapshot| snapshot.frequency_hz),
        before_band = before_snapshot.as_ref().map(|snapshot| snapshot.band.to_string()),
        before_mode = before_snapshot.as_ref().map(|snapshot| snapshot.mode.to_string()),
        "rig_config_apply_begin"
    );
    if frequency_hz > 0 {
        rig.set_frequency_hz(frequency_hz)?;
    }
    rig.set_mode(RigMode::Data)?;
    if let Some(power) = power.as_ref() {
        rig.apply_power_request(power)?;
    }
    match rig.read_snapshot() {
        Ok(after_snapshot) => {
            info!(
                rig_kind = %after_snapshot.kind,
                target_band = %band,
                target_frequency_hz = frequency_hz,
                after_frequency_hz = after_snapshot.frequency_hz,
                after_band = %after_snapshot.band,
                after_mode = %after_snapshot.mode,
                "rig_config_apply_complete"
            );
            if frequency_hz > 0
                && (after_snapshot.frequency_hz != frequency_hz || after_snapshot.band != band)
            {
                warn!(
                    rig_kind = %after_snapshot.kind,
                    target_band = %band,
                    target_frequency_hz = frequency_hz,
                    reported_band = %after_snapshot.band,
                    reported_frequency_hz = after_snapshot.frequency_hz,
                    "rig_config_apply_mismatch"
                );
            }
        }
        Err(error) => {
            warn!(message = %error, "rig_config_apply_readback_failed");
        }
    }
    Ok(())
}

fn start_tune_thread(
    rig: SharedRig,
    output_device: AudioDevice,
    playback_channels: usize,
    drive_level: f32,
    tx_busy: Arc<AtomicBool>,
    tune_active: Arc<AtomicBool>,
) -> Result<(), String> {
    if tx_busy
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return Err("transmit path busy".to_string());
    }
    tune_active.store(true, Ordering::Release);
    thread::spawn(move || {
        let _busy_guard = AtomicFlagGuard::new(tx_busy, false);
        let _tune_guard = AtomicFlagGuard::new(tune_active, false);
        if let Err(error) = run_tune_thread(rig, output_device, playback_channels, drive_level) {
            error!(message = %error, "rig_tune_failed");
        }
    });
    Ok(())
}

fn run_tune_thread(
    rig: SharedRig,
    output_device: AudioDevice,
    playback_channels: usize,
    drive_level: f32,
) -> Result<(), String> {
    let original_meter_mode = {
        let mut guard = rig.lock().expect("rig mutex poisoned");
        let rig = guard
            .as_mut()
            .ok_or_else(|| "rig unavailable".to_string())?;
        let original_meter_mode = if let Rig::K3s(k3) = rig {
            let original = k3.get_tx_meter_mode().unwrap_or(TxMeterMode::Rf);
            k3.set_tx_meter_mode(TxMeterMode::Rf)
                .map_err(|error| error.to_string())?;
            Some(original)
        } else {
            None
        };
        rig.enter_tx().map_err(|error| error.to_string())?;
        original_meter_mode
    };

    let playback_result = play_tone(
        &output_device,
        48_000,
        playback_channels,
        1_000.0,
        Duration::from_secs(10),
        drive_level,
    );

    let cleanup_result = {
        let mut guard = rig.lock().expect("rig mutex poisoned");
        let rig = guard
            .as_mut()
            .ok_or_else(|| "rig unavailable".to_string())?;
        rig.enter_rx().map_err(|error| error.to_string())?;
        if let (Rig::K3s(k3), Some(original_meter_mode)) = (rig, original_meter_mode) {
            k3.set_tx_meter_mode(original_meter_mode)
                .map_err(|error| error.to_string())
        } else {
            Ok(())
        }
    };

    playback_result
        .map_err(|error| error.to_string())
        .and(cleanup_result)
}

struct AtomicFlagGuard {
    flag: Arc<AtomicBool>,
    value: bool,
}

impl AtomicFlagGuard {
    fn new(flag: Arc<AtomicBool>, value: bool) -> Self {
        Self { flag, value }
    }
}

impl Drop for AtomicFlagGuard {
    fn drop(&mut self) {
        self.flag.store(self.value, Ordering::Release);
    }
}
fn extract_slot_capture(
    capture: &SampleStream,
    slot_start: SystemTime,
    mode: DecoderMode,
) -> Result<Vec<i16>, AppError> {
    Ok(capture.extract_window(
        slot_start,
        full_decode_sample_count(capture.config().sample_rate_hz, mode),
    )?)
}

fn extract_stage_capture(
    capture: &SampleStream,
    slot_start: SystemTime,
    mode: DecoderMode,
    stage: DecodeStage,
) -> Result<Vec<i16>, AppError> {
    Ok(capture.extract_window(
        slot_start,
        stage_sample_count(capture.config().sample_rate_hz, mode, stage),
    )?)
}

fn decode_slot_from_capture(
    capture: &SampleStream,
    slot_start: SystemTime,
    mode: DecoderMode,
    save_raw_wav: Option<&Path>,
    profile: DecodeProfile,
) -> Result<DecodeSummary, AppError> {
    let samples = extract_slot_capture(capture, slot_start, mode)?;
    let raw_path = save_raw_wav
        .map(Path::to_path_buf)
        .unwrap_or_else(|| temp_path("ft8rx-raw.wav"));
    decode_slot_from_samples_with_raw_path(
        &samples,
        capture.config().sample_rate_hz,
        &raw_path,
        save_raw_wav.is_some(),
        slot_start,
        mode,
        profile,
    )
}

fn decode_slot_from_samples_with_raw_path(
    samples: &[i16],
    sample_rate_hz: u32,
    raw_path: &Path,
    keep_raw: bool,
    slot_start: SystemTime,
    mode: DecoderMode,
    profile: DecodeProfile,
) -> Result<DecodeSummary, AppError> {
    if keep_raw {
        write_mono_wav(raw_path, sample_rate_hz, samples)?;
    }
    let decodes = decode_slot_from_samples(samples, sample_rate_hz, slot_start, mode, profile)?;
    if !keep_raw && raw_path.exists() {
        let _ = std::fs::remove_file(raw_path);
    }
    Ok(decodes)
}

fn decode_stage_from_samples(
    session: &mut DecoderSession,
    state: &mut DecoderState,
    samples: &[i16],
    sample_rate_hz: u32,
    mode: DecoderMode,
    stage: DecodeStage,
    slot_start: SystemTime,
    profile: DecodeProfile,
    raw_path: Option<&Path>,
) -> Result<StageDecodeReport, AppError> {
    if let Some(raw_path) = raw_path {
        write_mono_wav(raw_path, sample_rate_hz, samples)?;
    }
    let options = DecodeOptions {
        mode,
        profile,
        min_freq_hz: 200.0,
        max_freq_hz: 3_500.0,
        ..DecodeOptions::for_mode(mode)
    };
    let audio = AudioBuffer {
        sample_rate_hz: DECODER_SAMPLE_RATE_HZ,
        samples: resample_linear_f32(
            &samples
                .iter()
                .map(|&sample| sample as f32 / i16::MAX as f32)
                .collect::<Vec<_>>(),
            sample_rate_hz,
            DECODER_SAMPLE_RATE_HZ,
        ),
    };
    let (mut update, next_state) = session
        .decode_stage_with_state(&audio, &options, stage, Some(state))
        .map_err(|error| AppError::Decoder(error.to_string()))?;
    *state = next_state;
    relabel_stage_update(&mut update, slot_start, mode);
    Ok(update)
}

fn decode_slot_from_samples(
    samples: &[i16],
    sample_rate_hz: u32,
    slot_start: SystemTime,
    mode: DecoderMode,
    profile: DecodeProfile,
) -> Result<DecodeSummary, AppError> {
    let options = DecodeOptions {
        mode,
        profile,
        min_freq_hz: 200.0,
        max_freq_hz: 3_500.0,
        ..DecodeOptions::for_mode(mode)
    };
    let audio = AudioBuffer {
        sample_rate_hz: DECODER_SAMPLE_RATE_HZ,
        samples: resample_linear_f32(
            &samples
                .iter()
                .map(|&sample| sample as f32 / i16::MAX as f32)
                .collect::<Vec<_>>(),
            sample_rate_hz,
            DECODER_SAMPLE_RATE_HZ,
        ),
    };
    let mut session = DecoderSession::new();
    let updates = session
        .decode_available(&audio, &options)
        .map_err(|error| AppError::Decoder(error.to_string()))?;
    let mut final_decodes = Vec::new();
    for mut update in updates {
        relabel_stage_update(&mut update, slot_start, mode);
        match update.stage {
            DecodeStage::Early41 => {}
            DecodeStage::Early47 => {
                final_decodes = update.report.decodes.clone();
            }
            DecodeStage::Full => {
                final_decodes = update.report.decodes.clone();
            }
        }
    }
    Ok(DecodeSummary { final_decodes })
}

fn write_mono_wav(path: &Path, sample_rate_hz: u32, samples: &[i16]) -> Result<(), AppError> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: sample_rate_hz,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec)?;
    for &sample in samples {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    Ok(())
}

fn slot_progress_bar(now: SystemTime, mode: DecoderMode) -> String {
    let slot_start = current_slot_boundary_for_mode(mode, now);
    let elapsed = now
        .duration_since(slot_start)
        .unwrap_or_default()
        .as_secs_f32();
    let progress = (elapsed / slot_duration_for_mode(mode).as_secs_f32()).clamp(0.0, 1.0);
    let width: usize = 16;
    let filled = (progress * width as f32).round() as usize;
    format!(
        "[{}{}] {:>3}%",
        "#".repeat(filled),
        ".".repeat(width.saturating_sub(filled)),
        (progress * 100.0).round() as i32
    )
}

fn render(display: &DisplayState) {
    let now_local: DateTime<Local> = SystemTime::now().into();
    let now = SystemTime::now();
    let current_slot = current_slot_boundary_for_mode(display.app_mode, now);
    let rig_frequency = display
        .rig
        .as_ref()
        .map(|state| format!("{:.4} MHz", state.frequency_hz as f64 / 1_000_000.0))
        .unwrap_or_else(|| "unavailable".to_string());
    let rig_mode = display
        .rig
        .as_ref()
        .map(|state| state.mode.to_string())
        .unwrap_or_else(|| "unavailable".to_string());
    let rig_kind = display
        .rig
        .as_ref()
        .map(|state| state.kind.to_string())
        .unwrap_or_else(|| "?".to_string());
    let rig_band = display
        .rig
        .as_ref()
        .map(|state| state.band.to_string())
        .unwrap_or_else(|| "unavailable".to_string());
    let rig_direction = display
        .rig
        .as_ref()
        .and_then(|state| state.transmitting)
        .map(|tx| if tx { "TX" } else { "RX" })
        .unwrap_or("?");
    let rig_power = display
        .rig
        .as_ref()
        .map(|state| match &state.power {
            RigPowerState::Continuous { current_watts, .. } => current_watts
                .map(|watts| format!("{watts:.1}W"))
                .unwrap_or_else(|| "-".to_string()),
            RigPowerState::Discrete { current_label, .. } => {
                current_label.clone().unwrap_or_else(|| "-".to_string())
            }
        })
        .unwrap_or_else(|| "-".to_string());
    let rig_is_mchf = display
        .rig
        .as_ref()
        .is_some_and(|state| state.kind == RigKind::Mchf);
    let mut rig_telemetry = Vec::new();
    if !rig_is_mchf {
        let rig_bargraph = display
            .rig
            .as_ref()
            .and_then(|state| state.bar_graph)
            .map(|level| level.to_string())
            .unwrap_or_else(|| "-".to_string());
        rig_telemetry.push(format!("BG={rig_bargraph}"));
    }
    if let Some(value) = display
        .rig
        .as_ref()
        .and_then(|state| state.telemetry_rx_s_meter)
    {
        rig_telemetry.push(format!("S={value:.1}"));
    }
    if let Some(value) = display
        .rig
        .as_ref()
        .and_then(|state| state.telemetry_tx_forward_power_w)
    {
        rig_telemetry.push(format!("FWD={value:.1}W"));
    }
    if let Some(value) = display
        .rig
        .as_ref()
        .and_then(|state| state.telemetry_tx_swr)
    {
        rig_telemetry.push(format!("SWR={value:.1}"));
    }
    let rig_telemetry = if rig_telemetry.is_empty() {
        "-".to_string()
    } else {
        rig_telemetry.join("  ")
    };
    let display_decodes = preferred_stage_decodes(display);
    let dt_stats = summarize_dt(display_decodes);
    let composite_rows = composite_rows(display);
    let left = display
        .capture_channel_rms_dbfs
        .first()
        .copied()
        .unwrap_or(-120.0);
    let right = display
        .capture_channel_rms_dbfs
        .get(1)
        .copied()
        .unwrap_or(-120.0);
    let latest_sample = display
        .capture_latest_sample_time
        .map(|time| format_slot_time_for_mode(display.app_mode, time))
        .unwrap_or_else(|| "------".to_string());

    let mut output = String::new();
    let _ = writeln!(
        output,
        "\x1b[2J\x1b[HFT8RX    {}",
        now_local.format("%Y-%m-%d %H:%M:%S %Z")
    );
    let _ = writeln!(
        output,
        "Rig      {}  {}  {}  {}  {}  {}  P={}",
        rig_frequency,
        rig_kind,
        rig_mode,
        display.app_mode.as_str().to_uppercase(),
        rig_band,
        rig_direction,
        rig_power
    );
    let _ = writeln!(output, "RigTel   {}", rig_telemetry);
    let _ = writeln!(
        output,
        "Audio    {} ({})",
        display.audio.name, display.audio.spec
    );
    let _ = writeln!(
        output,
        "Chan     latest={} selected={} left={:.1} dBFS right={:.1} dBFS",
        latest_sample, display.capture_channel, left, right
    );
    let _ = writeln!(
        output,
        "Status   {}{}",
        display.decode_status,
        display
            .last_decode_wall_ms
            .map(|ms| format!(" last={:.2}s", ms as f32 / 1000.0))
            .unwrap_or_default()
    );
    if display.app_mode == DecoderMode::Ft8 {
        let _ = writeln!(
            output,
            "DecodeT  early={} mid={} late={} tx_margin={}",
            format_wall_time(display.early41_wall_ms),
            format_wall_time(display.early47_wall_ms),
            format_wall_time(display.full_wall_ms),
            format_signed_wall_time(display.tx_margin_ms)
        );
    } else {
        let _ = writeln!(
            output,
            "DecodeT  mid={} tx_margin={}",
            format_wall_time(display.full_wall_ms),
            format_signed_wall_time(display.tx_margin_ms)
        );
    }
    if let Some(slot_start) = display.last_slot_start {
        let _ = writeln!(
            output,
            "Slot     {} {} last_done={}",
            format_slot_time_for_mode(display.app_mode, current_slot),
            slot_progress_bar(now, display.app_mode),
            format_slot_time_for_mode(display.app_mode, slot_start)
        );
    } else {
        let _ = writeln!(
            output,
            "Slot     {} {}",
            format_slot_time_for_mode(display.app_mode, current_slot),
            slot_progress_bar(now, display.app_mode)
        );
    }
    let _ = writeln!(
        output,
        "AudioLvl last={:.1} dBFS recoveries={}",
        display.capture_rms_dbfs, display.capture_recoveries
    );
    let _ = writeln!(
        output,
        "dT stats avg={:+.2}s med={:+.2}s stddev={:.2}s count={}",
        dt_stats.mean.unwrap_or(0.0),
        dt_stats.median.unwrap_or(0.0),
        dt_stats.stddev.unwrap_or(0.0),
        dt_stats.count
    );
    let _ = writeln!(output);
    let _ = writeln!(output, "Seen    UTC    SNR   dT(s)   Freq(Hz)  Message");
    let _ = writeln!(output, "------  -----  ----  ------  --------  -------");
    if composite_rows.is_empty() {
        let _ = writeln!(output, "no decodes yet");
    } else {
        for row in &composite_rows {
            let _ = writeln!(
                output,
                "{:<6}  {:<5}  {:>4}  {:+6.2}  {:>8.0}  {}",
                row.seen,
                row.display.utc,
                row.display.snr_db,
                row.display.dt_seconds,
                row.display.freq_hz,
                row.display.text
            );
        }
    }
    print!("{output}");
}

#[derive(Debug, Clone, Copy, Default)]
struct DtSummary {
    mean: Option<f32>,
    median: Option<f32>,
    stddev: Option<f32>,
    count: usize,
}

fn summarize_dt(decodes: &[DecodedMessage]) -> DtSummary {
    if decodes.is_empty() {
        return DtSummary::default();
    }
    let values: Vec<f32> = decodes.iter().map(|decode| decode.dt_seconds).collect();
    summarize_dt_values(&values)
}

fn summarize_dt_history(frame_history: &VecDeque<Vec<DecodedMessage>>) -> DtSummary {
    let values = frame_history
        .iter()
        .flat_map(|frame| frame.iter().map(|decode| decode.dt_seconds))
        .collect::<Vec<_>>();
    summarize_dt_values(&values)
}

fn summarize_dt_values(values: &[f32]) -> DtSummary {
    if values.is_empty() {
        return DtSummary::default();
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values
        .iter()
        .map(|value| {
            let delta = *value - mean;
            delta * delta
        })
        .sum::<f32>()
        / values.len() as f32;
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));
    let median = if sorted.len() % 2 == 1 {
        sorted[sorted.len() / 2]
    } else {
        let upper = sorted.len() / 2;
        (sorted[upper - 1] + sorted[upper]) * 0.5
    };
    DtSummary {
        mean: Some(mean),
        median: Some(median),
        stddev: Some(variance.sqrt()),
        count: values.len(),
    }
}

fn format_wall_time(wall_ms: Option<u128>) -> String {
    wall_ms
        .map(|ms| format!("{:.2}s", ms as f32 / 1000.0))
        .unwrap_or_else(|| "-".to_string())
}

fn format_signed_wall_time(wall_ms: Option<i128>) -> String {
    wall_ms
        .map(|ms| format!("{:+.2}s", ms as f32 / 1000.0))
        .unwrap_or_else(|| "-".to_string())
}

fn ms_to_seconds(ms: u128) -> f32 {
    ms as f32 / 1000.0
}

fn ms_to_signed_seconds(ms: i128) -> f32 {
    ms as f32 / 1000.0
}

fn option_u128_i128(value: Option<u128>) -> i128 {
    value.map(|value| value as i128).unwrap_or(-1)
}

fn option_usize_i64(value: Option<usize>) -> i64 {
    value.map(|value| value as i64).unwrap_or(-1)
}

fn stage_display_label(_mode: DecoderMode, stage: DecodeStage) -> &'static str {
    match stage {
        DecodeStage::Early41 => "early",
        DecodeStage::Early47 => "mid",
        DecodeStage::Full => "full",
    }
}

fn slot_millis_for_mode(mode: DecoderMode) -> u64 {
    match mode {
        DecoderMode::Ft8 => 15_000,
        DecoderMode::Ft4 => 7_500,
        DecoderMode::Ft2 => unreachable!("FT2 is not supported in ft8rx"),
    }
}

fn decode_stage_enabled_for_profile(
    mode: DecoderMode,
    profile: DecodeProfile,
    stage: DecodeStage,
) -> bool {
    match stage {
        DecodeStage::Full => true,
        DecodeStage::Early41 => mode == DecoderMode::Ft8 && profile != DecodeProfile::Quick,
        DecodeStage::Early47 => mode == DecoderMode::Ft8 && profile != DecodeProfile::Quick,
    }
}

fn should_poll_rig_on_main_loop(
    now: SystemTime,
    last_rig_poll: SystemTime,
    next_slot: SystemTime,
    next_slot_stages: SlotStageState,
    mode: DecoderMode,
    decode_profile: DecodeProfile,
    active_decode: Option<ActiveDecodeJob>,
    qso_snapshot: &WebQsoSnapshot,
    tx_busy: bool,
) -> bool {
    if now.duration_since(last_rig_poll).unwrap_or_default() < Duration::from_secs(2) {
        return false;
    }
    if active_decode.is_some() || qso_snapshot.active || qso_snapshot.tx_active || tx_busy {
        return false;
    }
    let Some(next_stage_ready_at) =
        next_unhandled_decode_stage_ready_at(mode, decode_profile, next_slot, next_slot_stages)
    else {
        return true;
    };
    next_stage_ready_at.duration_since(now).unwrap_or_default() >= Duration::from_millis(1_500)
}

fn should_poll_rig_telemetry_during_tx(
    now: SystemTime,
    last_poll: SystemTime,
    active_decode: Option<ActiveDecodeJob>,
    tx_path_active: bool,
) -> bool {
    tx_path_active
        && active_decode.is_none()
        && now.duration_since(last_poll).unwrap_or_default() >= TX_TELEMETRY_REFRESH_INTERVAL
}

fn next_unhandled_decode_stage_ready_at(
    mode: DecoderMode,
    profile: DecodeProfile,
    slot_start: SystemTime,
    stages: SlotStageState,
) -> Option<SystemTime> {
    for stage in DecodeStage::ordered() {
        if !decode_stage_enabled_for_profile(mode, profile, stage) || stages.is_handled(stage) {
            continue;
        }
        return stage_capture_end(mode, slot_start, stage).ok();
    }
    None
}

pub(crate) fn slot_duration_for_mode(mode: DecoderMode) -> Duration {
    Duration::from_millis(slot_millis_for_mode(mode))
}

pub(crate) fn next_slot_boundary_for_mode(mode: DecoderMode, now: SystemTime) -> SystemTime {
    current_slot_boundary_for_mode(mode, now) + slot_duration_for_mode(mode)
}

pub(crate) fn current_slot_boundary_for_mode(mode: DecoderMode, now: SystemTime) -> SystemTime {
    let since_epoch = now.duration_since(UNIX_EPOCH).unwrap_or_default();
    let slot_millis = slot_millis_for_mode(mode) as u128;
    let elapsed_millis = since_epoch.as_millis();
    let current_millis = (elapsed_millis / slot_millis) * slot_millis;
    UNIX_EPOCH + Duration::from_millis(current_millis as u64)
}

pub(crate) fn slot_index_for_mode(mode: DecoderMode, time: SystemTime) -> u64 {
    let slot_millis = slot_millis_for_mode(mode) as u128;
    (time
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        / slot_millis) as u64
}

fn latest_slot_index_for_family(
    mode: DecoderMode,
    now: SystemTime,
    family: qso::SlotFamily,
) -> u64 {
    let current_slot = current_slot_boundary_for_mode(mode, now);
    let current = slot_index_for_mode(mode, current_slot);
    match (family, is_even_slot_family_for_mode(mode, current_slot)) {
        (qso::SlotFamily::Even, true) | (qso::SlotFamily::Odd, false) => current,
        _ => current.saturating_sub(1),
    }
}

pub(crate) fn is_even_slot_family_for_mode(mode: DecoderMode, time: SystemTime) -> bool {
    slot_index_for_mode(mode, time).is_multiple_of(2)
}

fn parse_slot_family_name(value: &str) -> Option<qso::SlotFamily> {
    match value.trim().to_ascii_lowercase().as_str() {
        "even" => Some(qso::SlotFamily::Even),
        "odd" => Some(qso::SlotFamily::Odd),
        _ => None,
    }
}

fn format_slot_time(time: SystemTime) -> String {
    let utc: DateTime<Utc> = time.into();
    utc.format("%H%M%S").to_string()
}

fn format_slot_time_for_mode(mode: DecoderMode, time: SystemTime) -> String {
    if mode == DecoderMode::Ft8 {
        return format_slot_time(time);
    }
    let utc: DateTime<Utc> = time.into();
    format!(
        "{}.{}",
        utc.format("%H%M%S"),
        utc.timestamp_subsec_millis() / 100
    )
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("{}-{}", std::process::id(), name))
}

fn full_decode_sample_count(sample_rate_hz: u32, mode: DecoderMode) -> usize {
    let decoder_samples = DecodeStage::Full.required_samples(mode.spec()) as u64;
    let numerator = decoder_samples * sample_rate_hz as u64 + (DECODER_SAMPLE_RATE_HZ as u64 / 2);
    (numerator / DECODER_SAMPLE_RATE_HZ as u64) as usize
}

fn samples_to_duration(sample_rate_hz: u32, sample_count: usize) -> Duration {
    Duration::from_secs_f64(sample_count as f64 / sample_rate_hz as f64)
}

fn stage_sample_count(sample_rate_hz: u32, mode: DecoderMode, stage: DecodeStage) -> usize {
    (((stage.required_samples(mode.spec()) as u64 * sample_rate_hz as u64)
        + (DECODER_SAMPLE_RATE_HZ as u64 / 2))
        / DECODER_SAMPLE_RATE_HZ as u64) as usize
}

fn capture_window_duration(sample_rate_hz: u32, mode: DecoderMode) -> Duration {
    Duration::from_secs_f64(
        full_decode_sample_count(sample_rate_hz, mode) as f64 / sample_rate_hz as f64,
    )
}

fn slot_capture_end(
    slot_start: SystemTime,
    sample_rate_hz: u32,
    mode: DecoderMode,
) -> Result<SystemTime, AppError> {
    slot_start
        .checked_add(capture_window_duration(sample_rate_hz, mode))
        .ok_or(AppError::Clock)
}

fn stage_capture_end(
    mode: DecoderMode,
    slot_start: SystemTime,
    stage: DecodeStage,
) -> Result<SystemTime, AppError> {
    slot_start
        .checked_add(Duration::from_secs_f64(
            stage.required_samples(mode.spec()) as f64 / DECODER_SAMPLE_RATE_HZ as f64,
        ))
        .ok_or(AppError::Clock)
}

fn tx_margin_after_stage_decode_ms(
    mode: DecoderMode,
    slot_start: SystemTime,
    stage: DecodeStage,
    wall_ms: u128,
) -> Result<i128, AppError> {
    let next_slot = slot_start
        .checked_add(slot_duration_for_mode(mode))
        .ok_or(AppError::Clock)?;
    let tx_start = qso::tx_key_time_for_mode(next_slot, mode);
    let capture_end = stage_capture_end(mode, slot_start, stage)?;
    let capture_to_tx_ms = tx_start
        .duration_since(capture_end)
        .map_err(|_| AppError::Clock)?
        .as_millis() as i128;
    Ok(capture_to_tx_ms - wall_ms as i128)
}

fn format_status(
    active_decode: Option<ActiveDecodeJob>,
    next_slot: SystemTime,
    dropped_slots: u64,
    sample_rate_hz: u32,
    app_mode: DecoderMode,
) -> String {
    let now = SystemTime::now();
    let capture_active = match slot_capture_end(
        current_slot_boundary_for_mode(app_mode, now),
        sample_rate_hz,
        app_mode,
    ) {
        Ok(capture_end) => now < capture_end,
        Err(_) => false,
    };
    let capture_state = if capture_active { "active" } else { "idle" };
    match active_decode {
        Some(active) => format!(
            "capture={} decode={} slot={} drops={} next={}",
            capture_state,
            stage_display_label(app_mode, active.stage),
            format_slot_time_for_mode(app_mode, active.slot_start),
            dropped_slots,
            format_slot_time_for_mode(app_mode, next_slot)
        ),
        None => format!(
            "capture={} decode=idle drops={} next={}",
            capture_state,
            dropped_slots,
            format_slot_time_for_mode(app_mode, next_slot)
        ),
    }
}

fn relabel_stage_update(update: &mut StageDecodeReport, slot_start: SystemTime, mode: DecoderMode) {
    let slot_label = format_slot_time_for_mode(mode, slot_start);
    for decode in &mut update.report.decodes {
        decode.utc = slot_label.clone();
    }
    for decode in &mut update.new_decodes {
        decode.utc = slot_label.clone();
    }
    for decode in &mut update.updated_decodes {
        decode.utc = slot_label.clone();
    }
}

fn preferred_stage_decodes(display: &DisplayState) -> &[DecodedMessage] {
    if !display.full_decodes.is_empty() {
        &display.full_decodes
    } else if !display.early47_decodes.is_empty() {
        &display.early47_decodes
    } else {
        &display.early41_decodes
    }
}

fn composite_rows(display: &DisplayState) -> Vec<CompositeDecodeRow> {
    let mut rows = BTreeMap::<String, CompositeDecodeRow>::new();
    for decode in &display.early41_decodes {
        let key = decode.text.clone();
        rows.insert(
            key,
            CompositeDecodeRow {
                display: decode.clone(),
                seen: "early",
            },
        );
    }
    for decode in &display.early47_decodes {
        let entry = rows
            .entry(decode.text.clone())
            .or_insert_with(|| CompositeDecodeRow {
                display: decode.clone(),
                seen: "mid",
            });
        entry.display = decode.clone();
    }
    for decode in &display.full_decodes {
        let entry = rows
            .entry(decode.text.clone())
            .or_insert_with(|| CompositeDecodeRow {
                display: decode.clone(),
                seen: "full",
            });
        entry.display = decode.clone();
    }
    let mut rows: Vec<_> = rows.into_values().collect();
    rows.sort_by(|left, right| {
        left.display
            .freq_hz
            .total_cmp(&right.display.freq_hz)
            .then_with(|| left.display.text.cmp(&right.display.text))
    });
    rows
}

fn decode_columns(decode: &DecodedMessage) -> (String, String, String, String) {
    match &decode.message {
        StructuredMessage::Standard {
            first,
            second,
            acknowledge,
            info,
            ..
        } => (
            "std".to_string(),
            structured_call_text(first),
            structured_call_text(second),
            structured_info_text(*acknowledge, info),
        ),
        StructuredMessage::Nonstandard {
            hashed_call,
            plain_call,
            hashed_is_second,
            reply,
            cq,
            ..
        } => {
            let hashed = hashed_call_text(hashed_call);
            let (field1, field2) = if *cq {
                ("CQ".to_string(), plain_call.callsign.clone())
            } else if *hashed_is_second {
                (plain_call.callsign.clone(), hashed)
            } else {
                (hashed, plain_call.callsign.clone())
            };
            let info = if *cq {
                "CQ".to_string()
            } else {
                reply_word_text(*reply).to_string()
            };
            ("nonstd".to_string(), field1, field2, info)
        }
        StructuredMessage::Dxpedition {
            completed_call,
            next_call,
            hashed_call10,
            report_db,
            ..
        } => (
            "dxped".to_string(),
            structured_call_text(completed_call),
            structured_call_text(next_call),
            format!(
                "RR73; {} {report_db:+03}",
                hashed_call10
                    .resolved_callsign
                    .as_ref()
                    .map(|callsign| format!("<{callsign}>"))
                    .unwrap_or_else(|| "<...>".to_string())
            ),
        ),
        StructuredMessage::FieldDay {
            first,
            second,
            acknowledge,
            transmitter_count,
            class,
            section,
            ..
        } => (
            "fday".to_string(),
            structured_call_text(first),
            structured_call_text(second),
            if *acknowledge {
                format!("R {transmitter_count}{class} {section}")
            } else {
                format!("{transmitter_count}{class} {section}")
            },
        ),
        StructuredMessage::RttyContest {
            tu,
            first,
            second,
            acknowledge,
            report,
            exchange,
            ..
        } => (
            "rtty".to_string(),
            if *tu {
                format!("TU; {}", structured_call_text(first))
            } else {
                structured_call_text(first)
            },
            structured_call_text(second),
            format!(
                "{}{:03} {}",
                if *acknowledge { "R " } else { "" },
                report,
                match exchange {
                    ft8_decoder::StructuredRttyExchange::Multiplier { value } => value.clone(),
                    ft8_decoder::StructuredRttyExchange::Serial { value } => format!("{value:04}"),
                }
            ),
        ),
        StructuredMessage::EuVhf {
            hashed_call12,
            hashed_call22,
            acknowledge,
            report,
            serial,
            grid6,
            ..
        } => (
            "euvhf".to_string(),
            hashed_call_text(hashed_call12),
            hashed22_call_text(hashed_call22),
            format!(
                "{}{:02}{:04} {}",
                if *acknowledge { "R " } else { "" },
                report,
                serial,
                grid6
            ),
        ),
        StructuredMessage::FreeText { .. } => (
            "free".to_string(),
            String::new(),
            String::new(),
            String::new(),
        ),
        StructuredMessage::Unsupported { .. } => (
            "unsup".to_string(),
            String::new(),
            String::new(),
            String::new(),
        ),
    }
}

fn semantic_sender_call(message: &StructuredMessage) -> Option<String> {
    match message {
        StructuredMessage::Standard { second, .. } => structured_call_station_name(second),
        StructuredMessage::Nonstandard {
            hashed_call,
            plain_call,
            hashed_is_second,
            cq,
            ..
        } => {
            if *cq {
                Some(plain_call.callsign.clone())
            } else if *hashed_is_second {
                hashed_call.resolved_callsign.clone()
            } else {
                Some(plain_call.callsign.clone())
            }
        }
        StructuredMessage::Dxpedition { hashed_call10, .. } => {
            hashed_call10.resolved_callsign.clone()
        }
        StructuredMessage::FieldDay { second, .. }
        | StructuredMessage::RttyContest { second, .. } => structured_call_station_name(second),
        StructuredMessage::EuVhf { hashed_call22, .. } => hashed_call22.resolved_callsign.clone(),
        StructuredMessage::FreeText { .. } | StructuredMessage::Unsupported { .. } => None,
    }
}

fn semantic_first_call_display_call(message: &StructuredMessage) -> Option<String> {
    match message {
        StructuredMessage::Standard { first, .. } => structured_call_station_name(first),
        StructuredMessage::Nonstandard {
            hashed_call,
            plain_call,
            hashed_is_second,
            cq,
            ..
        } => {
            if *cq {
                None
            } else if *hashed_is_second {
                Some(plain_call.callsign.clone())
            } else {
                hashed_call.resolved_callsign.clone()
            }
        }
        StructuredMessage::Dxpedition { completed_call, .. } => {
            structured_call_station_name(completed_call)
        }
        StructuredMessage::FieldDay { first, .. }
        | StructuredMessage::RttyContest { first, .. } => structured_call_station_name(first),
        StructuredMessage::EuVhf { hashed_call12, .. } => hashed_call12.resolved_callsign.clone(),
        StructuredMessage::FreeText { .. } | StructuredMessage::Unsupported { .. } => None,
    }
}

fn structured_call_station_name(field: &StructuredCallField) -> Option<String> {
    match &field.value {
        StructuredCallValue::StandardCall { callsign } => Some(callsign.clone()),
        StructuredCallValue::Hash22 {
            resolved_callsign: Some(callsign),
            ..
        } => Some(callsign.clone()),
        StructuredCallValue::Token { .. } | StructuredCallValue::Hash22 { .. } => None,
    }
}

fn message_related_calls(message: &StructuredMessage) -> Vec<String> {
    let mut related = BTreeSet::<String>::new();
    match message {
        StructuredMessage::Standard { first, second, .. } => {
            if let Some(call) = structured_call_station_name(first) {
                related.insert(call);
            }
            if let Some(call) = structured_call_station_name(second) {
                related.insert(call);
            }
        }
        StructuredMessage::Nonstandard {
            hashed_call,
            plain_call,
            cq,
            ..
        } => {
            related.insert(plain_call.callsign.clone());
            if !*cq {
                if let Some(call) = &hashed_call.resolved_callsign {
                    related.insert(call.clone());
                }
            }
        }
        StructuredMessage::Dxpedition {
            completed_call,
            next_call,
            hashed_call10,
            ..
        } => {
            if let Some(call) = structured_call_station_name(completed_call) {
                related.insert(call);
            }
            if let Some(call) = structured_call_station_name(next_call) {
                related.insert(call);
            }
            if let Some(call) = &hashed_call10.resolved_callsign {
                related.insert(call.clone());
            }
        }
        StructuredMessage::FieldDay { first, second, .. }
        | StructuredMessage::RttyContest { first, second, .. } => {
            if let Some(call) = structured_call_station_name(first) {
                related.insert(call);
            }
            if let Some(call) = structured_call_station_name(second) {
                related.insert(call);
            }
        }
        StructuredMessage::EuVhf {
            hashed_call12,
            hashed_call22,
            ..
        } => {
            if let Some(call) = &hashed_call12.resolved_callsign {
                related.insert(call.clone());
            }
            if let Some(call) = &hashed_call22.resolved_callsign {
                related.insert(call.clone());
            }
        }
        StructuredMessage::FreeText { .. } | StructuredMessage::Unsupported { .. } => {}
    }
    related.into_iter().collect()
}

fn bandmap_detail(message: &StructuredMessage) -> Option<String> {
    match message {
        StructuredMessage::Standard {
            first,
            second,
            acknowledge,
            info,
            ..
        } => {
            let mut parts = Vec::new();
            if let Some(token) = bandmap_token(first).or_else(|| bandmap_token(second)) {
                parts.push(token);
            }
            let info_text = structured_info_text(*acknowledge, info);
            if !info_text.is_empty() {
                parts.push(info_text);
            }
            (!parts.is_empty()).then(|| parts.join(" "))
        }
        StructuredMessage::Nonstandard { reply, cq, .. } => {
            let mut parts = Vec::new();
            if *cq {
                parts.push("CQ".to_string());
            }
            let reply_text = reply_word_text(*reply);
            if !reply_text.is_empty() {
                parts.push(reply_text.to_string());
            }
            (!parts.is_empty()).then(|| parts.join(" "))
        }
        StructuredMessage::Dxpedition { report_db, .. } => Some(format!("RR73 {report_db:+03}")),
        StructuredMessage::FieldDay {
            acknowledge,
            transmitter_count,
            class,
            section,
            ..
        } => Some(if *acknowledge {
            format!("R {transmitter_count}{class} {section}")
        } else {
            format!("{transmitter_count}{class} {section}")
        }),
        StructuredMessage::RttyContest {
            tu,
            acknowledge,
            report,
            exchange,
            ..
        } => Some(format!(
            "{}{}{:03} {}",
            if *tu { "TU; " } else { "" },
            if *acknowledge { "R " } else { "" },
            report,
            match exchange {
                ft8_decoder::StructuredRttyExchange::Multiplier { value } => value.clone(),
                ft8_decoder::StructuredRttyExchange::Serial { value } => format!("{value:04}"),
            }
        )),
        StructuredMessage::EuVhf {
            acknowledge,
            report,
            serial,
            grid6,
            ..
        } => Some(format!(
            "{}{:02}{:04} {}",
            if *acknowledge { "R " } else { "" },
            report,
            serial,
            grid6
        )),
        StructuredMessage::FreeText { .. } | StructuredMessage::Unsupported { .. } => None,
    }
}

fn bandmap_token(field: &StructuredCallField) -> Option<String> {
    match &field.value {
        StructuredCallValue::Token { token } => token
            .split_whitespace()
            .next()
            .map(ToOwned::to_owned)
            .or_else(|| (!token.is_empty()).then(|| token.clone())),
        StructuredCallValue::StandardCall { .. } | StructuredCallValue::Hash22 { .. } => None,
    }
}

fn structured_call_text(field: &StructuredCallField) -> String {
    let base = match &field.value {
        StructuredCallValue::Token { token } => token.clone(),
        StructuredCallValue::StandardCall { callsign } => callsign.clone(),
        StructuredCallValue::Hash22 {
            resolved_callsign: Some(callsign),
            ..
        } => callsign.clone(),
        StructuredCallValue::Hash22 {
            resolved_callsign: None,
            ..
        } => "<...>".to_string(),
    };
    match field.modifier {
        Some(CallModifier::R) => format!("{base}/R"),
        Some(CallModifier::P) => format!("{base}/P"),
        None => base,
    }
}

fn hashed_call_text(field: &ft8_decoder::HashedCallField12) -> String {
    field
        .resolved_callsign
        .as_ref()
        .map(|callsign| format!("<{callsign}>"))
        .unwrap_or_else(|| "<...>".to_string())
}

fn hashed22_call_text(field: &ft8_decoder::HashedCallField22) -> String {
    field
        .resolved_callsign
        .as_ref()
        .map(|callsign| format!("<{callsign}>"))
        .unwrap_or_else(|| "<...>".to_string())
}

fn structured_info_text(acknowledge: bool, info: &StructuredInfoField) -> String {
    match &info.value {
        StructuredInfoValue::Blank => {
            if acknowledge {
                "R".to_string()
            } else {
                String::new()
            }
        }
        StructuredInfoValue::Grid { locator } => {
            if acknowledge {
                format!("R {locator}")
            } else {
                locator.clone()
            }
        }
        StructuredInfoValue::SignalReport { db } => {
            let value = format!("{db:+03}");
            if acknowledge {
                format!("R{value}")
            } else {
                value
            }
        }
        StructuredInfoValue::Reply { word } => reply_word_text(*word).to_string(),
    }
}

fn reply_word_text(word: ft8_decoder::ReplyWord) -> &'static str {
    match word {
        ft8_decoder::ReplyWord::Blank => "",
        ft8_decoder::ReplyWord::Rrr => "RRR",
        ft8_decoder::ReplyWord::Rr73 => "RR73",
        ft8_decoder::ReplyWord::SeventyThree => "73",
    }
}

impl StationTracker {
    fn reset_for_mode(&mut self, mode: DecoderMode) {
        self.mode = mode;
        self.stations.clear();
        self.logs.clear();
        self.hash12_resolutions.clear();
        self.hash22_resolutions.clear();
    }

    fn ingest_stage(
        &mut self,
        received_at: SystemTime,
        stage: DecodeStage,
        decodes: &[DecodedMessage],
    ) {
        let mut grouped = BTreeMap::<String, Vec<&DecodedMessage>>::new();
        let mut passthrough = Vec::<&DecodedMessage>::new();
        for decode in decodes {
            self.observe_message_resolutions(&decode.message);
            if let Some(sender_call) = semantic_sender_call(&decode.message) {
                grouped.entry(sender_call).or_default().push(decode);
            } else {
                passthrough.push(decode);
            }
        }

        for decode in passthrough {
            self.ingest_decode_stage(received_at, stage, decode);
        }

        for sender_decodes in grouped.values() {
            let mut ordered = sender_decodes.clone();
            ordered.sort_by_key(|decode| {
                if message_ends_qso(&decode.message) {
                    1_u8
                } else {
                    0_u8
                }
            });
            for decode in ordered {
                self.ingest_decode_stage(received_at, stage, decode);
            }
        }
        self.prune(received_at);
    }

    fn ingest_frame(&mut self, received_at: SystemTime, decodes: &[DecodedMessage]) {
        self.ingest_stage(received_at, DecodeStage::Full, decodes);
    }

    fn ingest_decode_stage(
        &mut self,
        received_at: SystemTime,
        stage: DecodeStage,
        decode: &DecodedMessage,
    ) {
        let Some(sender_call) = semantic_sender_call(&decode.message) else {
            return;
        };
        let current_slot_index = slot_index_for_mode(self.mode, received_at);
        if let Some(existing) = self.stations.get(&sender_call) {
            if current_slot_index < existing.last_heard_slot_index
                || (current_slot_index == existing.last_heard_slot_index
                    && stage < existing.last_heard_decode_stage)
            {
                return;
            }
        }
        let observed_peer = qso_peer_from_first_field(&decode.message, self);
        let existing_active = self
            .stations
            .get(&sender_call)
            .and_then(|state| state.active_qso.clone());
        let peer_before = existing_active.as_ref().map(|qso| qso.peer.clone());

        enum QsoTransition {
            None,
            End,
            Keep(PeerRef),
            Start(PeerRef),
            Replace { old: ActiveQso, new_peer: PeerRef },
        }

        let transition = if message_ends_qso(&decode.message) {
            QsoTransition::End
        } else if let Some(peer) = observed_peer.clone() {
            match existing_active {
                Some(active) if self.same_peer_identity(&active.peer, &peer) => {
                    QsoTransition::Keep(self.prefer_peer_identity(active.peer, peer))
                }
                Some(active) => QsoTransition::Replace {
                    old: active,
                    new_peer: peer,
                },
                None => QsoTransition::Start(peer),
            }
        } else {
            QsoTransition::None
        };

        let entry = self
            .stations
            .entry(sender_call.clone())
            .or_insert_with(|| StationState {
                last_heard_at: received_at,
                last_heard_slot_index: current_slot_index,
                last_heard_decode_stage: stage,
                last_heard_freq_hz: decode.freq_hz,
                last_heard_snr_db: decode.snr_db,
                last_heard_slot_family: qso::slot_family_for_mode(self.mode, received_at),
                last_message_kind: station_message_kind(&decode.message),
                last_text: decode.text.clone(),
                last_structured_json: serde_json::to_string(&decode.message).unwrap_or_default(),
                active_qso: None,
                last_qso_ended_at: None,
                qso_history: Vec::new(),
            });
        entry.last_heard_at = received_at;
        entry.last_heard_slot_index = current_slot_index;
        entry.last_heard_decode_stage = stage;
        entry.last_heard_freq_hz = decode.freq_hz;
        entry.last_heard_snr_db = decode.snr_db;
        entry.last_heard_slot_family = qso::slot_family_for_mode(self.mode, received_at);
        entry.last_message_kind = station_message_kind(&decode.message);
        entry.last_text = decode.text.clone();
        entry.last_structured_json = serde_json::to_string(&decode.message).unwrap_or_default();

        match transition {
            QsoTransition::None => {}
            QsoTransition::End => {
                if let Some(active) = entry.active_qso.take() {
                    entry.qso_history.push(CompletedQso {
                        started_at: active.since,
                        ended_at: received_at,
                        peer: active.peer,
                    });
                    entry.last_qso_ended_at = Some(received_at);
                }
            }
            QsoTransition::Keep(peer) => {
                if let Some(active) = &mut entry.active_qso {
                    active.peer = peer;
                } else {
                    entry.active_qso = Some(ActiveQso {
                        since: received_at,
                        peer,
                    });
                }
            }
            QsoTransition::Start(peer) => {
                entry.active_qso = Some(ActiveQso {
                    since: received_at,
                    peer,
                });
            }
            QsoTransition::Replace { old, new_peer } => {
                entry.qso_history.push(CompletedQso {
                    started_at: old.since,
                    ended_at: received_at,
                    peer: old.peer,
                });
                entry.active_qso = Some(ActiveQso {
                    since: received_at,
                    peer: new_peer,
                });
            }
        }

        let peer_after = entry.active_qso.as_ref().map(|active| active.peer.clone());
        let (kind, field1, field2, info) = decode_columns(decode);
        let peer = if message_is_cq(&decode.message) {
            None
        } else {
            observed_peer
                .clone()
                .or_else(|| peer_after.clone())
                .or_else(|| peer_before.clone())
        };
        let tracker_tag_peer = if message_ends_qso(&decode.message) {
            peer_before.clone()
        } else {
            peer_after.clone()
        };
        if let Some(observed) = &observed_peer {
            if let Some(logged_peer) = &peer {
                debug_assert!(
                    self.same_peer_identity(observed, logged_peer),
                    "explicit field1 peer should match logged peer"
                );
                if !self.same_peer_identity(observed, logged_peer) {
                    tracing::warn!(
                        sender_call,
                        text = %decode.text,
                        observed_peer = %self.peer_display(observed),
                        logged_peer = %self.peer_display(logged_peer),
                        "station_tracker_logged_peer_mismatch"
                    );
                }
            }
            if let Some(tracker_peer) = &tracker_tag_peer {
                debug_assert!(
                    self.same_peer_identity(observed, tracker_peer),
                    "explicit field1 peer should match tracker state"
                );
                if !self.same_peer_identity(observed, tracker_peer) {
                    tracing::warn!(
                        sender_call,
                        text = %decode.text,
                        observed_peer = %self.peer_display(observed),
                        tracker_peer = %self.peer_display(tracker_peer),
                        message_ends_qso = message_ends_qso(&decode.message),
                        "station_tracker_state_mismatch"
                    );
                }
            }
        }
        let logged = LoggedDecode {
            received_at,
            slot_index: current_slot_index,
            decode_stage: stage,
            sender_call: sender_call.clone(),
            peer,
            peer_before,
            peer_after,
            related_calls: message_related_calls(&decode.message),
            snr_db: decode.snr_db,
            dt_seconds: decode.dt_seconds,
            freq_hz: decode.freq_hz,
            kind,
            field1,
            field2,
            info,
            text: decode.text.clone(),
        };
        if let Some(existing) = self.logs.iter_mut().find(|entry| {
            entry.sender_call == sender_call && entry.slot_index == current_slot_index
        }) {
            if stage >= existing.decode_stage {
                *existing = logged;
            }
        } else {
            self.logs.push_back(logged);
        }
    }

    fn observe_message_resolutions(&mut self, message: &StructuredMessage) {
        match message {
            StructuredMessage::Standard { first, second, .. } => {
                self.observe_structured_call(first);
                self.observe_structured_call(second);
            }
            StructuredMessage::Nonstandard { hashed_call, .. } => {
                if let Some(callsign) = &hashed_call.resolved_callsign {
                    self.hash12_resolutions
                        .insert(hashed_call.raw, callsign.clone());
                }
            }
            StructuredMessage::Dxpedition { .. }
            | StructuredMessage::FieldDay { .. }
            | StructuredMessage::RttyContest { .. } => {}
            StructuredMessage::EuVhf {
                hashed_call12,
                hashed_call22,
                ..
            } => {
                if let Some(callsign) = &hashed_call12.resolved_callsign {
                    self.hash12_resolutions
                        .insert(hashed_call12.raw, callsign.clone());
                }
                if let Some(callsign) = &hashed_call22.resolved_callsign {
                    self.hash22_resolutions
                        .insert(hashed_call22.raw, callsign.clone());
                }
            }
            StructuredMessage::FreeText { .. } | StructuredMessage::Unsupported { .. } => {}
        }
    }

    fn observe_structured_call(&mut self, field: &StructuredCallField) {
        if let StructuredCallValue::Hash22 {
            hash,
            resolved_callsign: Some(callsign),
        } = &field.value
        {
            self.hash22_resolutions.insert(*hash, callsign.clone());
        }
    }

    fn resolve_peer(&self, peer: PeerRef) -> PeerRef {
        match peer {
            PeerRef::Callsign(_) => peer,
            PeerRef::Hash12(raw) => self
                .hash12_resolutions
                .get(&raw)
                .cloned()
                .map(PeerRef::Callsign)
                .unwrap_or(PeerRef::Hash12(raw)),
            PeerRef::Hash22(raw) => self
                .hash22_resolutions
                .get(&raw)
                .cloned()
                .map(PeerRef::Callsign)
                .unwrap_or(PeerRef::Hash22(raw)),
        }
    }

    fn same_peer_identity(&self, left: &PeerRef, right: &PeerRef) -> bool {
        match (
            self.resolve_peer(left.clone()),
            self.resolve_peer(right.clone()),
        ) {
            (PeerRef::Callsign(left), PeerRef::Callsign(right)) => left == right,
            (PeerRef::Hash12(left), PeerRef::Hash12(right)) => left == right,
            (PeerRef::Hash22(left), PeerRef::Hash22(right)) => left == right,
            _ => false,
        }
    }

    fn prefer_peer_identity(&self, existing: PeerRef, observed: PeerRef) -> PeerRef {
        match (
            self.resolve_peer(existing.clone()),
            self.resolve_peer(observed.clone()),
        ) {
            (PeerRef::Callsign(_), PeerRef::Callsign(_)) => self.resolve_peer(observed),
            (PeerRef::Callsign(_), _) => self.resolve_peer(existing),
            (_, PeerRef::Callsign(_)) => self.resolve_peer(observed),
            _ => self.resolve_peer(observed),
        }
    }

    fn peer_display(&self, peer: &PeerRef) -> String {
        match self.resolve_peer(peer.clone()) {
            PeerRef::Callsign(call) => call,
            PeerRef::Hash12(raw) => format!("<h12:{raw:03X}>"),
            PeerRef::Hash22(raw) => format!("<h22:{raw:06X}>"),
        }
    }

    fn prune(&mut self, now: SystemTime) {
        while matches!(
            self.logs.front(),
            Some(entry) if now.duration_since(entry.received_at).unwrap_or_default() > STATION_RETENTION
        ) {
            self.logs.pop_front();
        }
        self.stations.retain(|_, state| {
            let keep =
                now.duration_since(state.last_heard_at).unwrap_or_default() <= STATION_RETENTION;
            if keep {
                state.qso_history.retain(|qso| {
                    now.duration_since(qso.ended_at).unwrap_or_default() <= STATION_RETENTION
                });
            }
            keep
        });
    }

    fn web_station_summaries(&self) -> Vec<WebStationSummary> {
        self.stations
            .iter()
            .map(|(callsign, state)| WebStationSummary {
                callsign: callsign.clone(),
                last_heard_at: format_time(state.last_heard_at),
                last_heard_freq_hz: state.last_heard_freq_hz,
                last_heard_snr_db: state.last_heard_snr_db,
                last_heard_slot_family: state.last_heard_slot_family.as_str().to_string(),
                is_in_qso: state.active_qso.is_some(),
                in_qso_since: state.active_qso.as_ref().map(|qso| format_time(qso.since)),
                qso_with: state
                    .active_qso
                    .as_ref()
                    .map(|qso| self.peer_display(&qso.peer)),
                last_qso_ended_at: state.last_qso_ended_at.map(format_time),
                qso_history: state
                    .qso_history
                    .iter()
                    .map(|qso| WebCompletedQso {
                        started_at: format_time(qso.started_at),
                        ended_at: format_time(qso.ended_at),
                        peer: self.peer_display(&qso.peer),
                    })
                    .collect(),
            })
            .collect()
    }

    fn start_info(&self, callsign: &str) -> Option<StationStartInfo> {
        self.stations.get(callsign).map(|state| StationStartInfo {
            callsign: callsign.to_string(),
            last_heard_at: state.last_heard_at,
            last_heard_slot_family: state.last_heard_slot_family,
            last_snr_db: state.last_heard_snr_db,
            last_text: (!state.last_text.is_empty()).then(|| state.last_text.clone()),
            last_structured_json: (!state.last_structured_json.is_empty())
                .then(|| state.last_structured_json.clone()),
        })
    }

    fn web_logs(&self) -> Vec<WebStationLog> {
        self.logs
            .iter()
            .map(|entry| WebStationLog {
                timestamp: format_time(entry.received_at),
                sender_call: entry.sender_call.clone(),
                peer: entry.peer.as_ref().map(|peer| self.peer_display(peer)),
                peer_before: entry
                    .peer_before
                    .as_ref()
                    .map(|peer| self.peer_display(peer)),
                peer_after: entry
                    .peer_after
                    .as_ref()
                    .map(|peer| self.peer_display(peer)),
                related_calls: entry.related_calls.clone(),
                snr_db: entry.snr_db,
                dt_seconds: entry.dt_seconds,
                freq_hz: entry.freq_hz,
                kind: entry.kind.clone(),
                field1: entry.field1.clone(),
                field2: entry.field2.clone(),
                info: entry.info.clone(),
                text: entry.text.clone(),
            })
            .collect()
    }

    fn web_direct_calls(&self, our_call: &str, since: SystemTime) -> Vec<WebDirectCallLog> {
        self.logs
            .iter()
            .filter(|entry| {
                entry.received_at >= since
                    && entry.peer.as_ref().map(|peer| self.peer_display(peer))
                        == Some(our_call.to_string())
            })
            .map(|entry| WebDirectCallLog {
                sort_epoch_ms: system_time_to_epoch_ms(entry.received_at),
                timestamp: format_time(entry.received_at),
                from_call: entry.sender_call.clone(),
                to_call: our_call.to_string(),
                snr_db: Some(entry.snr_db),
                dt_seconds: Some(entry.dt_seconds),
                freq_hz: Some(entry.freq_hz),
                text: entry.text.clone(),
                is_ours: false,
            })
            .collect()
    }

    fn unique_sender_count_since_excluding(&self, since: SystemTime, excluded_call: &str) -> usize {
        let mut calls = BTreeSet::new();
        for entry in &self.logs {
            if entry.received_at >= since && !entry.sender_call.eq_ignore_ascii_case(excluded_call)
            {
                calls.insert(entry.sender_call.clone());
            }
        }
        calls.len()
    }

    fn sender_decode_count_since(&self, since: SystemTime, callsign: &str) -> usize {
        self.logs
            .iter()
            .filter(|entry| {
                entry.received_at >= since && entry.sender_call.eq_ignore_ascii_case(callsign)
            })
            .count()
    }
}

fn qso_peer_from_first_field(
    message: &StructuredMessage,
    tracker: &StationTracker,
) -> Option<PeerRef> {
    match message {
        StructuredMessage::Standard { first, .. } => match &first.value {
            StructuredCallValue::StandardCall { callsign } => {
                Some(PeerRef::Callsign(callsign.clone()))
            }
            StructuredCallValue::Hash22 { hash, .. } => {
                Some(tracker.resolve_peer(PeerRef::Hash22(*hash)))
            }
            StructuredCallValue::Token { .. } => None,
        },
        StructuredMessage::Nonstandard {
            hashed_call,
            plain_call,
            hashed_is_second,
            cq,
            ..
        } => {
            if *cq {
                None
            } else if *hashed_is_second {
                Some(PeerRef::Callsign(plain_call.callsign.clone()))
            } else {
                Some(tracker.resolve_peer(PeerRef::Hash12(hashed_call.raw)))
            }
        }
        StructuredMessage::Dxpedition { completed_call, .. } => match &completed_call.value {
            StructuredCallValue::StandardCall { callsign } => {
                Some(PeerRef::Callsign(callsign.clone()))
            }
            StructuredCallValue::Hash22 { hash, .. } => {
                Some(tracker.resolve_peer(PeerRef::Hash22(*hash)))
            }
            StructuredCallValue::Token { .. } => None,
        },
        StructuredMessage::FieldDay { first, .. }
        | StructuredMessage::RttyContest { first, .. } => match &first.value {
            StructuredCallValue::StandardCall { callsign } => {
                Some(PeerRef::Callsign(callsign.clone()))
            }
            StructuredCallValue::Hash22 { hash, .. } => {
                Some(tracker.resolve_peer(PeerRef::Hash22(*hash)))
            }
            StructuredCallValue::Token { .. } => None,
        },
        StructuredMessage::EuVhf { hashed_call12, .. } => {
            Some(tracker.resolve_peer(PeerRef::Hash12(hashed_call12.raw)))
        }
        StructuredMessage::FreeText { .. } | StructuredMessage::Unsupported { .. } => None,
    }
}

fn message_ends_qso(message: &StructuredMessage) -> bool {
    if message_is_cq(message) {
        return true;
    }
    match message {
        StructuredMessage::Standard { info, .. } => matches!(
            info.value,
            StructuredInfoValue::Reply {
                word: ft8_decoder::ReplyWord::Rr73 | ft8_decoder::ReplyWord::SeventyThree
            }
        ),
        StructuredMessage::Nonstandard { reply, .. } => matches!(
            reply,
            ft8_decoder::ReplyWord::Rr73 | ft8_decoder::ReplyWord::SeventyThree
        ),
        StructuredMessage::Dxpedition { .. } => true,
        StructuredMessage::FieldDay { .. }
        | StructuredMessage::RttyContest { .. }
        | StructuredMessage::EuVhf { .. } => false,
        StructuredMessage::FreeText { .. } | StructuredMessage::Unsupported { .. } => false,
    }
}

fn message_is_cq(message: &StructuredMessage) -> bool {
    match message {
        StructuredMessage::Standard { first, .. } => matches!(
            &first.value,
            StructuredCallValue::Token { token } if token == "CQ" || token.starts_with("CQ ")
        ),
        StructuredMessage::Nonstandard { cq, .. } => *cq,
        StructuredMessage::Dxpedition { .. }
        | StructuredMessage::FieldDay { .. }
        | StructuredMessage::RttyContest { .. }
        | StructuredMessage::EuVhf { .. } => false,
        StructuredMessage::FreeText { .. } | StructuredMessage::Unsupported { .. } => false,
    }
}

fn station_message_kind(message: &StructuredMessage) -> StationLastMessageKind {
    if message_is_cq(message) {
        return StationLastMessageKind::Cq;
    }
    match message {
        StructuredMessage::Standard { info, .. } => match &info.value {
            StructuredInfoValue::Grid { locator } if locator.eq_ignore_ascii_case("RR73") => {
                StationLastMessageKind::Rr73
            }
            StructuredInfoValue::Reply {
                word: ft8_decoder::ReplyWord::Rr73,
            } => StationLastMessageKind::Rr73,
            StructuredInfoValue::Reply {
                word: ft8_decoder::ReplyWord::SeventyThree,
            } => StationLastMessageKind::SeventyThree,
            _ => StationLastMessageKind::Other,
        },
        StructuredMessage::Nonstandard { reply, .. } => match reply {
            ft8_decoder::ReplyWord::Rr73 => StationLastMessageKind::Rr73,
            ft8_decoder::ReplyWord::SeventyThree => StationLastMessageKind::SeventyThree,
            _ => StationLastMessageKind::Other,
        },
        StructuredMessage::Dxpedition { .. }
        | StructuredMessage::FieldDay { .. }
        | StructuredMessage::RttyContest { .. }
        | StructuredMessage::EuVhf { .. } => StationLastMessageKind::Other,
        StructuredMessage::FreeText { .. } | StructuredMessage::Unsupported { .. } => {
            StationLastMessageKind::Other
        }
    }
}

fn direct_call_observation_from_decode(
    decode: &DecodedMessage,
    our_call: &str,
    observed_at: SystemTime,
    mode: DecoderMode,
) -> Option<DirectCallObservation> {
    let sender_call = semantic_sender_call(&decode.message)?;
    let (start_state, compound_eligible) = match &decode.message {
        StructuredMessage::Standard {
            first,
            acknowledge,
            info,
            ..
        } => {
            if structured_call_station_name(first).as_deref() != Some(our_call) {
                return None;
            }
            match &info.value {
                StructuredInfoValue::Grid { locator } if locator.eq_ignore_ascii_case("RR73") => {
                    return None;
                }
                StructuredInfoValue::Reply {
                    word: ft8_decoder::ReplyWord::Rr73 | ft8_decoder::ReplyWord::SeventyThree,
                } => return None,
                StructuredInfoValue::Reply {
                    word: ft8_decoder::ReplyWord::Rrr,
                } => (QsoState::Send73, false),
                StructuredInfoValue::Reply {
                    word: ft8_decoder::ReplyWord::Blank,
                } => {
                    if *acknowledge {
                        (QsoState::Send73, false)
                    } else {
                        (QsoState::SendSig, false)
                    }
                }
                StructuredInfoValue::Grid { .. } => {
                    if *acknowledge {
                        (QsoState::Send73, false)
                    } else {
                        (QsoState::SendSig, true)
                    }
                }
                StructuredInfoValue::SignalReport { .. } => {
                    if *acknowledge {
                        (QsoState::SendRR73, false)
                    } else {
                        (QsoState::SendSigAck, false)
                    }
                }
                StructuredInfoValue::Blank => {
                    if *acknowledge {
                        (QsoState::Send73, false)
                    } else {
                        (QsoState::SendSig, false)
                    }
                }
            }
        }
        StructuredMessage::Nonstandard { reply, cq, .. } => {
            if *cq || semantic_first_call_display_call(&decode.message).as_deref() != Some(our_call)
            {
                return None;
            }
            match reply {
                ft8_decoder::ReplyWord::Blank => (QsoState::SendSig, false),
                ft8_decoder::ReplyWord::Rrr => (QsoState::Send73, false),
                ft8_decoder::ReplyWord::Rr73 | ft8_decoder::ReplyWord::SeventyThree => {
                    return None;
                }
            }
        }
        StructuredMessage::Dxpedition {
            completed_call,
            next_call,
            ..
        } => {
            if structured_call_station_name(completed_call).as_deref() == Some(our_call) {
                return None;
            }
            if structured_call_station_name(next_call).as_deref() != Some(our_call) {
                return None;
            }
            (QsoState::SendSigAck, false)
        }
        StructuredMessage::FieldDay { .. }
        | StructuredMessage::RttyContest { .. }
        | StructuredMessage::EuVhf { .. } => return None,
        StructuredMessage::FreeText { .. } | StructuredMessage::Unsupported { .. } => return None,
    };
    Some(DirectCallObservation {
        callsign: sender_call,
        observed_at,
        slot_index: slot_index_for_mode(mode, observed_at),
        slot_family: qso::slot_family_for_mode(mode, observed_at),
        snr_db: decode.snr_db,
        start_state,
        compound_eligible,
        text: decode.text.clone(),
        structured_json: serde_json::to_string(&decode.message).unwrap_or_default(),
    })
}

fn maybe_track_priority_directs(
    work_queue: &mut WorkQueueState,
    qso_controller: &mut QsoController,
    decodes: &[DecodedMessage],
    our_call: &str,
    slot_start: SystemTime,
    slot_skip_calls: &BTreeSet<String>,
) {
    if work_queue.auto_add_direct_calls {
        let active_partner = qso_controller.active_partner_call();
        for decode in decodes {
            let Some(observation) = direct_call_observation_from_decode(
                decode,
                our_call,
                slot_start,
                work_queue.current_mode,
            ) else {
                continue;
            };
            if slot_skip_calls
                .iter()
                .any(|callsign| callsign.eq_ignore_ascii_case(observation.callsign.as_str()))
            {
                continue;
            }
            if active_partner.as_deref() == Some(observation.callsign.as_str()) {
                continue;
            }
            let reserved_station = qso::StationStartInfo {
                callsign: observation.callsign.clone(),
                last_heard_at: observation.observed_at,
                last_heard_slot_family: observation.slot_family,
                last_snr_db: observation.snr_db,
                last_text: Some(observation.text.clone()),
                last_structured_json: Some(observation.structured_json.clone()),
            };
            if qso_controller
                .refresh_reserved_compound_next_station(reserved_station, observation.observed_at)
            {
                continue;
            }
            if let Err(message) = work_queue.add_direct_observation(observation, slot_start) {
                warn!(message, "queue_direct_add_failed");
            }
        }
    }
}

fn maybe_auto_add_decoded_calls(
    work_queue: &mut WorkQueueState,
    station_tracker: &StationTracker,
    qso_controller: &QsoController,
    decodes: &[DecodedMessage],
    now: SystemTime,
) {
    if !work_queue.auto_add_all_decoded_calls {
        return;
    }
    let active_partner = qso_controller.active_partner_call();
    let mut callsigns = BTreeSet::new();
    let since = now.checked_sub(CQ_ACTIVITY_WINDOW).unwrap_or(now);
    for decode in decodes {
        let Some(callsign) = semantic_sender_call(&decode.message) else {
            continue;
        };
        if active_partner.as_deref() == Some(callsign.as_str()) {
            continue;
        }
        callsigns.insert(callsign);
    }
    for callsign in callsigns {
        if station_tracker.sender_decode_count_since(since, &callsign)
            < work_queue.auto_add_decoded_min_count_5m as usize
        {
            continue;
        }
        let Some(info) = station_tracker.start_info(&callsign) else {
            continue;
        };
        if let Err(message) = work_queue.add_station(&callsign, info.last_heard_at, now) {
            warn!(callsign, message, "queue_auto_add_decoded_failed");
        }
    }
}

fn station_info_from_dispatch(
    tracker: &StationTracker,
    kind: &QueueDispatchKind,
) -> Option<StationStartInfo> {
    let QueueDispatchKind::Station {
        callsign,
        context_last_heard_at,
        context_last_heard_slot_family,
        context_text,
        context_structured_json,
        context_snr_db,
        ..
    } = kind
    else {
        return None;
    };
    let mut station_info = tracker.start_info(callsign);
    if let (Some(last_heard_at), Some(last_heard_slot_family)) =
        (*context_last_heard_at, *context_last_heard_slot_family)
    {
        match &mut station_info {
            Some(info) if last_heard_at >= info.last_heard_at => {
                info.last_heard_at = last_heard_at;
                info.last_heard_slot_family = last_heard_slot_family;
                info.last_snr_db = context_snr_db.unwrap_or(info.last_snr_db);
                info.last_text = context_text.clone().or_else(|| info.last_text.clone());
                info.last_structured_json = context_structured_json
                    .clone()
                    .or_else(|| info.last_structured_json.clone());
            }
            None => {
                station_info = Some(StationStartInfo {
                    callsign: callsign.clone(),
                    last_heard_at,
                    last_heard_slot_family,
                    last_snr_db: context_snr_db.unwrap_or(0),
                    last_text: context_text.clone(),
                    last_structured_json: context_structured_json.clone(),
                });
            }
            Some(info) => {
                if let Some(text) = context_text.clone() {
                    info.last_text = Some(text);
                }
                if let Some(structured_json) = context_structured_json.clone() {
                    info.last_structured_json = Some(structured_json);
                }
                if let Some(snr_db) = *context_snr_db {
                    info.last_snr_db = snr_db;
                }
            }
        }
    }
    station_info
}

fn qso_start_command_from_dispatch(dispatch: &QueueDispatch) -> qso::QsoCommand {
    match &dispatch.kind {
        QueueDispatchKind::Station {
            callsign,
            initial_state,
            start_mode,
            ..
        } => qso::QsoCommand::Start {
            partner_call: callsign.clone(),
            tx_freq_hz: dispatch.tx_freq_hz,
            initial_state: *initial_state,
            start_mode: *start_mode,
            tx_slot_family_override: Some(dispatch.tx_slot_family),
        },
        QueueDispatchKind::Cq {
            tx_slot_family_override,
        } => qso::QsoCommand::Start {
            partner_call: "CQ".to_string(),
            tx_freq_hz: dispatch.tx_freq_hz,
            initial_state: QsoState::SendCq,
            start_mode: QsoStartMode::Cq,
            tx_slot_family_override: *tx_slot_family_override,
        },
    }
}

fn maybe_arm_compound_handoff_from_queue(
    work_queue: &mut WorkQueueState,
    qso_controller: &mut QsoController,
    tracker: &StationTracker,
    slot_start: SystemTime,
    now: SystemTime,
) {
    let active_partner = qso_controller.active_partner_call();
    let Some(candidate) =
        work_queue.peek_compound_handoff_candidate(now, active_partner.as_deref())
    else {
        return;
    };
    let Some(station_info) = station_info_from_dispatch(tracker, &candidate.kind) else {
        return;
    };
    if qso_controller.maybe_arm_compound_handoff(
        slot_start,
        qso::CompoundHandoffPlan {
            next_station: station_info,
        },
        work_queue.use_compound_73_once_handoff(),
        now,
    ) {
        work_queue.remove_station(&candidate.callsign, "compound_handoff_reserved");
        info!(
            finished_call = qso_controller.active_partner_call().unwrap_or_default(),
            next_call = candidate.callsign,
            slot = %format_slot_time(slot_start),
            "qso_compound_handoff_reserved"
        );
    }
}

fn format_time(time: SystemTime) -> String {
    let utc: DateTime<Utc> = time.into();
    utc.format("%H:%M:%S").to_string()
}

fn format_datetime(time: SystemTime) -> String {
    let utc: DateTime<Utc> = time.into();
    utc.format("%Y-%m-%d %H:%M:%S").to_string()
}

fn format_relative_age(age: Duration) -> String {
    let seconds = age.as_secs();
    if seconds < 90 {
        format!("{} secs ago", seconds.max(1))
    } else if seconds < 90 * 60 {
        format!("{} mins ago", ((seconds as f64) / 60.0).round() as u64)
    } else if seconds < 36 * 60 * 60 {
        format!("{} hours ago", ((seconds as f64) / 3600.0).round() as u64)
    } else {
        format!("{} days ago", ((seconds as f64) / 86400.0).round() as u64)
    }
}

fn system_time_to_epoch_ms(time: SystemTime) -> u64 {
    time.duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u64::MAX as u128) as u64
}

fn resample_linear_f32(samples: &[f32], src_rate_hz: u32, dst_rate_hz: u32) -> Vec<f32> {
    if samples.is_empty() || src_rate_hz == dst_rate_hz {
        return samples.to_vec();
    }

    let output_len = ((samples.len() as u64 * dst_rate_hz as u64) + (src_rate_hz as u64 / 2))
        / src_rate_hz as u64;
    let mut output = Vec::with_capacity(output_len as usize);
    let scale = src_rate_hz as f64 / dst_rate_hz as f64;
    for index in 0..output_len as usize {
        let position = index as f64 * scale;
        let left = position.floor() as usize;
        let right = (left + 1).min(samples.len().saturating_sub(1));
        let frac = (position - left as f64) as f32;
        output.push(samples[left] * (1.0 - frac) + samples[right] * frac);
    }
    output
}

fn should_refresh_waterfall(
    now: SystemTime,
    last_update: SystemTime,
    latest_sample_time: Option<SystemTime>,
    last_sample_time: SystemTime,
) -> bool {
    if now.duration_since(last_update).unwrap_or_default()
        < Duration::from_millis(WATERFALL_UPDATE_MS)
    {
        return false;
    }
    latest_sample_time.is_some_and(|latest| latest > last_sample_time)
}

fn compute_latest_waterfall_row(
    capture: &SampleStream,
    latest_sample_time: SystemTime,
) -> Result<Vec<u8>, AppError> {
    let sample_rate_hz = capture.config().sample_rate_hz;
    let start = latest_sample_time
        .checked_sub(samples_to_duration(sample_rate_hz, WATERFALL_SAMPLES))
        .ok_or(AppError::Clock)?;
    let samples = capture.extract_window(start, WATERFALL_SAMPLES)?;
    Ok(compute_waterfall_row(&samples, sample_rate_hz))
}

fn compute_waterfall_row(samples: &[i16], sample_rate_hz: u32) -> Vec<u8> {
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(WATERFALL_SAMPLES);
    let mut buffer = vec![Complex32::new(0.0, 0.0); WATERFALL_SAMPLES];
    for (index, slot) in buffer.iter_mut().enumerate() {
        let window = 0.5
            - 0.5 * ((2.0 * std::f32::consts::PI * index as f32) / WATERFALL_SAMPLES as f32).cos();
        let sample = samples.get(index).copied().unwrap_or_default() as f32 / i16::MAX as f32;
        *slot = Complex32::new(sample * window, 0.0);
    }
    fft.process(&mut buffer);

    let bin_hz = sample_rate_hz as f32 / WATERFALL_SAMPLES as f32;
    let bucket_hz = WATERFALL_MAX_HZ / WATERFALL_BUCKETS as f32;
    let mut bucket_db = vec![0.0f32; WATERFALL_BUCKETS];
    let mut min_db = f32::INFINITY;
    let mut max_db = f32::NEG_INFINITY;
    let mut sum_db = 0.0f32;
    let mut row = vec![0u8; WATERFALL_BUCKETS];
    for (bucket_index, db_slot) in bucket_db.iter_mut().enumerate() {
        let start_hz = bucket_index as f32 * bucket_hz;
        let end_hz = start_hz + bucket_hz;
        let start_bin = (start_hz / bin_hz).floor() as usize;
        let end_bin = ((end_hz / bin_hz).ceil() as usize).min(buffer.len() / 2);
        let mut peak = 1.0e-12f32;
        for bin in start_bin..end_bin.max(start_bin + 1) {
            peak = peak.max(buffer[bin].norm_sqr());
        }
        let db = 10.0 * peak.log10();
        *db_slot = db;
        min_db = min_db.min(db);
        max_db = max_db.max(db);
        sum_db += db;
    }

    let mean_db = sum_db / WATERFALL_BUCKETS as f32;
    let floor_db = min_db.max(mean_db - 10.0);
    let ceiling_db = max_db.min(mean_db + 26.0);
    let span_db = (ceiling_db - floor_db).max(8.0);
    for (bucket_index, cell) in row.iter_mut().enumerate() {
        let normalized = ((bucket_db[bucket_index] - floor_db) / span_db).clamp(0.0, 1.0);
        let curved = normalized.powf(0.72);
        *cell = (curved * 255.0).round() as u8;
    }
    row
}

fn push_waterfall_row(rows: &mut VecDeque<Vec<u8>>, row: Vec<u8>) {
    if rows.len() == WATERFALL_HISTORY_ROWS {
        rows.pop_back();
    }
    rows.push_front(row);
}

fn seeded_waterfall_rows() -> VecDeque<Vec<u8>> {
    let mut rows = VecDeque::with_capacity(WATERFALL_HISTORY_ROWS);
    for _ in 0..WATERFALL_HISTORY_ROWS {
        rows.push_back(vec![0u8; WATERFALL_BUCKETS]);
    }
    rows
}

fn update_bandmaps(
    store: &mut BandMapStore,
    mode: DecoderMode,
    slot_start: SystemTime,
    decodes: &[DecodedMessage],
) {
    let slot_idx = slot_index_for_mode(mode, slot_start);
    prune_bandmap(&mut store.even, slot_idx);
    prune_bandmap(&mut store.odd, slot_idx);
    let map = if is_even_slot_family_for_mode(mode, slot_start) {
        store.even_last_updated_at = Some(slot_start);
        &mut store.even
    } else {
        store.odd_last_updated_at = Some(slot_start);
        &mut store.odd
    };
    for decode in decodes {
        for (callsign, detail) in bandmap_calls_from_decode(decode) {
            map.insert(
                callsign.clone(),
                BandMapEntry {
                    callsign,
                    detail,
                    freq_hz: decode.freq_hz,
                    last_seen_slot_index: slot_idx,
                },
            );
        }
    }
}

fn clear_bandmaps(store: &mut BandMapStore) {
    store.even.clear();
    store.odd.clear();
    store.even_last_updated_at = None;
    store.odd_last_updated_at = None;
}

fn bandmap_age_seconds(last_updated_at: Option<SystemTime>, now: SystemTime) -> Option<u64> {
    last_updated_at.map(|updated_at| now.duration_since(updated_at).unwrap_or_default().as_secs())
}

fn prune_bandmap(map: &mut BTreeMap<String, BandMapEntry>, current_slot_index: u64) {
    map.retain(|_, entry| {
        current_slot_index.saturating_sub(entry.last_seen_slot_index) < BANDMAP_MAX_AGE_SLOTS
    });
}

fn bandmap_calls_from_decode(decode: &DecodedMessage) -> Vec<(String, Option<String>)> {
    let detail = bandmap_detail(&decode.message);
    semantic_sender_call(&decode.message)
        .into_iter()
        .map(|call| (call, detail.clone()))
        .collect()
}

fn build_bandmap_grid(
    map: &BTreeMap<String, BandMapEntry>,
    current_slot_index: u64,
    work_queue: &WorkQueueState,
    now: SystemTime,
) -> Vec<Vec<Vec<WebBandMapCall>>> {
    let mut cells = vec![vec![Vec::<(f32, WebBandMapCall)>::new(); BANDMAP_COLUMNS]; BANDMAP_ROWS];
    for entry in map.values() {
        let age_slots = current_slot_index.saturating_sub(entry.last_seen_slot_index);
        if age_slots >= BANDMAP_MAX_AGE_SLOTS {
            continue;
        }
        if !(0.0..WATERFALL_MAX_HZ).contains(&entry.freq_hz) {
            continue;
        }
        let column = (entry.freq_hz / BANDMAP_CELL_HZ).floor() as usize;
        let row = 0usize;
        if row < BANDMAP_ROWS && column < BANDMAP_COLUMNS {
            cells[row][column].push((
                entry.freq_hz,
                WebBandMapCall {
                    callsign: entry.callsign.clone(),
                    detail: entry.detail.clone(),
                    age_slots,
                    worked_recently: work_queue
                        .was_worked_recently_on_current_band(&entry.callsign, now),
                },
            ));
        }
    }

    cells
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|mut cell| {
                    cell.sort_by(|left, right| {
                        left.0
                            .total_cmp(&right.0)
                            .then_with(|| left.1.callsign.cmp(&right.1.callsign))
                    });
                    cell.into_iter().map(|(_, call)| call).collect()
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        FsmConfig, LoggingConfig, NoFwdThreshold, QueueConfig, RetryThresholds, RigConfig,
        StationConfig, TxConfig,
    };
    use crate::qso::{SlotFamily, TxBackend};

    fn ingest_decode(
        tracker: &mut StationTracker,
        received_at: SystemTime,
        decode: &DecodedMessage,
    ) {
        tracker.ingest_frame(received_at, std::slice::from_ref(decode));
    }

    fn ft8_slot_index(time: SystemTime) -> u64 {
        slot_index_for_mode(DecoderMode::Ft8, time)
    }

    fn ft8_slot_family(time: SystemTime) -> SlotFamily {
        qso::slot_family_for_mode(DecoderMode::Ft8, time)
    }

    fn ft8_next_slot_boundary(now: SystemTime) -> SystemTime {
        next_slot_boundary_for_mode(DecoderMode::Ft8, now)
    }

    fn sample_app_config() -> AppConfig {
        AppConfig {
            station: StationConfig {
                our_call: "N1VF".to_string(),
                our_grid: "CM97".to_string(),
            },
            rig: None,
            tx: TxConfig {
                base_freq_hz: 1000.0,
                drive_level: 0.28,
                playback_channels: 2,
                output_device: None,
                power_w: Some(20.0),
                tx_freq_min_hz: 200.0,
                tx_freq_max_hz: 3500.0,
            },
            queue: QueueConfig {
                auto_add_all_decoded_calls_default: false,
                auto_add_decoded_min_count_5m_default: 2,
                auto_add_direct_calls_default: true,
                ignore_direct_calls_from_recently_worked_default: true,
                cq_enabled_default: false,
                cq_percent_default: 80,
                pause_cq_when_few_unique_calls_default: false,
                cq_pause_min_unique_calls_5m_default: 3,
                use_compound_rr73_handoff_default: true,
                use_compound_73_once_handoff_default: false,
                use_compound_for_direct_signal_callers_default: false,
                no_message_retry_delay_seconds_default: 35,
                no_forward_retry_delay_seconds_default: 300,
            },
            fsm: FsmConfig {
                rr73_enabled: true,
                timeout_seconds: 600,
                send_grid: RetryThresholds {
                    no_fwd: 3,
                    no_msg: 3,
                },
                send_sig: RetryThresholds {
                    no_fwd: 3,
                    no_msg: 3,
                },
                send_sig_ack: RetryThresholds {
                    no_fwd: 5,
                    no_msg: 2,
                },
                send_rr73: NoFwdThreshold { no_fwd: 3 },
                send_rrr: RetryThresholds {
                    no_fwd: 5,
                    no_msg: 2,
                },
                send_73: RetryThresholds {
                    no_fwd: 3,
                    no_msg: 2,
                },
            },
            logging: LoggingConfig {
                fsm_log_path: "logs/test-qso.jsonl".to_string(),
                app_log_path: "logs/test-ft8rx.log".to_string(),
            },
        }
    }

    #[test]
    fn configured_power_request_uses_continuous_k3s_power() {
        let config = sample_app_config();
        assert_eq!(
            configured_power_request(&config, RigKind::K3s),
            Some(RigPowerRequest::ContinuousWatts(20.0))
        );
    }

    #[test]
    fn configured_power_request_uses_discrete_mchf_setting() {
        let mut config = sample_app_config();
        config.rig = Some(RigConfig {
            kind: Some("mchf".to_string()),
            port_path: None,
            input_device: None,
            output_device: None,
            power_setting: Some("2w".to_string()),
            power_w: None,
        });
        assert_eq!(
            configured_power_request(&config, RigKind::Mchf),
            Some(RigPowerRequest::SettingId("2w".to_string()))
        );
    }

    fn standard_call(callsign: &str) -> StructuredCallField {
        StructuredCallField {
            raw: 0,
            value: StructuredCallValue::StandardCall {
                callsign: callsign.to_string(),
            },
            modifier: None,
        }
    }

    fn blank_info() -> StructuredInfoField {
        StructuredInfoField {
            raw: 0,
            value: StructuredInfoValue::Blank,
        }
    }

    fn token_call(token: &str) -> StructuredCallField {
        StructuredCallField {
            raw: 0,
            value: StructuredCallValue::Token {
                token: token.to_string(),
            },
            modifier: None,
        }
    }

    fn grid_info(locator: &str) -> StructuredInfoField {
        StructuredInfoField {
            raw: 0,
            value: StructuredInfoValue::Grid {
                locator: locator.to_string(),
            },
        }
    }

    fn directed_decode(from: &str, to: &str) -> DecodedMessage {
        let message = StructuredMessage::Standard {
            i3: 0,
            first: standard_call(to),
            second: standard_call(from),
            acknowledge: false,
            info: blank_info(),
        };
        DecodedMessage {
            utc: "00:00:00".to_string(),
            snr_db: -10,
            dt_seconds: 0.1,
            freq_hz: 1000.0,
            text: message.to_text(),
            candidate_score: 0.0,
            ldpc_iterations: 0,
            message,
        }
    }

    fn cq_decode(from: &str) -> DecodedMessage {
        let message = StructuredMessage::Standard {
            i3: 0,
            first: token_call("CQ"),
            second: standard_call(from),
            acknowledge: false,
            info: grid_info("FN20"),
        };
        DecodedMessage {
            utc: "00:00:00".to_string(),
            snr_db: -10,
            dt_seconds: 0.1,
            freq_hz: 1000.0,
            text: message.to_text(),
            candidate_score: 0.0,
            ldpc_iterations: 0,
            message,
        }
    }

    fn dxpedition_direct_decode(
        sender: &str,
        completed: &str,
        next: &str,
        report_db: i16,
    ) -> DecodedMessage {
        let message = StructuredMessage::Dxpedition {
            i3: 0,
            n3: 1,
            completed_call: standard_call(completed),
            next_call: standard_call(next),
            hashed_call10: ft8_decoder::HashedCallField10 {
                raw: 0,
                resolved_callsign: Some(sender.to_string()),
            },
            report_db,
        };
        DecodedMessage {
            utc: "00:00:00".to_string(),
            snr_db: -10,
            dt_seconds: 0.1,
            freq_hz: 1000.0,
            text: message.to_text(),
            candidate_score: 0.0,
            ldpc_iterations: 0,
            message,
        }
    }

    #[derive(Default)]
    struct NoopTxBackend;

    impl TxBackend for NoopTxBackend {
        fn start(&mut self, _request: qso::TxRequest) -> Result<(), String> {
            Ok(())
        }

        fn abort(&mut self) {}

        fn poll_event(&mut self) -> Option<qso::TxEvent> {
            None
        }

        fn is_active(&self) -> bool {
            false
        }
    }

    #[test]
    fn calling_frequency_uses_mode_specific_qrgs() {
        assert_eq!(
            calling_frequency_hz(Band::M20, DecoderMode::Ft8),
            14_074_000
        );
        assert_eq!(
            calling_frequency_hz(Band::M20, DecoderMode::Ft4),
            14_080_000
        );
        assert_eq!(
            calling_frequency_hz(Band::M60, DecoderMode::Ft4),
            calling_frequency_hz(Band::M60, DecoderMode::Ft8)
        );
    }

    #[test]
    fn full_decode_sample_window_matches_decoder_mode_spec() {
        assert_eq!(
            full_decode_sample_count(DECODER_SAMPLE_RATE_HZ, DecoderMode::Ft8),
            172_800
        );
        assert_eq!(
            full_decode_sample_count(DECODER_SAMPLE_RATE_HZ, DecoderMode::Ft4),
            72_576
        );
        assert_eq!(
            capture_window_duration(DECODER_SAMPLE_RATE_HZ, DecoderMode::Ft4),
            Duration::from_secs_f64(72_576.0 / 12_000.0)
        );
    }

    #[test]
    fn tx_telemetry_polling_is_allowed_during_tx_without_active_decode() {
        let now = UNIX_EPOCH + Duration::from_secs(10);
        assert!(should_poll_rig_telemetry_during_tx(
            now, UNIX_EPOCH, None, true
        ));
        assert!(!should_poll_rig_telemetry_during_tx(
            now,
            now - Duration::from_millis(100),
            None,
            true
        ));
        assert!(!should_poll_rig_telemetry_during_tx(
            now,
            UNIX_EPOCH,
            Some(ActiveDecodeJob {
                slot_start: now,
                stage: DecodeStage::Full,
            }),
            true
        ));
        assert!(!should_poll_rig_telemetry_during_tx(
            now, UNIX_EPOCH, None, false
        ));
    }

    #[test]
    fn ft8_live_scheduler_uses_medium_early47_residual_prep_before_full() {
        let slot_start = UNIX_EPOCH + Duration::from_secs(30);
        let spec = DecoderMode::Ft8.spec();
        let early41_ready = stage_capture_end(DecoderMode::Ft8, slot_start, DecodeStage::Early41)
            .expect("early41 ready");
        let early47_ready = stage_capture_end(DecoderMode::Ft8, slot_start, DecodeStage::Early47)
            .expect("early47 ready");
        let full_ready =
            stage_capture_end(DecoderMode::Ft8, slot_start, DecodeStage::Full).expect("full ready");
        assert_eq!(
            full_ready
                .duration_since(slot_start)
                .expect("full after slot")
                .as_millis(),
            14_400
        );
        assert_eq!(spec.full_decode_samples(), 172_800);

        let mut stages = SlotStageState::default();
        assert_eq!(
            stages.next_due_stage(
                DecoderMode::Ft8,
                DecodeProfile::Medium,
                slot_start,
                Some(early41_ready)
            ),
            Some(DecodeStage::Early41)
        );
        stages.mark_handled(DecodeStage::Early41);
        assert_eq!(
            stages.next_due_stage(
                DecoderMode::Ft8,
                DecodeProfile::Medium,
                slot_start,
                Some(early47_ready)
            ),
            Some(DecodeStage::Early47)
        );
        stages.mark_handled(DecodeStage::Early47);
        assert_eq!(
            stages.next_due_stage(
                DecoderMode::Ft8,
                DecodeProfile::Medium,
                slot_start,
                Some(full_ready)
            ),
            Some(DecodeStage::Full)
        );
    }

    #[test]
    fn ft8_deepest_live_scheduler_uses_early47_residual_prep_before_full() {
        let slot_start = UNIX_EPOCH + Duration::from_secs(30);
        let early41_ready = stage_capture_end(DecoderMode::Ft8, slot_start, DecodeStage::Early41)
            .expect("early41 ready");
        let early47_ready = stage_capture_end(DecoderMode::Ft8, slot_start, DecodeStage::Early47)
            .expect("early47 ready");
        let full_ready =
            stage_capture_end(DecoderMode::Ft8, slot_start, DecodeStage::Full).expect("full ready");

        let mut stages = SlotStageState::default();
        assert_eq!(
            stages.next_due_stage(
                DecoderMode::Ft8,
                DecodeProfile::Deepest,
                slot_start,
                Some(early41_ready)
            ),
            Some(DecodeStage::Early41)
        );
        stages.mark_handled(DecodeStage::Early41);
        assert_eq!(
            stages.next_due_stage(
                DecoderMode::Ft8,
                DecodeProfile::Deepest,
                slot_start,
                Some(early47_ready)
            ),
            Some(DecodeStage::Early47)
        );
        stages.mark_handled(DecodeStage::Early47);
        assert_eq!(
            stages.next_due_stage(
                DecoderMode::Ft8,
                DecodeProfile::Deepest,
                slot_start,
                Some(full_ready)
            ),
            Some(DecodeStage::Full)
        );
    }

    #[test]
    fn ft8_live_scheduler_skips_early_stages_for_quick_profile() {
        let slot_start = UNIX_EPOCH + Duration::from_secs(30);
        let early41_ready = stage_capture_end(DecoderMode::Ft8, slot_start, DecodeStage::Early41)
            .expect("early41 ready");
        let early47_ready = stage_capture_end(DecoderMode::Ft8, slot_start, DecodeStage::Early47)
            .expect("early47 ready");
        let full_ready =
            stage_capture_end(DecoderMode::Ft8, slot_start, DecodeStage::Full).expect("full ready");

        assert_eq!(
            SlotStageState::default().next_due_stage(
                DecoderMode::Ft8,
                DecodeProfile::Quick,
                slot_start,
                Some(early41_ready)
            ),
            None
        );
        assert_eq!(
            SlotStageState::default().next_due_stage(
                DecoderMode::Ft8,
                DecodeProfile::Quick,
                slot_start,
                Some(early47_ready)
            ),
            None
        );
        assert_eq!(
            SlotStageState::default().next_due_stage(
                DecoderMode::Ft8,
                DecodeProfile::Quick,
                slot_start,
                Some(full_ready)
            ),
            Some(DecodeStage::Full)
        );
    }

    #[test]
    fn full_decode_margin_is_against_next_slot_pre_key_deadline() {
        let slot_start = UNIX_EPOCH + Duration::from_secs(30);
        assert_eq!(
            tx_margin_after_stage_decode_ms(DecoderMode::Ft8, slot_start, DecodeStage::Full, 500)
                .expect("margin"),
            450
        );
    }

    #[test]
    fn ft4_full_stage_decode_silence_does_not_panic() {
        let slot_start = UNIX_EPOCH + Duration::from_secs(30);
        let samples =
            vec![0i16; full_decode_sample_count(DECODER_SAMPLE_RATE_HZ, DecoderMode::Ft4)];
        let result = std::panic::catch_unwind(|| {
            let mut session = DecoderSession::new();
            let mut state = DecoderState::new();
            decode_stage_from_samples(
                &mut session,
                &mut state,
                &samples,
                DECODER_SAMPLE_RATE_HZ,
                DecoderMode::Ft4,
                DecodeStage::Full,
                slot_start,
                DecodeProfile::Medium,
                None,
            )
        });
        assert!(
            result.is_ok(),
            "ft4 full-stage decode panicked inside decode worker path"
        );
        let report = result
            .expect("panic-free decode")
            .expect("decode stage result");
        assert_eq!(report.stage, DecodeStage::Full);
        assert!(report.report.decodes.is_empty());
    }

    #[test]
    fn ft4_full_stage_decode_resampled_silence_does_not_panic() {
        let slot_start = UNIX_EPOCH + Duration::from_secs(30);
        let source_rate = 48_000;
        let samples = vec![0i16; full_decode_sample_count(source_rate, DecoderMode::Ft4)];
        let result = std::panic::catch_unwind(|| {
            let mut session = DecoderSession::new();
            let mut state = DecoderState::new();
            decode_stage_from_samples(
                &mut session,
                &mut state,
                &samples,
                source_rate,
                DecoderMode::Ft4,
                DecodeStage::Full,
                slot_start,
                DecodeProfile::Medium,
                None,
            )
        });
        assert!(
            result.is_ok(),
            "ft4 full-stage resampled decode panicked inside decode worker path"
        );
        let report = result
            .expect("panic-free decode")
            .expect("decode stage result");
        assert_eq!(report.stage, DecodeStage::Full);
        assert!(report.report.decodes.is_empty());
    }

    #[test]
    fn ft4_only_schedules_full_decode_stage() {
        let slot_start = UNIX_EPOCH + Duration::from_secs(30);
        let before_full_ready = slot_start + Duration::from_secs(1);
        let ft4_full_ready = stage_capture_end(DecoderMode::Ft4, slot_start, DecodeStage::Full)
            .expect("ft4 full ready");

        assert_eq!(
            SlotStageState::default().next_due_stage(
                DecoderMode::Ft4,
                DecodeProfile::Medium,
                slot_start,
                Some(before_full_ready)
            ),
            None
        );
        assert_eq!(
            SlotStageState::default().next_due_stage(
                DecoderMode::Ft4,
                DecodeProfile::Medium,
                slot_start,
                Some(ft4_full_ready)
            ),
            Some(DecodeStage::Full)
        );
    }

    #[test]
    fn station_tracker_reset_for_mode_clears_previous_state() {
        let now = UNIX_EPOCH + Duration::from_secs(30);
        let mut tracker = StationTracker::default();
        ingest_decode(&mut tracker, now, &cq_decode("K1ABC"));
        assert!(tracker.start_info("K1ABC").is_some());

        tracker.reset_for_mode(DecoderMode::Ft4);

        assert_eq!(tracker.mode, DecoderMode::Ft4);
        assert!(tracker.start_info("K1ABC").is_none());
        assert!(tracker.web_logs().is_empty());
    }

    #[test]
    fn queue_scheduler_can_dispatch_in_ft4_mode() {
        let now = UNIX_EPOCH + Duration::from_secs(30);
        let config = sample_app_config();
        let mut tracker = StationTracker::default();
        ingest_decode(&mut tracker, now, &cq_decode("K1ABC"));
        let mut queue = WorkQueueState::new(&config, 900.0, BTreeMap::new());
        queue.auto_enabled = true;
        queue.set_current_band(Some("20m".to_string()));
        queue.set_current_mode(DecoderMode::Ft4);
        queue.add_station("K1ABC", now, now).expect("queued");

        let dispatch = queue
            .scheduler_pick(now + Duration::from_secs(8), &tracker, false, false)
            .expect("dispatch");
        assert_eq!(dispatch.callsign, "K1ABC");
        assert_eq!(queue.scheduler_status, "dispatching K1ABC");
    }

    #[test]
    fn ft4_slot_boundaries_preserve_half_second_alignment() {
        let before_half = UNIX_EPOCH + Duration::from_millis(7_499);
        let at_half = UNIX_EPOCH + Duration::from_millis(7_500);
        let later = UNIX_EPOCH + Duration::from_millis(22_900);

        assert_eq!(
            current_slot_boundary_for_mode(DecoderMode::Ft4, before_half),
            UNIX_EPOCH
        );
        assert_eq!(
            current_slot_boundary_for_mode(DecoderMode::Ft4, at_half),
            UNIX_EPOCH + Duration::from_millis(7_500)
        );
        assert_eq!(
            next_slot_boundary_for_mode(DecoderMode::Ft4, at_half),
            UNIX_EPOCH + Duration::from_millis(15_000)
        );
        assert_eq!(
            current_slot_boundary_for_mode(DecoderMode::Ft4, later),
            UNIX_EPOCH + Duration::from_millis(22_500)
        );
    }

    #[test]
    fn mode_aware_slot_time_formatting_is_stable() {
        assert_eq!(
            format_slot_time_for_mode(DecoderMode::Ft4, UNIX_EPOCH + Duration::from_millis(7_500)),
            "000007.5"
        );
        assert_eq!(
            format_slot_time_for_mode(DecoderMode::Ft4, UNIX_EPOCH + Duration::from_secs(15)),
            "000015.0"
        );
        assert_eq!(
            format_slot_time_for_mode(DecoderMode::Ft8, UNIX_EPOCH + Duration::from_secs(15)),
            "000015"
        );
        assert_eq!(
            format_slot_time(UNIX_EPOCH + Duration::from_secs(15)),
            "000015"
        );
    }

    #[test]
    fn full_decode_is_labeled_full_for_all_modes() {
        assert_eq!(
            stage_display_label(DecoderMode::Ft8, DecodeStage::Full),
            "full"
        );
        assert_eq!(
            stage_display_label(DecoderMode::Ft4, DecodeStage::Full),
            "full"
        );
    }

    #[test]
    fn parse_supported_app_mode_rejects_ft2() {
        assert_eq!(parse_supported_app_mode("ft8"), Ok(DecoderMode::Ft8));
        assert_eq!(parse_supported_app_mode("ft4"), Ok(DecoderMode::Ft4));
        assert_eq!(
            parse_supported_app_mode("ft2"),
            Err("FT2 is not supported in ft8rx; use FT8 or FT4".to_string())
        );
    }

    #[test]
    fn related_calls_ignore_previous_qso_peer() {
        let mut tracker = StationTracker::default();
        let now = UNIX_EPOCH + Duration::from_secs(1);

        ingest_decode(&mut tracker, now, &directed_decode("B", "A"));
        ingest_decode(
            &mut tracker,
            now + Duration::from_secs(15),
            &directed_decode("B", "C"),
        );

        let logs = tracker.web_logs();
        assert_eq!(logs.len(), 2);
        assert_eq!(
            logs[0].related_calls,
            vec!["A".to_string(), "B".to_string()]
        );
        assert_eq!(
            logs[1].related_calls,
            vec!["B".to_string(), "C".to_string()]
        );
        assert_eq!(logs[1].peer.as_deref(), Some("C"));
        assert_eq!(logs[1].peer.as_deref(), Some("C"));
        assert_eq!(logs[1].peer_before.as_deref(), Some("A"));
        assert_eq!(logs[1].peer_after.as_deref(), Some("C"));
        assert!(!logs[1].related_calls.iter().any(|call| call == "A"));
    }

    #[test]
    fn cq_clears_display_peer_after_qso() {
        let mut tracker = StationTracker::default();
        let now = UNIX_EPOCH + Duration::from_secs(1);

        ingest_decode(&mut tracker, now, &directed_decode("B", "A"));
        ingest_decode(&mut tracker, now + Duration::from_secs(15), &cq_decode("B"));

        let logs = tracker.web_logs();
        assert_eq!(logs.len(), 2);
        assert_eq!(logs[1].peer_before.as_deref(), Some("A"));
        assert_eq!(logs[1].peer, None);
        assert_eq!(logs[1].peer_after, None);
    }

    #[test]
    fn queue_pick_uses_oldest_ready_entry() {
        let now = UNIX_EPOCH + Duration::from_secs(30);
        let mut tracker = StationTracker::default();
        ingest_decode(&mut tracker, now, &cq_decode("OLDER"));
        ingest_decode(&mut tracker, now, &cq_decode("NEWER"));

        let config = sample_app_config();
        let mut queue = WorkQueueState::new(&config, 900.0, BTreeMap::new());
        queue.auto_enabled = true;
        queue.add_station("OLDER", now, now).expect("older queued");
        queue
            .add_station(
                "NEWER",
                now + Duration::from_secs(1),
                now + Duration::from_secs(1),
            )
            .expect("newer queued");

        let dispatch = queue
            .scheduler_pick(now + Duration::from_secs(1), &tracker, false, false)
            .expect("dispatch");
        assert_eq!(dispatch.callsign, "OLDER");
    }

    #[test]
    fn direct_observation_same_slot_does_not_increment_count() {
        let config = sample_app_config();
        let mut queue = WorkQueueState::new(&config, 900.0, BTreeMap::new());
        let now = UNIX_EPOCH + Duration::from_secs(30);
        let observation = DirectCallObservation {
            callsign: "K1ABC".to_string(),
            observed_at: now,
            slot_index: ft8_slot_index(now),
            slot_family: ft8_slot_family(now),
            snr_db: -5,
            start_state: QsoState::SendSig,
            compound_eligible: true,
            text: "N1VF K1ABC FN20".to_string(),
            structured_json: "{}".to_string(),
        };
        queue
            .add_direct_observation(observation.clone(), now)
            .expect("first add");
        queue
            .add_direct_observation(observation, now)
            .expect("duplicate stage add");
        let entry = queue.entries.front().expect("queued entry");
        assert_eq!(entry.direct_count, 1);
    }

    #[test]
    fn dxpedition_direct_to_us_starts_as_send_sig_ack() {
        let now = UNIX_EPOCH + Duration::from_secs(30);
        let decode = dxpedition_direct_decode("PY7ZZ", "SP4MCH", "N1VF", -18);
        let observation =
            direct_call_observation_from_decode(&decode, "N1VF", now, DecoderMode::Ft8)
                .expect("direct observation");
        assert_eq!(observation.callsign, "PY7ZZ");
        assert_eq!(observation.start_state, QsoState::SendSigAck);
        assert!(!observation.compound_eligible);
    }

    #[test]
    fn acknowledged_signal_direct_to_us_starts_as_send_rr73() {
        let now = UNIX_EPOCH + Duration::from_secs(30);
        let decode = DecodedMessage {
            utc: "00:00:00".to_string(),
            snr_db: 2,
            dt_seconds: 0.1,
            freq_hz: 1000.0,
            text: "N1VF JA1IST R-21".to_string(),
            candidate_score: 0.0,
            ldpc_iterations: 0,
            message: StructuredMessage::Standard {
                i3: 0,
                first: standard_call("N1VF"),
                second: standard_call("JA1IST"),
                acknowledge: true,
                info: StructuredInfoField {
                    raw: 0,
                    value: StructuredInfoValue::SignalReport { db: -21 },
                },
            },
        };
        let observation =
            direct_call_observation_from_decode(&decode, "N1VF", now, DecoderMode::Ft8)
                .expect("direct observation");
        assert_eq!(observation.callsign, "JA1IST");
        assert_eq!(observation.start_state, QsoState::SendRR73);
        assert!(!observation.compound_eligible);
    }

    #[test]
    fn compound_handoff_candidate_only_uses_grid_style_directs() {
        let config = sample_app_config();
        let mut queue = WorkQueueState::new(&config, 900.0, BTreeMap::new());
        let now = UNIX_EPOCH + Duration::from_secs(30);
        queue
            .add_direct_observation(
                DirectCallObservation {
                    callsign: "GRID".to_string(),
                    observed_at: now,
                    slot_index: ft8_slot_index(now),
                    slot_family: ft8_slot_family(now),
                    snr_db: -8,
                    start_state: QsoState::SendSig,
                    compound_eligible: true,
                    text: "N1VF GRID FN20".to_string(),
                    structured_json: "{}".to_string(),
                },
                now,
            )
            .expect("grid direct");
        queue
            .add_direct_observation(
                DirectCallObservation {
                    callsign: "GRID".to_string(),
                    observed_at: now + Duration::from_secs(15),
                    slot_index: ft8_slot_index(now + Duration::from_secs(15)),
                    slot_family: ft8_slot_family(now + Duration::from_secs(15)),
                    snr_db: -8,
                    start_state: QsoState::SendSig,
                    compound_eligible: true,
                    text: "N1VF GRID FN20".to_string(),
                    structured_json: "{}".to_string(),
                },
                now + Duration::from_secs(15),
            )
            .expect("grid repeat");
        queue
            .add_direct_observation(
                DirectCallObservation {
                    callsign: "SIG".to_string(),
                    observed_at: now + Duration::from_secs(15),
                    slot_index: ft8_slot_index(now + Duration::from_secs(15)),
                    slot_family: ft8_slot_family(now + Duration::from_secs(15)),
                    snr_db: -2,
                    start_state: QsoState::SendSigAck,
                    compound_eligible: false,
                    text: "N1VF SIG -02".to_string(),
                    structured_json: "{}".to_string(),
                },
                now + Duration::from_secs(15),
            )
            .expect("signal direct");
        let candidate = queue
            .peek_compound_handoff_candidate(now + Duration::from_secs(16), None)
            .expect("compound candidate");
        assert_eq!(candidate.callsign, "GRID");

        let mut queue = WorkQueueState::new(&config, 900.0, BTreeMap::new());
        queue
            .add_direct_observation(
                DirectCallObservation {
                    callsign: "SIG".to_string(),
                    observed_at: now,
                    slot_index: ft8_slot_index(now),
                    slot_family: ft8_slot_family(now),
                    snr_db: -2,
                    start_state: QsoState::SendSigAck,
                    compound_eligible: false,
                    text: "N1VF SIG -02".to_string(),
                    structured_json: "{}".to_string(),
                },
                now,
            )
            .expect("signal direct");
        assert!(
            queue
                .peek_compound_handoff_candidate(now + Duration::from_secs(1), None)
                .is_none()
        );
    }

    #[test]
    fn compound_handoff_candidate_can_include_signal_direct_when_enabled() {
        let config = sample_app_config();
        let mut queue = WorkQueueState::new(&config, 900.0, BTreeMap::new());
        queue.set_use_compound_for_direct_signal_callers(true);
        let now = UNIX_EPOCH + Duration::from_secs(30);
        queue
            .add_direct_observation(
                DirectCallObservation {
                    callsign: "SIG".to_string(),
                    observed_at: now,
                    slot_index: ft8_slot_index(now),
                    slot_family: ft8_slot_family(now),
                    snr_db: -2,
                    start_state: QsoState::SendSigAck,
                    compound_eligible: false,
                    text: "N1VF SIG -02".to_string(),
                    structured_json: "{}".to_string(),
                },
                now,
            )
            .expect("signal direct");
        let candidate = queue
            .peek_compound_handoff_candidate(now + Duration::from_secs(1), None)
            .expect("compound candidate");
        assert_eq!(candidate.callsign, "SIG");
        match candidate.kind {
            QueueDispatchKind::Station { initial_state, .. } => {
                assert_eq!(initial_state, QsoState::SendSigAck);
            }
            QueueDispatchKind::Cq { .. } => panic!("expected station dispatch"),
        }
    }

    #[test]
    fn reserved_compound_next_call_is_not_readded_to_queue() {
        let config = sample_app_config();
        let mut queue = WorkQueueState::new(&config, 900.0, BTreeMap::new());
        let mut controller = qso::QsoController::new(config.clone(), Box::new(NoopTxBackend));
        let now = UNIX_EPOCH + Duration::from_secs(30);
        let rx_slot_start = now + Duration::from_secs(15);

        controller.handle_command(
            qso::QsoCommand::Start {
                partner_call: "OLD1".to_string(),
                tx_freq_hz: 900.0,
                initial_state: QsoState::SendSig,
                start_mode: qso::QsoStartMode::Direct,
                tx_slot_family_override: Some(SlotFamily::Even),
            },
            Some(qso::StationStartInfo {
                callsign: "OLD1".to_string(),
                last_heard_at: now,
                last_heard_slot_family: SlotFamily::Odd,
                last_snr_db: -5,
                last_text: None,
                last_structured_json: None,
            }),
            now,
        );
        controller.on_full_decode(
            rx_slot_start,
            &[DecodedMessage {
                utc: "00:00:00".to_string(),
                snr_db: -10,
                dt_seconds: 0.1,
                freq_hz: 1000.0,
                text: StructuredMessage::Standard {
                    i3: 0,
                    first: standard_call("N1VF"),
                    second: standard_call("OLD1"),
                    acknowledge: true,
                    info: StructuredInfoField {
                        raw: 0,
                        value: StructuredInfoValue::Blank,
                    },
                }
                .to_text(),
                candidate_score: 0.0,
                ldpc_iterations: 0,
                message: StructuredMessage::Standard {
                    i3: 0,
                    first: standard_call("N1VF"),
                    second: standard_call("OLD1"),
                    acknowledge: true,
                    info: StructuredInfoField {
                        raw: 0,
                        value: StructuredInfoValue::Blank,
                    },
                },
            }],
            rx_slot_start + Duration::from_secs(15),
        );
        assert!(controller.maybe_arm_compound_handoff(
            rx_slot_start,
            qso::CompoundHandoffPlan {
                next_station: qso::StationStartInfo {
                    callsign: "NEW1".to_string(),
                    last_heard_at: rx_slot_start,
                    last_heard_slot_family: SlotFamily::Odd,
                    last_snr_db: -7,
                    last_text: Some("N1VF NEW1 FN20".to_string()),
                    last_structured_json: Some("{}".to_string()),
                },
            },
            true,
            rx_slot_start + Duration::from_secs(15),
        ));
        let decode = DecodedMessage {
            utc: "00:00:00".to_string(),
            snr_db: -7,
            dt_seconds: 0.1,
            freq_hz: 1000.0,
            text: "N1VF NEW1 FN20".to_string(),
            candidate_score: 0.0,
            ldpc_iterations: 0,
            message: StructuredMessage::Standard {
                i3: 0,
                first: standard_call("N1VF"),
                second: standard_call("NEW1"),
                acknowledge: false,
                info: grid_info("FN20"),
            },
        };
        maybe_track_priority_directs(
            &mut queue,
            &mut controller,
            &[decode],
            "N1VF",
            rx_slot_start,
            &BTreeSet::from(["NEW1".to_string()]),
        );
        assert!(queue.entries.is_empty());
    }

    #[test]
    fn later_decode_stage_does_not_requeue_same_slot_active_partner_direct() {
        let config = sample_app_config();
        let mut queue = WorkQueueState::new(&config, 900.0, BTreeMap::new());
        queue.set_auto_add_direct_calls(true);
        let mut controller = qso::QsoController::new(config, Box::new(NoopTxBackend));

        let start_at = UNIX_EPOCH + Duration::from_secs(13);
        controller.handle_command(
            qso::QsoCommand::Start {
                partner_call: "JA1IST".to_string(),
                tx_freq_hz: 800.0,
                initial_state: QsoState::Send73,
                start_mode: qso::QsoStartMode::Direct,
                tx_slot_family_override: Some(SlotFamily::Odd),
            },
            Some(qso::StationStartInfo {
                callsign: "JA1IST".to_string(),
                last_heard_at: start_at,
                last_heard_slot_family: SlotFamily::Even,
                last_snr_db: 2,
                last_text: Some("N1VF JA1IST R-21".to_string()),
                last_structured_json: Some("{}".to_string()),
            }),
            start_at,
        );

        let slot_1 = UNIX_EPOCH + Duration::from_secs(15);
        let slot_2 = UNIX_EPOCH + Duration::from_secs(45);
        let slot_3 = UNIX_EPOCH + Duration::from_secs(75);
        let ack_21 = DecodedMessage {
            utc: "00:00:00".to_string(),
            snr_db: 2,
            dt_seconds: 0.1,
            freq_hz: 1000.0,
            text: "N1VF JA1IST R-21".to_string(),
            candidate_score: 0.0,
            ldpc_iterations: 0,
            message: StructuredMessage::Standard {
                i3: 0,
                first: standard_call("N1VF"),
                second: standard_call("JA1IST"),
                acknowledge: true,
                info: StructuredInfoField {
                    raw: 0,
                    value: StructuredInfoValue::SignalReport { db: -21 },
                },
            },
        };
        let ack_08 = DecodedMessage {
            utc: "00:00:30".to_string(),
            snr_db: 1,
            dt_seconds: 0.1,
            freq_hz: 1000.0,
            text: "N1VF JA1IST R-08".to_string(),
            candidate_score: 0.0,
            ldpc_iterations: 0,
            message: StructuredMessage::Standard {
                i3: 0,
                first: standard_call("N1VF"),
                second: standard_call("JA1IST"),
                acknowledge: true,
                info: StructuredInfoField {
                    raw: 0,
                    value: StructuredInfoValue::SignalReport { db: -8 },
                },
            },
        };

        controller.on_decode_stage(
            slot_1,
            DecodeStage::Early41,
            std::slice::from_ref(&ack_21),
            slot_1,
        );
        controller.on_decode_stage(
            slot_2,
            DecodeStage::Early41,
            std::slice::from_ref(&ack_08),
            slot_2,
        );
        assert_eq!(controller.active_partner_call().as_deref(), Some("JA1IST"));

        let slot_skip_calls = BTreeSet::from(["JA1IST".to_string()]);
        maybe_track_priority_directs(
            &mut queue,
            &mut controller,
            std::slice::from_ref(&ack_08),
            "N1VF",
            slot_3,
            &slot_skip_calls,
        );
        assert!(queue.entries.is_empty());

        controller.on_decode_stage(
            slot_3,
            DecodeStage::Early41,
            std::slice::from_ref(&ack_08),
            slot_3,
        );
        controller.handle_command(
            qso::QsoCommand::Stop {
                reason: "test_stop".to_string(),
            },
            None,
            slot_3,
        );
        assert!(controller.active_partner_call().is_none());

        maybe_track_priority_directs(
            &mut queue,
            &mut controller,
            std::slice::from_ref(&ack_08),
            "N1VF",
            slot_3,
            &slot_skip_calls,
        );
        assert!(queue.entries.is_empty());
    }

    #[test]
    fn queue_requeue_after_early_no_msg_adds_retry_delay() {
        let now = UNIX_EPOCH + Duration::from_secs(30);
        let mut tracker = StationTracker::default();
        ingest_decode(&mut tracker, now, &cq_decode("K1ABC"));
        let config = sample_app_config();
        let mut queue = WorkQueueState::new(&config, 900.0, BTreeMap::new());
        queue.set_current_band(Some("40m".to_string()));
        queue.handle_qso_outcome(
            &QsoOutcome {
                partner_call: "K1ABC".to_string(),
                exit_reason: "send_grid_no_msg_limit".to_string(),
                finished_at: now,
                rig_band: Some("40m".to_string()),
                sent_terminal_73: false,
            },
            &tracker,
        );
        let entry = queue.entries.front().expect("requeued entry");
        assert_eq!(entry.callsign, "K1ABC");
        assert_eq!(entry.queued_at, now);
        assert_eq!(
            entry.ok_to_schedule_after,
            now + Duration::from_secs(config.queue.no_message_retry_delay_seconds_default)
        );
    }

    #[test]
    fn queue_requeue_after_no_fwd_uses_no_fwd_retry_delay() {
        let now = UNIX_EPOCH + Duration::from_secs(30);
        let mut tracker = StationTracker::default();
        ingest_decode(&mut tracker, now, &cq_decode("K1ABC"));
        let config = sample_app_config();
        let mut queue = WorkQueueState::new(&config, 900.0, BTreeMap::new());
        queue.set_current_band(Some("40m".to_string()));
        queue.handle_qso_outcome(
            &QsoOutcome {
                partner_call: "K1ABC".to_string(),
                exit_reason: "send_sig_no_fwd_limit".to_string(),
                finished_at: now,
                rig_band: Some("40m".to_string()),
                sent_terminal_73: false,
            },
            &tracker,
        );
        let entry = queue.entries.front().expect("requeued entry");
        assert_eq!(entry.callsign, "K1ABC");
        assert_eq!(entry.queued_at, now);
        assert_eq!(
            entry.ok_to_schedule_after,
            now + Duration::from_secs(config.queue.no_forward_retry_delay_seconds_default)
        );
    }

    #[test]
    fn queue_drops_recently_worked_station_immediately() {
        let now = UNIX_EPOCH + Duration::from_secs(30);
        let config = sample_app_config();
        let mut queue = WorkQueueState::new(&config, 900.0, BTreeMap::new());
        queue.set_current_band(Some("40m".to_string()));
        queue.mark_worked("K1ABC", "40m", now);
        let added = queue.add_station("K1ABC", now, now);
        assert!(added.is_err());
        assert!(queue.entries.is_empty());
    }

    #[test]
    fn queue_allows_recently_worked_station_on_different_band() {
        let now = UNIX_EPOCH + Duration::from_secs(30);
        let config = sample_app_config();
        let mut queue = WorkQueueState::new(&config, 900.0, BTreeMap::new());
        queue.mark_worked("K1ABC", "20m", now);
        queue.set_current_band(Some("40m".to_string()));
        let added = queue.add_station("K1ABC", now, now);
        assert!(added.is_ok());
        assert_eq!(queue.entries.len(), 1);
    }

    #[test]
    fn queue_rejects_our_own_call() {
        let now = UNIX_EPOCH + Duration::from_secs(30);
        let config = sample_app_config();
        let mut queue = WorkQueueState::new(&config, 900.0, BTreeMap::new());
        queue.set_current_band(Some("40m".to_string()));
        assert!(queue.add_station("N1VF", now, now).is_err());
        assert!(queue.entries.is_empty());
        let observation = DirectCallObservation {
            callsign: "N1VF".to_string(),
            observed_at: now,
            slot_index: ft8_slot_index(now),
            slot_family: ft8_slot_family(now),
            snr_db: -10,
            start_state: QsoState::SendSig,
            compound_eligible: false,
            text: "N1VF N1VF FN20".to_string(),
            structured_json: "{}".to_string(),
        };
        assert!(queue.add_direct_observation(observation, now).is_err());
        assert!(queue.entries.is_empty());
    }

    #[test]
    fn next_cq_parity_flip_is_consumed_once() {
        let now = UNIX_EPOCH + Duration::from_secs(31);
        let config = sample_app_config();
        let mut queue = WorkQueueState::new(&config, 900.0, BTreeMap::new());
        let tracker = StationTracker::default();
        queue.auto_enabled = true;
        queue.cq_enabled = true;
        queue.cq_percent = 100;
        queue.next_cq_parity_flipped = true;

        let dispatch = queue
            .scheduler_pick(now, &tracker, false, false)
            .expect("cq dispatch");
        match dispatch.kind {
            QueueDispatchKind::Cq {
                tx_slot_family_override,
            } => {
                assert_eq!(
                    tx_slot_family_override,
                    Some(ft8_slot_family(ft8_next_slot_boundary(now)).opposite())
                );
            }
            _ => panic!("expected cq dispatch"),
        }
        assert!(!queue.next_cq_parity_flipped);

        let second = queue
            .scheduler_pick(now + Duration::from_secs(15), &tracker, false, false)
            .expect("second cq dispatch");
        match second.kind {
            QueueDispatchKind::Cq {
                tx_slot_family_override,
            } => assert_eq!(
                tx_slot_family_override,
                Some(ft8_slot_family(ft8_next_slot_boundary(
                    now + Duration::from_secs(15)
                )))
            ),
            _ => panic!("expected cq dispatch"),
        }
    }

    #[test]
    fn auto_add_decoded_calls_queues_unique_non_active_callers() {
        let now = UNIX_EPOCH + Duration::from_secs(30);
        let config = sample_app_config();
        let mut queue = WorkQueueState::new(&config, 900.0, BTreeMap::new());
        queue.set_current_band(Some("40m".to_string()));
        queue.set_auto_add_all_decoded_calls(true);
        queue.set_auto_add_decoded_min_count_5m(1);
        let mut tracker = StationTracker::default();
        let controller = qso::QsoController::new(config, Box::new(NoopTxBackend));

        let alpha = cq_decode("ALPHA");
        let bravo = cq_decode("BRAVO");
        tracker.ingest_frame(now, &[alpha.clone(), bravo.clone(), alpha]);

        maybe_auto_add_decoded_calls(
            &mut queue,
            &tracker,
            &controller,
            &[cq_decode("ALPHA"), cq_decode("BRAVO"), cq_decode("ALPHA")],
            now,
        );

        let calls = queue
            .entries
            .iter()
            .map(|entry| entry.callsign.clone())
            .collect::<Vec<_>>();
        assert_eq!(calls, vec!["ALPHA".to_string(), "BRAVO".to_string()]);
    }

    #[test]
    fn auto_add_decoded_calls_requires_repeat_count_within_5m() {
        let config = sample_app_config();
        let mut queue = WorkQueueState::new(&config, 900.0, BTreeMap::new());
        queue.set_current_band(Some("40m".to_string()));
        queue.set_auto_add_all_decoded_calls(true);
        let mut tracker = StationTracker::default();
        let controller = qso::QsoController::new(config, Box::new(NoopTxBackend));

        let first = UNIX_EPOCH + Duration::from_secs(30);
        let second = first + Duration::from_secs(30);

        ingest_decode(&mut tracker, first, &cq_decode("ALPHA"));
        maybe_auto_add_decoded_calls(
            &mut queue,
            &tracker,
            &controller,
            &[cq_decode("ALPHA")],
            first,
        );
        assert!(queue.entries.is_empty());

        ingest_decode(&mut tracker, second, &cq_decode("ALPHA"));
        maybe_auto_add_decoded_calls(
            &mut queue,
            &tracker,
            &controller,
            &[cq_decode("ALPHA")],
            second,
        );

        let calls = queue
            .entries
            .iter()
            .map(|entry| entry.callsign.clone())
            .collect::<Vec<_>>();
        assert_eq!(calls, vec!["ALPHA".to_string()]);
    }

    #[test]
    fn auto_add_decoded_calls_ignores_old_repeat_outside_5m_window() {
        let config = sample_app_config();
        let mut queue = WorkQueueState::new(&config, 900.0, BTreeMap::new());
        queue.set_current_band(Some("40m".to_string()));
        queue.set_auto_add_all_decoded_calls(true);
        let mut tracker = StationTracker::default();
        let controller = qso::QsoController::new(config, Box::new(NoopTxBackend));

        let first = UNIX_EPOCH + Duration::from_secs(30);
        let second = first + Duration::from_secs(301);

        ingest_decode(&mut tracker, first, &cq_decode("ALPHA"));
        ingest_decode(&mut tracker, second, &cq_decode("ALPHA"));
        maybe_auto_add_decoded_calls(
            &mut queue,
            &tracker,
            &controller,
            &[cq_decode("ALPHA")],
            second,
        );

        assert!(queue.entries.is_empty());
    }

    #[test]
    fn cq_pause_low_activity_blocks_cq_dispatch_until_threshold_met() {
        let now = UNIX_EPOCH + Duration::from_secs(60);
        let config = sample_app_config();
        let mut queue = WorkQueueState::new(&config, 900.0, BTreeMap::new());
        let mut tracker = StationTracker::default();
        queue.auto_enabled = true;
        queue.cq_enabled = true;
        queue.cq_percent = 100;
        queue.set_pause_cq_when_few_unique_calls(true);
        queue.set_cq_pause_min_unique_calls_5m(3);

        ingest_decode(
            &mut tracker,
            now - Duration::from_secs(30),
            &cq_decode("A1"),
        );
        ingest_decode(
            &mut tracker,
            now - Duration::from_secs(15),
            &cq_decode("A2"),
        );

        assert!(queue.scheduler_pick(now, &tracker, false, false).is_none());
        assert!(queue.scheduler_status.contains("cq paused"));

        ingest_decode(&mut tracker, now, &cq_decode("A3"));
        let dispatch = queue
            .scheduler_pick(now + Duration::from_secs(1), &tracker, false, false)
            .expect("cq dispatch once threshold is met");
        match dispatch.kind {
            QueueDispatchKind::Cq { .. } => {}
            _ => panic!("expected cq dispatch"),
        }
    }

    #[test]
    fn unique_sender_count_excludes_our_own_call() {
        let now = UNIX_EPOCH + Duration::from_secs(60);
        let mut tracker = StationTracker::default();
        ingest_decode(
            &mut tracker,
            now - Duration::from_secs(30),
            &cq_decode("A1"),
        );
        ingest_decode(
            &mut tracker,
            now - Duration::from_secs(15),
            &cq_decode("N1VF"),
        );
        ingest_decode(&mut tracker, now, &cq_decode("A2"));

        assert_eq!(
            tracker.unique_sender_count_since_excluding(
                now.checked_sub(CQ_ACTIVITY_WINDOW).unwrap_or(now),
                "N1VF",
            ),
            2
        );
    }

    #[test]
    fn station_tracker_uses_early_stage_for_start_info() {
        let slot_start = UNIX_EPOCH + Duration::from_secs(45);
        let mut tracker = StationTracker::default();
        tracker.ingest_stage(
            slot_start,
            DecodeStage::Early47,
            &[DecodedMessage {
                utc: "00:00:00".to_string(),
                snr_db: -4,
                dt_seconds: 0.1,
                freq_hz: 1000.0,
                text: "N1VF K7VAY DM42".to_string(),
                candidate_score: 0.0,
                ldpc_iterations: 0,
                message: StructuredMessage::Standard {
                    i3: 0,
                    first: standard_call("N1VF"),
                    second: standard_call("K7VAY"),
                    acknowledge: false,
                    info: grid_info("DM42"),
                },
            }],
        );

        let info = tracker.start_info("K7VAY").expect("start info");
        assert_eq!(info.last_heard_at, slot_start);
        assert_eq!(info.last_heard_slot_family, SlotFamily::Odd);
        assert_eq!(info.last_snr_db, -4);
        assert_eq!(info.last_text.as_deref(), Some("N1VF K7VAY DM42"));
    }

    #[test]
    fn station_tracker_replaces_same_slot_log_with_later_stage() {
        let slot_start = UNIX_EPOCH + Duration::from_secs(45);
        let mut tracker = StationTracker::default();
        let early = DecodedMessage {
            utc: "00:00:00".to_string(),
            snr_db: -9,
            dt_seconds: 0.1,
            freq_hz: 1000.0,
            text: "N1VF K7VAY DM42".to_string(),
            candidate_score: 0.0,
            ldpc_iterations: 0,
            message: StructuredMessage::Standard {
                i3: 0,
                first: standard_call("N1VF"),
                second: standard_call("K7VAY"),
                acknowledge: false,
                info: grid_info("DM42"),
            },
        };
        let full = DecodedMessage {
            snr_db: -4,
            ..early.clone()
        };

        tracker.ingest_stage(slot_start, DecodeStage::Early47, &[early]);
        tracker.ingest_stage(slot_start, DecodeStage::Full, &[full]);

        let logs = tracker.web_logs();
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0].sender_call, "K7VAY");
        assert_eq!(logs[0].snr_db, -4);
        let info = tracker.start_info("K7VAY").expect("start info");
        assert_eq!(info.last_snr_db, -4);
        assert_eq!(info.last_heard_slot_family, SlotFamily::Odd);
    }

    #[test]
    fn station_dispatch_preserves_scheduler_selected_tx_parity() {
        let dispatch = QueueDispatch {
            kind: QueueDispatchKind::Station {
                callsign: "K7VAY".to_string(),
                initial_state: QsoState::SendSig,
                start_mode: QsoStartMode::Direct,
                context_last_heard_at: None,
                context_last_heard_slot_family: None,
                context_text: None,
                context_structured_json: None,
                context_snr_db: None,
            },
            callsign: "K7VAY".to_string(),
            tx_slot_family: SlotFamily::Even,
            tx_freq_hz: 720.0,
        };

        match qso_start_command_from_dispatch(&dispatch) {
            qso::QsoCommand::Start {
                tx_slot_family_override,
                ..
            } => assert_eq!(tx_slot_family_override, Some(SlotFamily::Even)),
            _ => panic!("expected start command"),
        }
    }

    #[test]
    fn direct_dispatch_context_overrides_stale_tracker_slot_family() {
        let older = UNIX_EPOCH + Duration::from_secs(30);
        let newer = older + Duration::from_secs(15);
        let mut tracker = StationTracker::default();
        ingest_decode(&mut tracker, older, &cq_decode("K7VAY"));

        let station_info = station_info_from_dispatch(
            &tracker,
            &QueueDispatchKind::Station {
                callsign: "K7VAY".to_string(),
                initial_state: QsoState::SendSig,
                start_mode: QsoStartMode::Direct,
                context_last_heard_at: Some(newer),
                context_last_heard_slot_family: Some(SlotFamily::Odd),
                context_text: Some("N1VF K7VAY DM42".to_string()),
                context_structured_json: Some("{}".to_string()),
                context_snr_db: Some(-4),
            },
        )
        .expect("station info");

        assert_eq!(station_info.last_heard_at, newer);
        assert_eq!(station_info.last_heard_slot_family, SlotFamily::Odd);
        assert_eq!(station_info.last_snr_db, -4);
        assert_eq!(station_info.last_text.as_deref(), Some("N1VF K7VAY DM42"));
    }

    #[test]
    fn qso_jsonl_scan_does_not_merge_reused_session_ids() {
        let contents = r#"{"timestamp":"2026-04-03T02:26:37Z","level":"INFO","fields":{"message":"qso_fsm","event":"start","session_id":2,"partner_call":"W0ONN","state_before":"idle","state_after":"send_grid","last_rx_event":"start","rx_text":"","tx_text":""}}
{"timestamp":"2026-04-03T02:30:00Z","level":"INFO","fields":{"message":"qso_fsm","event":"exit","session_id":2,"partner_call":"W0ONN","state_before":"send_grid","state_after":"idle","last_rx_event":"send_grid_no_msg_limit","rx_text":"","tx_text":""}}
{"timestamp":"2026-04-04T01:24:12Z","level":"INFO","fields":{"message":"qso_fsm","event":"start","session_id":2,"partner_call":"K4SYT","state_before":"idle","state_after":"send_grid","last_rx_event":"start","rx_text":"","tx_text":""}}
{"timestamp":"2026-04-04T01:26:14Z","level":"INFO","fields":{"message":"qso_fsm","event":"tx_launch","session_id":2,"partner_call":"K4SYT","state_before":"send_73","state_after":"send_73","last_rx_event":"to_us_reply_rr73","rx_text":"","tx_text":"K4SYT N1VF 73"}}
{"timestamp":"2026-04-04T01:27:15Z","level":"INFO","fields":{"message":"qso_fsm","event":"exit","session_id":2,"partner_call":"K4SYT","state_before":"send_73","state_after":"idle","last_rx_event":"send_73_no_msg_limit","rx_text":"","tx_text":""}}"#;
        let now = UNIX_EPOCH + Duration::from_secs(10 * 365 * 24 * 60 * 60);
        let scan = scan_qso_jsonl(contents, now, UNIX_EPOCH);
        assert_eq!(scan.history.len(), 2);
        assert_eq!(scan.history[0].callsign, "K4SYT");
        assert_eq!(scan.history[1].callsign, "W0ONN");
        assert!(scan.history[0].reached_73);
        assert!(!scan.history[1].reached_73);
    }

    #[test]
    fn qso_jsonl_scan_uses_start_context_rx_text_for_received_info() {
        let contents = r#"{"timestamp":"2026-04-04T04:54:34Z","level":"INFO","fields":{"message":"qso_fsm","event":"start","session_id":7,"partner_call":"KJ7JJ","state_before":"idle","state_after":"send_grid","last_rx_event":"start_context","rx_text":"CQ KJ7JJ DN17","tx_text":""}}
{"timestamp":"2026-04-04T04:54:58Z","level":"INFO","fields":{"message":"qso_fsm","event":"exit","session_id":7,"partner_call":"KJ7JJ","state_before":"send_grid","state_after":"idle","last_rx_event":"tx_error","rx_text":"","tx_text":""}}"#;
        let now = UNIX_EPOCH + Duration::from_secs(10 * 365 * 24 * 60 * 60);
        let scan = scan_qso_jsonl(contents, now, UNIX_EPOCH);
        assert_eq!(scan.history.len(), 1);
        assert_eq!(scan.history[0].callsign, "KJ7JJ");
        assert_eq!(scan.history[0].received_info, "DN17");
    }

    #[test]
    fn qso_jsonl_scan_attributes_compound_report_only_to_follow_on_qso() {
        let contents = r#"{"timestamp":"2026-04-04T05:00:00Z","level":"INFO","fields":{"message":"qso_fsm","event":"start","session_id":10,"partner_call":"OLD1","state_before":"idle","state_after":"send_sig","last_rx_event":"start","rx_text":"","tx_text":""}}
{"timestamp":"2026-04-04T05:00:15Z","level":"INFO","fields":{"message":"qso_fsm","event":"tx_launch","session_id":10,"partner_call":"OLD1","state_before":"send_rr73","state_after":"send_rr73","last_rx_event":"to_us_ack","rx_text":"","tx_text":"OLD1 RR73; NEW1 <N1VF> -07","compound_finished_call":"OLD1","compound_next_call":"NEW1"}}
{"timestamp":"2026-04-04T05:00:30Z","level":"INFO","fields":{"message":"qso_fsm","event":"exit","session_id":10,"partner_call":"OLD1","state_before":"send_rr73","state_after":"idle","last_rx_event":"compound_handoff_sent","rx_text":"","tx_text":"","compound_finished_call":"OLD1","compound_next_call":"NEW1"}}
{"timestamp":"2026-04-04T05:00:30Z","level":"INFO","fields":{"message":"qso_fsm","event":"start","session_id":11,"partner_call":"NEW1","state_before":"idle","state_after":"send_sig","last_rx_event":"start_context","rx_text":"N1VF NEW1 FN20","tx_text":""}}
{"timestamp":"2026-04-04T05:00:30Z","level":"INFO","fields":{"message":"qso_fsm","event":"compound_start","session_id":11,"partner_call":"NEW1","state_before":"idle","state_after":"send_sig","last_rx_event":"start_context","rx_text":"","tx_text":"OLD1 RR73; NEW1 <N1VF> -07","compound_finished_call":"OLD1","compound_next_call":"NEW1"}}"#;
        let now = UNIX_EPOCH + Duration::from_secs(10 * 365 * 24 * 60 * 60);
        let scan = scan_qso_jsonl(contents, now, UNIX_EPOCH);
        assert_eq!(scan.history.len(), 2);
        let old = scan
            .history
            .iter()
            .find(|entry| entry.callsign == "OLD1")
            .expect("old qso");
        let new = scan
            .history
            .iter()
            .find(|entry| entry.callsign == "NEW1")
            .expect("new qso");
        assert_eq!(new.sent_info, "-07");
        assert_eq!(old.sent_info, "-");
        assert!(old.reached_73);
    }

    #[test]
    fn qso_jsonl_scan_tracks_roger_responses_separately_from_any_reply() {
        let contents = r#"{"timestamp":"2026-04-04T05:00:00Z","level":"INFO","fields":{"message":"qso_fsm","event":"start","session_id":12,"partner_call":"K1ABC","state_before":"idle","state_after":"send_grid","last_rx_event":"start","rx_text":"","tx_text":""}}
{"timestamp":"2026-04-04T05:00:15Z","level":"INFO","fields":{"message":"qso_fsm","event":"rx_slot_early41","session_id":12,"partner_call":"K1ABC","state_before":"send_grid","state_after":"send_grid","last_rx_event":"to_us_report_like","rx_text":"N1VF K1ABC -08","tx_text":""}}
{"timestamp":"2026-04-04T05:00:30Z","level":"INFO","fields":{"message":"qso_fsm","event":"rx_slot_early41","session_id":12,"partner_call":"K1ABC","state_before":"send_sig_ack","state_after":"send_sig_ack","last_rx_event":"to_us_ack","rx_text":"N1VF K1ABC R-07","tx_text":""}}
{"timestamp":"2026-04-04T05:00:45Z","level":"INFO","fields":{"message":"qso_fsm","event":"exit","session_id":12,"partner_call":"K1ABC","state_before":"send_sig_ack","state_after":"idle","last_rx_event":"stopped","rx_text":"","tx_text":""}}"#;
        let now = UNIX_EPOCH + Duration::from_secs(10 * 365 * 24 * 60 * 60);
        let scan = scan_qso_jsonl(contents, now, UNIX_EPOCH);
        assert_eq!(scan.history.len(), 1);
        assert!(scan.history[0].got_reply);
        assert!(scan.history[0].got_roger);
    }

    #[test]
    fn qso_jsonl_scan_surfaces_logged_app_mode() {
        let contents = r#"{"timestamp":"2026-04-04T05:00:00Z","level":"INFO","fields":{"message":"qso_fsm","event":"start","session_id":13,"partner_call":"K1ABC","state_before":"idle","state_after":"send_grid","last_rx_event":"start","app_mode":"ft4","rx_text":"","tx_text":""}}
{"timestamp":"2026-04-04T05:00:15Z","level":"INFO","fields":{"message":"qso_fsm","event":"exit","session_id":13,"partner_call":"K1ABC","state_before":"send_grid","state_after":"idle","last_rx_event":"stopped","app_mode":"ft4","rx_text":"","tx_text":""}}"#;
        let now = UNIX_EPOCH + Duration::from_secs(10 * 365 * 24 * 60 * 60);
        let scan = scan_qso_jsonl(contents, now, UNIX_EPOCH);
        assert_eq!(scan.history.len(), 1);
        assert_eq!(scan.history[0].mode, "FT4");
    }

    #[test]
    fn collect_adif_records_exports_only_award_valid_qsos() {
        let contents = r#"{"timestamp":"2026-04-04T05:00:00Z","level":"INFO","fields":{"message":"qso_fsm","event":"start","session_id":20,"partner_call":"K1ABC","state_before":"idle","state_after":"send_grid","last_rx_event":"start_context","rig_band":"20m","rig_frequency_hz":14074000,"app_mode":"ft8","rx_text":"CQ K1ABC FN20","tx_text":""}}
{"timestamp":"2026-04-04T05:00:15Z","level":"INFO","fields":{"message":"qso_fsm","event":"rx_slot_early41","session_id":20,"partner_call":"K1ABC","state_before":"send_grid","state_after":"send_sig_ack","last_rx_event":"to_us_report_like","rig_band":"20m","rig_frequency_hz":14074000,"app_mode":"ft8","rx_text":"N1VF K1ABC -08","tx_text":""}}
{"timestamp":"2026-04-04T05:00:30Z","level":"INFO","fields":{"message":"qso_fsm","event":"rx_slot_early41","session_id":20,"partner_call":"K1ABC","state_before":"send_sig_ack","state_after":"send_rr73","last_rx_event":"to_us_ack","rig_band":"20m","rig_frequency_hz":14074000,"app_mode":"ft8","rx_text":"N1VF K1ABC R-08","tx_text":""}}
{"timestamp":"2026-04-04T05:00:45Z","level":"INFO","fields":{"message":"qso_fsm","event":"tx_launch","session_id":20,"partner_call":"K1ABC","state_before":"send_sig_ack","state_after":"send_sig_ack","last_rx_event":"to_us_report_like","rig_band":"20m","rig_frequency_hz":14074000,"app_mode":"ft8","rx_text":"","tx_text":"K1ABC N1VF R-07"}}
{"timestamp":"2026-04-04T05:01:00Z","level":"INFO","fields":{"message":"qso_fsm","event":"tx_launch","session_id":20,"partner_call":"K1ABC","state_before":"send_rr73","state_after":"send_rr73","last_rx_event":"to_us_ack","rig_band":"20m","rig_frequency_hz":14074000,"app_mode":"ft8","rx_text":"","tx_text":"K1ABC N1VF RR73"}}
{"timestamp":"2026-04-04T05:01:15Z","level":"INFO","fields":{"message":"qso_fsm","event":"exit","session_id":20,"partner_call":"K1ABC","state_before":"send_rr73","state_after":"idle","last_rx_event":"send_rr73_no_msg_limit","rig_band":"20m","rig_frequency_hz":14074000,"app_mode":"ft8","rx_text":"","tx_text":""}}
{"timestamp":"2026-04-04T05:02:00Z","level":"INFO","fields":{"message":"qso_fsm","event":"start","session_id":21,"partner_call":"W1XYZ","state_before":"idle","state_after":"send_grid","last_rx_event":"start_context","rig_band":"20m","rig_frequency_hz":14074000,"app_mode":"ft8","rx_text":"CQ W1XYZ FN31","tx_text":""}}
{"timestamp":"2026-04-04T05:02:15Z","level":"INFO","fields":{"message":"qso_fsm","event":"rx_slot_early41","session_id":21,"partner_call":"W1XYZ","state_before":"send_grid","state_after":"send_sig_ack","last_rx_event":"to_us_report_like","rig_band":"20m","rig_frequency_hz":14074000,"app_mode":"ft8","rx_text":"N1VF W1XYZ -05","tx_text":""}}
{"timestamp":"2026-04-04T05:02:30Z","level":"INFO","fields":{"message":"qso_fsm","event":"exit","session_id":21,"partner_call":"W1XYZ","state_before":"send_sig_ack","state_after":"idle","last_rx_event":"stopped","rig_band":"20m","rig_frequency_hz":14074000,"app_mode":"ft8","rx_text":"","tx_text":""}}"#;
        let records = collect_adif_records(contents, "N1VF", "CM97");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].call, "K1ABC");
        assert_eq!(records[0].band, "20M");
        assert_eq!(records[0].freq.as_deref(), Some("14.074000"));
        assert_eq!(records[0].mode, "MFSK");
        assert_eq!(records[0].submode.as_deref(), Some("FT8"));
        assert_eq!(records[0].rst_sent, "-07");
        assert_eq!(records[0].rst_rcvd, "-08");
        assert_eq!(records[0].gridsquare.as_deref(), Some("FN20"));
    }

    #[test]
    fn collect_adif_records_requires_both_signal_reports() {
        let contents = r#"{"timestamp":"2026-04-04T05:00:00Z","level":"INFO","fields":{"message":"qso_fsm","event":"start","session_id":22,"partner_call":"K1ABC","state_before":"idle","state_after":"send_grid","last_rx_event":"start_context","rig_band":"20m","rig_frequency_hz":14074000,"app_mode":"ft8","rx_text":"CQ K1ABC FN20","tx_text":""}}
{"timestamp":"2026-04-04T05:00:15Z","level":"INFO","fields":{"message":"qso_fsm","event":"rx_slot_early41","session_id":22,"partner_call":"K1ABC","state_before":"send_grid","state_after":"send_sig_ack","last_rx_event":"to_us_ack","rig_band":"20m","rig_frequency_hz":14074000,"app_mode":"ft8","rx_text":"N1VF K1ABC R-08","tx_text":""}}
{"timestamp":"2026-04-04T05:00:30Z","level":"INFO","fields":{"message":"qso_fsm","event":"tx_launch","session_id":22,"partner_call":"K1ABC","state_before":"send_rr73","state_after":"send_rr73","last_rx_event":"to_us_ack","rig_band":"20m","rig_frequency_hz":14074000,"app_mode":"ft8","rx_text":"","tx_text":"K1ABC N1VF RR73"}}
{"timestamp":"2026-04-04T05:00:45Z","level":"INFO","fields":{"message":"qso_fsm","event":"exit","session_id":22,"partner_call":"K1ABC","state_before":"send_rr73","state_after":"idle","last_rx_event":"send_rr73_no_msg_limit","rig_band":"20m","rig_frequency_hz":14074000,"app_mode":"ft8","rx_text":"","tx_text":""}}"#;
        let records = collect_adif_records(contents, "N1VF", "CM97");
        assert!(records.is_empty());
    }

    #[test]
    fn build_adif_export_report_lists_ineligible_reasons() {
        let contents = r#"{"timestamp":"2026-04-04T05:00:00Z","level":"INFO","fields":{"message":"qso_fsm","event":"start","session_id":23,"partner_call":"W1XYZ","state_before":"idle","state_after":"send_grid","last_rx_event":"start_context","rig_band":"20m","rig_frequency_hz":14074000,"app_mode":"ft8","rx_text":"CQ W1XYZ FN31","tx_text":""}}
{"timestamp":"2026-04-04T05:00:15Z","level":"INFO","fields":{"message":"qso_fsm","event":"rx_slot_early41","session_id":23,"partner_call":"W1XYZ","state_before":"send_grid","state_after":"send_sig_ack","last_rx_event":"to_us_report_like","rig_band":"20m","rig_frequency_hz":14074000,"app_mode":"ft8","rx_text":"N1VF W1XYZ -05","tx_text":""}}
{"timestamp":"2026-04-04T05:00:30Z","level":"INFO","fields":{"message":"qso_fsm","event":"exit","session_id":23,"partner_call":"W1XYZ","state_before":"send_sig_ack","state_after":"idle","last_rx_event":"stopped","rig_band":"20m","rig_frequency_hz":14074000,"app_mode":"ft8","rx_text":"","tx_text":""}}"#;
        let report = build_adif_export_report(contents, "N1VF", "CM97");
        assert!(report.eligible_records.is_empty());
        assert_eq!(report.ineligible_sessions.len(), 1);
        assert_eq!(report.ineligible_sessions[0].session_id, 23);
        assert_eq!(
            report.ineligible_sessions[0].reason,
            "missing sent report".to_string()
        );
    }

    #[test]
    fn exporter_allows_remote_station_named_like_previous_home_call() {
        let contents = r#"{"timestamp":"2026-04-04T05:00:00Z","level":"INFO","fields":{"message":"qso_fsm","event":"start","session_id":24,"partner_call":"N1VF","state_before":"idle","state_after":"send_grid","last_rx_event":"start_context","rig_band":"20m","rig_frequency_hz":14074000,"app_mode":"ft8","rx_text":"CQ N1VF CM97","tx_text":""}}
{"timestamp":"2026-04-04T05:00:15Z","level":"INFO","fields":{"message":"qso_fsm","event":"rx_slot_early41","session_id":24,"partner_call":"N1VF","state_before":"send_grid","state_after":"send_sig_ack","last_rx_event":"to_us_report_like","rig_band":"20m","rig_frequency_hz":14074000,"app_mode":"ft8","rx_text":"E51RWK N1VF -08","tx_text":""}}
{"timestamp":"2026-04-04T05:00:30Z","level":"INFO","fields":{"message":"qso_fsm","event":"rx_slot_early41","session_id":24,"partner_call":"N1VF","state_before":"send_sig_ack","state_after":"send_rr73","last_rx_event":"to_us_ack","rig_band":"20m","rig_frequency_hz":14074000,"app_mode":"ft8","rx_text":"E51RWK N1VF R-08","tx_text":""}}
{"timestamp":"2026-04-04T05:00:45Z","level":"INFO","fields":{"message":"qso_fsm","event":"tx_launch","session_id":24,"partner_call":"N1VF","state_before":"send_sig_ack","state_after":"send_sig_ack","last_rx_event":"to_us_report_like","rig_band":"20m","rig_frequency_hz":14074000,"app_mode":"ft8","rx_text":"","tx_text":"N1VF E51RWK R-07"}}
{"timestamp":"2026-04-04T05:01:00Z","level":"INFO","fields":{"message":"qso_fsm","event":"tx_launch","session_id":24,"partner_call":"N1VF","state_before":"send_rr73","state_after":"send_rr73","last_rx_event":"to_us_ack","rig_band":"20m","rig_frequency_hz":14074000,"app_mode":"ft8","rx_text":"","tx_text":"N1VF E51RWK RR73"}}
{"timestamp":"2026-04-04T05:01:15Z","level":"INFO","fields":{"message":"qso_fsm","event":"exit","session_id":24,"partner_call":"N1VF","state_before":"send_rr73","state_after":"idle","last_rx_event":"send_rr73_no_msg_limit","rig_band":"20m","rig_frequency_hz":14074000,"app_mode":"ft8","rx_text":"","tx_text":""}}"#;
        let records = collect_adif_records(contents, "E51RWK", "BG08");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].call, "N1VF");
        assert_eq!(records[0].station_callsign, "E51RWK");
    }

    #[test]
    fn exporter_rejects_sessions_logged_under_different_station_call() {
        let contents = r#"{"timestamp":"2026-04-04T05:00:00Z","level":"INFO","fields":{"message":"qso_fsm","event":"start","session_id":25,"partner_call":"K1ABC","state_before":"idle","state_after":"send_grid","last_rx_event":"start_context","rig_band":"20m","rig_frequency_hz":14074000,"app_mode":"ft8","rx_text":"CQ K1ABC FN20","tx_text":""}}
{"timestamp":"2026-04-04T05:00:15Z","level":"INFO","fields":{"message":"qso_fsm","event":"rx_slot_early41","session_id":25,"partner_call":"K1ABC","state_before":"send_grid","state_after":"send_sig_ack","last_rx_event":"to_us_report_like","rig_band":"20m","rig_frequency_hz":14074000,"app_mode":"ft8","rx_text":"N1VF K1ABC -08","tx_text":""}}
{"timestamp":"2026-04-04T05:00:30Z","level":"INFO","fields":{"message":"qso_fsm","event":"rx_slot_early41","session_id":25,"partner_call":"K1ABC","state_before":"send_sig_ack","state_after":"send_rr73","last_rx_event":"to_us_ack","rig_band":"20m","rig_frequency_hz":14074000,"app_mode":"ft8","rx_text":"N1VF K1ABC R-08","tx_text":""}}
{"timestamp":"2026-04-04T05:00:45Z","level":"INFO","fields":{"message":"qso_fsm","event":"tx_launch","session_id":25,"partner_call":"K1ABC","state_before":"send_sig_ack","state_after":"send_sig_ack","last_rx_event":"to_us_report_like","rig_band":"20m","rig_frequency_hz":14074000,"app_mode":"ft8","rx_text":"","tx_text":"K1ABC N1VF R-07"}}
{"timestamp":"2026-04-04T05:01:00Z","level":"INFO","fields":{"message":"qso_fsm","event":"tx_launch","session_id":25,"partner_call":"K1ABC","state_before":"send_rr73","state_after":"send_rr73","last_rx_event":"to_us_ack","rig_band":"20m","rig_frequency_hz":14074000,"app_mode":"ft8","rx_text":"","tx_text":"K1ABC N1VF RR73"}}
{"timestamp":"2026-04-04T05:01:15Z","level":"INFO","fields":{"message":"qso_fsm","event":"exit","session_id":25,"partner_call":"K1ABC","state_before":"send_rr73","state_after":"idle","last_rx_event":"send_rr73_no_msg_limit","rig_band":"20m","rig_frequency_hz":14074000,"app_mode":"ft8","rx_text":"","tx_text":""}}"#;
        let report = build_adif_export_report(contents, "E51RWK", "BG08");
        assert!(report.eligible_records.is_empty());
        assert_eq!(report.ineligible_sessions.len(), 1);
        assert!(
            report.ineligible_sessions[0]
                .reason
                .contains("different station callsign(s): N1VF")
        );
    }

    #[test]
    fn render_adif_writes_expected_fields() {
        let record = AdifRecord {
            session_id: 1,
            call: "K1ABC".to_string(),
            qso_date: "20260404".to_string(),
            time_on: "050000".to_string(),
            qso_date_off: Some("20260404".to_string()),
            time_off: Some("050115".to_string()),
            band: "20M".to_string(),
            freq: Some("14.074000".to_string()),
            mode: "MFSK".to_string(),
            submode: Some("FT8".to_string()),
            rst_sent: "-07".to_string(),
            rst_rcvd: "-08".to_string(),
            station_callsign: "N1VF".to_string(),
            my_gridsquare: "CM97".to_string(),
            gridsquare: Some("FN20".to_string()),
        };
        let adif = render_adif(&[record]);
        assert!(adif.contains("<ADIF_VER:5>3.1.4<PROGRAMID:5>ft8rx"));
        assert!(adif.contains("<CALL:5>K1ABC"));
        assert!(adif.contains("<MODE:4>MFSK<SUBMODE:3>FT8"));
        assert!(adif.contains("<RST_SENT:3>-07<RST_RCVD:3>-08"));
        assert!(
            adif.contains("<STATION_CALLSIGN:4>N1VF<MY_GRIDSQUARE:4>CM97<GRIDSQUARE:4>FN20<EOR>")
        );
    }
}
