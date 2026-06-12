import type { Attributes, Context, Counter, Gauge, Histogram, Span, Tracer } from "@opentelemetry/api"
import type { LogRecord } from "@opentelemetry/api-logs"

/** Numeric priority map for log levels; higher value = higher severity. */
export const LEVELS = { debug: 0, info: 1, warn: 2, error: 3 } as const

/** Union of supported log level names. */
export type Level = keyof typeof LEVELS

/** Maximum number of entries kept in `pendingToolSpans`, `pendingReasoningSpans`, and `pendingPermissions` maps. */
export const MAX_PENDING = 500

/** Structured logger forwarded to the opencode `client.app.log` API. */
export type PluginLogger = (
  level: Level,
  message: string,
  extra?: Record<string, unknown>,
) => Promise<void>

/** OTel span/metric attributes common to every emitted signal. */
export type CommonAttrs = Attributes & { readonly "project.id": string }

/** In-flight tool execution tracked between `running` and `completed`/`error` part updates. */
export type PendingToolSpan = {
  tool: string
  sessionID: string
  messageID: string
  startMs: number
  span?: Span
}

/** Tool calls requested by one assistant message before their tool spans run. */
export type MessageToolCall = {
  id: string
  name: string
  input?: Record<string, unknown>
}

/** In-flight reasoning content tracked between reasoning part updates and stream deltas. */
export type PendingReasoningSpan = {
  sessionID: string
  messageID: string
  partID: string
  startMs: number
  text: string
  truncated: boolean
  span?: Span
}

/** Permission prompt tracked between `permission.updated` and `permission.replied`. */
export type PendingPermission = {
  type: string
  title: string
  sessionID: string
}

/** OTel metric instruments created once at plugin startup and shared via `HandlerContext`. */
export type Instruments = {
  sessionCounter: Counter
  tokenCounter: Counter
  costCounter: Counter
  linesCounter: Counter
  linesTotalGauge: Gauge
  commitCounter: Counter
  toolDurationHistogram: Histogram
  cacheCounter: Counter
  sessionDurationHistogram: Histogram
  messageCounter: Counter
  sessionTokenGauge: Histogram
  sessionCostGauge: Histogram
  modelUsageCounter: Counter
  retryCounter: Counter
  subtaskCounter: Counter
}

/** Accumulated per-session totals used for gauge snapshots on session.idle. */
export type SessionTotals = {
  startMs: number
  tokens: number
  cost: number
  messages: number
  agent: string
}

/** Shared context threaded through every event handler. */
export type HandlerContext = {
  log: PluginLogger
  debugStderr: (message: string, extra?: Record<string, unknown>) => void
  emitLog: (record: LogRecord) => void
  instruments: Instruments
  commonAttrs: CommonAttrs
  pendingToolSpans: Map<string, PendingToolSpan>
  pendingReasoningSpans: Map<string, PendingReasoningSpan>
  pendingPermissions: Map<string, PendingPermission>
  sessionTotals: Map<string, SessionTotals>
  sessionDiffTotals: Map<string, { additions: number; deletions: number }>
  disabledMetrics: Set<string>
  disabledTraces: Set<string>
  tracer: Tracer
  tracePrefix: string
  rootContext: () => Context
  sessionSpans: Map<string, Span>
  messageSpans: Map<string, Span>
  finalizedMessageSpans: Set<string>
  sessionInputs: Map<string, string>
  messageOutputs: Map<string, string>
  messageToolCalls: Map<string, MessageToolCall[]>
}
