import { SeverityNumber } from "@opentelemetry/api-logs"
import { SpanStatusCode, trace } from "@opentelemetry/api"
import type { EventSessionCreated, EventSessionIdle, EventSessionError, EventSessionStatus } from "@opencode-ai/sdk"
import { AGENT_NAME, MimeType, OpenInferenceSpanKind, OUTPUT_MIME_TYPE, OUTPUT_VALUE, SemanticConventions, SESSION_ID } from "@arizeai/openinference-semantic-conventions"
import { errorSummary, setBoundedMap, isMetricEnabled, isTraceEnabled } from "../util.ts"
import type { HandlerContext } from "../types.ts"

const OPENINFERENCE_SPAN_KIND = SemanticConventions.OPENINFERENCE_SPAN_KIND

/** Increments the session counter, records start time, starts the root session span, and emits a `session.created` log event. */
export function handleSessionCreated(e: EventSessionCreated, ctx: HandlerContext) {
  const { id: sessionID, time, parentID } = e.properties.info
  const createdAt = time.created
  const isSubagent = !!parentID
  if (isMetricEnabled("session.count", ctx)) {
    ctx.instruments.sessionCounter.add(1, { ...ctx.commonAttrs, "session.id": sessionID, is_subagent: isSubagent })
  }
  setBoundedMap(ctx.sessionTotals, sessionID, { startMs: createdAt, tokens: 0, cost: 0, messages: 0, agent: "unknown" })

  // WARNING: disabling "session" traces while "llm" or "tool" traces remain enabled
  // leaves those child spans without a local session parent. If OPENCODE_TRACEPARENT
  // is set, they fall back to that remote parent; otherwise they become root spans.
  if (isTraceEnabled("session", ctx)) {
    const parentSpan = parentID ? ctx.sessionSpans.get(parentID) : undefined
    const baseCtx = ctx.rootContext()
    const spanCtx = parentSpan
      ? trace.setSpan(baseCtx, parentSpan)
      : baseCtx

    const sessionSpan = ctx.tracer.startSpan(
      `${ctx.tracePrefix}session`,
      {
        startTime: createdAt,
        attributes: {
          [OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.AGENT,
          [SESSION_ID]: sessionID,
          [AGENT_NAME]: "unknown",
          "session.is_subagent": isSubagent,
          ...ctx.commonAttrs,
        },
      },
      spanCtx,
    )
    setBoundedMap(ctx.sessionSpans, sessionID, sessionSpan)
  }

  ctx.emitLog({
    severityNumber: SeverityNumber.INFO,
    severityText: "INFO",
    timestamp: createdAt,
    observedTimestamp: Date.now(),
    body: "session.created",
    attributes: { "event.name": "session.created", "session.id": sessionID, is_subagent: isSubagent, ...ctx.commonAttrs },
  })
  return ctx.log("info", "otel: session.created", { sessionID, createdAt, isSubagent })
}

function sweepSession(sessionID: string, ctx: HandlerContext) {
  for (const [id, perm] of ctx.pendingPermissions) {
    if (perm.sessionID === sessionID) ctx.pendingPermissions.delete(id)
  }
  for (const [key, span] of ctx.pendingToolSpans) {
    if (span.sessionID === sessionID) {
      span.span?.setStatus({ code: SpanStatusCode.ERROR, message: "session ended before tool completed" })
      span.span?.end()
      ctx.pendingToolSpans.delete(key)
    }
  }
  for (const [key, span] of ctx.pendingReasoningSpans) {
    if (span.sessionID === sessionID) {
      span.span?.setAttributes({
        [OUTPUT_VALUE]: span.text,
        [OUTPUT_MIME_TYPE]: MimeType.TEXT,
        "opencode.reasoning.text_length": span.text.length,
        "opencode.reasoning.truncated": span.truncated,
      })
      span.span?.setStatus({ code: SpanStatusCode.ERROR, message: "session ended before reasoning completed" })
      span.span?.end()
      ctx.pendingReasoningSpans.delete(key)
    }
  }
  ctx.sessionInputs.delete(sessionID)
  const msgPrefix = `${sessionID}:`
  for (const [key, span] of ctx.messageSpans) {
    if (key.startsWith(msgPrefix)) {
      span.setStatus({ code: SpanStatusCode.ERROR, message: "session ended before message completed" })
      span.end()
      ctx.messageSpans.delete(key)
    }
  }
  for (const key of ctx.messageOutputs.keys()) {
    if (key.startsWith(msgPrefix)) ctx.messageOutputs.delete(key)
  }
  for (const key of ctx.messageToolCalls.keys()) {
    if (key.startsWith(msgPrefix)) ctx.messageToolCalls.delete(key)
  }
}

/** Emits a `session.idle` log event, records duration and session total histograms, ends the session span, and clears pending state. */
export function handleSessionIdle(e: EventSessionIdle, ctx: HandlerContext) {
  const sessionID = e.properties.sessionID
  const totals = ctx.sessionTotals.get(sessionID)
  ctx.sessionTotals.delete(sessionID)
  ctx.sessionDiffTotals.delete(sessionID)
  sweepSession(sessionID, ctx)

  const attrs = { ...ctx.commonAttrs, "session.id": sessionID }
  let duration_ms: number | undefined

  if (totals) {
    duration_ms = Date.now() - totals.startMs
    if (isMetricEnabled("session.duration", ctx)) {
      ctx.instruments.sessionDurationHistogram.record(duration_ms, attrs)
    }
    if (isMetricEnabled("session.token.total", ctx)) {
      ctx.instruments.sessionTokenGauge.record(totals.tokens, attrs)
    }
    if (isMetricEnabled("session.cost.total", ctx)) {
      ctx.instruments.sessionCostGauge.record(totals.cost, attrs)
    }
  }

  const sessionSpan = ctx.sessionSpans.get(sessionID)
  if (sessionSpan) {
    if (totals) {
      sessionSpan.setAttributes({
        [AGENT_NAME]: totals.agent,
        "session.total_tokens": totals.tokens,
        "session.total_cost_usd": totals.cost,
        "session.total_messages": totals.messages,
      })
    }
    sessionSpan.setStatus({ code: SpanStatusCode.OK })
    sessionSpan.end()
    ctx.sessionSpans.delete(sessionID)
  }

  ctx.emitLog({
    severityNumber: SeverityNumber.INFO,
    severityText: "INFO",
    timestamp: Date.now(),
    observedTimestamp: Date.now(),
    body: "session.idle",
    attributes: {
      "event.name": "session.idle",
      "session.id": sessionID,
      total_tokens: totals?.tokens ?? 0,
      total_cost_usd: totals?.cost ?? 0,
      total_messages: totals?.messages ?? 0,
      ...ctx.commonAttrs,
    },
  })
  ctx.log("debug", "otel: session.idle", {
    sessionID,
    ...(totals ? { duration_ms, total_tokens: totals.tokens, total_cost_usd: totals.cost, total_messages: totals.messages } : {}),
  })
}

/** Emits a `session.error` log event, ends the session span with error status, and clears any pending state for the session. */
export function handleSessionError(e: EventSessionError, ctx: HandlerContext) {
  const rawID = e.properties.sessionID
  const sessionID = rawID ?? "unknown"
  const error = errorSummary(e.properties.error)
  const totals = rawID ? ctx.sessionTotals.get(rawID) : undefined
  if (rawID) {
    ctx.sessionTotals.delete(rawID)
    ctx.sessionDiffTotals.delete(rawID)
  }
  sweepSession(sessionID, ctx)

  if (rawID) {
    const sessionSpan = ctx.sessionSpans.get(rawID)
    if (sessionSpan) {
      if (totals) sessionSpan.setAttribute(AGENT_NAME, totals.agent)
      sessionSpan.setStatus({ code: SpanStatusCode.ERROR, message: error })
      sessionSpan.setAttribute("error", error)
      sessionSpan.end()
      ctx.sessionSpans.delete(rawID)
    }
  }

  ctx.emitLog({
    severityNumber: SeverityNumber.ERROR,
    severityText: "ERROR",
    timestamp: Date.now(),
    observedTimestamp: Date.now(),
    body: "session.error",
    attributes: {
      "event.name": "session.error",
      "session.id": sessionID,
      error,
      ...ctx.commonAttrs,
    },
  })
  ctx.log("error", "otel: session.error", { sessionID, error })
}

/** Increments the retry counter when the session enters a retry state. */
export function handleSessionStatus(e: EventSessionStatus, ctx: HandlerContext) {
  if (e.properties.status.type !== "retry") return
  const { sessionID, status } = e.properties
  const { attempt, message: retryMessage } = status
  if (isMetricEnabled("retry.count", ctx)) {
    ctx.instruments.retryCounter.add(1, { ...ctx.commonAttrs, "session.id": sessionID })
    ctx.log("debug", "otel: retry counter incremented", { sessionID, attempt, retryMessage })
  }
}
