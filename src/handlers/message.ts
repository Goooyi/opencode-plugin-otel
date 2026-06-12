import { SeverityNumber } from "@opentelemetry/api-logs"
import { SpanStatusCode, SpanKind, trace } from "@opentelemetry/api"
import type { AssistantMessage, EventMessageUpdated, EventMessagePartUpdated, ToolPart } from "@opencode-ai/sdk"
import {
  AGENT_NAME,
  INPUT_MIME_TYPE,
  INPUT_VALUE,
  LLM_COST_TOTAL,
  LLM_INPUT_MESSAGES,
  LLM_MODEL_NAME,
  LLM_OUTPUT_MESSAGES,
  LLM_PROVIDER,
  LLM_SYSTEM,
  LLM_TOKEN_COUNT_COMPLETION,
  LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING,
  LLM_TOKEN_COUNT_PROMPT,
  LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ,
  LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE,
  LLM_TOKEN_COUNT_TOTAL,
  MimeType,
  OpenInferenceSpanKind,
  OUTPUT_MIME_TYPE,
  OUTPUT_VALUE,
  SemanticConventions,
  SESSION_ID,
  TOOL_ID,
  TOOL_NAME,
  TOOL_PARAMETERS,
} from "@arizeai/openinference-semantic-conventions"
import { errorSummary, setBoundedMap, accumulateSessionTotals, isMetricEnabled, isTraceEnabled } from "../util.ts"
import type { HandlerContext, MessageToolCall } from "../types.ts"

const OPENINFERENCE_SPAN_KIND = SemanticConventions.OPENINFERENCE_SPAN_KIND
const LLM_FINISH_REASON = "llm.finish_reason"
const MAX_REASONING_TEXT_CHARS = 12000

type SubtaskPart = {
  type: "subtask"
  sessionID: string
  messageID: string
  prompt: string
  description: string
  agent: string
}

type ReasoningPart = {
  id: string
  sessionID: string
  messageID: string
  type: "reasoning"
  text: string
  metadata?: Record<string, unknown>
  time: {
    start: number
    end?: number
  }
}

type StepStartPart = {
  id: string
  sessionID: string
  messageID: string
  type: "step-start"
}

type StepFinishPart = {
  id: string
  sessionID: string
  messageID: string
  type: "step-finish"
  reason?: string
  tokens?: {
    input?: number
    output?: number
    reasoning?: number
    total?: number
    cache?: {
      read?: number
      write?: number
    }
  }
  cost?: number
}

function messageKey(sessionID: string, messageID: string): string {
  return `${sessionID}:${messageID}`
}

function toolSpanKey(toolPart: ToolPart): string {
  return `${toolPart.sessionID}:${toolPart.callID}`
}

function nonEmptyToolInput(input: Record<string, unknown>): Record<string, unknown> | undefined {
  return Object.keys(input).length > 0 ? input : undefined
}

function recordMessageToolCall(toolPart: ToolPart, ctx: HandlerContext) {
  const key = messageKey(toolPart.sessionID, toolPart.messageID)
  const calls = ctx.messageToolCalls.get(key) ?? []
  const input = nonEmptyToolInput(toolPart.state.input)
  const existingIndex = calls.findIndex((call) => call.id === toolPart.callID)
  if (existingIndex >= 0) {
    if (input && !calls[existingIndex]!.input) {
      const next = [...calls]
      next[existingIndex] = { ...next[existingIndex]!, input }
      setBoundedMap(ctx.messageToolCalls, key, next)
    }
    return
  }
  const next: MessageToolCall[] = [
    ...calls,
    {
      id: toolPart.callID,
      name: toolPart.tool,
      ...(input ? { input } : {}),
    },
  ]
  setBoundedMap(ctx.messageToolCalls, key, next)
}

function toolCallOutput(toolCalls: MessageToolCall[]) {
  return {
    toolCalls: toolCalls.map((call) => ({
      id: call.id,
      name: call.name,
      ...(call.input ? { input: call.input } : {}),
    })),
  }
}

function openAiToolCalls(toolCalls: MessageToolCall[]) {
  return toolCalls.map((call) => ({
    id: call.id,
    type: "function",
    function: {
      name: call.name,
      arguments: JSON.stringify(call.input ?? {}),
    },
  }))
}

function outputAttributes(outputText: string | undefined, toolCalls: MessageToolCall[]) {
  const hasToolCalls = toolCalls.length > 0
  const outputMessage = hasToolCalls
    ? {
        role: "assistant",
        ...(outputText ? { content: outputText } : {}),
        tool_calls: openAiToolCalls(toolCalls),
      }
    : { role: "assistant", content: outputText ?? "" }

  if (outputText) {
    return {
      [OUTPUT_VALUE]: outputText,
      [OUTPUT_MIME_TYPE]: MimeType.TEXT,
      [LLM_OUTPUT_MESSAGES]: JSON.stringify([outputMessage]),
    }
  }
  if (hasToolCalls) {
    return {
      [OUTPUT_VALUE]: JSON.stringify(toolCallOutput(toolCalls)),
      [OUTPUT_MIME_TYPE]: MimeType.JSON,
      [LLM_OUTPUT_MESSAGES]: JSON.stringify([outputMessage]),
      "opencode.tool_calls": JSON.stringify(toolCallOutput(toolCalls)),
    }
  }
  return {}
}

function parentContextForMessage(sessionID: string, messageID: string, ctx: HandlerContext) {
  const baseCtx = ctx.rootContext()
  const msgSpan = ctx.messageSpans.get(messageKey(sessionID, messageID))
  if (msgSpan) return trace.setSpan(baseCtx, msgSpan)
  const sessionSpan = ctx.sessionSpans.get(sessionID)
  return sessionSpan ? trace.setSpan(baseCtx, sessionSpan) : baseCtx
}

export type EventMessagePartDelta = {
  id?: string
  type: "message.part.delta"
  properties: {
    sessionID: string
    messageID: string
    partID: string
    field: string
    delta: string
  }
}

function reasoningKey(sessionID: string, messageID: string, partID: string): string {
  return `${sessionID}:${messageID}:${partID}`
}

function boundedReasoningText(text: string): { text: string; truncated: boolean } {
  if (text.length <= MAX_REASONING_TEXT_CHARS) return { text, truncated: false }
  return { text: text.slice(0, MAX_REASONING_TEXT_CHARS), truncated: true }
}

function appendReasoningText(existing: string, delta: string): { text: string; truncated: boolean } {
  if (existing.length >= MAX_REASONING_TEXT_CHARS) return { text: existing, truncated: true }
  const next = `${existing}${delta}`
  return boundedReasoningText(next)
}

function ensureReasoningPending(
  sessionID: string,
  messageID: string,
  partID: string,
  startMs: number,
  ctx: HandlerContext,
) {
  const key = reasoningKey(sessionID, messageID, partID)
  const existing = ctx.pendingReasoningSpans.get(key)
  if (existing) return existing

  const pending = {
    sessionID,
    messageID,
    partID,
    startMs,
    text: "",
    truncated: false,
    span: undefined,
  }
  setBoundedMap(ctx.pendingReasoningSpans, key, pending)
  ctx.log("debug", "otel: reasoning part observed", { sessionID, messageID, partID, key })
  return pending
}

function ensureReasoningOtelSpan(pending: ReturnType<typeof ensureReasoningPending>, ctx: HandlerContext) {
  if (pending.span || !isTraceEnabled("reasoning", ctx) || !pending.text) return
  const messageSpan = ctx.messageSpans.get(`${pending.sessionID}:${pending.messageID}`)
  const sessionSpan = ctx.sessionSpans.get(pending.sessionID)
  const baseCtx = ctx.rootContext()
  const parentCtx = messageSpan
    ? trace.setSpan(baseCtx, messageSpan)
    : sessionSpan
      ? trace.setSpan(baseCtx, sessionSpan)
      : baseCtx
  pending.span = ctx.tracer.startSpan(
    `${ctx.tracePrefix}reasoning`,
    {
      startTime: pending.startMs,
      kind: SpanKind.INTERNAL,
      attributes: {
        [OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.CHAIN,
        [SESSION_ID]: pending.sessionID,
        [INPUT_VALUE]: JSON.stringify({
          sessionID: pending.sessionID,
          messageID: pending.messageID,
          partID: pending.partID,
        }),
        [INPUT_MIME_TYPE]: MimeType.JSON,
        "opencode.message.id": pending.messageID,
        "opencode.reasoning.part_id": pending.partID,
        "opencode.reasoning.text_length": pending.text.length,
        "opencode.reasoning.truncated": pending.truncated,
        ...ctx.commonAttrs,
      },
    },
    parentCtx,
  )
  ctx.log("debug", "otel: reasoning span started", {
    sessionID: pending.sessionID,
    messageID: pending.messageID,
    partID: pending.partID,
  })
}

function endReasoningSpan(key: string, pending: ReturnType<typeof ensureReasoningPending>, endMs: number) {
  pending.span?.setAttributes({
    [OUTPUT_VALUE]: pending.text,
    [OUTPUT_MIME_TYPE]: MimeType.TEXT,
    "opencode.reasoning.text_length": pending.text.length,
    "opencode.reasoning.truncated": pending.truncated,
  })
  pending.span?.setStatus({ code: SpanStatusCode.OK })
  pending.span?.end(endMs)
}

/**
 * Handles a completed assistant message: increments token and cost counters, emits
 * either an `api_request` or `api_error` log event, and ends the LLM span for this message.
 * The `agent` attribute is sourced from the session totals, which are populated by the
 * `chat.message` hook when the user prompt is received.
 */
export function handleMessageUpdated(e: EventMessageUpdated, ctx: HandlerContext) {
  const msg = e.properties.info
  if (msg.role !== "assistant") return
  const assistant = msg as AssistantMessage
  if (!assistant.time.completed) return

  const { sessionID, modelID, providerID } = assistant
  const duration = assistant.time.completed - assistant.time.created
  const agent = ctx.sessionTotals.get(sessionID)?.agent ?? "unknown"

  const totalTokens = assistant.tokens.input + assistant.tokens.output + assistant.tokens.reasoning
    + assistant.tokens.cache.read + assistant.tokens.cache.write

  if (isMetricEnabled("token.usage", ctx)) {
    const { tokenCounter } = ctx.instruments
    tokenCounter.add(assistant.tokens.input, { ...ctx.commonAttrs, "session.id": sessionID, model: modelID, agent, type: "input" })
    tokenCounter.add(assistant.tokens.output, { ...ctx.commonAttrs, "session.id": sessionID, model: modelID, agent, type: "output" })
    tokenCounter.add(assistant.tokens.reasoning, { ...ctx.commonAttrs, "session.id": sessionID, model: modelID, agent, type: "reasoning" })
    tokenCounter.add(assistant.tokens.cache.read, { ...ctx.commonAttrs, "session.id": sessionID, model: modelID, agent, type: "cacheRead" })
    tokenCounter.add(assistant.tokens.cache.write, { ...ctx.commonAttrs, "session.id": sessionID, model: modelID, agent, type: "cacheCreation" })
  }

  if (isMetricEnabled("cost.usage", ctx)) {
    ctx.instruments.costCounter.add(assistant.cost, { ...ctx.commonAttrs, "session.id": sessionID, model: modelID, agent })
  }

  if (isMetricEnabled("cache.count", ctx)) {
    if (assistant.tokens.cache.read > 0) {
      ctx.instruments.cacheCounter.add(1, { ...ctx.commonAttrs, "session.id": sessionID, model: modelID, agent, type: "cacheRead" })
    }
    if (assistant.tokens.cache.write > 0) {
      ctx.instruments.cacheCounter.add(1, { ...ctx.commonAttrs, "session.id": sessionID, model: modelID, agent, type: "cacheCreation" })
    }
  }

  if (isMetricEnabled("message.count", ctx)) {
    ctx.instruments.messageCounter.add(1, { ...ctx.commonAttrs, "session.id": sessionID, model: modelID, agent })
  }

  if (isMetricEnabled("model.usage", ctx)) {
    ctx.instruments.modelUsageCounter.add(1, { ...ctx.commonAttrs, "session.id": sessionID, model: modelID, provider: providerID, agent })
  }

  accumulateSessionTotals(sessionID, totalTokens, assistant.cost, ctx)

  ctx.log("debug", "otel: token+cost counters incremented", {
    sessionID,
    model: modelID,
    agent,
    input: assistant.tokens.input,
    output: assistant.tokens.output,
    reasoning: assistant.tokens.reasoning,
    cacheRead: assistant.tokens.cache.read,
    cacheWrite: assistant.tokens.cache.write,
    cost_usd: assistant.cost,
  })

  const msgKey = messageKey(sessionID, assistant.id)
  if (!ctx.finalizedMessageSpans.has(msgKey)) {
    if (!ctx.messageSpans.has(msgKey)) {
      startMessageSpan(sessionID, assistant.id, modelID ?? "unknown", providerID ?? "unknown", assistant.time.created, ctx)
    }
    const msgSpan = ctx.messageSpans.get(msgKey)
    if (msgSpan) {
      const outputText = ctx.messageOutputs.get(msgKey)
      const toolCalls = ctx.messageToolCalls.get(msgKey) ?? []
      msgSpan.setAttributes({
        [LLM_SYSTEM]: providerID,
        [LLM_PROVIDER]: providerID,
        [LLM_MODEL_NAME]: modelID,
        [LLM_TOKEN_COUNT_PROMPT]: assistant.tokens.input,
        [LLM_TOKEN_COUNT_COMPLETION]: assistant.tokens.output,
        [LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING]: assistant.tokens.reasoning,
        [LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ]: assistant.tokens.cache.read,
        [LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE]: assistant.tokens.cache.write,
        [LLM_TOKEN_COUNT_TOTAL]: totalTokens,
        [LLM_FINISH_REASON]: assistant.error ? "error" : (assistant.finish ?? "stop"),
        [LLM_COST_TOTAL]: assistant.cost,
        ...outputAttributes(outputText, toolCalls),
        cost_usd: assistant.cost,
        duration_ms: duration,
      })
      if (assistant.error) {
        msgSpan.setStatus({ code: SpanStatusCode.ERROR, message: errorSummary(assistant.error) })
      } else {
        msgSpan.setStatus({ code: SpanStatusCode.OK })
      }
      msgSpan.end(assistant.time.completed)
      ctx.messageSpans.delete(msgKey)
      ctx.messageOutputs.delete(msgKey)
      ctx.messageToolCalls.delete(msgKey)
      ctx.finalizedMessageSpans.add(msgKey)
    }
  }

  if (assistant.error) {
    ctx.emitLog({
      severityNumber: SeverityNumber.ERROR,
      severityText: "ERROR",
      timestamp: assistant.time.created,
      observedTimestamp: Date.now(),
      body: "api_error",
      attributes: {
        "event.name": "api_error",
        "session.id": sessionID,
        model: modelID,
        provider: providerID,
        agent,
        error: errorSummary(assistant.error),
        duration_ms: duration,
        ...ctx.commonAttrs,
      },
    })
    return ctx.log("error", "otel: api_error", {
      sessionID,
      model: modelID,
      agent,
      error: errorSummary(assistant.error),
      duration_ms: duration,
    })
  }

  ctx.emitLog({
    severityNumber: SeverityNumber.INFO,
    severityText: "INFO",
    timestamp: assistant.time.created,
    observedTimestamp: Date.now(),
    body: "api_request",
    attributes: {
      "event.name": "api_request",
      "session.id": sessionID,
      model: modelID,
      provider: providerID,
      agent,
      cost_usd: assistant.cost,
      duration_ms: duration,
      input_tokens: assistant.tokens.input,
      output_tokens: assistant.tokens.output,
      reasoning_tokens: assistant.tokens.reasoning,
      cache_read_tokens: assistant.tokens.cache.read,
      cache_creation_tokens: assistant.tokens.cache.write,
      ...ctx.commonAttrs,
    },
  })
  return ctx.log("info", "otel: api_request", {
    sessionID,
    model: modelID,
    agent,
    cost_usd: assistant.cost,
    duration_ms: duration,
    input_tokens: assistant.tokens.input,
    output_tokens: assistant.tokens.output,
  })
}

/**
 * Tracks tool execution time between `running` and `completed`/`error` part updates,
 * records a `tool.duration` histogram measurement, manages the tool child span, and emits
 * a `tool_result` log event. Also handles `subtask` parts, incrementing the sub-agent
 * invocation counter and emitting a `subtask_invoked` log event.
 *
 * For tool spans: on `running` a child span of the current session span is started and stored
 * in `pendingToolSpans`. On `completed`/`error` the span is ended with appropriate status.
 * If no `running` event was seen (out-of-order), a best-effort span is started and immediately ended.
 */
export function handleMessagePartUpdated(e: EventMessagePartUpdated, ctx: HandlerContext) {
  const part = e.properties.part

  if (part.type === "step-start") {
    const stepStart = part as unknown as StepStartPart
    startMessageSpan(stepStart.sessionID, stepStart.messageID, "unknown", "unknown", Date.now(), ctx)
    return
  }

  if (part.type === "step-finish") {
    const stepFinish = part as unknown as StepFinishPart
    finalizeMessageSpanFromStep(stepFinish, ctx)
    return
  }

  if (part.type === "reasoning") {
    const reasoning = part as unknown as ReasoningPart
    const key = reasoningKey(reasoning.sessionID, reasoning.messageID, reasoning.id)
    const pending = ensureReasoningPending(reasoning.sessionID, reasoning.messageID, reasoning.id, reasoning.time.start, ctx)
    if (reasoning.text || !pending.text) {
      const next = boundedReasoningText(reasoning.text)
      pending.text = next.text
      pending.truncated = next.truncated
    }
    ensureReasoningOtelSpan(pending, ctx)
    pending.span?.setAttributes({
      "opencode.reasoning.text_length": pending.text.length,
      "opencode.reasoning.truncated": pending.truncated,
      "opencode.reasoning.has_metadata": !!reasoning.metadata,
    })
    if (reasoning.time.end !== undefined) {
      endReasoningSpan(key, pending, reasoning.time.end)
      ctx.pendingReasoningSpans.delete(key)
      ctx.log("debug", "otel: reasoning span ended", { sessionID: reasoning.sessionID, messageID: reasoning.messageID, partID: reasoning.id })
    }
    return
  }

  if (part.type === "text") {
    const key = messageKey(part.sessionID, part.messageID)
    ctx.messageOutputs.set(key, `${ctx.messageOutputs.get(key) ?? ""}${part.text}`)
    return
  }

  if (part.type === "subtask") {
    const subtask = part as unknown as SubtaskPart
    if (isMetricEnabled("subtask.count", ctx)) {
      ctx.instruments.subtaskCounter.add(1, {
        ...ctx.commonAttrs,
        "session.id": subtask.sessionID,
        agent: subtask.agent,
      })
    }
    ctx.emitLog({
      severityNumber: SeverityNumber.INFO,
      severityText: "INFO",
      timestamp: Date.now(),
      observedTimestamp: Date.now(),
      body: "subtask_invoked",
      attributes: {
        "event.name": "subtask_invoked",
        "session.id": subtask.sessionID,
        agent: subtask.agent,
        description: subtask.description,
        prompt_length: subtask.prompt.length,
        ...ctx.commonAttrs,
      },
    })
    return ctx.log("info", "otel: subtask_invoked", {
      sessionID: subtask.sessionID,
      agent: subtask.agent,
      description: subtask.description,
    })
  }

  if (part.type === "tool") {
    const toolPart = part as ToolPart
    const key = toolSpanKey(toolPart)
    recordMessageToolCall(toolPart, ctx)

    if (toolPart.state.status === "running") {
      const toolSpan = isTraceEnabled("tool", ctx)
        ? (() => {
            const parentCtx = parentContextForMessage(toolPart.sessionID, toolPart.messageID, ctx)
            return ctx.tracer.startSpan(
              `${ctx.tracePrefix}tool.${toolPart.tool}`,
              {
                startTime: toolPart.state.time.start,
                kind: SpanKind.INTERNAL,
                attributes: {
                  [OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.TOOL,
                  [SESSION_ID]: toolPart.sessionID,
                  [TOOL_ID]: toolPart.callID,
                  [TOOL_NAME]: toolPart.tool,
                  [TOOL_PARAMETERS]: JSON.stringify(toolPart.state.input),
                  [INPUT_VALUE]: JSON.stringify(toolPart.state.input),
                  [INPUT_MIME_TYPE]: MimeType.JSON,
                  ...ctx.commonAttrs,
                },
              },
              parentCtx,
            )
          })()
        : undefined
      setBoundedMap(ctx.pendingToolSpans, key, {
        tool: toolPart.tool,
        sessionID: toolPart.sessionID,
        messageID: toolPart.messageID,
        startMs: toolPart.state.time.start,
        span: toolSpan,
      })
      ctx.log("debug", "otel: tool span started", { sessionID: toolPart.sessionID, tool: toolPart.tool, key })
      return
    }

    if (toolPart.state.status !== "completed" && toolPart.state.status !== "error") return

    const pending = ctx.pendingToolSpans.get(key)
    ctx.pendingToolSpans.delete(key)
    const start = pending?.startMs ?? toolPart.state.time.start
    const end = toolPart.state.time.end
    if (end === undefined) return
    const duration_ms = end - start
    const success = toolPart.state.status === "completed"

    if (isMetricEnabled("tool.duration", ctx)) {
      ctx.instruments.toolDurationHistogram.record(duration_ms, {
        ...ctx.commonAttrs,
        "session.id": toolPart.sessionID,
        tool_name: toolPart.tool,
        success,
      })
    }

    if (isTraceEnabled("tool", ctx)) {
      const toolSpan = pending?.span ?? (() => {
        const parentCtx = parentContextForMessage(toolPart.sessionID, toolPart.messageID, ctx)
        return ctx.tracer.startSpan(
          `${ctx.tracePrefix}tool.${toolPart.tool}`,
          {
            startTime: start,
            kind: SpanKind.INTERNAL,
            attributes: {
              [OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.TOOL,
              [SESSION_ID]: toolPart.sessionID,
              [TOOL_ID]: toolPart.callID,
              [TOOL_NAME]: toolPart.tool,
              [TOOL_PARAMETERS]: JSON.stringify(toolPart.state.input),
              [INPUT_VALUE]: JSON.stringify(toolPart.state.input),
              [INPUT_MIME_TYPE]: MimeType.JSON,
              ...ctx.commonAttrs,
            },
          },
          parentCtx,
        )
      })()
      toolSpan.setAttribute("tool.success", success)
      if (success) {
        const output = (toolPart.state as { output: string }).output
        toolSpan.setAttributes({
          [OUTPUT_VALUE]: output,
          [OUTPUT_MIME_TYPE]: MimeType.TEXT,
        })
        toolSpan.setAttribute("tool.result_size_bytes", Buffer.byteLength(output, "utf8"))
        toolSpan.setStatus({ code: SpanStatusCode.OK })
      } else {
        const err = (toolPart.state as { error: string }).error
        toolSpan.setAttributes({
          [OUTPUT_VALUE]: err,
          [OUTPUT_MIME_TYPE]: MimeType.TEXT,
        })
        toolSpan.setAttribute("tool.error", err)
        toolSpan.setStatus({ code: SpanStatusCode.ERROR, message: err })
      }
      toolSpan.end(end)
    }

    const sizeAttr = success
      ? { tool_result_size_bytes: Buffer.byteLength((toolPart.state as { output: string }).output, "utf8") }
      : { error: (toolPart.state as { error: string }).error }

    ctx.emitLog({
      severityNumber: success ? SeverityNumber.INFO : SeverityNumber.ERROR,
      severityText: success ? "INFO" : "ERROR",
      timestamp: start,
      observedTimestamp: Date.now(),
      body: "tool_result",
      attributes: {
        "event.name": "tool_result",
        "session.id": toolPart.sessionID,
        tool_name: toolPart.tool,
        success,
        duration_ms,
        ...sizeAttr,
        ...ctx.commonAttrs,
      },
    })
    ctx.log("debug", "otel: tool.duration histogram recorded", {
      sessionID: toolPart.sessionID,
      tool_name: toolPart.tool,
      duration_ms,
      success,
    })
    return ctx.log(success ? "info" : "error", "otel: tool_result", {
      sessionID: toolPart.sessionID,
      tool_name: toolPart.tool,
      success,
      duration_ms,
    })
  }
}

function finalizeMessageSpanFromStep(step: StepFinishPart, ctx: HandlerContext) {
  const msgKey = messageKey(step.sessionID, step.messageID)
  if (ctx.finalizedMessageSpans.has(msgKey)) return
  const msgSpan = ctx.messageSpans.get(msgKey)
  if (!msgSpan) return

  const outputText = ctx.messageOutputs.get(msgKey)
  const toolCalls = ctx.messageToolCalls.get(msgKey) ?? []
  const inputTokens = step.tokens?.input ?? 0
  const outputTokens = step.tokens?.output ?? 0
  const reasoningTokens = step.tokens?.reasoning ?? 0
  const cacheRead = step.tokens?.cache?.read ?? 0
  const cacheWrite = step.tokens?.cache?.write ?? 0
  const totalTokens = step.tokens?.total ?? inputTokens + outputTokens + reasoningTokens + cacheRead + cacheWrite
  const cost = step.cost ?? 0
  msgSpan.setAttributes({
    [LLM_TOKEN_COUNT_PROMPT]: inputTokens,
    [LLM_TOKEN_COUNT_COMPLETION]: outputTokens,
    [LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING]: reasoningTokens,
    [LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ]: cacheRead,
    [LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE]: cacheWrite,
    [LLM_TOKEN_COUNT_TOTAL]: totalTokens,
    [LLM_FINISH_REASON]: step.reason ?? "stop",
    [LLM_COST_TOTAL]: cost,
    ...outputAttributes(outputText, toolCalls),
    cost_usd: cost,
  })
  msgSpan.setStatus({ code: SpanStatusCode.OK })
  msgSpan.end(Date.now())
  ctx.messageSpans.delete(msgKey)
  ctx.messageOutputs.delete(msgKey)
  ctx.messageToolCalls.delete(msgKey)
  ctx.finalizedMessageSpans.add(msgKey)
  ctx.log("debug", "otel: llm span finalized from step-finish", { sessionID: step.sessionID, messageID: step.messageID })
}

/**
 * Accumulates streamed reasoning text emitted by OpenCode before the full reasoning part
 * is written back. The span itself remains a single child observation of the LLM span.
 */
export function handleMessagePartDelta(e: EventMessagePartDelta, ctx: HandlerContext) {
  const { sessionID, messageID, partID, field, delta } = e.properties
  if (field !== "text") return
  const key = reasoningKey(sessionID, messageID, partID)
  const pending = ctx.pendingReasoningSpans.get(key)
  if (!pending) return
  const next = appendReasoningText(pending.text, delta)
  pending.text = next.text
  pending.truncated = pending.truncated || next.truncated
  ensureReasoningOtelSpan(pending, ctx)
  pending.span?.setAttributes({
    "opencode.reasoning.text_length": pending.text.length,
    "opencode.reasoning.truncated": pending.truncated,
  })
  ctx.log("debug", "otel: reasoning delta recorded", { sessionID, messageID, partID, key, deltaLength: delta.length })
}

/**
 * Starts an LLM span for an assistant message when it first appears in `message.updated`.
 * The span is parented to the session span and carries `gen_ai.*` semantic attributes for
 * the model and provider. It is ended in `handleMessageUpdated` once the message completes.
 *
 * Only called for assistant messages that have not yet completed (`time.completed` absent).
 */
export function startMessageSpan(
  sessionID: string,
  messageID: string,
  modelID: string,
  providerID: string,
  startTime: number,
  ctx: HandlerContext,
) {
  if (!isTraceEnabled("llm", ctx)) return
  const msgKey = `${sessionID}:${messageID}`
  if (ctx.messageSpans.has(msgKey)) return
  const sessionSpan = ctx.sessionSpans.get(sessionID)
  const baseCtx = ctx.rootContext()
  const parentCtx = sessionSpan
    ? trace.setSpan(baseCtx, sessionSpan)
    : baseCtx

  const msgSpan = ctx.tracer.startSpan(
    `${ctx.tracePrefix}llm`,
    {
      startTime,
      kind: SpanKind.CLIENT,
      attributes: {
        [OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
        [SESSION_ID]: sessionID,
        [AGENT_NAME]: ctx.sessionTotals.get(sessionID)?.agent ?? "unknown",
        [LLM_SYSTEM]: providerID,
        [LLM_PROVIDER]: providerID,
        [LLM_MODEL_NAME]: modelID,
        ...(ctx.sessionInputs.has(sessionID)
          ? {
              [INPUT_VALUE]: ctx.sessionInputs.get(sessionID)!,
              [INPUT_MIME_TYPE]: MimeType.TEXT,
              [LLM_INPUT_MESSAGES]: JSON.stringify([{ role: "user", content: ctx.sessionInputs.get(sessionID)! }]),
            }
          : {}),
        ...ctx.commonAttrs,
      },
    },
    parentCtx,
  )
  setBoundedMap(ctx.messageSpans, msgKey, msgSpan)
}
