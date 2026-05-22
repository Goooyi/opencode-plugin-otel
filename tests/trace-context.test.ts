import { describe, expect, test } from "bun:test"
import { context, ROOT_CONTEXT, trace, TraceFlags } from "@opentelemetry/api"
import { remoteParentContext } from "../src/trace-context.ts"

describe("remoteParentContext", () => {
  test("returns undefined when traceparent is absent", () => {
    expect(remoteParentContext(undefined, undefined)).toBeUndefined()
  })

  test("returns undefined for malformed traceparent", () => {
    expect(remoteParentContext("not-a-traceparent", undefined)).toBeUndefined()
  })

  test("does not inherit the active context for malformed traceparent", () => {
    const active = trace.setSpanContext(ROOT_CONTEXT, {
      traceId: "0af7651916cd43dd8448eb211c80319c",
      spanId: "b7ad6b7169203331",
      traceFlags: TraceFlags.SAMPLED,
    })
    const ctx = context.with(active, () => remoteParentContext("not-a-traceparent", undefined))
    expect(ctx).toBeUndefined()
  })

  test("returns a remote sampled parent context", () => {
    const ctx = remoteParentContext(
      "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
      "vendor=value",
    )
    const spanCtx = ctx ? trace.getSpanContext(ctx) : undefined
    expect(spanCtx?.traceId).toBe("0af7651916cd43dd8448eb211c80319c")
    expect(spanCtx?.spanId).toBe("b7ad6b7169203331")
    expect(spanCtx?.traceFlags).toBe(TraceFlags.SAMPLED)
    expect(spanCtx?.isRemote).toBe(true)
    expect(spanCtx?.traceState?.serialize()).toBe("vendor=value")
  })

  test("rejects zero trace and span ids", () => {
    expect(remoteParentContext("00-00000000000000000000000000000000-b7ad6b7169203331-01", undefined)).toBeUndefined()
    expect(remoteParentContext("00-0af7651916cd43dd8448eb211c80319c-0000000000000000-01", undefined)).toBeUndefined()
  })

  test("rejects version 00 with trailing data", () => {
    expect(remoteParentContext("00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01-extra", undefined)).toBeUndefined()
  })

  test("accepts trailing data for future versions", () => {
    const ctx = remoteParentContext("01-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01-extra", undefined)
    expect(ctx ? trace.getSpanContext(ctx)?.traceId : undefined).toBe("0af7651916cd43dd8448eb211c80319c")
  })

  test("preserves parsed trace flags", () => {
    const ctx = remoteParentContext("00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-03", undefined)
    expect(ctx ? trace.getSpanContext(ctx)?.traceFlags : undefined).toBe(3)
  })
})
