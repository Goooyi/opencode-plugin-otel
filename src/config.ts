import type { Attributes, AttributeValue } from "@opentelemetry/api"
import { LEVELS, type Level } from "./types.ts"

/** Accepted values for `OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE`. */
export type MetricsTemporality = "cumulative" | "delta" | "lowmemory"

/** Valid trace types emitted by the plugin. */
export const TRACE_TYPES = ["session", "llm", "tool", "reasoning"] as const

const VALID_TEMPORALITIES: ReadonlySet<MetricsTemporality> = new Set<MetricsTemporality>(["cumulative", "delta", "lowmemory"])
const TRACE_DISABLE_ALL_VALUES = new Set(["all", "*", "true", "1"])
const TRACE_ATTRIBUTE_KEY = /^[a-zA-Z0-9_.-]{1,128}$/
const MAX_TRACE_ATTRIBUTES = 64
const MAX_TRACE_ATTRIBUTE_STRING_LENGTH = 1000
const MAX_TRACE_ATTRIBUTE_ARRAY_LENGTH = 32

/** Configuration values resolved from `OPENCODE_*` environment variables. */
export type PluginConfig = {
  enabled: boolean
  logsEnabled: boolean
  endpoint: string
  protocol: "grpc" | "http/protobuf" | "http/json"
  metricsInterval: number
  logsInterval: number
  metricPrefix: string
  otlpHeaders: string | undefined
  otlpHeadersHelper: string | undefined
  resourceAttributes: string | undefined
  traceAttributes: Attributes
  traceparent: string | undefined
  tracestate: string | undefined
  metricsTemporality: MetricsTemporality | undefined
  disabledMetrics: Set<string>
  disabledTraces: Set<string>
}

/** Parses a positive integer from an environment variable, returning `fallback` if absent or invalid. */
export function parseEnvInt(key: string, fallback: number): number {
  const raw = process.env[key]
  if (!raw) return fallback
  if (!/^[1-9]\d*$/.test(raw)) return fallback
  const n = Number(raw)
  return Number.isSafeInteger(n) ? n : fallback
}

/** Returns `true` when the environment variable is present and non-empty. */
function hasNonEmptyEnv(key: string): boolean {
  return !!process.env[key]
}

/** Parses `OPENCODE_DISABLE_TRACES`, expanding explicit global values like `all`. */
function parseDisabledTraces(raw: string | undefined): Set<string> {
  const values = (raw ?? "")
    .split(",")
    .map(s => s.trim().toLowerCase())
    .filter(Boolean)

  if (values.some(value => TRACE_DISABLE_ALL_VALUES.has(value))) {
    return new Set(TRACE_TYPES)
  }

  return new Set(values)
}

/** Parses safe span attributes from OPENCODE_TRACE_ATTRIBUTES JSON. */
export function parseTraceAttributes(raw: string | undefined): Attributes {
  if (!raw) return {}
  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch {
    console.warn("[opencode-plugin-otel] Invalid OPENCODE_TRACE_ATTRIBUTES JSON. Value ignored.")
    return {}
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return {}

  const attributes: Attributes = {}
  for (const [key, value] of Object.entries(parsed).slice(0, MAX_TRACE_ATTRIBUTES)) {
    if (!TRACE_ATTRIBUTE_KEY.test(key)) continue
    const normalized = normalizeTraceAttributeValue(value)
    if (normalized !== undefined) attributes[key] = normalized
  }
  return attributes
}

function normalizeTraceAttributeValue(value: unknown): AttributeValue | undefined {
  if (typeof value === "string") return truncateTraceAttributeString(value)
  if (typeof value === "number") return Number.isFinite(value) ? value : undefined
  if (typeof value === "boolean") return value
  if (!Array.isArray(value)) return undefined

  const normalized = value
    .slice(0, MAX_TRACE_ATTRIBUTE_ARRAY_LENGTH)
    .filter((item): item is string => typeof item === "string")
    .map(truncateTraceAttributeString)
  return normalized.length > 0 ? normalized : undefined
}

function truncateTraceAttributeString(value: string): string {
  return value.length > MAX_TRACE_ATTRIBUTE_STRING_LENGTH ? value.slice(0, MAX_TRACE_ATTRIBUTE_STRING_LENGTH) : value
}

/**
 * Reads all `OPENCODE_*` environment variables and returns the resolved plugin config.
 * Copies `OPENCODE_OTLP_HEADERS` → `OTEL_EXPORTER_OTLP_HEADERS`,
 * `OPENCODE_RESOURCE_ATTRIBUTES` → `OTEL_RESOURCE_ATTRIBUTES`, and
 * `OPENCODE_OTLP_METRICS_TEMPORALITY` → `OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE`
 * so the OTel SDK picks them up automatically when initialised.
 */
export function loadConfig(): PluginConfig {
  const otlpHeaders = process.env["OPENCODE_OTLP_HEADERS"]
  const otlpHeadersHelper = process.env["OPENCODE_OTLP_HEADERS_HELPER"]
  const resourceAttributes = process.env["OPENCODE_RESOURCE_ATTRIBUTES"]
  const traceAttributes = parseTraceAttributes(process.env["OPENCODE_TRACE_ATTRIBUTES"])
  const traceparent = process.env["OPENCODE_TRACEPARENT"]
  const tracestate = process.env["OPENCODE_TRACESTATE"]
  const rawTemporality = process.env["OPENCODE_OTLP_METRICS_TEMPORALITY"]
  const protocol = process.env["OPENCODE_OTLP_PROTOCOL"]

  let metricsTemporality: MetricsTemporality | undefined
  if (rawTemporality) {
    const normalized = rawTemporality.toLowerCase()
    if (VALID_TEMPORALITIES.has(normalized as MetricsTemporality)) {
      metricsTemporality = normalized as MetricsTemporality
      process.env["OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE"] = normalized
    } else {
      console.warn(
        `[opencode-plugin-otel] Invalid OPENCODE_OTLP_METRICS_TEMPORALITY="${rawTemporality}". ` +
          `Expected one of: cumulative, delta, lowmemory. Value ignored.`,
      )
    }
  }

  if (otlpHeaders) process.env["OTEL_EXPORTER_OTLP_HEADERS"] = otlpHeaders
  if (resourceAttributes) process.env["OTEL_RESOURCE_ATTRIBUTES"] = resourceAttributes

  const disabledMetrics = new Set(
    (process.env["OPENCODE_DISABLE_METRICS"] ?? "")
      .split(",")
      .map(s => s.trim())
      .filter(Boolean),
  )

  const disabledTraces = parseDisabledTraces(process.env["OPENCODE_DISABLE_TRACES"])

  return {
    enabled: hasNonEmptyEnv("OPENCODE_ENABLE_TELEMETRY"),
    logsEnabled: !hasNonEmptyEnv("OPENCODE_DISABLE_LOGS"),
    endpoint: process.env["OPENCODE_OTLP_ENDPOINT"] ?? "http://localhost:4317",
    protocol: protocol === "http/protobuf"
      ? "http/protobuf"
      : protocol === "http/json"
        ? "http/json"
        : "grpc",
    metricsInterval: parseEnvInt("OPENCODE_OTLP_METRICS_INTERVAL", 60000),
    logsInterval: parseEnvInt("OPENCODE_OTLP_LOGS_INTERVAL", 5000),
    metricPrefix: process.env["OPENCODE_METRIC_PREFIX"] ?? "opencode.",
    otlpHeaders,
    otlpHeadersHelper,
    resourceAttributes,
    traceAttributes,
    traceparent,
    tracestate,
    metricsTemporality,
    disabledMetrics,
    disabledTraces,
  }
}

export function resolveHelperPath(
  helper: string | undefined,
  directory: string | undefined,
  worktree: string | undefined,
): string | undefined {
  if (!helper) return helper
  const projectRoot = worktree ?? directory ?? process.cwd()
  return helper
    .replaceAll("${PROJECT_ROOT}", projectRoot)
    .replaceAll("${WORKTREE}", worktree ?? projectRoot)
    .replaceAll("${DIRECTORY}", directory ?? projectRoot)
}

/**
 * Resolves an opencode log level string to a `Level`.
 * Returns `current` unchanged when the input does not match a known level.
 */
export function resolveLogLevel(logLevel: string, current: Level): Level {
  const candidate = logLevel.toLowerCase()
  if (candidate in LEVELS) return candidate as Level
  return current
}
