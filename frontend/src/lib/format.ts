import type { Outcome, Triple } from "@/types/data";

/** Replace `{name}` placeholders. Unknown placeholders are left in place so they're visible. */
export function fill(template: string, params: Record<string, string | number>): string {
  return template.replace(/\{(\w+)\}/g, (match, key: string) =>
    key in params ? String(params[key]) : match,
  );
}

// Intl formatters are expensive to construct and a results page formats thousands of values,
// so each (locale, options) pair is built once.
const numberFormats = new Map<string, Intl.NumberFormat>();
const dateFormats = new Map<string, Intl.DateTimeFormat>();

function numberFormat(intl: string, options: Intl.NumberFormatOptions): Intl.NumberFormat {
  const key = `${intl}|${JSON.stringify(options)}`;
  let format = numberFormats.get(key);
  if (!format) {
    format = new Intl.NumberFormat(intl, options);
    numberFormats.set(key, format);
  }
  return format;
}

function dateFormat(intl: string, options: Intl.DateTimeFormatOptions): Intl.DateTimeFormat {
  const key = `${intl}|${JSON.stringify(options)}`;
  let format = dateFormats.get(key);
  if (!format) {
    format = new Intl.DateTimeFormat(intl, options);
    dateFormats.set(key, format);
  }
  return format;
}

export function percent(p: number, intl: string, digits = 0): string {
  return numberFormat(intl, {
    style: "percent",
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(p);
}

export function decimal(value: number, intl: string, digits = 3): string {
  return numberFormat(intl, { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(
    value,
  );
}

export function integer(value: number, intl: string): string {
  return numberFormat(intl, { maximumFractionDigits: 0 }).format(value);
}

/** Signed percentage-point difference, e.g. "+4" or "−2". */
export function points(delta: number, intl: string): string {
  const value = Math.round(delta * 100);
  return numberFormat(intl, { signDisplay: "exceptZero" }).format(value).replace("-", "−");
}

export function kickoffDay(iso: string, intl: string): string {
  return dateFormat(intl, { weekday: "short", day: "numeric", month: "short" }).format(new Date(iso));
}

export function kickoffTime(iso: string, intl: string): string {
  return dateFormat(intl, { hour: "2-digit", minute: "2-digit" }).format(new Date(iso));
}

export function longDate(iso: string, intl: string): string {
  return dateFormat(intl, { day: "numeric", month: "short", year: "numeric" }).format(new Date(iso));
}

export function dateTime(iso: string, intl: string): string {
  return dateFormat(intl, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }).format(
    new Date(iso),
  );
}

export const OUTCOME_INDEX: Record<Outcome, 0 | 1 | 2> = { H: 0, D: 1, A: 2 };

export function probabilityOf(probs: Triple, outcome: Outcome): number {
  return probs[OUTCOME_INDEX[outcome]];
}
