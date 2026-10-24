import { describe, expect, it } from "vitest";
import { LOCALES, detectLocale } from ".";

type Tree = { [key: string]: string | readonly string[] | Tree };

function leaves(tree: Tree, prefix = ""): Map<string, string> {
  const out = new Map<string, string>();
  for (const [key, value] of Object.entries(tree)) {
    const path = prefix ? `${prefix}.${key}` : key;
    if (typeof value === "string") out.set(path, value);
    else if (Array.isArray(value)) value.forEach((v, i) => out.set(`${path}.${i}`, v as string));
    else for (const [k, v] of leaves(value as Tree, path)) out.set(k, v);
  }
  return out;
}

const placeholders = (s: string) => [...s.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort();

describe("locales", () => {
  const reference = leaves(LOCALES.en.messages as unknown as Tree);
  const others = Object.keys(LOCALES).filter((l) => l !== "en") as (keyof typeof LOCALES)[];

  it.each(others)("%s has every key, non-empty, with the same placeholders", (code) => {
    const messages = leaves(LOCALES[code].messages as unknown as Tree);
    expect([...messages.keys()].sort()).toEqual([...reference.keys()].sort());
    for (const [key, english] of reference) {
      const translated = messages.get(key) ?? "";
      expect(translated.trim(), key).not.toBe("");
      expect(placeholders(translated), key).toEqual(placeholders(english));
    }
  });
});

describe("detectLocale", () => {
  it("matches on the base language", () => {
    expect(detectLocale(["pt-BR", "en-US"])).toBe("pt");
    expect(detectLocale(["de-AT"])).toBe("de");
  });

  it("falls back to English", () => {
    expect(detectLocale(["ja-JP", "nl"])).toBe("en");
    expect(detectLocale([])).toBe("en");
  });
});
