import { describe, expect, it } from "vitest";
import { fill, percent, points, probabilityOf } from "./format";

describe("fill", () => {
  it("replaces known placeholders and keeps unknown ones visible", () => {
    expect(fill("Round {n} of {total}", { n: 3 })).toBe("Round 3 of {total}");
  });
});

describe("percent", () => {
  it("formats per locale", () => {
    expect(percent(0.462, "en-GB")).toBe("46%");
    expect(percent(0.462, "fr-FR")).toMatch(/^46\s?%$/);
    expect(percent(0.4625, "de-DE", 1)).toMatch(/^46,3\s?%$/);
  });
});

describe("points", () => {
  it("signs differences in percentage points with a true minus", () => {
    expect(points(0.034, "en-GB")).toBe("+3");
    expect(points(-0.051, "en-GB")).toBe("−5");
    expect(points(0.001, "en-GB")).toBe("0");
  });
});

describe("probabilityOf", () => {
  it("reads the probability of an outcome", () => {
    expect(probabilityOf([0.5, 0.3, 0.2], "D")).toBe(0.3);
    expect(probabilityOf([0.5, 0.3, 0.2], "A")).toBe(0.2);
  });
});
