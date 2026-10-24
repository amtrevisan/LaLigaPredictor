import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { I18nProvider } from "@/i18n";
import type { UpcomingFixture } from "@/types/data";
import { FixtureField } from "./FixtureField";

const team = (name: string) => ({
  name,
  short: name.slice(0, 3).toUpperCase(),
  crest: null,
  photos: [],
  stadium: null,
});

const fixture = (id: string, home: string, away: string): UpcomingFixture => ({
  id,
  round: 10,
  kickoff: "2026-10-25T19:00:00Z",
  home: team(home),
  away: team(away),
  probs: [0.49, 0.23, 0.28],
  pick: "H",
  market: [0.59, 0.17, 0.24],
  xg: [1.92, 1.31],
  components: { classifier: null, goals: null },
  elo: [1760, 1674],
  scorelines: [],
  meetings: [],
});

describe("FixtureField", () => {
  it("marks the selected fixture and reports clicks", () => {
    const onSelect = vi.fn();
    render(
      <I18nProvider locale="en">
        <FixtureField
          fixtures={[fixture("a", "Barcelona", "Real Madrid"), fixture("b", "Sevilla", "Osasuna")]}
          selectedId="a"
          onSelect={onSelect}
          describedBy="legend"
        />
      </I18nProvider>,
    );

    const first = screen.getByRole("button", { name: /Barcelona – Real Madrid/ });
    const second = screen.getByRole("button", { name: /Sevilla – Osasuna/ });
    expect(first).toHaveAttribute("aria-pressed", "true");
    expect(second).toHaveAttribute("aria-pressed", "false");

    fireEvent.click(second);
    expect(onSelect).toHaveBeenCalledWith("b");
  });

  it("names each fixture with teams, kick-off and probabilities, in the visitor's language", () => {
    render(
      <I18nProvider locale="es">
        <FixtureField
          fixtures={[fixture("a", "Barcelona", "Real Madrid")]}
          selectedId="a"
          onSelect={() => {}}
          describedBy="legend"
        />
      </I18nProvider>,
    );
    expect(
      screen.getByRole("button", {
        name: /Barcelona – Real Madrid\. .+\. Local 49\s?%, Empate 23\s?%, Visitante 28\s?%/,
      }),
    ).toBeInTheDocument();
  });
});
