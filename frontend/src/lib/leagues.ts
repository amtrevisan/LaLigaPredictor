import type { Locale } from "@/i18n";

export const LEAGUE_IDS = [
  "premier-league",
  "la-liga",
  "serie-a",
  "bundesliga",
  "ligue-1",
  "brasileirao",
] as const;

export type LeagueId = (typeof LEAGUE_IDS)[number];

export function isLeagueId(value: string | undefined): value is LeagueId {
  return (LEAGUE_IDS as readonly string[]).includes(value ?? "");
}

/** A first-time visitor lands on the league that speaks their language. */
const HOME_LEAGUE: Record<Locale, LeagueId> = {
  en: "premier-league",
  es: "la-liga",
  it: "serie-a",
  de: "bundesliga",
  fr: "ligue-1",
  pt: "brasileirao",
};

export function homeLeague(locale: Locale): LeagueId {
  return HOME_LEAGUE[locale];
}
