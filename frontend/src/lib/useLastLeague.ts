import { useEffect } from "react";
import { isLeagueId, type LeagueId } from "./leagues";

const STORAGE_KEY = "matchfield.league";

function read(): LeagueId | undefined {
  try {
    const value = window.localStorage.getItem(STORAGE_KEY) ?? undefined;
    return isLeagueId(value) ? value : undefined;
  } catch {
    return undefined;
  }
}

/** The league currently in view, or the last one visited (so the Model page keeps context). */
export function useLastLeague(current: LeagueId | undefined): LeagueId | undefined {
  useEffect(() => {
    if (!current) return;
    try {
      window.localStorage.setItem(STORAGE_KEY, current);
    } catch {
      // Not persisted; only affects which league the Model page links back to.
    }
  }, [current]);

  return current ?? read();
}
