import { useQuery } from "@tanstack/react-query";
import type { LeagueData, Manifest, ModelReport } from "@/types/data";

const BASE = `${import.meta.env.BASE_URL}data`;

export class DataError extends Error {
  constructor(
    readonly path: string,
    readonly status: number,
  ) {
    super(`GET ${path} failed with ${status}`);
  }
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${BASE}/${path}`, { headers: { Accept: "application/json" } });
  if (!response.ok) throw new DataError(path, response.status);
  return (await response.json()) as T;
}

// The data changes a few times a day at most; keep it for the session.
const STALE = 10 * 60 * 1000;

export function useManifest() {
  return useQuery({ queryKey: ["manifest"], queryFn: () => getJson<Manifest>("manifest.json"), staleTime: STALE });
}

export function useLeague(id: string | undefined) {
  return useQuery({
    queryKey: ["league", id],
    queryFn: () => getJson<LeagueData>(`leagues/${id}.json`),
    enabled: Boolean(id),
    staleTime: STALE,
  });
}

export function useModelReport() {
  return useQuery({ queryKey: ["model"], queryFn: () => getJson<ModelReport>("model.json"), staleTime: STALE });
}
