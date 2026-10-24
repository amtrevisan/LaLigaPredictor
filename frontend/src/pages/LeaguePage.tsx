import { useParams } from "react-router";
import { ErrorState, FieldSkeleton } from "@/components/States";
import { useLeague } from "@/lib/api";
import { isLeagueId } from "@/lib/leagues";
import { FixturesView } from "@/views/FixturesView";
import { ResultsView } from "@/views/ResultsView";
import { TableView } from "@/views/TableView";
import { NotFound } from "./NotFound";

const VIEWS = { fixtures: FixturesView, results: ResultsView, table: TableView } as const;

export function LeaguePage({ view }: { view: keyof typeof VIEWS }) {
  const { leagueId } = useParams();
  const valid = isLeagueId(leagueId);
  const league = useLeague(valid ? leagueId : undefined);

  if (!valid) return <NotFound />;
  // A failed background refetch keeps showing the data we already have.
  if (!league.data) {
    return league.isError ? <ErrorState onRetry={() => void league.refetch()} /> : <FieldSkeleton />;
  }

  const View = VIEWS[view];
  // Keyed by league so view-local state (round filter, selection) resets on a league switch.
  return <View key={leagueId} data={league.data} />;
}
