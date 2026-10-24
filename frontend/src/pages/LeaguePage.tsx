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
  if (league.isPending) return <FieldSkeleton />;
  if (league.isError) return <ErrorState onRetry={() => void league.refetch()} />;

  const View = VIEWS[view];
  return <View data={league.data} />;
}
