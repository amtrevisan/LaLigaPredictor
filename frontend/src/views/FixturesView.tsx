import { useSearchParams } from "react-router";
import { FixtureDetail } from "@/components/FixtureDetail";
import { FixtureField } from "@/components/FixtureField";
import { OutcomeKey } from "@/components/OutcomeBar";
import { RecordRail } from "@/components/RecordRail";
import { EmptyState } from "@/components/States";
import { useI18n } from "@/i18n";
import { fill } from "@/lib/format";
import type { LeagueData } from "@/types/data";
import styles from "./views.module.css";

const DETAIL_ID = "fixture-detail";
const LEGEND_ID = "fixture-legend";

export function FixturesView({ data }: { data: LeagueData }) {
  const { t } = useI18n();
  const [params, setParams] = useSearchParams();
  const fixtures = data.upcoming;
  const selected = fixtures.find((f) => f.id === params.get("match")) ?? fixtures[0];
  const rounds = [...new Set(fixtures.map((f) => f.round))].sort((a, b) => a - b);
  const roundLabel =
    rounds.length > 1
      ? fill(t.fixtures.round, { n: `${rounds[0]}–${rounds[rounds.length - 1]}` })
      : fill(t.fixtures.round, { n: rounds[0] ?? "" });

  return (
    <div className={styles.withRail}>
      <div className={styles.primary}>
        <header className={styles.viewHead}>
          <h1 className={styles.viewTitle}>
            {data.league.name} <span>{fixtures.length ? roundLabel : data.season}</span>
          </h1>
          {fixtures.length > 0 && (
            <div className={styles.viewMeta}>
              <span>{fill(t.fixtures.count, { n: fixtures.length })}</span>
              <OutcomeKey />
            </div>
          )}
        </header>
        {fixtures.length === 0 || !selected ? (
          <EmptyState>{t.fixtures.empty}</EmptyState>
        ) : (
          <>
            <p id={LEGEND_ID} className="visually-hidden">
              {t.fixtures.legend}
            </p>
            <FixtureField
              fixtures={fixtures}
              selectedId={selected.id}
              describedBy={LEGEND_ID}
              onSelect={(id) =>
                setParams(
                  (current) => {
                    const next = new URLSearchParams(current);
                    next.set("match", id);
                    return next;
                  },
                  { replace: true, preventScrollReset: true },
                )
              }
            />
            {/* Persistent live region: announces the newly selected fixture, not the whole band. */}
            <p className="visually-hidden" role="status">
              {selected.home.name} – {selected.away.name}
            </p>
            <FixtureDetail key={selected.id} fixture={selected} id={DETAIL_ID} />
          </>
        )}
      </div>
      <RecordRail leagueId={data.league.id} season={data.season} record={data.record} results={data.results} />
    </div>
  );
}
