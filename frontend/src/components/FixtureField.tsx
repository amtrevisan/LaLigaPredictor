import { useI18n } from "@/i18n";
import { decimal, kickoffDay, kickoffTime, percent } from "@/lib/format";
import type { UpcomingFixture } from "@/types/data";
import { Crest } from "./Crest";
import { OutcomeBar } from "./OutcomeBar";
import styles from "./FixtureField.module.css";

interface Props {
  fixtures: UpcomingFixture[];
  selectedId: string | undefined;
  onSelect: (id: string) => void;
  detailId: string;
}

/**
 * The round as one field of columns, one per fixture. Selecting a column inverts it (the
 * signature move) and drives the detail band below.
 */
export function FixtureField({ fixtures, selectedId, onSelect, detailId }: Props) {
  const { t, intl } = useI18n();
  return (
    <ol className={styles.field} style={{ "--columns": Math.min(fixtures.length, 10) } as React.CSSProperties}>
      {fixtures.map((fixture) => {
        const selected = fixture.id === selectedId;
        return (
          <li key={fixture.id}>
            <button
              type="button"
              className={styles.column}
              aria-expanded={selected}
              aria-controls={detailId}
              onClick={() => onSelect(fixture.id)}
            >
              <span className="visually-hidden">
                {fixture.home.name} – {fixture.away.name}
              </span>
              <span className={styles.when} aria-hidden="true">
                <span>{kickoffDay(fixture.kickoff, intl)}</span>
                <span>{kickoffTime(fixture.kickoff, intl)}</span>
              </span>
              <span className={styles.teams} aria-hidden="true">
                <span className={styles.team}>
                  <Crest team={fixture.home} size={26} />
                  <b>{fixture.home.short}</b>
                </span>
                <span className={styles.team}>
                  <Crest team={fixture.away} size={26} />
                  <b>{fixture.away.short}</b>
                </span>
              </span>
              <OutcomeBar probs={fixture.probs} className={styles.bar} />
              <span className={styles.numbers}>
                {(["H", "D", "A"] as const).map((o, i) => (
                  <span key={o} className={fixture.pick === o ? styles.pick : undefined}>
                    <span aria-hidden="true">{t.outcome.short[o]}</span>
                    <span>{percent(fixture.probs[i] ?? 0, intl)}</span>
                  </span>
                ))}
              </span>
              {fixture.xg && (
                <span className={styles.xg}>
                  {t.fixtures.xg} {decimal(fixture.xg[0], intl, 2)}–{decimal(fixture.xg[1], intl, 2)}
                </span>
              )}
            </button>
          </li>
        );
      })}
    </ol>
  );
}
