import { memo, useMemo } from "react";
import { useI18n } from "@/i18n";
import { probabilityOf } from "@/lib/format";
import type { PlayedFixture } from "@/types/data";
import styles from "./SeasonBarcode.module.css";

interface Props {
  /** Played fixtures, any order; drawn oldest to newest. */
  results: PlayedFixture[];
  limit?: number;
  height?: number;
  label: string;
}

const BAR = 3;
const GAP = 1;

/**
 * The season as a barcode: one bar per match, height = probability the model gave to what
 * actually happened. Solid bars were called correctly, outlined bars were misses.
 */
export const SeasonBarcode = memo(function SeasonBarcode({ results, limit, height = 64, label }: Props) {
  const { t } = useI18n();
  const shown = useMemo(() => {
    const ordered = [...results].sort((a, b) => a.kickoff.localeCompare(b.kickoff));
    return limit ? ordered.slice(-limit) : ordered;
  }, [results, limit]);
  const width = Math.max(shown.length * (BAR + GAP) - GAP, 1);
  const hits = shown.filter((r) => r.correct).length;

  return (
    <figure className={styles.figure}>
      <svg
        className={styles.svg}
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        role="img"
        aria-label={`${label}: ${hits} ${t.results.hit} / ${shown.length - hits} ${t.results.miss}`}
      >
        <line x1="0" x2={width} y1={height - 0.5} y2={height - 0.5} className={styles.base} />
        {shown.map((r, i) => {
          const p = probabilityOf(r.probs, r.result.outcome);
          const h = Math.max(p * height, 1);
          const x = i * (BAR + GAP);
          return r.correct ? (
            <rect key={r.id} x={x} y={height - h} width={BAR} height={h} className={styles.hit} />
          ) : (
            <rect
              key={r.id}
              x={x + 0.5}
              y={height - h + 0.5}
              width={BAR - 1}
              height={Math.max(h - 1, 0.5)}
              className={styles.miss}
            />
          );
        })}
      </svg>
    </figure>
  );
});
