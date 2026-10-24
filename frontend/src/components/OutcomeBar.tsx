import { useI18n } from "@/i18n";
import { percent } from "@/lib/format";
import type { Triple } from "@/types/data";
import styles from "./OutcomeBar.module.css";

interface Props {
  probs: Triple;
  orientation?: "vertical" | "horizontal";
  /** Thin variant for tables and inline comparisons. */
  compact?: boolean;
  className?: string;
}

/**
 * Home / draw / away as one stacked bar. Outcomes are told apart by pattern, never colour:
 * home is solid, draw is a hairline barcode, away is an outline. Every segment is drawn in
 * `currentColor`, so the bar inverts with its container.
 */
export function OutcomeBar({ probs, orientation = "vertical", compact = false, className }: Props) {
  const { t, intl } = useI18n();
  const [home, draw, away] = probs;
  const label = `${t.outcome.H} ${percent(home, intl)}, ${t.outcome.D} ${percent(draw, intl)}, ${t.outcome.A} ${percent(away, intl)}`;
  return (
    <span
      role="img"
      aria-label={label}
      className={[styles.bar, styles[orientation], compact && styles.compact, className]
        .filter(Boolean)
        .join(" ")}
    >
      <span className={styles.home} style={{ flexGrow: home }} />
      <span className={styles.draw} style={{ flexGrow: draw }} />
      <span className={styles.away} style={{ flexGrow: away }} />
    </span>
  );
}

const SEGMENT = { H: styles.home, D: styles.draw, A: styles.away } as const;

export function OutcomeKey() {
  const { t } = useI18n();
  return (
    <ul className={styles.key}>
      {(["H", "D", "A"] as const).map((o) => (
        <li key={o}>
          <span className={`${styles.swatch} ${SEGMENT[o]}`} />
          {t.outcome[o]}
        </li>
      ))}
    </ul>
  );
}
